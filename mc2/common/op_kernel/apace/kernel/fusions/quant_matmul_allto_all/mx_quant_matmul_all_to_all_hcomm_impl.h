/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mx_quant_matmul_all_to_all_hcomm_impl.h
 * \brief QuantMatmul AllToAll — 计算→通信融合实现（纯C核通信），带流水掩盖
 *
 * 流水掩盖机制（与主线pipeline类似）：
 *   - Init() 中提前异步下发所有 AllToAll 通信任务（AlltoAll<true> → handle）
 *   - Run() 中逐 tile 循环：计算 tile N 的同时，上一轮 tile N-1 的通信在后台并行执行
 *   - 通信数据为 mmResult（matmul 输出），通信完成后直接写入 y
 *
 * 数据流方向：MatMul → AllToAll（与 allto_all_matmul 相反）
 *   - 无 scale 通信：scale 在 matmul 阶段本地消费
 *   - 通信数据类型为 YType（matmul 输出）
 *   - 无 splitK：每卡本地完整 K 维度
 *   - matmul 按 tile 分批调用 Blaze kernel，每次计算一个 M tile
 *
 * Workspace 布局: [mmResult (rankM * rankN * sizeof(YType))]
 * AllToAll: sendBuffer=mmResult, recvBuffer=y
 *
 * \note 本实现仅启动 AIC 核（cube-only）。
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h"
#include "include/tensor_api/tensor.h"

namespace Apace {

// PostComputeCommPolicy: 在Blaze kernel内部每个tile计算开始前，
// 异步下发上一个tile的AllToAll通信（上一个tile的matmul结果已写完到GM）
// 同时Wait上上个tile的AllToAll（pipeline深度2），实现逐tile通算掩盖
template <typename HcclType, typename YType, typename TilingDataType>
struct PostComputeCommPolicy {
    HcclType *hccl_{nullptr};
    TilingDataType *tilingData_{nullptr};
    GM_ADDR mmResultGM_{0};
    GM_ADDR y_{0};
    uint64_t rankDim_{0};
    uint64_t nonSplitAxisSize_{0};
    uint64_t yHcclDataType_{0};
    // pipeline深度2: 当前handle和前一个handle
    AscendC::HcclHandle curHandle_{0};
    uint32_t lastCommTileIdx_{0xFFFFFFFF}; // 去重: 已处理的最大tileIdx
    uint32_t lastIssuedTileIdx_{0xFFFFFFFF}; // 已下发AllToAll的最大tileIdx

    __aicore__ inline uint64_t CalcTileOffset(uint32_t tileIdx)
    {
        // 累计偏移: 头块部分 + 尾块部分
        auto &&ct = tilingData_->commTilingData;
        uint32_t headCnt = ct.splitAxisTileCnt;
        if (tileIdx < headCnt) {
            return static_cast<uint64_t>(tileIdx) * ct.splitAxisTileSize;
        } else {
            return static_cast<uint64_t>(headCnt) * ct.splitAxisTileSize +
                   static_cast<uint64_t>(tileIdx - headCnt) * ct.splitAxisTailSize;
        }
    }

    __aicore__ inline uint32_t CalcTileM(uint32_t tileIdx)
    {
        auto &&ct = tilingData_->commTilingData;
        return (tileIdx < ct.splitAxisTileCnt) ? ct.splitAxisTileSize : ct.splitAxisTailSize;
    }

    __aicore__ inline void IssueAllToAll(uint32_t tileIdx)
    {
        uint32_t tileM = CalcTileM(tileIdx);
        uint64_t perRankN = nonSplitAxisSize_ / rankDim_;
        uint64_t tileOffset = CalcTileOffset(tileIdx);
        // 逐行发送: sendCount=perRankN(每rank的数据量), strideCount=perRankN(行内相邻rank间距), batchCnt=tileM(行数)
        curHandle_ = hccl_->template AlltoAll<true>(
            mmResultGM_ + tileOffset * nonSplitAxisSize_ * sizeof(YType),
            y_ + tileOffset * perRankN * sizeof(YType),
            perRankN, static_cast<AscendC::HcclDataType>(yHcclDataType_),
            perRankN, static_cast<uint8_t>(tileM));
        lastIssuedTileIdx_ = tileIdx;
    }

    // 记录已下发AllToAll的最大tileIdx
    __aicore__ inline void WaitTile(uint32_t tileIdx)
    {
        // 去重: 同一tileIdx只处理一次
        if (tileIdx <= lastCommTileIdx_) {
            return;
        }

        // WaitTile(tileIdx)下发tileIdx-1的AllToAll(异步,不Wait)
        // 不能在此处Wait——Blaze kernel内部未做SyncAll,rank间可能不同步,
        // Wait会导致快的rank等待慢的rank下发AllToAll,形成死锁
        if (tileIdx > 0) {
            uint32_t prevTileIdx = tileIdx - 1;
            // 主循环(连续递增): 下发prevTileIdx的AllToAll
            if (prevTileIdx == lastCommTileIdx_ || lastCommTileIdx_ == 0xFFFFFFFF) {
                IssueAllToAll(prevTileIdx);
            }
            // 末尾循环(跳跃): 跳过,在PrepareLastTileAllToAll中处理
        }
        lastCommTileIdx_ = tileIdx;
    }
};

// 获取 YType 对应的 HCCL 数据类型
template <typename T>
constexpr uint64_t GetYHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        return AscendC::HCCL_DATA_TYPE_FP16;
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP32;
    }
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
class QuantMatmulAllToAllHcommImpl {
public:
    explicit __aicore__ inline QuantMatmulAllToAllHcommImpl(QuantMatmulAllToAllTilingDataType *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义 — 使用PostComputeCommPolicy，逐tile通算掩盖
    using CommPolicy = PostComputeCommPolicy<AscendC::Hccl<ServerType>, YType, QuantMatmulAllToAllTilingDataType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::AllToAllQbmmMxKernel<
        ProblemShape, BlockMmad, BlockScheduler, CommPolicy>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    CommPolicy commPolicy_;

private:
    QuantMatmulAllToAllTilingDataType *tilingData_;

    GM_ADDR x1_;
    GM_ADDR x2_;
    GM_ADDR y_;
    GM_ADDR bias_;
    GM_ADDR x1Scale_;
    GM_ADDR x2Scale_;
    GM_ADDR workspaceGM_;
    GM_ADDR mmResultGM_;  // matmul 输出缓冲区

    AscendC::Hccl<ServerType> hccl_;
    uint32_t rankId_{0};
    uint32_t rankDim_{0};
    uint64_t splitAxisSize_{0};  // 本卡 M 大小（= rankM = M/rankDim）
    uint32_t totalTileCnt_{0};   // 总 tile 数 = tileCnt + tailCnt
    uint32_t headTileCnt_{0};    // 头块 tile 数
    bool isBias_{false};

    static constexpr uint64_t yHcclDataType_ = GetYHcclDataType<YType>();
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    // 流水掩盖：逐 tile 计算+通信交替
    __aicore__ inline void SetupTileParams(Params &out, uint32_t tileIdx, uint32_t curTileM, GM_ADDR aGMOffset,
                                           GM_ADDR cGMOffset, GM_ADDR scaleAOffset);
    __aicore__ inline void PrepareLastTileAllToAll();
};

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM)
{
    auto &&commTiling = tilingData_->commTilingData;
    splitAxisSize_ = static_cast<uint64_t>(commTiling.splitAxisTileSize) * commTiling.splitAxisTileCnt +
                     static_cast<uint64_t>(commTiling.splitAxisTailSize) * commTiling.splitAxisTailCnt;
    headTileCnt_ = commTiling.splitAxisTileCnt;
    totalTileCnt_ = commTiling.splitAxisTileCnt + commTiling.splitAxisTailCnt;

    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;

    // workspace 布局: mmResult 用于存储 matmul 输出
    mmResultGM_ = workspaceGM;

    // 初始化 HCCL
    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(QuantMatmulAllToAllTilingDataType, mc2CcTiling)));
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    // 配置PostComputeCommPolicy：绑定hccl、tilingData、GM地址
    // Blaze kernel内部逐tile调用WaitTile时，自动下发上一个tile的AllToAll
    commPolicy_.hccl_ = &hccl_;
    commPolicy_.tilingData_ = tilingData_;
    commPolicy_.mmResultGM_ = mmResultGM_;
    commPolicy_.y_ = y_;
    commPolicy_.rankDim_ = rankDim_;
    commPolicy_.nonSplitAxisSize_ = tilingData_->commTilingData.nonSplitAxisSize;
    commPolicy_.yHcclDataType_ = yHcclDataType_;
    quantMatmulKernelImpl_.GetCommPolicy() = commPolicy_;
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::PrepareLastTileAllToAll()
{
    // Blaze kernel完成后，下发剩余未下发的tile通信
    // 主循环中已通过WaitTile下发了部分tile的AllToAll(异步,未Wait)
    // 这里统一下发剩余tile并Wait所有通信
    auto &&commTiling = tilingData_->commTilingData;

    // 下发最后一个tile的AllToAll（WaitTile只下发tileIdx-1,最后一个tile无后续触发）
    uint32_t lastTileIdx = totalTileCnt_ - 1;
    if (lastTileIdx != commPolicy_.lastIssuedTileIdx_) {
        commPolicy_.IssueAllToAll(lastTileIdx);
    }

    // Wait所有已下发的AllToAll
    // 注意: 所有rank必须先完成下发(SyncAll保证),再统一Wait,避免死锁
    AscendC::SyncAll();
    hccl_.Wait(commPolicy_.curHandle_);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::SetupTileParams(
    Params &out, uint32_t tileIdx, uint32_t curTileM, GM_ADDR aGMOffset, GM_ADDR cGMOffset, GM_ADDR scaleAOffset)
{
    auto &&commTiling = tilingData_->commTilingData;
    const auto &mmTile = tilingData_->tileQbmmTilingData;
    auto &&nonSplitAxisSize = commTiling.nonSplitAxisSize;  // rankN

    // ProblemShape: M=curTileM(当前tile的M), N=nonSplitAxisSize, K=mmTile.k, batch=1
    ProblemShape problemShape{static_cast<int64_t>(curTileM),
                              static_cast<int64_t>(nonSplitAxisSize),
                              static_cast<int64_t>(mmTile.k), 1UL};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = aGMOffset;
    mmadParams.scaleAGmAddr = scaleAOffset;
    mmadParams.bGmAddr = x2_;
    mmadParams.scaleBGmAddr = x2Scale_;
    mmadParams.cGmAddr = cGMOffset;
    mmadParams.biasGmAddr = bias_;

    LocalParams localParams{0,                     // rankId=0
                            1,                     // rankSize=1
                            static_cast<uint64_t>(curTileM), // originalM = 当前 tile M
                            x1_,                   // localAGmAddr
                            x1Scale_,              // localScaleAGmAddr
                            0,                     // localMatmul=0
                            1,                     // splitKNum=1
                            MatmulMode::REMOTE,    // REMOTE
                            static_cast<uint32_t>(commTiling.splitAxisTileSize)};

    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1,
                      mmTile.nBufferNum};

    BlockSchedulerParams schedulerParams{
        mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};

    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c, isBias_};

    out = {problemShape, mmadParams, l1Params, schedulerParams, qbmmParams, localParams};
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    auto &&commTiling = tilingData_->commTilingData;

    // 逐tile通算掩盖流程（通过PostComputeCommPolicy在Blaze kernel内部实现）:
    // Blaze kernel遍历每个tile时，在matmul开始前调用WaitTile(tileIdx)
    // PostComputeCommPolicy::WaitTile(tileIdx)异步下发tileIdx-1的AllToAll
    // 这样tile N的matmul与tile N-1的AllToAll并行执行
    //
    // 时间轴:
    // AIC(计算):   [matmul t0] [matmul t1] [matmul t2] ... [matmul tN-1]
    // AICPU(通信):              [comm t0  ] [comm t1  ] ... [comm tN-2] [comm tN-1]

    // Phase 1: 一次性调用Blaze kernel算完所有tile
    // kernel内部逐tile调用WaitTile，自动下发前一个tile的AllToAll
    if (totalTileCnt_ > 0) {
        // 设置完整M，让kernel内部遍历所有tile
        Params params;
        SetupTileParams(params, 0, static_cast<uint32_t>(splitAxisSize_), x1_, mmResultGM_, x1Scale_);
        quantMatmulKernelImpl_(params);
    }

    // Phase 2: Blaze kernel完成后，下发剩余tile并Wait所有通信
    // PrepareLastTileAllToAll内部会做SyncAll(确保所有rank完成下发)再Wait
    if (totalTileCnt_ > 0) {
        PrepareLastTileAllToAll();
    }

    // Phase 3: Finalize
    AscendC::SyncAll();
    hccl_.Finalize();
    AscendC::SyncAll();
    hccl_.Finalize();
}

} // namespace Apace
