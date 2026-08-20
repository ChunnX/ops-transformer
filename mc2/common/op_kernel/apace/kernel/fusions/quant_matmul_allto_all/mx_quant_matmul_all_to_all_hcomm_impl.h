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
#include "quant_matmul_all_to_all_tiling_data.h"

namespace Apace {

// NopCommWaitPolicy: matmul 阶段不需要通信等待（通信在计算之后）
struct NopCommWaitPolicy {
    __aicore__ inline void WaitTile(uint32_t tileIdx) {}
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

    // 组件定义 — 使用 NopCommWaitPolicy，matmul 阶段无通信依赖
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::AllToAllQbmmMxKernel<
        ProblemShape, BlockMmad, BlockScheduler, NopCommWaitPolicy>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;
    using MatmulMode = typename QuantMatmulKernelImpl::MatmulMode;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;

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
    __aicore__ inline void MatmulTileProcess(uint32_t tileIdx, uint32_t curTileM);
    // 逐tile异步下发通信，返回handle
    __aicore__ inline AscendC::HcclHandle PrepareTileAllToAll(uint32_t tileIdx, uint32_t curTileM);
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
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline AscendC::HcclHandle
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::PrepareTileAllToAll(
    uint32_t tileIdx, uint32_t curTileM)
{
    auto &&commTiling = tilingData_->commTilingData;
    auto &&nonSplitAxisSize = commTiling.nonSplitAxisSize;  // rankN
    uint64_t perRankN = nonSplitAxisSize / rankDim_;
    uint64_t dataStrideCount = static_cast<uint64_t>(splitAxisSize_) * perRankN;

    // 计算当前tile在mmResult和y中的偏移
    uint64_t tileOffset = static_cast<uint64_t>(tileIdx) * commTiling.splitAxisTileSize;
    uint64_t sendCount = static_cast<uint64_t>(curTileM) * perRankN;

    return hccl_.template AlltoAll<true>(
        mmResultGM_ + tileOffset * nonSplitAxisSize * sizeof(YType),
        y_ + tileOffset * perRankN * sizeof(YType),
        sendCount, static_cast<AscendC::HcclDataType>(yHcclDataType_),
        dataStrideCount, 1);
}
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
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::MatmulTileProcess(
    uint32_t tileIdx, uint32_t curTileM)
{
    auto &&commTiling = tilingData_->commTilingData;
    const auto &mmTile = tilingData_->tileQbmmTilingData;
    auto &&nonSplitAxisSize = commTiling.nonSplitAxisSize;

    // 计算 x1 偏移: tileIdx * tileM * rankK
    uint64_t x1ElemOffset = static_cast<uint64_t>(tileIdx) * commTiling.splitAxisTileSize * mmTile.k;
    GM_ADDR aGMOffset = x1_ + x1ElemOffset * sizeof(X1Type) / (IsMxFp4 ? MXFP_DATA_NUM_PER_BYTE : 1);

    // 计算 mmResult 偏移: tileIdx * tileM * rankN
    uint64_t mmResultElemOffset = static_cast<uint64_t>(tileIdx) * commTiling.splitAxisTileSize * nonSplitAxisSize;
    GM_ADDR cGMOffset = mmResultGM_ + mmResultElemOffset * sizeof(YType);

    // 计算 scale 偏移: tileIdx * tileM * ceilDiv(rankK, GROUP_SIZE) * 2
    uint64_t scaleKLen = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(mmTile.k), MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint64_t scaleElemOffset = static_cast<uint64_t>(tileIdx) * commTiling.splitAxisTileSize * scaleKLen;
    GM_ADDR scaleAOffset = x1Scale_ + scaleElemOffset * sizeof(AscendC::fp8_e8m0_t);

    Params params;
    SetupTileParams(params, tileIdx, curTileM, aGMOffset, cGMOffset, scaleAOffset);
    quantMatmulKernelImpl_(params);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    auto &&commTiling = tilingData_->commTilingData;

    // 逐tile计算通信流水掩盖流程：
    // HCCL通信在AICPU上执行，AIC核仅触发和等待，不占用Cube计算资源
    // 因此计算tile N+1与通信tile N可真正并行
    //
    // 时间轴:
    // AIC(计算): [matmul tile0] [matmul tile1] [matmul tile2] ...
    // AICPU(通信):              [comm tile0  ] [comm tile1  ] ...
    //                              ↑ 计算tile1时通信tile0并行执行

    // Phase 1: 逐tile交替计算+通信
    // tile 0: 先计算（无前序通信可等）
    AscendC::HcclHandle handles[64];
    if (totalTileCnt_ > 0) {
        uint32_t curTileM = (0 < headTileCnt_) ? commTiling.splitAxisTileSize : commTiling.splitAxisTailSize;
        MatmulTileProcess(0, curTileM);
        AscendC::SyncAll(); // 确保tile0计算全核完成
        handles[0] = PrepareTileAllToAll(0, curTileM);
    }

    // tile 1~N-1: 计算当前tile（与前一个tile的通信并行）→ 异步下发当前tile通信
    for (uint32_t tileIdx = 1; tileIdx < totalTileCnt_; tileIdx++) {
        uint32_t curTileM = (tileIdx < headTileCnt_) ? commTiling.splitAxisTileSize : commTiling.splitAxisTailSize;
        MatmulTileProcess(tileIdx, curTileM);
        AscendC::SyncAll(); // 确保当前tile计算全核完成
        handles[tileIdx] = PrepareTileAllToAll(tileIdx, curTileM);
    }

    // Phase 2: 等待所有tile通信完成
    AscendC::SyncAll();
    for (uint32_t tileIdx = 0; tileIdx < totalTileCnt_; tileIdx++) {
        hccl_.Wait(handles[tileIdx]);
    }

    // Phase 3: Finalize
    AscendC::SyncAll();
    hccl_.Finalize();
}

} // namespace Apace
