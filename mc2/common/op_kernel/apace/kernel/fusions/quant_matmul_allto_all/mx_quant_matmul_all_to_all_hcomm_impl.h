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
 * \file quant_matmul_allto_all_mx_quant_matmul_hcomm_impl.h
 * \brief QuantMatmul AllToAll — 计算→通信融合实现（纯C核通信）
 *
 * 与 all_to_all_mx_quant_matmul_hcomm_impl.h 的关键区别：
 *   - 数据流方向相反：MatMul → AllToAll（参考文件是 AllToAll → MatMul）
 *   - 无 scale 通信：scale 在 matmul 阶段本地消费，不需要跨卡通信
 *   - 通信数据类型为 YType（matmul 输出），而非 X1Type（matmul 输入）
 *   - 无 splitK：每卡本地完整 K 维度，不需要跨卡 K 累加
 *   - matmul 阶段不需要 commPolicy wait（通信在计算之后）
 *
 * Init():
 *   AIC: 初始化 HCCL，获取 rankId/rankDim，设置 GM 地址和 workspace 布局。
 *
 * Run():
 *   Phase 1 — MatmulProcess(): Blaze QBMM MX kernel 执行 x1×x2 → mmResult（workspace）
 *   Phase 2 — SyncAll(): 确保所有核完成 matmul
 *   Phase 3 — RunAllToAll(): 批量下发 AllToAll 通信任务，将 mmResult 按 M 轴重分布 → y
 *   Phase 4 — Finalize HCCL
 *
 * Workspace 布局: [mmResult (rankM * rankN * sizeof(YType))]
 * AllToAll: sendBuffer=mmResult, recvBuffer=y（或 all2all_out）
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
        return AscendC::HCCL_DATA_TYPE_BF16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FLOAT;
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

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out,
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
    bool isBias_{false};

    static constexpr uint64_t yHcclDataType_ = GetYHcclDataType<YType>();
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    __aicore__ inline void MatmulProcess();
    __aicore__ inline void RunAllToAll();
    __aicore__ inline void SetupParams(Params &out);
};

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Init(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR x1_scale, GM_ADDR x2_scale,
    GM_ADDR workspaceGM)
{
    auto &&commTiling = tilingData_->commTilingData;
    // splitAxisSize = 本卡 M 大小 = tileM * tileCnt + tailM * tailCnt
    splitAxisSize_ = static_cast<uint64_t>(commTiling.splitAxisTileSize) * commTiling.splitAxisTileCnt +
                     static_cast<uint64_t>(commTiling.splitAxisTailSize) * commTiling.splitAxisTailCnt;

    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;

    // workspace 布局: mmResult 用于存储 matmul 输出
    // mmResult 大小 = splitAxisSize_ (rankM) * nonSplitAxisSize (rankN) * sizeof(YType)
    mmResultGM_ = workspaceGM;
    // 若 all2all_out 有效，可作为 AllToAll 接收缓冲区；否则复用 workspace 后段
    if (all2all_out != nullptr) {
        y_ = all2all_out;  // AllToAll 接收到 all2all_out，后续由框架拷贝到 y
    }

    // 初始化 HCCL
    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(QuantMatmulAllToAllTilingDataType, mc2CcTiling)));
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    // Phase 1: Matmul — x1 × x2 → mmResult（workspace）
    MatmulProcess();

    // Phase 2: 确保所有核完成 matmul
    AscendC::SyncAll();

    // Phase 3: AllToAll — 将 mmResult 按 M 轴重分布 → y
    RunAllToAll();

    // Phase 4: Finalize
    AscendC::SyncAll();
    hccl_.Finalize();
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::SetupParams(Params &out)
{
    auto &&commTiling = tilingData_->commTilingData;
    const auto &mmTile = tilingData_->tileQbmmTilingData;
    auto &&nonSplitAxisSize = commTiling.nonSplitAxisSize;  // rankN

    // ProblemShape: M=splitAxisSize(本卡M), N=nonSplitAxisSize, K=mmTile.k, batch=1
    // matmul 仅计算本卡本地数据，不涉及跨卡 K 累加
    ProblemShape problemShape{static_cast<int64_t>(splitAxisSize_),
                              static_cast<int64_t>(nonSplitAxisSize),
                              static_cast<int64_t>(mmTile.k), 1UL};

    BlockMmadParams mmadParams;
    mmadParams.aGmAddr = x1_;             // matmul 输入: 本地 x1
    mmadParams.scaleAGmAddr = x1Scale_;   // x1 scale (本地)
    mmadParams.bGmAddr = x2_;             // matmul 输入: x2 (全量)
    mmadParams.scaleBGmAddr = x2Scale_;   // x2 scale
    mmadParams.cGmAddr = mmResultGM_;     // matmul 输出: mmResult (workspace)
    mmadParams.biasGmAddr = bias_;

    // matmul 阶段: rankId=0, rankSize=1, splitKNum=1
    // 不需要 rank 区分和 splitK 累加 — 纯本地计算
    LocalParams localParams{0,                     // rankId=0 (本地计算，无 rank 区分)
                            1,                     // rankSize=1 (单卡视角)
                            splitAxisSize_,        // originalM = 本卡 M
                            x1_,                   // localAGmAddr
                            x1Scale_,              // localScaleAGmAddr
                            0,                     // localMatmul=0 (不使用 LOCAL/REMOTE 区分)
                            1,                     // splitKNum=1 (无 K 切分)
                            MatmulMode::REMOTE,    // REMOTE 模式 (rankSize=1 时仅遍历本 rank)
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
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::MatmulProcess()
{
    Params params;
    SetupParams(params);
    quantMatmulKernelImpl_(params);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::RunAllToAll()
{
    auto &&commTiling = tilingData_->commTilingData;
    auto &&nonSplitAxisSize = commTiling.nonSplitAxisSize;  // rankN

    // AllToAll: 将 mmResult[rankM, rankN] 按 N 轴重分布
    // 每卡持有完整 mmResult[rankM, rankN]，需将 rankN 拆成 rankDim 份分发给各卡
    // 每份大小 = rankM * (rankN / rankDim)
    // strideCount = rankM * rankN / rankDim (源缓冲区中同一目标卡的两个 tile 间步长)

    uint64_t perRankN = nonSplitAxisSize / rankDim_;

    // 头块 AllToAll
    if (commTiling.splitAxisTileCnt > 0) {
        uint64_t headSendCount =
            static_cast<uint64_t>(commTiling.splitAxisTileSize) * perRankN;
        uint64_t headStrideCount = static_cast<uint64_t>(splitAxisSize_) * perRankN;
        hccl_.template AlltoAll<false>(
            mmResultGM_, y_,
            headSendCount, static_cast<AscendC::HcclDataType>(yHcclDataType_),
            headStrideCount, static_cast<uint8_t>(commTiling.splitAxisTileCnt));
    }

    // 尾块 AllToAll
    if (commTiling.splitAxisTailCnt > 0) {
        uint64_t headOffset = static_cast<uint64_t>(commTiling.splitAxisTileCnt) *
                              commTiling.splitAxisTileSize * nonSplitAxisSize * sizeof(YType);
        uint64_t tailSendCount =
            static_cast<uint64_t>(commTiling.splitAxisTailSize) * perRankN;
        uint64_t tailStrideCount = static_cast<uint64_t>(splitAxisSize_) * perRankN;
        hccl_.template AlltoAll<false>(
            mmResultGM_ + headOffset, y_ + headOffset,
            tailSendCount, static_cast<AscendC::HcclDataType>(yHcclDataType_),
            tailStrideCount, static_cast<uint8_t>(commTiling.splitAxisTailCnt));
    }
}

} // namespace Apace
