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
 * \brief QuantMatmul AllToAll — 计算→通信融合实现（HCCL纯C通信）
 *
 * 基于PR#10054 urma方案重新设计:
 * - 使用QuantBatchMatmulMxKernel(纯计算kernel,无CommPolicy/WaitTile)
 * - AIC侧for循环逐tile调用kernel,每个tile算完后异步下发AllToAll
 * - rankId/rankSize用实际值(非写死0/1)
 * - B矩阵按rank偏移: bGm + rankId * N * K
 * - 逐tile通算掩盖: tile N的matmul与tile N-1的AllToAll并行
 *
 * 数据流: MatMul → AllToAll(通信在后)
 * - 无scale通信(scale在matmul阶段本地消费)
 * - 通信数据类型为YType(matmul输出)
 * - 无splitK(每卡本地完整K维度)
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/quant_batch_matmul_mx_kernel.h"
#include "include/tensor_api/tensor.h"

namespace Apace {

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

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::DNExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, X1Type>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, X1Type, LayoutA, X2Type, LayoutB, YType, LayoutC, BiasType, LayoutBias>;
    using QuantMatmulKernelImpl = Blaze::Gemm::Kernel::QuantBatchMatmulMxKernel<
        ProblemShape, BlockMmad, BlockScheduler>;

    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockMmadParams = typename QuantMatmulKernelImpl::BlockMmadParams;
    using L1Params = typename QuantMatmulKernelImpl::L1Params;
    using LocalParams = typename QuantMatmulKernelImpl::LocalParams;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename QuantMatmulKernelImpl::QBMMTiling;

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
    GM_ADDR mmResultGM_;

    AscendC::Hccl<ServerType> hccl_;
    uint32_t rankId_{0};
    uint32_t rankDim_{0};
    uint64_t splitAxisSize_{0};
    uint32_t totalTileCnt_{0};
    uint32_t headTileCnt_{0};
    uint32_t tileM_{0};
    uint32_t tailM_{0};
    uint32_t tailCnt_{0};
    bool isBias_{false};

    static constexpr uint64_t yHcclDataType_ = GetYHcclDataType<YType>();
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
    static constexpr uint64_t ALIGN_NUM = 512UL;

    // 逐tile流水: 当前handle和前一个handle(pipeline深度2)
    AscendC::HcclHandle curHandle_{0};
    AscendC::HcclHandle prevHandle_{0};
    bool hasPrevHandle_{false};

    __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile, uint32_t curTileM, Params &params);
    __aicore__ inline void RunMatmulTile(uint32_t tileIdx, uint32_t curTileM);
    __aicore__ inline AscendC::HcclHandle IssueAllToAll(uint32_t tileIdx, uint32_t curTileM);
    __aicore__ inline uint64_t CalcTileOffset(uint32_t tileIdx);
    __aicore__ inline uint32_t CalcTileM(uint32_t tileIdx);
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
    tileM_ = commTiling.splitAxisTileSize;
    tailM_ = commTiling.splitAxisTailSize;
    headTileCnt_ = commTiling.splitAxisTileCnt;
    tailCnt_ = commTiling.splitAxisTailCnt;
    totalTileCnt_ = headTileCnt_ + tailCnt_;
    splitAxisSize_ = static_cast<uint64_t>(tileM_) * headTileCnt_ +
                     static_cast<uint64_t>(tailM_) * tailCnt_;

    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    mmResultGM_ = workspaceGM;

    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(QuantMatmulAllToAllTilingDataType, mc2CcTiling)));
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline uint64_t
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::CalcTileOffset(
    uint32_t tileIdx)
{
    if (tileIdx < headTileCnt_) {
        return static_cast<uint64_t>(tileIdx) * tileM_;
    } else {
        return static_cast<uint64_t>(headTileCnt_) * tileM_ +
               static_cast<uint64_t>(tileIdx - headTileCnt_) * tailM_;
    }
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline uint32_t
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::CalcTileM(
    uint32_t tileIdx)
{
    return (tileIdx < headTileCnt_) ? tileM_ : tailM_;
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::SetupParams(
    const QuantMatmulTilingData &mmTile, uint32_t curTileM, Params &params)
{
    auto &&commTiling = tilingData_->commTilingData;
    uint64_t axisN = commTiling.nonSplitAxisSize;
    uint64_t axisK = mmTile.k;
    uint64_t perRankN = axisN / rankDim_;

    // 与urma方案保持一致:
    // ProblemShape: M=curTileM, N=perRankN(每卡N), K=完整K, batch=rankDim
    params.problemShape = ProblemShape{
        static_cast<int64_t>(curTileM),
        static_cast<int64_t>(perRankN),
        static_cast<int64_t>(axisK),
        static_cast<int64_t>(rankDim_)};

    constexpr uint64_t sizeShift = IsMxFp4 ? 1 : 0;
    uint64_t scaleKLen = Blaze::Gemm::CeilDiv(axisK, MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE;

    // A: 全局布局[rankDim*rankM, K], kernel内部按rank*oriM偏移
    params.mmadParams.aGmAddr = x1_;
    // B: 全局布局, kernel内部按rank*K偏移读取各rank的B段
    params.mmadParams.bGmAddr = x2_;
    // C: 写入mmResult(workspace)
    params.mmadParams.cGmAddr = mmResultGM_;
    params.mmadParams.biasGmAddr = bias_;
    params.mmadParams.scaleAGmAddr = x1Scale_;
    params.mmadParams.scaleBGmAddr = x2Scale_;

    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schParams = BlockSchedulerParams{
        mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};

    params.qbmmParams.batchCount = rankDim_;
    params.qbmmParams.baseM = mmTile.baseM;
    params.qbmmParams.baseN = mmTile.baseN;
    params.qbmmParams.baseK = mmTile.baseK;
    params.qbmmParams.isBias = isBias_ ? QBMMTiling::BIAS_ENABLED : QBMMTiling::BIAS_DISABLED;
    params.qbmmParams.dbL0c = mmTile.dbL0c;
    params.qbmmParams.skipSelfBatch = false;

    // rankId/rankSize用实际值,与urma方案一致
    params.localParams.rankId = rankId_;
    params.localParams.rankSize = rankDim_;
    params.localParams.originalM = curTileM;
    params.localParams.selfRankId = rankId_;
    params.localParams.cGmSelfAddr = 0;
    params.localParams.skipSelfBatch = false;
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::RunMatmulTile(
    uint32_t tileIdx, uint32_t curTileM)
{
    auto &&commTiling = tilingData_->commTilingData;
    const auto &mmTile = (tileIdx < headTileCnt_) ? tilingData_->tileQbmmTilingData : tilingData_->tileQbmmTilingData;

    Params params;
    SetupParams(mmTile, curTileM, params);

    // 计算当前tile在各rank的A中的M偏移
    // urma方案: A是全局布局[rankDim*rankM, K], 每个tile的A偏移 = tileOffset * K
    // kernel内部按 rank * curTileM + mPos 遍历各rank的A段
    uint64_t tileOffset = CalcTileOffset(tileIdx);

    // A矩阵偏移: x1_ + tileOffset * K (每个tile在M方向的起始偏移)
    uint64_t aByteOffset = tileOffset * mmTile.k * sizeof(X1Type) / (IsMxFp4 ? MXFP_DATA_NUM_PER_BYTE : 1);
    params.mmadParams.aGmAddr = x1_ + aByteOffset;

    // Scale A偏移
    uint64_t scaleKLen = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(mmTile.k), MXFP_GROUP_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint64_t scaleAByteOffset = tileOffset * scaleKLen * sizeof(AscendC::fp8_e8m0_t);
    params.mmadParams.scaleAGmAddr = x1Scale_ + scaleAByteOffset;

    // C输出偏移: mmResult_ + tileOffset * perRankN (每个tile输出perRankN列)
    uint64_t perRankN = commTiling.nonSplitAxisSize / rankDim_;
    params.mmadParams.cGmAddr = mmResultGM_ + tileOffset * perRankN * sizeof(YType);

    quantMatmulKernelImpl_.InitMmad(params);
    quantMatmulKernelImpl_(params);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline AscendC::HcclHandle
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::IssueAllToAll(
    uint32_t tileIdx, uint32_t curTileM)
{
    auto &&commTiling = tilingData_->commTilingData;
    uint64_t perRankN = commTiling.nonSplitAxisSize / rankDim_;
    uint64_t tileOffset = CalcTileOffset(tileIdx);

    // 逐行发送: sendCount=perRankN, strideCount=perRankN(行内相邻rank间距), batchCnt=curTileM(行数)
    return hccl_.template AlltoAll<true>(
        mmResultGM_ + tileOffset * perRankN * sizeof(YType),
        y_ + tileOffset * perRankN * sizeof(YType),
        perRankN, static_cast<AscendC::HcclDataType>(yHcclDataType_),
        perRankN, static_cast<uint8_t>(curTileM));
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    auto &&commTiling = tilingData_->commTilingData;

    // 逐tile通算掩盖流程:
    // tile 0: matmul → 异步下发AllToAll
    // tile 1: Wait(prevHandle) → matmul(与tile0通信并行) → 异步下发AllToAll
    // ...
    // tile N-1: Wait(prevHandle) → matmul → 异步下发AllToAll
    // 末尾: SyncAll → Wait(curHandle) → Wait(prevHandle) → Finalize

    for (uint32_t tid = 0; tid < totalTileCnt_; ++tid) {
        uint32_t curTileM = CalcTileM(tid);

        // pipeline深度2: Wait前一个tile的AllToAll(不阻塞当前tile的matmul)
        if (hasPrevHandle_) {
            hccl_.Wait(prevHandle_);
        }

        // matmul计算当前tile
        RunMatmulTile(tid, curTileM);

        // 确保matmul结果写入GM
        AscendC::SyncAll();

        // 异步下发当前tile的AllToAll
        prevHandle_ = curHandle_;
        curHandle_ = IssueAllToAll(tid, curTileM);
        hasPrevHandle_ = true;
    }

    // 等待所有通信完成
    AscendC::SyncAll();
    if (hasPrevHandle_) {
        hccl_.Wait(prevHandle_);
    }
    hccl_.Wait(curHandle_);

    AscendC::SyncAll();
    hccl_.Finalize();
}

} // namespace Apace
