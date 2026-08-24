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
 * \brief MatmulAllToAll MX hcomm实现(PR#10391方案)
 *
 * AIC_ONLY + 逐tile compute+SyncAll+comm:
 * - for循环逐tile: kernel(tileM, perRankN) → SyncAll → AllToAll(tileM*perRankN)
 * - kernel逐rank遍历x2的N列, 输出rank-major [rankDim, tileM, perRankN]
 * - AllToAll整块发送(batchCnt=1)
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/quant_batch_matmul_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "quant_matmul_all_to_all_tiling_data.h"

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
    uint32_t tileM_{0};
    uint32_t tailM_{0};
    uint32_t tileCnt_{0};
    uint32_t tailCnt_{0};
    uint32_t totalTileCnt_{0};
    uint64_t axisM_{0};
    uint64_t axisK_{0};
    uint64_t axisNPerRank_{0};
    uint64_t maxTileM_{0};
    uint64_t aByteStrideK_{0};
    uint64_t scaleAByteStride_{0};
    uint64_t scaleAElemCnt_{0};

    bool isBias_{false};

    static constexpr uint64_t yHcclDataType_ = GetYHcclDataType<YType>();
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2UL;
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;

    __aicore__ inline void InitBaseParams();
    __aicore__ inline void SetupParams(const QuantMatmulTilingData &mmTile, uint32_t curTileM, Params &params);
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
    x1_ = x1;
    x2_ = x2;
    y_ = y;
    bias_ = bias;
    isBias_ = (bias != nullptr);
    x1Scale_ = x1_scale;
    x2Scale_ = x2_scale;
    workspaceGM_ = workspaceGM;
    mmResultGM_ = workspaceGM;

    InitBaseParams();

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
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::InitBaseParams()
{
    auto &&commTD = tilingData_->commTilingData;
    axisK_ = tilingData_->tileQbmmTilingData.k;
    axisNPerRank_ = tilingData_->tileQbmmTilingData.n;
    tileM_ = static_cast<uint32_t>(commTD.splitAxisTileSize);
    tailM_ = static_cast<uint32_t>(commTD.splitAxisTailSize);
    tileCnt_ = static_cast<uint32_t>(commTD.splitAxisTileCnt);
    tailCnt_ = static_cast<uint32_t>(commTD.splitAxisTailCnt);
    totalTileCnt_ = tileCnt_ + tailCnt_;
    axisM_ = static_cast<uint64_t>(tileCnt_) * tileM_ + static_cast<uint64_t>(tailCnt_) * tailM_;
    maxTileM_ = (tileM_ > tailM_) ? tileM_ : tailM_;
    uint32_t scaleKGroups = (axisK_ + Blaze::Gemm::MXFP_DIVISOR_SIZE - 1) / Blaze::Gemm::MXFP_DIVISOR_SIZE;
    scaleAElemCnt_ = scaleKGroups * Blaze::Gemm::MXFP_MULTI_BASE_SIZE;
    aByteStrideK_ = axisK_ * sizeof(X1Type);
    scaleAByteStride_ = scaleAElemCnt_ * sizeof(AscendC::fp8_e8m0_t);
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::SetupParams(
    const QuantMatmulTilingData &mmTile, uint32_t curTileM, Params &params)
{
    uint64_t totalN = axisNPerRank_ * rankDim_;
    // ProblemShape: M=curTileM(tile级), N=perRankN, K, batch=1
    params.problemShape = ProblemShape{
        static_cast<int64_t>(curTileM),
        static_cast<int64_t>(axisNPerRank_),
        static_cast<int64_t>(axisK_),
        1};

    params.mmParams = BlockMmadParams{
        x1_, x2_, mmResultGM_, bias_,
        x1Scale_, x2Scale_};

    params.l1Params = L1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    params.schedulerParams = BlockSchedulerParams{mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    params.qbmmParams = QBMMTiling{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c,
        isBias_ ? 1u : 0u};
    // rankStride = curTileM * perRankN (每rank的C输出步长,以CType元素为单位)
    params.fullN = totalN;
    params.rankStride = static_cast<uint64_t>(curTileM) * axisNPerRank_;
    params.rankDim = rankDim_;
}

template <typename X1Type, typename X2Type, typename YType,
          typename QuantMatmulAllToAllTilingDataType,
          AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void
QuantMatmulAllToAllHcommImpl<X1Type, X2Type, YType,
                              QuantMatmulAllToAllTilingDataType, ServerType, IsMxFp4>::Run()
{
    // PR#10391方案: AIC_ONLY, 逐tile compute+SyncAll+comm
    // for循环逐tile: kernel(tileM, perRankN) → SyncAll → AllToAll(tileM*perRankN)
    if ASCEND_IS_AIC {
        GM_ADDR aCur = x1_;
        GM_ADDR aScaleCur = x1Scale_;
        uint32_t lastTiling = 0;
        const auto &headTile = tilingData_->tileQbmmTilingData;
        const auto &tailTile = tilingData_->tailQbmmTilingData;

        Params param;
        for (uint32_t tid = 0; tid < totalTileCnt_; ++tid) {
            bool isTail = (tid >= tileCnt_);
            uint32_t curTileM = isTail ? tailM_ : tileM_;

            // 每tile重新SetupParams(kernel内部mmOp_.Init每次调用)
            SetupParams(isTail ? tailTile : headTile, curTileM, param);

            // 更新A和scaleA偏移
            param.mmParams.aGmAddr = aCur;
            param.mmParams.scaleAGmAddr = aScaleCur;
            // C输出: mmResultGM_基址(kernel内部按rankIdx*rankStride偏移)
            param.mmParams.cGmAddr = mmResultGM_;

            // matmul (kernel内部调用mmOp_.Init + 逐rank遍历)
            quantMatmulKernelImpl_(param);

            AscendC::SyncAll();

            // AllToAll整块发送: sendCount=curTileM*perRankN, batchCnt=1
            uint64_t tileOffset = (tid < tileCnt_)
                ? static_cast<uint64_t>(tid) * tileM_
                : static_cast<uint64_t>(tileCnt_) * tileM_ + static_cast<uint64_t>(tid - tileCnt_) * tailM_;
            uint64_t sendCount = static_cast<uint64_t>(curTileM) * axisNPerRank_;
            AscendC::HcclHandle h = hccl_.template AlltoAll<true>(
                mmResultGM_,
                y_ + tileOffset * axisNPerRank_ * sizeof(YType),
                sendCount, static_cast<AscendC::HcclDataType>(yHcclDataType_),
                sendCount, 1);
            hccl_.Wait(h);

            aCur += static_cast<uint64_t>(curTileM) * aByteStrideK_;
            aScaleCur += static_cast<uint64_t>(curTileM) * scaleAByteStride_;
        }

        AscendC::SyncAll();
        hccl_.Finalize();
    }
}

} // namespace Apace
