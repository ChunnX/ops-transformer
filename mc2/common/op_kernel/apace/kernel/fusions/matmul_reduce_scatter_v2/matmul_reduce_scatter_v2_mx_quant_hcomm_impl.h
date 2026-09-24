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
 * \file matmul_reduce_scatter_v2_mx_quant_hcomm_impl.h
 * \brief APACE MXFP8/MXFP4/HiFP4 Matmul + ReduceScatter implementation using pure AIC Hcomm.
 */

#pragma once

#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/matmul_reduce_scatter_v2_qbmm_mx_kernel.h"
#include "include/tensor_api/tensor.h"
#include "matmul_reduce_scatter_v2_tiling_data.h"

namespace Apace {

template <bool BTrans>
struct MatmulReduceScatterV2BLayout {
    using Type = AscendC::Te::NDExtLayoutPtn;
};

template <>
struct MatmulReduceScatterV2BLayout<true> {
    using Type = AscendC::Te::DNExtLayoutPtn;
};

template <typename AType, typename BType, typename CType, bool BTrans,
          typename TilingDataType, AscendC::HcclServerType ServerType>
class MatmulReduceScatterV2MxQuantHcommImpl {
public:
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
    static_assert(!(AscendC::IsSameType<AType, hifloat4x2_t>::value ||
                    AscendC::IsSameType<BType, hifloat4x2_t>::value) ||
                  AscendC::IsSameType<CType, half>::value || AscendC::IsSameType<CType, bfloat16_t>::value,
                  "HiFP4 MMRS only supports float16 or bfloat16 output");
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = typename MatmulReduceScatterV2BLayout<BTrans>::Type;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    // Reuse MXFP4's B4 scheduler geometry for HiFP4; MMAD keeps the real type.
    using SchedulerAType = AscendC::Std::conditional_t<
        AscendC::IsSameType<AType, hifloat4x2_t>::value, fp4x2_e2m1_t, AType>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, SchedulerAType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias>;
    using Kernel = Blaze::Gemm::Kernel::MatmulReduceScatterV2QbmmMxKernel<
        ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename Kernel::Params;
    using BlockMmadParams = typename Kernel::BlockMmadParams;
    using L1Params = typename Kernel::L1Params;
    using BlockSchedulerParams = typename Kernel::BlockSchedulerParams;
    using QBMMTiling = typename Kernel::QBMMTiling;
    using SliceParams = typename Kernel::SliceParams;

    static constexpr uint32_t APACE_MAX_RANK_DIM = 64U;

    // Follow the proven AllToAll Hcomm pattern: keep the communication dtype
    // as an integral compile-time constant, then cast at the HCCL call site.
    static constexpr uint64_t outputHcclDataType_ =
        AscendC::IsSameType<CType, float>::value ? AscendC::HCCL_DATA_TYPE_FP32 :
        (AscendC::IsSameType<CType, bfloat16_t>::value ? AscendC::HCCL_DATA_TYPE_BFP16 :
                                                        AscendC::HCCL_DATA_TYPE_FP16);

    __aicore__ inline explicit MatmulReduceScatterV2MxQuantHcommImpl(TilingDataType *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM,
                                GM_ADDR x2ScaleGM, GM_ADDR yGM, GM_ADDR contextGM, GM_ADDR workspaceGM)
    {
        aGM_ = aGM;
        bGM_ = bGM;
        // HiFP4 的非空 bias 由 Host 拒绝；Kernel 不为该场景启用 bias 搬运与累加。
        biasGM_ = IS_HIF4 ? nullptr : biasGM;
        x1ScaleGM_ = x1ScaleGM;
        x2ScaleGM_ = x2ScaleGM;
        yGM_ = yGM;
        workspaceGM_ = workspaceGM;
        hccl_.InitV2(contextGM, &tilingData_->mc2InitTiling);
        hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(TilingDataType, mc2CcTiling)));
        rankDim_ = hccl_.GetRankDim();
    }

    __aicore__ inline void Run()
    {
        const auto &commTiling = tilingData_->commTilingData;
        const uint32_t tileCount = static_cast<uint32_t>(commTiling.splitAxisTileCnt);
        const uint32_t tailCount = static_cast<uint32_t>(commTiling.splitAxisTailCnt);
        // 与 Host 的卡数集合一致：2、4、8、16、32、64；卡数不要求小于 AIC 核数。
        const bool invalidRank = rankDim_ < 2 || rankDim_ > APACE_MAX_RANK_DIM ||
            (rankDim_ & (rankDim_ - 1U)) != 0;
        const bool invalidTiling = invalidRank || tileCount == 0 ||
            (tileCount > 0 && tilingData_->tileQbmmTilingData.m !=
                static_cast<uint64_t>(commTiling.splitAxisTileSize) * rankDim_) ||
            (tailCount > 0 && tilingData_->tailQbmmTilingData.m !=
                static_cast<uint64_t>(commTiling.splitAxisTailSize) * rankDim_);
        if (invalidTiling) {
            hccl_.Finalize();
            return;
        }

        const uint64_t tileM = static_cast<uint64_t>(commTiling.splitAxisTileSize);
        const uint64_t tailM = static_cast<uint64_t>(commTiling.splitAxisTailSize);
        const uint64_t perRankM = tileM * tileCount + tailM * tailCount;
        Params tileParams;
        Params tailParams;
        SetupMatmulParams(tileParams, &tilingData_->tileQbmmTilingData, tileM, perRankM);
        if (tailCount > 0) {
            SetupMatmulParams(tailParams, &tilingData_->tailQbmmTilingData, tailM, perRankM);
        }

        // One APACE QBMM invocation owns the complete per-rank M. The kernel
        // iterates communication slices internally and calls IssueSlice after
        // each slice becomes globally visible in workspace.
        kernel_.RunSlices(tileParams, tileCount, tailParams, tailCount, *this);
        WaitPendingReduceScatter();
        hccl_.Finalize();
    }

    __aicore__ inline void IssueSlice(uint64_t sliceStart, uint64_t sliceM)
    {
        const uint64_t rankN = static_cast<uint64_t>(tilingData_->commTilingData.nonSplitAxisSize);
        GM_ADDR sendBuffer = workspaceGM_ +
            sliceStart * rankDim_ * rankN * sizeof(CType);
        GM_ADDR recvBuffer = yGM_ + sliceStart * rankN * sizeof(CType);
        IssueReduceScatter(sendBuffer, recvBuffer, sliceM * rankN);
    }

    __aicore__ inline void SyncAicCores() {
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(AIC_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(AIC_SYNC_EVENT_ID);
    }

private:
    __aicore__ inline void SetupMatmulParams(
        Params &params, const QuantMatmulTilingData *tiling, uint64_t sliceM, uint64_t perRankM)
    {
        ProblemShape problemShape{static_cast<int64_t>(tiling->m), static_cast<int64_t>(tiling->n),
                                  static_cast<int64_t>(tiling->k), 1L};
        BlockMmadParams mmParams;
        mmParams.aGmAddr = aGM_;
        mmParams.scaleAGmAddr = x1ScaleGM_;
        mmParams.bGmAddr = bGM_;
        mmParams.scaleBGmAddr = x2ScaleGM_;
        // RunSlices derives every slice address from this workspace base.
        mmParams.cGmAddr = workspaceGM_;
        mmParams.biasGmAddr = biasGM_;
        L1Params l1Params{static_cast<uint64_t>(tiling->stepK) * tiling->baseK,
                          tiling->scaleKL1, tiling->nBufferNum};
        BlockSchedulerParams schedulerParams{
            tiling->baseM, tiling->baseN, tiling->mTailTile, tiling->nTailTile,
            tiling->mBaseTailSplitCnt, tiling->nBaseTailSplitCnt, tiling->mTailMain, tiling->nTailMain};
        QBMMTiling qbmmParams{
            tiling->baseM, tiling->baseN, tiling->baseK, tiling->dbL0c,
            biasGM_ == nullptr ? QBMMTiling::BIAS_DISABLED : QBMMTiling::BIAS_ENABLED};
        SliceParams sliceParams{rankDim_, perRankM, 0, sliceM};
        params = {problemShape, mmParams, l1Params, schedulerParams, qbmmParams, sliceParams};
    }

    __aicore__ inline void IssueReduceScatter(
        GM_ADDR sendBuffer, GM_ADDR recvBuffer, uint64_t recvCount)
    {
        // Keep at most two asynchronous ReduceScatter tasks in flight. Before
        // reusing a slot, wait for the task issued two rounds earlier. This
        // removes the fixed per-tile handle array while preserving a depth-2
        // compute/communication pipeline.
        const uint32_t slot = issuedHandleCount_ % HCOMM_PIPELINE_DEPTH;
        if (issuedHandleCount_ >= HCOMM_PIPELINE_DEPTH) {
            hccl_.Wait(pipelineHandles_[slot]);
            SyncAicCores();
        }
        pipelineHandles_[slot] = hccl_.template ReduceScatter<true>(
            sendBuffer, recvBuffer, recvCount,
            static_cast<AscendC::HcclDataType>(outputHcclDataType_),
            AscendC::HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
        ++issuedHandleCount_;
    }

    __aicore__ inline void WaitPendingReduceScatter()
    {
        const uint32_t pendingCount = issuedHandleCount_ < HCOMM_PIPELINE_DEPTH ?
            issuedHandleCount_ : HCOMM_PIPELINE_DEPTH;
        const uint32_t firstPending = issuedHandleCount_ - pendingCount;
        for (uint32_t i = 0; i < pendingCount; ++i) {
            const uint32_t slot = (firstPending + i) % HCOMM_PIPELINE_DEPTH;
            hccl_.Wait(pipelineHandles_[slot]);
        }
        SyncAicCores();
    }

private:
    TilingDataType *tilingData_{nullptr};
    Kernel kernel_;
    AscendC::Hccl<ServerType> hccl_;
    static constexpr uint32_t HCOMM_PIPELINE_DEPTH = 2U;
    static constexpr uint16_t AIC_SYNC_EVENT_ID = 3U;
    AscendC::HcclHandle pipelineHandles_[HCOMM_PIPELINE_DEPTH]{};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    uint32_t issuedHandleCount_{0};
    uint32_t rankDim_{0};
};

} // namespace Apace

