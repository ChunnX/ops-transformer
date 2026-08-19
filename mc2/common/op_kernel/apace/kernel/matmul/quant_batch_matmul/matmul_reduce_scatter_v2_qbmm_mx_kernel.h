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
 * \brief APACE MXFP8/MXFP4 Matmul + ReduceScatter implementation using pure AIC Hcomm.
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

constexpr uint32_t MAX_MATMUL_REDUCE_SCATTER_HCOMM_HANDLE = 64;

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
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = typename MatmulReduceScatterV2BLayout<BTrans>::Type;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;
    using BiasType = float;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, 0, LayoutA, LayoutB, AType>;
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

    __aicore__ inline explicit MatmulReduceScatterV2MxQuantHcommImpl(TilingDataType *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR x1ScaleGM,
                                GM_ADDR x2ScaleGM, GM_ADDR yGM, GM_ADDR contextGM, GM_ADDR workspaceGM)
    {
        aGM_ = aGM;
        bGM_ = bGM;
        biasGM_ = biasGM;
        x1ScaleGM_ = x1ScaleGM;
        x2ScaleGM_ = x2ScaleGM;
        yGM_ = yGM;
        workspaceGM_ = workspaceGM;
        hccl_.InitV2(contextGM, &tilingData_->mc2InitTiling);
        hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
    }

    __aicore__ inline void Run()
    {
        RunSlices(tilingData_->param.tileCnt, false, &tilingData_->tileQbmmTilingData);
        if (tilingData_->param.tailCnt > 0) {
            RunSlices(tilingData_->param.tailCnt, true, &tilingData_->tailQbmmTilingData);
        }
        if (AscendC::GetBlockIdx() == 0) {
            for (uint32_t i = 0; i < handleCount_; ++i) {
                hccl_.Wait(handles_[i]);
            }
        }
        AscendC::SyncAll();
        hccl_.Finalize();
    }

private:
    __aicore__ inline uint64_t GetInputBytes(uint64_t elements) const
    {
        if constexpr (Blaze::Gemm::IsFp4<AType>()) {
            return (elements + 1UL) / 2UL;
        }
        return elements * sizeof(AType);
    }

    __aicore__ inline uint64_t GetScaleKBytes(const QuantMatmulTilingData *tiling) const
    {
        constexpr uint64_t groupSize = 64UL;
        constexpr uint64_t scaleMultiBase = 2UL;
        return ((static_cast<uint64_t>(tiling->k) + groupSize - 1UL) / groupSize) * scaleMultiBase;
    }

    __aicore__ inline void RunSingleRank(GM_ADDR rankA, GM_ADDR rankScaleA, GM_ADDR rankC,
                                         uint64_t sliceM, const QuantMatmulTilingData *tiling)
    {
        ProblemShape problemShape{static_cast<int64_t>(sliceM), static_cast<int64_t>(tiling->n),
                                  static_cast<int64_t>(tiling->k), 1L};
        BlockMmadParams mmParams;
        mmParams.aGmAddr = rankA;
        mmParams.scaleAGmAddr = rankScaleA;
        mmParams.bGmAddr = bGM_;
        mmParams.scaleBGmAddr = x2ScaleGM_;
        mmParams.cGmAddr = rankC;
        mmParams.biasGmAddr = biasGM_;
        L1Params l1Params{static_cast<uint64_t>(tiling->stepK) * tiling->baseK,
                          tiling->scaleKL1, tiling->nBufferNum};
        BlockSchedulerParams schedulerParams{
            tiling->baseM, tiling->baseN, tiling->mTailTile, tiling->nTailTile,
            tiling->mBaseTailSplitCnt, tiling->nBaseTailSplitCnt, tiling->mTailMain, tiling->nTailMain};
        QBMMTiling qbmmParams{
            tiling->baseM, tiling->baseN, tiling->baseK, tiling->dbL0c,
            biasGM_ == nullptr ? QBMMTiling::BIAS_DISABLED : QBMMTiling::BIAS_ENABLED};
        Params params{problemShape, mmParams, l1Params, schedulerParams, qbmmParams};
        kernel_(params);
    }

    __aicore__ inline void RunSlices(uint32_t sliceCount, bool isTail, const QuantMatmulTilingData *tiling)
    {
        const auto &cfg = tilingData_->param;
        const uint64_t perRankM = static_cast<uint64_t>(cfg.rankM) / cfg.rankDim;
        const uint64_t tileM =
            (perRankM - static_cast<uint64_t>(cfg.tailM) * cfg.tailCnt) / cfg.tileCnt;
        const uint64_t sliceM = isTail ? cfg.tailM : tileM;
        const uint64_t outputBase = isTail ? static_cast<uint64_t>(cfg.tileCnt) * tileM * cfg.rankN : 0;
        const uint32_t handleShift = isTail ? cfg.tileCnt : 0;
        const uint64_t stride = perRankM * cfg.rankN;
        const auto dataType = static_cast<AscendC::HcclDataType>(tilingData_->dataType);

        for (uint32_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
            const uint64_t sliceStart = isTail
                ? static_cast<uint64_t>(cfg.tileCnt) * tileM + static_cast<uint64_t>(sliceIdx) * cfg.tailM
                : static_cast<uint64_t>(sliceIdx) * tileM;
            for (uint32_t rank = 0; rank < cfg.rankDim; ++rank) {
                const uint64_t globalRow = static_cast<uint64_t>(rank) * perRankM + sliceStart;
                GM_ADDR rankA = aGM_ + GetInputBytes(globalRow * tiling->k);
                GM_ADDR rankScaleA = x1ScaleGM_ + globalRow * GetScaleKBytes(tiling);
                GM_ADDR rankC = workspaceGM_ + globalRow * tiling->n * sizeof(CType);
                RunSingleRank(rankA, rankScaleA, rankC, sliceM, tiling);
            }
            AscendC::SyncAll();
            GM_ADDR sendBuffer = workspaceGM_ + sliceStart * cfg.rankN * sizeof(CType);
            GM_ADDR recvBuffer = yGM_ +
                (outputBase + static_cast<uint64_t>(sliceIdx) * sliceM * cfg.rankN) * sizeof(CType);
            handles_[sliceIdx + handleShift] = hccl_.template ReduceScatter<true>(
                sendBuffer, recvBuffer, sliceM * cfg.rankN, dataType,
                AscendC::HcclReduceOp::HCCL_REDUCE_SUM, stride, 1);
        }
        handleCount_ += sliceCount;
    }

private:
    TilingDataType *tilingData_{nullptr};
    Kernel kernel_;
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle handles_[MAX_MATMUL_REDUCE_SCATTER_HCOMM_HANDLE]{};
    GM_ADDR aGM_{nullptr};
    GM_ADDR bGM_{nullptr};
    GM_ADDR biasGM_{nullptr};
    GM_ADDR x1ScaleGM_{nullptr};
    GM_ADDR x2ScaleGM_{nullptr};
    GM_ADDR yGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    uint32_t handleCount_{0};
};

} // namespace Apace
