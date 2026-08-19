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

template <typename T>
constexpr AscendC::HcclDataType GetMatmulReduceScatterV2HcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float>::value) {
        return AscendC::HCCL_DATA_TYPE_FP32;
    } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::HCCL_DATA_TYPE_BFP16;
    } else {
        return AscendC::HCCL_DATA_TYPE_FP16;
    }
}

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
        const auto &commTiling = tilingData_->commTilingData;
        RunMatmul(&tilingData_->tileQbmmTilingData);
        AscendC::SyncAll();
        RunReduceScatterSlices(static_cast<uint32_t>(commTiling.splitAxisTileCnt), false);
        if (commTiling.splitAxisTailCnt > 0) {
            RunReduceScatterSlices(static_cast<uint32_t>(commTiling.splitAxisTailCnt), true);
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
    __aicore__ inline void RunMatmul(const QuantMatmulTilingData *tiling)
    {
        ProblemShape problemShape{static_cast<int64_t>(tiling->m), static_cast<int64_t>(tiling->n),
                                  static_cast<int64_t>(tiling->k), 1L};
        BlockMmadParams mmParams;
        mmParams.aGmAddr = aGM_;
        mmParams.scaleAGmAddr = x1ScaleGM_;
        mmParams.bGmAddr = bGM_;
        mmParams.scaleBGmAddr = x2ScaleGM_;
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
        Params params{problemShape, mmParams, l1Params, schedulerParams, qbmmParams};
        kernel_(params);
    }

    __aicore__ inline void RunReduceScatterSlices(uint32_t sliceCount, bool isTail)
    {
        const auto &commTiling = tilingData_->commTilingData;
        const uint64_t tileM = commTiling.splitAxisTileSize;
        const uint64_t tailM = commTiling.splitAxisTailSize;
        const uint64_t tileCnt = commTiling.splitAxisTileCnt;
        const uint64_t tailCnt = commTiling.splitAxisTailCnt;
        const uint64_t rankN = commTiling.nonSplitAxisSize;
        const uint64_t perRankM = tileM * tileCnt + tailM * tailCnt;
        const uint64_t sliceM = isTail ? tailM : tileM;
        const uint64_t outputBase = isTail ? tileCnt * tileM * rankN : 0;
        const uint32_t handleShift = isTail ? static_cast<uint32_t>(tileCnt) : 0;
        const uint64_t stride = perRankM * rankN;
        constexpr auto dataType = GetMatmulReduceScatterV2HcclDataType<CType>();

        for (uint32_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
            const uint64_t sliceStart = isTail
                ? tileCnt * tileM + static_cast<uint64_t>(sliceIdx) * tailM
                : static_cast<uint64_t>(sliceIdx) * tileM;
            GM_ADDR sendBuffer = workspaceGM_ + sliceStart * rankN * sizeof(CType);
            GM_ADDR recvBuffer = yGM_ +
                (outputBase + static_cast<uint64_t>(sliceIdx) * sliceM * rankN) * sizeof(CType);
            handles_[sliceIdx + handleShift] = hccl_.template ReduceScatter<true>(
                sendBuffer, recvBuffer, sliceM * rankN, dataType,
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
