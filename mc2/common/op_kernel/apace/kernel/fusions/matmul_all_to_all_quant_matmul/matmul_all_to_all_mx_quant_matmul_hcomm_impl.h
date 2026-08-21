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
 * \file matmul_all_to_all_mx_quant_matmul_hcomm_impl.h
 * \brief Apace Blaze computation stage for the MatmulAlltoAll Hcomm pipeline.
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "../../matmul/quant_batch_matmul/matmul_all_to_all_qbmm_mx_kernel.h"
#include "../../../../mc2_templates/computation/matmul/mc2_mx_quant_matmul.h"
#include "../../../tiling/quant_matmul_tiling_data.h"

namespace Apace {

/**
 * This class intentionally implements the same Init/GetContextPtr/Process/End
 * contract as the legacy MC2 computation component.  That lets the Apace Blaze
 * matmul reuse MatmulAlltoAll's proven transpose and Hcomm stages unchanged.
 */
template <typename X1Type, typename X2Type, typename YType>
class MatmulAllToAllMxQuantMatmulComputation {
public:
    using TilingType = QuantMatmulTilingData;
    using ContextType = MC2KernelTemplate::MC2MMContext<MC2KernelTemplate::MXQuantMMAdditionalData, TilingType>;

    explicit __aicore__ inline MatmulAllToAllMxQuantMatmulComputation(AscendC::TPipe *pipe)
    {
        (void)pipe;
    }

    __aicore__ inline void Init() {}
    __aicore__ inline ContextType *GetContextPtr() { return &context_; }
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void End() {}

private:
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
    using KernelImpl = Blaze::Gemm::Kernel::MatmulAllToAllQbmmMxKernel<ProblemShape, BlockMmad, BlockScheduler>;
    using Params = typename KernelImpl::Params;
    using BlockMmadParams = typename KernelImpl::BlockMmadParams;
    using L1Params = typename KernelImpl::L1Params;
    using BlockSchedulerParams = typename KernelImpl::BlockSchedulerParams;
    using QBMMTiling = typename KernelImpl::QBMMTiling;

    ContextType context_{};
    KernelImpl kernel_{};
};

template <typename X1Type, typename X2Type, typename YType>
__aicore__ inline void MatmulAllToAllMxQuantMatmulComputation<X1Type, X2Type, YType>::Process(uint32_t taskIndex)
{
    const TilingType &mmTile = *context_.tilingDataPtr;
    GM_ADDR aGM = context_.baseData.aGM + taskIndex * context_.baseData.aOffset;
    GM_ADDR cGM = context_.baseData.cGM + taskIndex * context_.baseData.cOffset;
    GM_ADDR x1Scale = context_.additionalData.x1Scale + taskIndex * context_.additionalData.x1ScaleOffset;

    ProblemShape problemShape{mmTile.m, mmTile.n, mmTile.k, 1UL};
    BlockMmadParams mmParams;
    mmParams.aGmAddr = aGM;
    mmParams.scaleAGmAddr = x1Scale;
    mmParams.bGmAddr = context_.baseData.bGM;
    mmParams.scaleBGmAddr = context_.additionalData.x2Scale;
    mmParams.cGmAddr = cGM;
    mmParams.biasGmAddr = context_.baseData.biasGM;

    L1Params l1Params{static_cast<uint64_t>(mmTile.stepK) * mmTile.baseK, mmTile.scaleKL1, mmTile.nBufferNum};
    BlockSchedulerParams schedulerParams{
        mmTile.baseM, mmTile.baseN, mmTile.mTailTile, mmTile.nTailTile,
        mmTile.mBaseTailSplitCnt, mmTile.nBaseTailSplitCnt, mmTile.mTailMain, mmTile.nTailMain};
    QBMMTiling qbmmParams{mmTile.baseM, mmTile.baseN, mmTile.baseK, mmTile.dbL0c,
                          context_.baseData.biasGM != nullptr};
    Params params{problemShape, mmParams, l1Params, schedulerParams, qbmmParams};
    kernel_(params);
}

} // namespace Apace

#ifndef DEFINE_APACE_MATMUL_CONTEXT_FOR_MATMUL_ALL_TO_ALL_MX_QUANT
// The macro argument must not be named ContextType. Otherwise the preprocessor
// also rewrites the nested ::ContextType on the right-hand side.
#define DEFINE_APACE_MATMUL_CONTEXT_FOR_MATMUL_ALL_TO_ALL_MX_QUANT(ContextTypeAlias)                                 \
    using ContextTypeAlias =                                                                                          \
        typename Apace::MatmulAllToAllMxQuantMatmulComputation<DTYPE_X1, DTYPE_X2, DTYPE_Y>::ContextType
#endif

#ifndef DEFINE_APACE_MATMUL_FOR_MATMUL_ALL_TO_ALL_MX_QUANT
#define DEFINE_APACE_MATMUL_FOR_MATMUL_ALL_TO_ALL_MX_QUANT(ComputationType)                                         \
    using ComputationType = Apace::MatmulAllToAllMxQuantMatmulComputation<DTYPE_X1, DTYPE_X2, DTYPE_Y>
#endif
