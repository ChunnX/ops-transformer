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
 * \file grouped_matmul_finalize_routing_wqgmm_mx_fr.h
 * \brief TensorAPI entry point for GroupedMatmulFinalizeRouting MX A8W4 weight quant
 *        (FP8 E4M3FN x FP4 E2M1, WeightNZ). Replaces the legacy
 *        GMMFRWeightQuantResplitController assembly with Blaze components.
 */

#ifndef GROUPED_MATMUL_FINALIZE_ROUTING_WQGMM_MX_FR_H
#define GROUPED_MATMUL_FINALIZE_ROUTING_WQGMM_MX_FR_H

#include "blaze/gemm/block/block_mmad_wqgmm_mx_mix.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mx_mix_finalize_routing.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "gmm_fr_weight_quant_tiling_data.h"

namespace GROUPED_MATMUL_FINALIZE_ROUTING {

template <typename xType, typename wType, typename scaleType, typename perTokenScaleType, typename biasType,
          typename yType, typename sharedInputDType, typename logitsType, typename rowIndexType, bool hasBias>
__aicore__ inline void GmmFrWeightQuantTensorApiMx(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
                                                   GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR logitsAddr,
                                                   GM_ADDR rowIndexAddr, GM_ADDR y, GM_ADDR shareInput,
                                                   const GMMFinalizeRoutingArch35Tiling::
                                                       GMMFinalizeRoutingWeightQuantTilingData *__restrict tiling)
{
    (void)sizeof(xType);
    (void)sizeof(wType);
    (void)sizeof(scaleType);
    (void)sizeof(perTokenScaleType);
    (void)sizeof(yType);
    using AType = xType;                 // fp8_e4m3fn_t activation
    using BType = wType;                 // fp4x2_e2m1_t packed weight
    using ScaleType = scaleType;         // fp8_e8m0_t
    using BiasType = biasType;           // bfloat16_t
    using SharedType = sharedInputDType; // bfloat16_t
    using LogitType = logitsType;        // float
    using RowIndexType = rowIndexType;   // int64_t
    static_assert(AscendC::IsSameType<LogitType, float>::value, "A8W4 FR kernel requires float logits");
    static_assert(AscendC::IsSameType<RowIndexType, int64_t>::value, "A8W4 FR kernel requires int64 row index");
    static_assert(AscendC::IsSameType<SharedType, bfloat16_t>::value, "A8W4 FR kernel requires bf16 shared input");

    using DispatchPolicy = Blaze::Gemm::GroupedMatmulWithWeightQuantMx;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutScaleA = asc::te::scalea_nd_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
    using ProblemShape = decltype(asc::te::make_shape(0UL, 0UL, 0UL));
    using BlockScheduler =
        Blaze::Gemm::Kernel::BlockSchedulerWqgmmNResplit<decltype(asc::te::make_shape(0UL, 0UL, 0UL))>;
    using BlockMmad =
        Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>,
                                      AscendC::Std::tuple<LayoutA, LayoutScaleA>,
                                      AscendC::Std::tuple<BType, ScaleType>,
                                      AscendC::Std::tuple<LayoutB, LayoutScaleB>, float, asc::te::nd_ext_layout_ptn,
                                      BiasType, asc::te::nd_ext_layout_ptn>;
    using BlockPrologue = Blaze::Gemm::Kernel::GmmWqgmmFrWeightPrologue<AType, BType, BiasType>;
    using BlockEpilogue = Blaze::Gemm::Kernel::GmmWqgmmFrEpilogue;
    using BlockOutputInit = Blaze::Gemm::Kernel::GmmWqgmmFrOutputInit;
    using KernelImpl =
        Blaze::Gemm::Kernel::GmmWqgmmMxFinalizeRoutingKernel<ProblemShape, BlockMmad, BlockScheduler, BlockPrologue,
                                                             BlockEpilogue, BlockOutputInit>;

    typename BlockMmad::Params mmadParams{};
    mmadParams.ptrA = x;
    mmadParams.ptrScaleA = perTokenScale;
    mmadParams.ptrScaleB = antiquantScale;
    mmadParams.ptrBias = bias;
    mmadParams.ptrC = y; // consumed by the AIV epilogue; AIC writes the UB fixpipe buffer

    const uint64_t nMainBlockCount = Blaze::Gemm::CeilDiv(tiling->nSize, 256UL);
    typename BlockScheduler::Params schedulerParams{nMainBlockCount,
                                                    256UL,
                                                    0UL,
                                                    0UL,
                                                    0UL,
                                                    0UL,
                                                    static_cast<uint64_t>(tiling->coreNum),
                                                    0UL,
                                                    static_cast<int32_t>(128),
                                                    static_cast<uint64_t>(tiling->nSize)};
    typename BlockPrologue::Params prologueParams{reinterpret_cast<__gm__ BType *>(weight)};
    typename BlockEpilogue::Params epilogueParams{};
    epilogueParams.ptrY = reinterpret_cast<__gm__ float *>(y);
    epilogueParams.ptrLogit = reinterpret_cast<__gm__ float *>(logitsAddr);
    epilogueParams.ptrRowIndex = reinterpret_cast<__gm__ int64_t *>(rowIndexAddr);
    epilogueParams.nSize = tiling->nSize;
    typename BlockOutputInit::Params outputInitParams{};
    outputInitParams.ptrY = reinterpret_cast<__gm__ float *>(y);
    outputInitParams.ptrSharedInput = reinterpret_cast<__gm__ SharedType *>(shareInput);
    outputInitParams.nSize = tiling->nSize;
    outputInitParams.initSize = tiling->initSize;
    outputInitParams.sharedInputOffset = tiling->sharedInputOffset;
    outputInitParams.sharedInputLen = tiling->sharedInputLen;
    outputInitParams.sharedInputWeight = tiling->sharedInputWeight;

    typename KernelImpl::Params params{
        ProblemShape{static_cast<uint64_t>(tiling->kSize), static_cast<uint64_t>(tiling->nSize),
                     static_cast<uint64_t>(tiling->groupNum)},
        mmadParams, schedulerParams, prologueParams, epilogueParams, outputInitParams, groupList,
        static_cast<uint32_t>(tiling->groupListType), static_cast<uint32_t>(tiling->hasBias)};
    KernelImpl kernelImpl;
    kernelImpl(params);
}

} // namespace GROUPED_MATMUL_FINALIZE_ROUTING

#endif // GROUPED_MATMUL_FINALIZE_ROUTING_WQGMM_MX_FR_H
