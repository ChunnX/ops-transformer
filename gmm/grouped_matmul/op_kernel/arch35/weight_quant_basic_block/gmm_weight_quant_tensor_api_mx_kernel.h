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
 * \file gmm_weight_quant_tensor_api_mx_kernel.h
 * \brief GMM pseudo-quantized MX A8W4 Tensor API adapter.
 */
#ifndef GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
#define GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H

#include "blaze/gemm/kernel/kernel_grouped_matmul_mix_weight_prologue.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

template <class XType, class WeightType, class ScaleType, class PerTokenScaleType, class BiasType, class YType>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam* gmmTiling,
    const TCubeTiling* mmTiling)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::ZNLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<XType, PerTokenScaleType>,
        AscendC::Std::tuple<LayoutA, LayoutScaleA>, AscendC::Std::tuple<WeightType, ScaleType>,
        AscendC::Std::tuple<LayoutB, LayoutScaleB>, YType, LayoutC, BiasType, LayoutC>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;
    using GroupedKernel = Blaze::Gemm::Kernel::GroupedMatmulMixWeightPrologue<MatmulKernel>;

    constexpr uint64_t L1_BUFFER_NUM = 4U;
    const int64_t m = 1;
    const int64_t n = static_cast<int64_t>(gmmTiling->nSize);
    const int64_t k = static_cast<int64_t>(gmmTiling->kSize);
    const int64_t baseM = static_cast<int64_t>(mmTiling->baseM);
    const int64_t baseN = static_cast<int64_t>(mmTiling->baseN);
    const int64_t baseK = static_cast<int64_t>(mmTiling->baseK);
    if (n <= 0 || k <= 0 || baseM <= 0 || baseN <= 0 || baseK <= 0) {
        return;
    }

    // Keep one complete L1 K tile in the NZ prologue UB. Splitting it between
    // AIV sub-blocks changes the packed-FP4 slice origin and is not required by
    // the Blaze prologue; the second sub-block participates in hand-off only.
    constexpr uint64_t K_UB_ALIGNMENT = 64U;
    const uint64_t kL1Size = static_cast<uint64_t>(baseK);
    const uint64_t kBubSize =
        (kL1Size + K_UB_ALIGNMENT - 1U) / K_UB_ALIGNMENT * K_UB_ALIGNMENT;

    GM_ADDR xAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<XType>(0, x));
    GM_ADDR yAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
    // per_token_scale is an optional Tensor in the operator ABI, not a
    // dynamic TensorList.  It is already the data address and must not be
    // decoded as a tensor-list descriptor.
    GM_ADDR perTokenScaleAddr = perTokenScale;
    typename GroupedKernel::Params params{
        {{m, n, k},
         {xAddr, perTokenScaleAddr, nullptr, yAddr,
          AscendC::Te::MakeShape(baseM, baseN, static_cast<int64_t>(kL1Size), static_cast<int64_t>(kL1Size)),
          AscendC::Te::MakeShape(baseM, baseN, baseK), L1_BUFFER_NUM, gmmTiling->hasBias != 0U},
         {nullptr, nullptr, kBubSize, static_cast<uint64_t>(baseN)},
         {static_cast<uint64_t>(baseM), static_cast<uint64_t>(baseN), 1U, 1U, 1U, 1U, 0U, 0U}},
        {gmmTiling->groupNum, gmmTiling->groupListType, gmmTiling->hasBias != 0U,
         gmmTiling->singleWeight != 0U},
        weight, antiquantScale, bias, groupList};
    GroupedKernel kernel;
    kernel(params);
}

#endif // GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
