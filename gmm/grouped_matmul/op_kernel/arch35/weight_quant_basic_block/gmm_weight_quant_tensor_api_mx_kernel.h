/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: CANN-2.0
 */
#pragma once

#include "blaze/gemm/kernel/kernel_grouped_matmul_mix_weight_prologue.h"
#include "../grouped_matmul_tiling_data_apt.h"
#include "../../grouped_matmul_utils.h"

namespace GROUPED_MATMUL {

template <class AType, class BType, class ScaleBType, class ScaleAType, class BiasType, class CType,
          class LayoutA, class LayoutB, class LayoutC>
__aicore__ inline void InvokeGmmWeightQuantTensorApiMx(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scaleB, GM_ADDR bias, GM_ADDR groupList, GM_ADDR scaleA, GM_ADDR y,
    const GroupedMatmulTilingData::GMMWeightQuantParam *gmm,
    const TCubeTiling *mm)
{
    using ScaleType = AscendC::fp8_e8m0_t;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>,
        CType, LayoutC, BiasType, LayoutC>;
    using Kernel = Blaze::Gemm::Kernel::KernelGroupedMatmulMixWeightPrologue<BlockMmad>;

    // x/y are TensorList descriptors at the public ABI but this tiling branch
    // has one logical concatenated X/Y tensor. Per-token scale and groupList
    // are direct addresses and must never pass through GetTensorAddr.
    GM_ADDR xData = reinterpret_cast<GM_ADDR>(GetTensorAddr<AType>(0, x));
    GM_ADDR yData = reinterpret_cast<GM_ADDR>(GetTensorAddr<CType>(0, y));
    GM_ADDR weightArg = gmm->singleWeight ?
        reinterpret_cast<GM_ADDR>(GetTensorAddr<BType>(0, weight)) : weight;
    GM_ADDR scaleBArg = gmm->singleWeight ?
        reinterpret_cast<GM_ADDR>(GetTensorAddr<ScaleBType>(0, scaleB)) : scaleB;
    GM_ADDR biasArg = nullptr;
    if (gmm->hasBias) {
        biasArg = gmm->singleWeight ? reinterpret_cast<GM_ADDR>(GetTensorAddr<BiasType>(0, bias)) : bias;
    }

    typename Kernel::Params params{
        xData, weightArg, scaleA, scaleBArg, biasArg, groupList, yData,
        gmm->coreNum, gmm->groupNum, gmm->groupType, gmm->groupListType,
        gmm->singleWeight, gmm->hasBias, gmm->groupSize, gmm->nSize, gmm->kSize,
        gmm->mainBlockCount, gmm->mainBlockSize,
        gmm->firstTailBlockCount, gmm->firstTailBlockSize,
        gmm->secondTailBlockCount, gmm->secondTailBlockSize,
        static_cast<uint32_t>(mm->baseM), static_cast<uint32_t>(mm->baseN),
        static_cast<uint32_t>(mm->baseK), static_cast<uint32_t>(mm->stepKb),
        256U, 2U, 0U, 1U};
    Kernel kernel;
    kernel(params);
}

} // namespace GROUPED_MATMUL
