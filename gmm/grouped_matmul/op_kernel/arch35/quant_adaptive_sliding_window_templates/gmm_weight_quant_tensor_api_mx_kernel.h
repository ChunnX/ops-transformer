/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
#define GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H

#include "../grouped_matmul_tiling_data_apt.h"
#include "blaze/gemm/block/block_mmad_weight_prologue_mx.h"
#include "blaze/gemm/kernel/kernel_grouped_matmul_mix_weight_prologue.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/common_utils.h"
#include "../../grouped_matmul_utils.h"

namespace GROUPED_MATMUL {

template <class xType, class wType, class biasType, class yType>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR antiquantScale,
    GM_ADDR antiquantOffset, GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
    const GroupedMatmulTilingData::GMMWeightQuantParam *gmmBaseParams,
    const TCubeTiling *mmTilingData, AscendC::TPipe *tPipe)
{
    (void)scale;
    (void)antiquantOffset;
    (void)tPipe;

    using AType = xType;
    using BType = wType;
    using ScaleType = AscendC::fp8_e8m0_t;
    using CType = yType;
    using BiasTypeT = biasType;
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Te::ZNLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;
    using LayoutBias = AscendC::Te::NDExtLayoutPtn;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx<Blaze::Gemm::KernelGroupedMmadWithWeightPrologue>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy,
        AscendC::Std::tuple<AType, ScaleType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>,
        CType, LayoutC, BiasTypeT, LayoutBias>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, void>;

    const int64_t n = static_cast<int64_t>(gmmBaseParams->nSize);
    const int64_t k = static_cast<int64_t>(gmmBaseParams->kSize);
    const uint64_t kAL1 = static_cast<uint64_t>(mmTilingData->stepKb * mmTilingData->baseK);

    GM_ADDR aDataAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<AType>(0, x));
    GM_ADDR yDataAddr = reinterpret_cast<GM_ADDR>(GetTensorAddr<CType>(0, y));
    GM_ADDR perTokenScaleDataAddr = perTokenScale;
    bool isSingleW = gmmBaseParams->singleWeight == 1;
    GM_ADDR weightDataAddr = isSingleW ?
        reinterpret_cast<GM_ADDR>(GetTensorAddr<BType>(0, weight)) : weight;
    GM_ADDR scaleBDataAddr = isSingleW ?
        reinterpret_cast<GM_ADDR>(GetTensorAddr<ScaleType>(0, antiquantScale)) : antiquantScale;
    GM_ADDR biasDataAddr = (gmmBaseParams->hasBias == 1) ?
        (isSingleW ? reinterpret_cast<GM_ADDR>(GetTensorAddr<BiasTypeT>(0, bias)) : bias) : nullptr;

    typename Kernel::Params params;
    params.problemShape = ProblemShape{0, n, k};
    params.mmadParams.aGmAddr = aDataAddr;
    params.mmadParams.scaleAGmAddr = perTokenScaleDataAddr;
    params.mmadParams.scaleBGmAddr = scaleBDataAddr;
    params.mmadParams.cGmAddr = yDataAddr;
    params.mmadParams.l1TileShape = AscendC::Te::MakeShape(
        static_cast<int64_t>(mmTilingData->baseM),
        static_cast<int64_t>(mmTilingData->baseN),
        static_cast<int64_t>(kAL1),
        static_cast<int64_t>(kAL1));
    params.mmadParams.l0TileShape = AscendC::Te::MakeShape(
        static_cast<int64_t>(mmTilingData->baseM),
        static_cast<int64_t>(mmTilingData->baseN),
        static_cast<int64_t>(mmTilingData->baseK));
    params.mmadParams.l1BufferNum = static_cast<uint64_t>(Blaze::Gemm::DOUBLE_BUFFER_COUNT);
    params.mmadParams.hasBias = gmmBaseParams->hasBias == 1;
    params.prologueParams.bGmAddr = weightDataAddr;
    params.prologueParams.biasGmAddr = biasDataAddr;
    params.prologueParams.kBubSize = kAL1;
    params.prologueParams.nBubSize = static_cast<uint64_t>(mmTilingData->baseN);
    params.groupListGmAddr = groupList;

    typename Kernel::GMMTiling gmmTiling;
    gmmTiling.groupNum = gmmBaseParams->groupNum;
    gmmTiling.coreNum = gmmBaseParams->coreNum;
    gmmTiling.n = n;
    gmmTiling.k = k;
    gmmTiling.baseM = mmTilingData->baseM;
    gmmTiling.baseN = mmTilingData->baseN;
    gmmTiling.baseK = mmTilingData->baseK;
    gmmTiling.isBias = gmmBaseParams->hasBias;
    gmmTiling.groupListType = gmmBaseParams->groupListType;
    gmmTiling.singleW = gmmBaseParams->singleWeight;
    gmmTiling.groupSize = gmmBaseParams->groupSize;
    gmmTiling.mainBlockSize = gmmBaseParams->mainBlockSize;
    gmmTiling.mainBlockCount = gmmBaseParams->mainBlockCount;
    gmmTiling.firstTailBlockSize = gmmBaseParams->firstTailBlockSize;
    gmmTiling.firstTailBlockCount = gmmBaseParams->firstTailBlockCount;
    gmmTiling.secondTailBlockSize = gmmBaseParams->secondTailBlockSize;
    gmmTiling.secondTailBlockCount = gmmBaseParams->secondTailBlockCount;
    params.gmmParams = gmmTiling;

    Kernel kernel;
    kernel(params);
}

} // namespace GROUPED_MATMUL

#endif // GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
