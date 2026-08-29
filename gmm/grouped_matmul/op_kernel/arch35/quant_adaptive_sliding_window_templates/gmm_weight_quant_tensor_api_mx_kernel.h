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
 * \brief Adapter from GroupedMatmul operator inputs and tiling data to the MX A8W4 Tensor API kernel.
 */

#ifndef GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
#define GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H

#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL {

template <class XType_, class WeightType_, class ScaleBType_, class BiasType_, class YType_, bool IsSingleMultiSingle_>
__aicore__ inline void GmmWeightQuantTensorApiMxKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y, const GroupedMatmulTilingData::GMMWeightQuantParam *gmmBaseParams, const TCubeTiling *mmTilingData)
{
    using XType = XType_;
    using WeightType = WeightType_;
    using ScaleBType = ScaleBType_;
    using BiasType = BiasType_;
    using YType = YType_;
    constexpr bool IS_SINGLE_MULTI_SINGLE = IsSingleMultiSingle_;

    Blaze::Gemm::Kernel::GmmWeightQuantMxTiling gmmTiling{};
    gmmTiling.groupNum = gmmBaseParams->groupNum;
    gmmTiling.coreNum = gmmBaseParams->coreNum;
    gmmTiling.kSize = gmmBaseParams->kSize;
    gmmTiling.nSize = gmmBaseParams->nSize;
    gmmTiling.cubeNumBlocksN = gmmBaseParams->cubeNumBlocksN;
    gmmTiling.mainBlockSize = gmmBaseParams->mainBlockSize;
    gmmTiling.mainBlockCount = gmmBaseParams->mainBlockCount;
    gmmTiling.firstTailBlockSize = gmmBaseParams->firstTailBlockSize;
    gmmTiling.secondTailBlockSize = gmmBaseParams->secondTailBlockSize;
    gmmTiling.firstTailBlockCount = gmmBaseParams->firstTailBlockCount;
    gmmTiling.secondTailBlockCount = gmmBaseParams->secondTailBlockCount;
    gmmTiling.baseM = mmTilingData->baseM;
    gmmTiling.groupListType = gmmBaseParams->groupListType;
    gmmTiling.hasBias = gmmBaseParams->hasBias;
    if constexpr (IS_SINGLE_MULTI_SINGLE) {
        // SMS: X/Y are packed single tensors; W/ScaleB/Bias remain tensor-list descriptors for per-group lookup.
        GM_ADDR xAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<XType>(0, x));
        GM_ADDR yAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
        Blaze::Gemm::Kernel::RunGmmWeightQuantMxKernel<WeightType, BiasType, YType, true>(
            xAddr, weight, perTokenScale, antiquantScale, bias, groupList, yAddr, gmmTiling);
    } else {
        // The MX A8W4 host checker rejects MMM, so a non-SMS tiling key is SSS. Resolve each one-element tensor
        // list once, then advance contiguous group payloads in the kernel.
        GM_ADDR xAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<XType>(0, x));
        GM_ADDR weightAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<WeightType>(0, weight));
        GM_ADDR scaleBAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleBType>(0, antiquantScale));
        GM_ADDR biasAddr = bias;
        if (gmmBaseParams->hasBias != 0) {
            biasAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<BiasType>(0, bias));
        }
        GM_ADDR yAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));
        Blaze::Gemm::Kernel::RunGmmWeightQuantMxKernel<WeightType, BiasType, YType, false>(
            xAddr, weightAddr, perTokenScale, scaleBAddr, biasAddr, groupList, yAddr, gmmTiling);
    }
}

} // namespace GROUPED_MATMUL

#endif // GMM_WEIGHT_QUANT_TENSOR_API_MX_KERNEL_H
