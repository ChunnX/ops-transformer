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
 * \file gmm_weight_quant_sample_mx_kernel.h
 * \brief Adapter from GroupedMatmul operator inputs and tiling data to the MXFP8/FP4 sample recipe.
 */

#ifndef GMM_WEIGHT_QUANT_SAMPLE_MX_KERNEL_H
#define GMM_WEIGHT_QUANT_SAMPLE_MX_KERNEL_H

#include "blaze/gemm/kernel/kernel_wqgmm_mxfp8fp4.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL {

template <class XType, class WeightType, class ScaleBType, class YType>
__aicore__ inline void GmmWeightQuantSampleMxKernel(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale,
                                                    GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                                    const GroupedMatmulTilingData::GMMWeightQuantParam *gmmBaseParams,
                                                    const TCubeTiling *mmTilingData)
{
    Blaze::Gemm::Kernel::WeightQuantGroupedMatmulMxfp8Fp4TilingData recipeTiling{};
    recipeTiling.groupNum = gmmBaseParams->groupNum;
    recipeTiling.coreNum = gmmBaseParams->coreNum;
    recipeTiling.kSize = gmmBaseParams->kSize;
    recipeTiling.nSize = gmmBaseParams->nSize;
    recipeTiling.cubeNumBlocksN = gmmBaseParams->cubeNumBlocksN;
    recipeTiling.mainBlockSize = gmmBaseParams->mainBlockSize;
    recipeTiling.mainBlockCount = gmmBaseParams->mainBlockCount;
    recipeTiling.firstTailBlockSize = gmmBaseParams->firstTailBlockSize;
    recipeTiling.secondTailBlockSize = gmmBaseParams->secondTailBlockSize;
    recipeTiling.firstTailBlockCount = gmmBaseParams->firstTailBlockCount;
    recipeTiling.secondTailBlockCount = gmmBaseParams->secondTailBlockCount;
    recipeTiling.baseM = mmTilingData->baseM;
    recipeTiling.groupListType = gmmBaseParams->groupListType;
    recipeTiling.singleWeight = gmmBaseParams->singleWeight;

    GM_ADDR xAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<XType>(0, x));
    // The recipe consumes a raw base in single-weight mode and resolves each tensor-list entry in multi-weight mode.
    GM_ADDR weightAddr = weight;
    GM_ADDR scaleBAddr = antiquantScale;
    if (gmmBaseParams->singleWeight != 0) {
        weightAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<WeightType>(0, weight));
        scaleBAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<ScaleBType>(0, antiquantScale));
    }
    GM_ADDR yAddr = reinterpret_cast<GM_ADDR>(GROUPED_MATMUL::GetTensorAddr<YType>(0, y));

    Blaze::Gemm::Kernel::RunWeightQuantGroupedMatmulMxfp8Fp4Recipe<WeightType, YType>(
        xAddr, weightAddr, perTokenScale, scaleBAddr, groupList, yAddr, recipeTiling);
}

} // namespace GROUPED_MATMUL

#endif // GMM_WEIGHT_QUANT_SAMPLE_MX_KERNEL_H
