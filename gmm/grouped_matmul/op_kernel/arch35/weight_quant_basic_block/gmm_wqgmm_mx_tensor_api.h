/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*! \file gmm_wqgmm_mx_tensor_api.h
 *  \brief Thin GroupedMatmul adaptor for the ops-tensor WQGMM MX A8W4 kernel.
 */
#pragma once

#include <cstdint>

#include "kernel_operator.h"
#include "blaze/gemm/kernel/kernel_wqgmm_mix_weight_prologue.h"
#include "../grouped_matmul_tiling_data_apt.h"

namespace GROUPED_MATMUL::WqgmmMxTensorApi {

#if (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510)
template <typename XType, typename WeightType, typename ScaleAType, typename ScaleBType, typename BiasType,
          typename YType>
__aicore__ inline void InvokeWqgmmMxTensorApi(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias,
                                              GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                              const GroupedMatmulTilingData::GMMWeightQuantParam *gmmTiling,
                                              const TCubeTiling *mmTiling)
{
    using Kernel = Blaze::Gemm::Kernel::DefaultKernelWqgmmMixWeightPrologue<XType, WeightType, ScaleAType, ScaleBType,
                                                                            BiasType, YType>;
    using Params = typename Kernel::Params;

    // Copy fields explicitly so transformer tiling ABI does not leak into ops-tensor.
    Params params{};
    params.problemShape =
        AscendC::Te::MakeShape(static_cast<int64_t>(mmTiling->M), static_cast<int64_t>(gmmTiling->nSize),
                               static_cast<int64_t>(gmmTiling->kSize), static_cast<int64_t>(1));

    params.operands.xGmAddr = x;
    params.operands.weightGmAddr = weight;
    params.operands.scaleAGmAddr = perTokenScale;
    params.operands.scaleBGmAddr = antiquantScale;
    params.operands.biasGmAddr = gmmTiling->hasBias != 0U ? bias : nullptr;
    params.operands.yGmAddr = y;
    params.operands.groupListGmAddr = groupList;

    params.gmmParams.groupNum = gmmTiling->groupNum;
    params.gmmParams.coreNum = gmmTiling->coreNum;
    params.gmmParams.kSize = gmmTiling->kSize;
    params.gmmParams.nSize = gmmTiling->nSize;
    params.gmmParams.singleX = gmmTiling->singleX;
    params.gmmParams.singleWeight = gmmTiling->singleWeight;
    params.gmmParams.singleY = gmmTiling->singleY;
    params.gmmParams.groupType = gmmTiling->groupType;
    params.gmmParams.groupListType = gmmTiling->groupListType;
    params.gmmParams.hasBias = gmmTiling->hasBias;
    params.gmmParams.cubeNumBlocksN = gmmTiling->cubeNumBlocksN;
    params.gmmParams.groupSize = gmmTiling->groupSize;
    params.gmmParams.mainBlockSize = gmmTiling->mainBlockSize;
    params.gmmParams.mainBlockCount = gmmTiling->mainBlockCount;
    params.gmmParams.firstTailBlockSize = gmmTiling->firstTailBlockSize;
    params.gmmParams.secondTailBlockSize = gmmTiling->secondTailBlockSize;
    params.gmmParams.firstTailBlockCount = gmmTiling->firstTailBlockCount;
    params.gmmParams.secondTailBlockCount = gmmTiling->secondTailBlockCount;

    params.mmTiling.baseM = mmTiling->baseM;
    params.mmTiling.baseN = mmTiling->baseN;
    params.mmTiling.baseK = mmTiling->baseK;
    params.mmTiling.stepKa = mmTiling->stepKa;
    params.mmTiling.stepKb = mmTiling->stepKb;

    Kernel kernel;
    kernel(params);
}
#endif

} // namespace GROUPED_MATMUL::WqgmmMxTensorApi
