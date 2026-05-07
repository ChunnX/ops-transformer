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
 * \file grouped_matmul_swiglu_quant_v2_mxfp4_weight_nz.h
 * \brief MXFP4 weight NZ grouped matmul entry for SwigluQuantV2.
 */

#ifndef GROUPED_MATMUL_SWIGLU_QUANT_V2_MXFP4_WEIGHT_NZ_H
#define GROUPED_MATMUL_SWIGLU_QUANT_V2_MXFP4_WEIGHT_NZ_H

#include "cgmct/block/block_mmad_mx.h"
#include "grouped_matmul_swiglu_quant_v2_mxquant.h"

template <typename layoutA, typename layoutB>
__aicore__ inline void GmmSwigluMxFp4WeightNz(GM_ADDR x, GM_ADDR weight, GM_ADDR weightScale, GM_ADDR xScale,
                                              GM_ADDR weightAssistanceMatrix, GM_ADDR smoothScale, GM_ADDR groupList,
                                              GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    GmmSwigluAswt<layoutA, layoutB>(x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y,
                                    yScale, workspace, tiling);
}

#endif
