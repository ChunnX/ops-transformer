/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_apt.cpp
 * \brief
 */

#include "grouped_matmul_utils.h"
#include "arch35/grouped_matmul_tiling_data_apt.h"
using GMMWeightQuantTilingData = GroupedMatmulTilingData::GMMWeightQuantTilingData;
using GMMNoQuantTilingData = GroupedMatmulTilingData::GMMNoQuantTilingData;
using GMMQuantTilingData = GroupedMatmulTilingData::GMMQuantTilingData;

#include "grouped_matmul_apt_quant.h"
#include "grouped_matmul_apt_antiquant.h"
#include "grouped_matmul_apt_no_quant.h"

using namespace AscendC;
using namespace matmul;
using namespace GROUPED_MATMUL;

#ifndef FORMAT_FRACTAL_NZ
    #define FORMAT_FRACTAL_NZ
#endif

namespace {
#if defined(FORMAT_WEIGHT) && FORMAT_WEIGHT == FORMAT_FRACTAL_NZ
constexpr CubeFormat wFormat = CubeFormat::NZ;
#else
constexpr CubeFormat wFormat = CubeFormat::ND;
#endif
}

template <bool trans = false>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X, trans>;

template <bool trans = false>
using weightType = MatmulType<AscendC::TPosition::GM, wFormat, DTYPE_X, trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, MM_DTYPE_Y>;

using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;

#if defined(V310_GMM_QUANT)
template <typename TDtypeX, typename TDtypeWeight, typename TDtypeBias, typename TDtypeY,
          typename TDtypeScale, typename TDtypeL0cLocal, CubeFormat kWFormat,
          int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#elif defined(V310_GMM_ANTI_QUANT)
template <typename TDtypeX, typename TDtypeWeight, typename TDtypeBias, typename TDtypeY,
          typename TDtypeScale, typename TDtypeAntiquantScale, typename TDtypePerTokenScale,
          int kOrigDtypeX, int kOrigDtypeAntiquantScale,
          int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE,
          int8_t WQ_B_TRANS, int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC,
          int8_t ALGORITHM_SUB_CATEGORY, int8_t ALGORITHM_CATEGORY>
#else
template <typename TDtypeY, CubeFormat kWFormat,
          int8_t NO_QUANT_B_TRANS, int8_t NO_QUANT_A_TRANS>
#endif
void grouped_matmul_impl(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                         GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                         GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                         GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GM_ADDR user1 = GetUserWorkspace(workspace);

#ifndef __CCE_KT_TEST__
#if defined(V310_GMM_QUANT)
    GROUPED_MATMUL_QUANT_IMPL_BODY(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY, TDtypeScale, TDtypeL0cLocal, kWFormat,
                                   QUANT_B_TRANS, QUANT_A_TRANS, KERNEL_TYPE);
#elif defined(V310_GMM_ANTI_QUANT)
    GROUPED_MATMUL_ANTIQUANT_IMPL_BODY(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY, TDtypeScale, TDtypeAntiquantScale,
                                       TDtypePerTokenScale, kOrigDtypeX, kOrigDtypeAntiquantScale,
                                       W_TYPE, OFFSET_OR_BIAS_EXIT, C_QUANT_TYPE, W_QUANT_TYPE, WQ_B_TRANS,
                                       WQ_A_TRANS, TEMPLATE_CUSTOM_SC, ALGORITHM_SUB_CATEGORY, ALGORITHM_CATEGORY);
#else
    GROUPED_MATMUL_NO_QUANT_IMPL_BODY(TDtypeY, kWFormat, NO_QUANT_B_TRANS, NO_QUANT_A_TRANS);
#endif
#endif
}

#if defined(V310_GMM_QUANT)
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, int8_t KERNEL_TYPE>
#elif defined(V310_GMM_ANTI_QUANT)
template <int8_t W_TYPE, int8_t OFFSET_OR_BIAS_EXIT, int8_t C_QUANT_TYPE, int8_t W_QUANT_TYPE, int8_t WQ_B_TRANS,
          int8_t WQ_A_TRANS, int8_t TEMPLATE_CUSTOM_SC, int8_t ALGORITHM_SUB_CATEGORY,
          int8_t ALGORITHM_CATEGORY>
#else
template <int8_t NO_QUANT_B_TRANS, int8_t NO_QUANT_A_TRANS>
#endif
__global__ __aicore__ void grouped_matmul(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                          GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                          GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,
                                          GM_ADDR workspace, GM_ADDR tiling)
{
#if defined(V310_GMM_QUANT)
    grouped_matmul_impl<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y,
                        DTYPE_SCALE, DTYPE_L0C_LOCAL, wFormat,
                        QUANT_B_TRANS, QUANT_A_TRANS, KERNEL_TYPE>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
        groupList, perTokenScale, y, workspace, tiling);
#elif defined(V310_GMM_ANTI_QUANT)
    grouped_matmul_impl<DTYPE_X, DTYPE_WEIGHT, DTYPE_BIAS, DTYPE_Y,
                        DTYPE_SCALE, DTYPE_ANTIQUANT_SCALE, DTYPE_PER_TOKEN_SCALE,
                        ORIG_DTYPE_X, ORIG_DTYPE_ANTIQUANT_SCALE,
                        W_TYPE, OFFSET_OR_BIAS_EXIT, C_QUANT_TYPE, W_QUANT_TYPE,
                        WQ_B_TRANS, WQ_A_TRANS, TEMPLATE_CUSTOM_SC,
                        ALGORITHM_SUB_CATEGORY, ALGORITHM_CATEGORY>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
        groupList, perTokenScale, y, workspace, tiling);
#else
    grouped_matmul_impl<DTYPE_Y, wFormat,
                        NO_QUANT_B_TRANS, NO_QUANT_A_TRANS>(
        x, weight, bias, scale, offset, antiquantScale, antiquantOffset,
        groupList, perTokenScale, y, workspace, tiling);
#endif
}