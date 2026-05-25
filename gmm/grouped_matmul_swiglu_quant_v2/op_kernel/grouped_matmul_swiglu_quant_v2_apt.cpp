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
 * \file grouped_matmul_swiglu_quant_v2_apt.cpp
 * \brief
 */

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0

#if (ORIG_DTYPE_X == DT_FLOAT8_E4M3FN && ORIG_DTYPE_WEIGHT == DT_FLOAT4_E2M1)
#define GMM_SWIGLU_QUANT_WEIGHT_QUANT
// 伪量化MxA8W4场景识别宏：x=FP8_E4M3FN, w=FP4_E2M1
    #include "arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_data.h"
    #include "arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h"
#else
// 全量化Mx量化
    #include "arch35/grouped_matmul_swiglu_quant_v2_mxquant.h"
#endif

#elif ORIG_DTYPE_X_SCALE == DT_FLOAT
    #include "arch35/grouped_matmul_swiglu_quant_v2_pertoken_quant.h"
#endif

#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;
using namespace matmul;

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

template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS>
__global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight,
                                                          GM_ADDR weightScale, GM_ADDR weightAssistanceMatrix,
                                                          GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    int64_t oriOverflowMode = 0;
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);

#if defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
    #ifndef DTYPE_BIAS
    #define DTYPE_BIAS bfloat16_t
    #endif
    // MxA8W4场景：完整Controller实现
    AscendC::InitSocState();

    oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData);
    GET_TILING_DATA_WITH_STRUCT(GMMSQArch35Tiling::GMMSQWeightQuantTilingData, tilingDataIn, tiling);

    GMMSQWeightQuant::GMMSQWeightQuantResplitController<
        DTYPE_X, DTYPE_WEIGHT, DTYPE_WEIGHT_SCALE,
        DTYPE_X_SCALE, DTYPE_BIAS, DTYPE_Y, DTYPE_Y_SCALE, false,
        GMMSQWeightQuant::MXA8W4_NZNK,
        GMMSQWeightQuant::VEC_ANTIQUANT_CONFIG_DYNAMIC>
        controller;

    controller.Init(x, weight, weightScale, nullptr, groupList, xScale, y, yScale, &tilingDataIn);
    controller.Process();
#else
    TPipe tPipe;

    oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // enable overflow mode to avoid nan/inf value
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    
    GM_ADDR userWorkspace = GetUserWorkspace(workspace);
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
    if (wFormat == CubeFormat::NZ) {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz>(x, weight, weightScale, xScale,
                weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(x, weight, weightScale, xScale,
                weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
        }
    } else {
        if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(x, weight, weightScale,
                xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
        } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
            GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(x, weight, weightScale,
                xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
        }
    }
#elif ORIG_DTYPE_X_SCALE == DT_FLOAT
    if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS &&
        QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) { // transX = false, transW = false
        GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling, &tPipe);
    } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS &&
               QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) { // transX = false, transW = true
        GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
            x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
            tiling, &tPipe);
    }
#endif
#endif
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}
