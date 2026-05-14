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

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
    #include "arch35/grouped_matmul_swiglu_quant_v2_mxquant.h"
#elif ORIG_DTYPE_X_SCALE == DT_FLOAT
    #include "arch35/grouped_matmul_swiglu_quant_v2_pertoken_quant.h"
#endif
#include "arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#include "arch35/gmm_swiglu_quant_mx_vec_compute.h"

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

__aicore__ inline void ExampleCallMxVecCompute(GmmSwigluQuantMxA8W4::GmmSwigluQuantMxVecCompute<float, fp8_e4m3fn_t, float> &mxCompute,
                                               uint16_t mSize, uint16_t nSize)
{
    // constexpr uint64_t mxUbMaxSize = 244 * 1024;
    constexpr uint64_t KB_8bit = 1024;
    AscendC::LocalTensor<float> l0cOutUbFirst;
    l0cOutUbFirst = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, KB_8bit * 64,
        KB_8bit * 64);
    AscendC::LocalTensor<float> l0cOutUbSecond;
    l0cOutUbSecond = AscendC::LocalTensor<float>(
        AscendC::TPosition::VECCALC, KB_8bit * 128,
        KB_8bit * 128);

    AscendC::Duplicate<float>(l0cOutUbFirst, static_cast<float>(1), KB_8bit * 64);
    AscendC::Duplicate<float>(l0cOutUbSecond, static_cast<float>(1), KB_8bit * 128);

    mxCompute.VFDoSwigluAndQuantForMX(l0cOutUbFirst, l0cOutUbSecond, mSize, nSize);
}

template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS>
__global__ __aicore__ void grouped_matmul_swiglu_quant_v2(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight,
                                                          GM_ADDR weightScale, GM_ADDR weightAssistanceMatrix,
                                                          GM_ADDR bias, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR yScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
//     TPipe tPipe;
//     GM_ADDR userWorkspace = GetUserWorkspace(workspace);
//     int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
//     // enable overflow mode to avoid nan/inf value
//     AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
// #if ORIG_DTYPE_X_SCALE == DT_FLOAT8_E8M0
//     if (wFormat == CubeFormat::NZ) {
//         // if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
//         //     GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Nz>(x, weight, weightScale, xScale,
//         //         weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
//         // } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
//         //     GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::Zn>(x, weight, weightScale, xScale,
//         //         weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
//         // }
        
//     } else {
//         if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
//             GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(x, weight, weightScale,
//                 xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
//         } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS && QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) {
//             GmmSwigluAswt<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(x, weight, weightScale,
//                 xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace, tiling);
//         }
//     }
// #elif ORIG_DTYPE_X_SCALE == DT_FLOAT
//     if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_NO_TRANS &&
//         QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) { // transX = false, transW = false
//         GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor>(
//             x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
//             tiling, &tPipe);
//     } else if (QUANT_B_TRANS == GMM_SWIGLU_QUANT_TRANS &&
//                QUANT_A_TRANS == GMM_SWIGLU_QUANT_NO_TRANS) { // transX = false, transW = true
//         GmmSwigluAswtPertoken<Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor>(
//             x, weight, weightScale, xScale, weightAssistanceMatrix, smoothScale, groupList, y, yScale, workspace,
//             tiling, &tPipe);
//     }
// #endif
//     AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);

    // 示例：调用 GmmSwigluQuantMxVecCompute 进行 SwiGLU + MX 量化
        {
            constexpr uint16_t exampleM = 8;
            constexpr uint16_t exampleN = 8;
            constexpr uint64_t mxUbMaxSize = 244 * 1024;

            GmmSwigluQuantMxA8W4::GmmSwigluQuantMxVecCompute<float, fp8_e4m3fn_t, float> mxCompute;
            mxCompute.Init();
            mxCompute.UpdateGlobalAddr((__gm__ int8_t *)y, (__gm__ int8_t *)yScale);
            mxCompute.UpdateOutputShape(exampleN,
                                        (exampleN + GmmSwigluQuantMxA8W4::MXFP_DIVISOR_SIZE - 1) /
                                            GmmSwigluQuantMxA8W4::MXFP_DIVISOR_SIZE *
                                            GmmSwigluQuantMxA8W4::MXFP_MULTI_BASE_SIZE);

            ExampleCallMxVecCompute(mxCompute, exampleM, exampleN);
        }
}
