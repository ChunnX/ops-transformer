/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_APT_QUANT_H
#define GROUPED_MATMUL_APT_QUANT_H

#if defined(V310_GMM_QUANT)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_tiling_key.h"
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cube_on_the_fly.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_init_output.h"
#endif
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_mix_online_dynamic.h"
#endif
#if defined(V310_GMM_QUANT_PERTILE)
#include "arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_pertile_kernel.h"
#endif

#define GMM_QUANT_IMPL_CLASS(T_X, T_W, T_BIAS, T_SC, T_Y, W_FMT, transposeX1, transposeX2, templateClass)             \
    do {                                                                                                               \
        templateClass<T_X, T_W, T_BIAS, T_SC, T_Y, W_FMT,                                                             \
                      transposeX1, transposeX2>                                                                        \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_MIX_IMPL_CLASS(T_X, T_W, T_BIAS, T_SC, T_Y, W_FMT, T_L0C,                                          \
                                 transposeX1, transposeX2, templateClass)                                              \
    do {                                                                                                               \
        templateClass<T_X, T_W, T_BIAS, T_SC, float, T_Y, W_FMT,                                                      \
                      transposeX1, transposeX2, T_L0C>                                                                 \
            op;                                                                                                        \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,          \
                gmmArrayAddr_, &tPipe);                                                                                \
        op.Process();                                                                                                  \
    } while (0)

#define GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(T_X, T_W, T_BIAS, T_SC, T_Y, W_FMT,                                   \
                                               transposeX1, transposeX2, templateClass)                                \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GET_TILING_DATA_MEMBER_ADDR(GMMQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                              \
        if ASCEND_IS_AIC {                                                                                             \
            templateClass<T_X, T_W, T_BIAS, T_SC, T_Y, W_FMT,                                                         \
                          transposeX1, transposeX2>                                                                    \
                op;                                                                                                    \
            op.Init(x, weight, bias, scale, groupList, perTokenScale, y, user1, &gmmQuantParams_, &mmTilingData_,      \
                    gmmArrayAddr_, &tPipe);                                                                            \
            op.Process();                                                                                              \
        }                                                                                                              \
        if ASCEND_IS_AIV {                                                                                             \
            GQmmEmptyTensor<T_Y>(groupList, y, &gmmQuantParams_, mmTilingData_.M, mmTilingData_.N);                    \
        }                                                                                                              \
    } while (0)

#define GMM_QUANT_GB_IMPL_CLASS(T_X, T_W, T_BIAS, T_SC, T_Y, T_L0C, xLayout, wLayout, yLayout)                        \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, gmmQuantParams, gmmQuantParams_, tiling);                           \
        GET_TILING_DATA_MEMBER(GMMQuantTilingData, mmTilingData, mmTilingData_, tiling);                               \
        GmmCgmctPerTileKernel<T_X, T_W, T_BIAS, T_SC, float, T_Y, xLayout, wLayout, yLayout,                          \
                            T_L0C>(x, weight, bias, scale, groupList, perTokenScale, y, user1,                         \
                                             &gmmQuantParams_, &mmTilingData_, &tPipe);                                \
    } while (0)

#define GROUPED_MATMUL_QUANT_IMPL_BODY(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY, TDtypeScale, TDtypeL0cLocal, kWFormat, \
                                       QUANT_B_TRANS, QUANT_A_TRANS, KERNEL_TYPE)                                       \
    REGISTER_TILING_DEFAULT(GMMQuantTilingData);                                                                       \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                                                    \
#if defined(V310_GMM_QUANT_MX)                                                                                         \
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                       \
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {                                                                          \
        GMM_QUANT_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat,                        \
                             false, false, GmmASWKernel);                                                              \
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                   \
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {                                                                          \
        GMM_QUANT_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat,                        \
                             false, true, GmmASWKernel);                                                               \
    }                                                                                                                  \
#endif                                                                                                                 \
#if defined(V310_GMM_QUANT_CUBE) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)                                            \
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                       \
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {                                                                          \
        GMM_QUANT_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat,                        \
                             false, false, GmmASWKernel);                                                              \
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                   \
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {                                                                          \
        GMM_QUANT_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat,                        \
                             false, true, GmmASWKernel);                                                               \
    }                                                                                                                  \
#endif                                                                                                                 \
#if defined(V310_GMM_QUANT_MX) || defined(V310_GMM_QUANT_PERTENSOR_CUBE)                                              \
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS                                          \
        && KERNEL_TYPE == GMM_DEQUANT_FIXP) {                                                                          \
        GMM_QUANT_WITH_EMPTY_TENSOR_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat,      \
                                               true, false, GmmASWKernel);                                            \
    }                                                                                                                  \
#endif                                                                                                                 \
#if defined(V310_GMM_QUANT_MIX)                                                                                        \
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                       \
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {                                                                        \
        GMM_QUANT_MIX_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat, TDtypeL0cLocal,    \
                                 false, false, GQmmMixRegbaseKernel);                                                  \
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                   \
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {                                                                        \
        GMM_QUANT_MIX_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat, TDtypeL0cLocal,    \
                                 false, true, GQmmMixRegbaseKernel);                                                   \
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS                                   \
        && KERNEL_TYPE == GMM_DEQUANT_VECTOR) {                                                                        \
        GMM_QUANT_MIX_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, kWFormat, TDtypeL0cLocal,    \
                                 true, false, GQmmMixRegbaseKernel);                                                   \
    }                                                                                                                  \
#endif                                                                                                                 \
#if defined(V310_GMM_QUANT_PERTILE)                                                                                    \
    if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                       \
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {                                                                     \
        GMM_QUANT_GB_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, TDtypeL0cLocal,               \
                                Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::RowMajor,                          \
                                Cgmct::Gemm::layout::RowMajorAlign);                                                   \
    } else if constexpr (QUANT_B_TRANS == GMM_TRANS && QUANT_A_TRANS == GMM_NO_TRANS                                   \
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {                                                                     \
        GMM_QUANT_GB_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, TDtypeL0cLocal,               \
                                Cgmct::Gemm::layout::RowMajor, Cgmct::Gemm::layout::ColumnMajor,                       \
                                Cgmct::Gemm::layout::RowMajorAlign);                                                   \
    } else if constexpr (QUANT_B_TRANS == GMM_NO_TRANS && QUANT_A_TRANS == GMM_TRANS                                   \
        && KERNEL_TYPE == GMM_PERGROUP_PERBLOCK) {                                                                     \
        GMM_QUANT_GB_IMPL_CLASS(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeScale, TDtypeY, TDtypeL0cLocal,               \
                                Cgmct::Gemm::layout::ColumnMajor, Cgmct::Gemm::layout::RowMajor,                       \
                                Cgmct::Gemm::layout::RowMajorAlign);                                                   \
    }                                                                                                                  \
#endif

#endif
#endif