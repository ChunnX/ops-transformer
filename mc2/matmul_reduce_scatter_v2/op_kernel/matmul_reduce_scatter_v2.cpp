/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_v2.cpp
 * \brief
 */

#include "matmul_reduce_scatter_v2_tiling_key.h"
#ifdef __DAV_C310__
#include "lib/matmul_intf.h"
#include "common_def.h"

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "matmul_reduce_scatter_fp16_bf16.h"
using namespace Mc2Tiling;
#endif
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||        \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
#include "quant_bmm_reduce_scatter_fp8_hif8.h"
using namespace Mc2Tiling;
#endif
#else
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "matmul_reduce_scatter_aiv_mode.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
using namespace matmulReduceScatterV2_aivmode_tiling;
#endif

using namespace AscendC;
using namespace MatmulReduceScatterV2Impl;

#define INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(templateClass)                             \
    do {                                                                                    \
        using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>;   \
        using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;          \
        REGISTER_TILING_DEFAULT(Mc2Tiling::MatmulReduceScatterV2TilingData);                           \
        auto tiling = (__gm__ Mc2Tiling::MatmulReduceScatterV2TilingData*)tilingGM;                                   \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                      \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                         \
        templateClass<AType, BType, BiasType, CType> op;                                    \
        op.Init(aGM, bGM, biasGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData, mc2InitTiling, mc2CcTiling, &pipe); \
        op.Process();                                                                       \
    } while (0)

#define INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                                             \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);                                          \
        auto tiling = (__gm__ Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData*)tilingGM;                                   \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                      \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                         \
        if (tilingData.msg.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) {                                \
            using mmClass = MatMulASWKernel<DTYPE_X1, DTYPE_X2, float, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND,          \
                                            CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>;                            \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, \
                          mmClass, false, __VA_ARGS__> op;                                                   \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData,         \
                    mc2InitTiling, mc2CcTiling, &pipe);                                                             \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

#define INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                              \
    do {                                                                                                             \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);                                         \
        auto tiling = (__gm__ Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData*)tilingGM;                                   \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                      \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                         \
        if (tilingData.msg.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) {                                \
            using mmClass = MatMulASWKernel<DTYPE_X1, DTYPE_X2, fp8_e8m0_t, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND,     \
                                            CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>;                            \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, fp8_e8m0_t, mmClass, false, __VA_ARGS__> op;                  \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData,          \
                    mc2InitTiling, mc2CcTiling, &pipe);                                                             \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

#define INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                                      \
    do {                                                                                                              \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);                                           \
        auto tiling = (__gm__ Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData*)tilingGM;                                   \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                      \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                         \
        if (tilingData.msg.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) {                                 \           
            using mmClass =                                                                                           \
                Mc2QuantBatchMatmulV3::MatMulPerBlockASW<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y,                        \
                                                         CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>;   \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, mmClass, true, __VA_ARGS__> op;                         \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData,           \
                    mc2InitTiling, mc2CcTiling, &pipe);                                                             \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

template<TPL_AIV_MODE_TILING_PARAMS_COMM, TPL_PARAMS_COMM, TPL_BASE_TILING_PARAMS_COMM, TPL_QUANT_BMM_TILING_PARAMS_COMM> 
__global__ __aicore__ void matmul_reduce_scatter_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                                               GM_ADDR quantScaleGM, GM_ADDR cGM, GM_ADDR amaxOutGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
#ifdef __DAV_C310__
// david算子模板
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    auto context = GetHcclContext<0>();
    GET_TILING_DATA(tilingData, tilingGM); 
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    // bf16/fp16 场景
    using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>;
    if constexpr (TPL_COMMALG == TPL_COMM_ALG_FULL_MESH && TPL_TRANSPOSE == TPL_X1_X2_NO_TRANSPOSE) {  // david + fullmesh + no nd2nz + bais not cast
        using BType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, false>;
        INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulReduceScatterFP16BF16);
    } else if constexpr (TPL_COMMALG == TPL_COMM_ALG_FULL_MESH && TPL_TRANSPOSE == TPL_X2_TRANSPOSE) {    //  david + transb + fullmesh + no nd2nz + bais not cast
        using BType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, true>;
        INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulReduceScatterFP16BF16);
    }
#elif (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_HIFLOAT8))) ||        \
       (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
        ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
    // float8/hif8
    #if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
        if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_FP8E8M0 && TPL_TRANSPOSE == TPL_X2_TRANSPOSE) {
            INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, true);
        } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_FP8E8M0 && TPL_TRANSPOSE == TPL_X1_X2_NO_TRANSPOSE) {
            INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, false);
        }
    #endif
    if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                TPL_TRANSPOSE == TPL_X1_X2_NO_TRANSPOSE && !TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, false);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X1_TRANSPOSE && !TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, true, false);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X2_TRANSPOSE && !TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, true);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X1_X2_TRANSPOSE && !TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, true, true);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X1_X2_NO_TRANSPOSE && TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, false);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X1_TRANSPOSE && TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, true, false);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X2_TRANSPOSE && TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, true);
    } else if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER && \
                        TPL_TRANSPOSE == TPL_X1_X2_TRANSPOSE && TPL_ISPERBLOCK) {
        INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, true, true);
    }
#endif

#else
//aiv算子模板

    #define INVOKE_MMREDUCESCATTER_AIV_MODE_OP_IMPL(templateClass, ...)                                              \
        do {                                                                                                    \
            GET_TILING_DATA_WITH_STRUCT(MatmulReduceScatterV2AivModeTilingData, tilingData, tilingGM);          \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X2_SCALE, DTYPE_Y, __VA_ARGS__> op;                             \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, workspaceGM, tilingGM);                        \
            op.Process();                                                                                       \
        } while (0)

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MatmulReduceScatterV2AivModeTilingData);
    if constexpr (!TPL_IS_TRANSPOSE_B) {
        //aivMode，非transB
        INVOKE_MMREDUCESCATTER_AIV_MODE_OP_IMPL(MatmulReduceScatterAivMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, false);
    } else if constexpr (TPL_IS_TRANSPOSE_B) {
        //aivMode，transB
        INVOKE_MMREDUCESCATTER_AIV_MODE_OP_IMPL(MatmulReduceScatterAivMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, true);
    }
#endif
}