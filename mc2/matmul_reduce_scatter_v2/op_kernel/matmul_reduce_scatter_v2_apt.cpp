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
 * \file matmul_reduce_scatter_v2_apt.cpp
 * \brief
 */

#include "matmul_reduce_scatter_v2_apt_tiling_key.h"
#include "lib/matmul_intf.h"
#include "common_def.h"

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "arch35/matmul_a2a_vec_reduce_fp16_bf16.h" // 新实现：All2All + Vec Reduce
#include "arch35/matmul_reduce_scatter_fp16_bf16.h" // 旧实现：原生 ReduceScatter
using namespace Mc2Tiling;
#endif
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||        \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
#include "arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h" // 新实现：All2All + Vec Reduce
#include "arch35/quant_bmm_reduce_scatter_fp8_hif8.h" // 旧实现：原生 ReduceScatter
using namespace Mc2Tiling;
#endif

using namespace MatmulReduceScatterV2Impl;

#define INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(templateClass)                                                        \
    do {                                                                                                               \
        using AType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>;                              \
        using CType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;                                     \
        REGISTER_TILING_DEFAULT(Mc2Tiling::MatmulReduceScatterV2TilingData);                                           \
        auto tiling = (__gm__ Mc2Tiling::MatmulReduceScatterV2TilingData*)tilingGM;                                    \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                         \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                             \
        templateClass<AType, BType, BiasType, CType> op;                                                               \
        op.Init(aGM, bGM, biasGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData, mc2InitTiling, mc2CcTiling, &pipe); \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                                             \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);                               \
        auto tiling = (__gm__ Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData*)tilingGM;                        \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                       \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                           \
        if (tilingData.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) {                                \
            using mmClass = MatMulASWKernel<DTYPE_X1, DTYPE_X2, float, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND,          \
                                            CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>;                            \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, float,                                                        \
                          mmClass, false, __VA_ARGS__> op;                                                           \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData,         \
                    mc2InitTiling, mc2CcTiling, &pipe);                                                              \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

#define INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                              \
    do {                                                                                                             \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);                               \
        auto tiling = (__gm__ Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData*)tilingGM;                        \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                       \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                           \
        if (tilingData.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) {                                \
            using mmClass = MatMulASWKernel<DTYPE_X1, DTYPE_X2, fp8_e8m0_t, DTYPE_BIAS, DTYPE_Y, CubeFormat::ND,     \
                                            CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>;                            \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, fp8_e8m0_t, mmClass, false, __VA_ARGS__> op;                  \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData,         \
                    mc2InitTiling, mc2CcTiling, &pipe);                                                              \
            op.Process();                                                                                            \
        }                                                                                                            \
    } while (0)

#define INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(templateClass, ...)                                      \
    do {                                                                                                              \
        REGISTER_TILING_DEFAULT(Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData);                                \
        auto tiling = (__gm__ Mc2Tiling::QuantBatchMatmulV3ReduceScatterTilingData*)tilingGM;                         \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                        \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                            \
        if (tilingData.debugMode != static_cast<uint8_t>(MC2_DEBUG_ONLY_AICPU)) {                                 \
            using mmClass =                                                                                           \
                Mc2QuantBatchMatmulV3::MatMulPerBlockASWNonContiguous<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y,        \
                                                         CubeFormat::ND, CubeFormat::ND, CubeFormat::ND, __VA_ARGS__>;\
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_Y, float, mmClass, true, __VA_ARGS__> op;                         \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, (GM_ADDR)context, workspaceGM, &tilingData,          \
                    mc2InitTiling, mc2CcTiling, &pipe);                                                               \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

template<bool TPL_ISPERBLOCK, bool TPL_TRANSA, bool TPL_TRANSB, bool TPL_INPUT, uint8_t TPL_OUTPUTDTYPE, uint8_t TPL_SCALETYPE, uint8_t TPL_COMMALG>
__global__ __aicore__ void matmul_reduce_scatter_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                    GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                                    GM_ADDR quantScaleGM, GM_ADDR cGM, GM_ADDR amaxOutGM,
                                                    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
// david算子模板
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    auto context = GetHcclContext<0>();
    GET_TILING_DATA(tilingData, tilingGM); 
    
    AscendC::printf("[APT_DBG] Template params: TPL_INPUT=%d, TPL_OUTPUTDTYPE=%d, TPL_SCALETYPE=%d, TPL_COMMALG=%d\n",
                    TPL_INPUT, TPL_OUTPUTDTYPE, TPL_SCALETYPE, TPL_COMMALG);
    AscendC::printf("[APT_DBG] TPL_ISPERBLOCK=%d, TPL_TRANSA=%d, TPL_TRANSB=%d\n",
                    TPL_ISPERBLOCK, TPL_TRANSA, TPL_TRANSB);
    AscendC::printf("[APT_DBG] DTYPE: X1=%d, X2=%d, Y=%d, BIAS=%d\n",
                    ORIG_DTYPE_X1, ORIG_DTYPE_X2, DTYPE_Y, DTYPE_BIAS);
    AscendC::printf("[APT_DBG] Constants: DT_BF16=%d, DT_FLOAT16=%d, INPUT_TYPE_IS_FP16_BF16=%d, OUTPUT_TYPE_IS_FP8=%d\n",
                    DT_BF16, DT_FLOAT16, INPUT_TYPE_IS_FP16_BF16, OUTPUT_TYPE_IS_FP8);
    
#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
    AscendC::printf("[APT_DBG] Enter BF16/FP16 branch: ORIG_DTYPE_X1=%d\n", ORIG_DTYPE_X1);
    using BiasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename MatmulReduceScatterV2Impl::BiasType<BIAS_DTYPE>::type>;
    AscendC::printf("[APT_DBG] BIAS_DTYPE=%d, BiasType_is_bf16=%d\n", 
                    BIAS_DTYPE, std::is_same_v<typename MatmulReduceScatterV2Impl::BiasType<BIAS_DTYPE>::type, bfloat16_t>);
    if constexpr (!TPL_ISPERBLOCK && TPL_INPUT == INPUT_TYPE_IS_FP16_BF16 && \
                TPL_OUTPUTDTYPE == OUTPUT_TYPE_IS_FP8 && TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER) {
        AscendC::printf("[APT_DBG] Enter MatmulReduceScatterFP16BF16 kernel\n");
        using BType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, TPL_TRANSB>;
        if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
            INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulA2AVecReduceFP16BF16);
        } else {
            INVOKE_MMREDUCESCATTER_FP16_BF16_OP_IMPL(MatmulReduceScatterFP16BF16);
        }
    } else {
        AscendC::printf("[APT_DBG] NOT enter MatmulReduceScatterFP16BF16 kernel, conditions: ISPERBLOCK=%d, INPUT=%d==%d?, OUTPUT=%d==%d?, SCALE=%d==%d?\n",
                        TPL_ISPERBLOCK, TPL_INPUT, INPUT_TYPE_IS_FP16_BF16, TPL_OUTPUTDTYPE, OUTPUT_TYPE_IS_FP8, TPL_SCALETYPE, TPL_X1_X2_DTYPE_IS_OTHER);
    }
#elif (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_HIFLOAT8))) ||        \
       (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
        ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))))
    AscendC::printf("[APT_DBG] Enter FP8/HiF8 branch\n");
    #if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
        if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_FP8E8M0) {
            if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, false, TPL_TRANSB);
            } else {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, TPL_TRANSB);
            }
        }
    #endif
    if constexpr (!TPL_ISPERBLOCK && TPL_INPUT == INPUT_TYPE_IS_FP8 && TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER) {
        if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
            INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, TPL_TRANSA, TPL_TRANSB);
        } else {
            INVOKE_QUANT_BATCHMM_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, TPL_TRANSA, TPL_TRANSB);
        }
    } else if constexpr (TPL_ISPERBLOCK && TPL_INPUT == INPUT_TYPE_IS_FP8 && TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_OTHER) {
        if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
            INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, TPL_TRANSA, TPL_TRANSB);
        } else {
            INVOKE_QUANT_BATCHMM_PERBLOCK_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, TPL_TRANSA, TPL_TRANSB);
        }
    }
#endif
}