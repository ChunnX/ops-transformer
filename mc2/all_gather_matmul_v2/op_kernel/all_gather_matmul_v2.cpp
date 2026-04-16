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
 * \file all_gather_matmul_v2.cpp
 * \brief
 */

#ifdef __DAV_C310__
#include "lib/matmul_intf.h"
#include "common.h"

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "all_gather_matmul_fp16_bf16.h"
#endif
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||       \
     ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
         ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))
#include "all_gather_quant_bmm.h"
#include "../3rd/quant_batch_matmul_v3/op_kernel/arch35/qbmm_mix_perblock.h"
#include "all_gather_quant_bmm_perblock.h"
#endif

#else
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "all_gather_matmul_aiv_mode.h"
#include "all_gather_matmul_aiv_mode_tiling.h"
#endif

#ifdef __DAV_C310__
using namespace Mc2Tiling;
using namespace AllGatherMatmulImpl;
#else
using namespace AllGatherMatmulAIVModeImpl;
#endif
using namespace AscendC;

#define INVOKE_ALLGATHERMM_FP16_BF16_V2_OP_IMPL(templateClass, isTransB, ...)                                     \
    do {                                                                                                          \
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>;                         \
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, isTransB>;                      \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>; \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;                                \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataV2);                                          \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataV2*)tilingGM;                                   \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                    \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                        \
        GET_TILING_DATA(tilingData, tilingGM);                                                                    \
        templateClass<aType, bType, biasType, cType> op;                                                          \
        op.Init(aGM, bGM, biasGM, cGM, (__gm__ uint8_t*)context, workspaceGM, gatherOut, &tilingData,             \
                                        mc2InitTiling, mc2CcTiling, &pipe);                                       \
        op.Process();                                                                                             \
    } while (0)

#define INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(templateClass, ...)                                               \
    do {                                                                                                              \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataFp8);                                             \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataFp8*)tilingGM;                                      \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                        \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                        \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, uint64_t, DTYPE_Y, __VA_ARGS__> op;                             \
        op.Init(aGM, bGM, scaleInv1, scaleInv2, biasGM, scale, cGM, gatherOut, workspaceGM, (__gm__ uint8_t*)context, \
                &tilingData, mc2InitTiling, mc2CcTiling, &pipe);                                                      \
        op.Process();                                                                                                 \
    } while (0)

#define INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL(templateClass, ...)                                            \
    do {                                                                                                              \
        REGISTER_TILING_DEFAULT(Mc2Tiling::AllGatherMatmulTilingDataFp8);                                             \
        auto tiling = (__gm__ Mc2Tiling::AllGatherMatmulTilingDataFp8*)tilingGM;                                      \
        __gm__ void* mc2InitTiling = (__gm__ void*)(&(tiling->mc2InitTiling));                                        \
        __gm__ void* mc2CcTiling = (__gm__ void*)(&(tiling->mc2CcTiling));                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                        \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, fp8_e8m0_t, DTYPE_Y, __VA_ARGS__> op;                           \
        op.Init(aGM, bGM, scaleInv1, scaleInv2, biasGM, scale, cGM, gatherOut, workspaceGM, (__gm__ uint8_t*)context, \
                &tilingData, mc2InitTiling, mc2CcTiling, &pipe);                                                      \
        op.Process();                                                                                                 \
    } while (0)

extern "C" __global__ __aicore__ void all_gather_matmul_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scaleInv1,
                                                           GM_ADDR scaleInv2, GM_ADDR scale, GM_ADDR cGM,
                                                           GM_ADDR gatherOut, GM_ADDR amax, GM_ADDR workspaceGM,
                                                           GM_ADDR tilingGM)
{
#ifdef __DAV_C310__
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    __gm__ HcclCombinOpParam* context = (__gm__ HcclCombinOpParam*)(GetHcclContext<0>());

#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_HIFLOAT8)) ||       \
     ((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
         ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2)))
        // MX
        #if (ORIG_DTYPE_X1 != DT_HIFLOAT8)
            if (TILING_KEY_IS(1000000000012001100UL) || TILING_KEY_IS(1000000000012021100UL)) {
                INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL(AllGatherQuantBmm, false, true);
            } else if (TILING_KEY_IS(1000000000010001100UL) || TILING_KEY_IS(1000000000010021100UL)) {
                INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_MX_OP_IMPL(AllGatherQuantBmm, false, false);
            }
        #endif
    if (TILING_KEY_IS(1000000000000001100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(AllGatherQuantBmm, false, false);
    } else if (TILING_KEY_IS(1000000000000021100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(AllGatherQuantBmm, false, false);
    } else if (TILING_KEY_IS(1000000000002021100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(AllGatherQuantBmm, false, true);
    } else if (TILING_KEY_IS(1000000000002001100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_OP_IMPL(AllGatherQuantBmm, false, true);
    } else if (TILING_KEY_IS(1000000000000101100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_PERBLOCK_OP_IMPL(Mc2QuantBatchMatmulV3::MatMulPerBlockASW,
                                                             Mc2CoreType::ON_CUBE, false, false);
    } else if (TILING_KEY_IS(1000000000000121100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_PERBLOCK_OP_IMPL(Mc2QuantBatchMatmulV3::MatMulPerBlockASW,
                                                             Mc2CoreType::ON_CUBE, false, false);
    } else if (TILING_KEY_IS(1000000000002121100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_PERBLOCK_OP_IMPL(Mc2QuantBatchMatmulV3::MatMulPerBlockASW,
                                                             Mc2CoreType::ON_CUBE, false, true);
    } else if (TILING_KEY_IS(1000000000002101100UL)) {
        INVOKE_ALL_GATHER_QUANT_BATCHMATMUL_PERBLOCK_OP_IMPL(Mc2QuantBatchMatmulV3::MatMulPerBlockASW,
                                                             Mc2CoreType::ON_CUBE, false, true);
    }
#elif ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
      // B矩阵根据tilingkey第7位判断, A矩阵不支持转置
    if (TILING_KEY_IS(1000000000000000100UL)) {    // full mesh+ no nd2nz +biasNoNeedCast
        INVOKE_ALLGATHERMM_FP16_BF16_V2_OP_IMPL(AllGatherMatmulFP16BF16, false);
    } else if(TILING_KEY_IS(1000000000002000100UL)) {
        INVOKE_ALLGATHERMM_FP16_BF16_V2_OP_IMPL(AllGatherMatmulFP16BF16, true);
    }
#endif

#else
//aiv算子模板

    #define INVOKE_ALLGATHERMATMUL_AIV_MODE_OP_IMPL(templateClass, ...)                                              \
    do {                                                                                                    \
        GET_TILING_DATA_WITH_STRUCT(AllGatherMatmulAIVModeTilingData, tilingData, tilingGM);          \
        templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X2_SCALE, DTYPE_Y, __VA_ARGS__> op;                             \
        op.Init(aGM, bGM, biasGM, scaleInv1, scaleInv2, cGM, gatherOut, workspaceGM, tilingGM);                        \
        op.Process();                                                                                       \
    } while (0)

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(AllGatherMatmulAIVModeTilingData);
    if (TILING_KEY_IS(10000)) {
        //aivMode，非transB
        INVOKE_ALLGATHERMATMUL_AIV_MODE_OP_IMPL(AllGatherMatmulAIVMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, false);
    } else if (TILING_KEY_IS(10010)) {
        //aivMode，transB
        INVOKE_ALLGATHERMATMUL_AIV_MODE_OP_IMPL(AllGatherMatmulAIVMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, true);
    }
#endif
}