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
 * \file mla_prolog_v2.cpp
 * \brief
 */

#include "../../mla_prolog/op_kernel/kernel_mla_prolog_split_n.h"

using namespace MlaProlog;

#define INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(templateClass, ...)                                                        \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(MlaPrologTilingData, baseParams, tilingDataIn, tiling);                                 \
        const MlaPrologTilingData* __restrict tilingData = nullptr;                                                    \
        const MlaPrologBaseParams *__restrict tilingDataBaseParams = &tilingDataIn;                                    \
        templateClass<MLAPType<__VA_ARGS__>> op(&pipe, tilingData, tilingDataBaseParams);                              \
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,         \
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,        \
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, queryOut, queryRopeOut,              \
                dequantScaleQNopeOut, workspace);                                                                      \
        op.Process();                                                                                                  \
    } while (0)

extern "C" __global__ __aicore__ void mla_prolog_v2(
    __gm__ uint8_t *tokenX,
    __gm__ uint8_t *weightDq,
    __gm__ uint8_t *weightUqQr, 
    __gm__ uint8_t *weightUk, 
    __gm__ uint8_t *weightDkvKr, 
    __gm__ uint8_t *rmsnormGammaCq,
    __gm__ uint8_t *rmsnormGammaCkv, 
    __gm__ uint8_t *ropeSin, 
    __gm__ uint8_t *ropeCos, 
    __gm__ uint8_t *cacheIndex,
    __gm__ uint8_t *kvCache, 
    __gm__ uint8_t *krCache, 
    __gm__ uint8_t *dequantScaleX, 
    __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *dequantScaleWUqQr, 
    __gm__ uint8_t *dequantScaleWDkvKr, 
    __gm__ uint8_t *quantScaleCkv,
    __gm__ uint8_t *quantScaleCkr, 
    __gm__ uint8_t *smoothScalesCq, 
    __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut, 
    __gm__ uint8_t *kvCacheOut, 
    __gm__ uint8_t *krCacheOut,
    __gm__ uint8_t *dequantScaleQNopeOut, 
    __gm__ uint8_t *workspace, 
    __gm__ uint8_t *tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    // 个位代表 CACHE_MOD 0-CACHE_MODE_BNSD(预留)   1-PA_BSND    2-PA_NZ
    // 十位代表场景    0-FP16(预留)     1-BF16      2-量化场景
    // 百位代表量化场景    0-MMQcQr量化    1-MMQcQr量化+KVcache量化    2-MMcqCkvKr量化+MMQcQr量化    3-MMCqCkvkr量化+MMQcQr量化+KVcache量化
    // 千位代表反量化使能场景  0-关闭  1-开启
    // 万位代表量化的算力分组场景  0-关闭  1-开启
    // 十万位代表空tensor场景 NON_EMPTY 无空tensor EMPTY_CACHE kv_cache, kr_cache为空 EMPTY_QUERY query为空, cache不更新    
    // MlaPrologV2<inputType1, inputType2, input3Type3, CACHE_MODE>
    // inputType1: token_x weight_dq weight_dkv_kr
    // inputType2: weight_uq_qr
    // input3Type3: kv_cache kr_cache
    if (TILING_KEY_IS(10000000000200000)) {
        // B/S1/T = 0, emtpy query
        return;
    }

    // 非量化场景
#if (ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_BF16)
    TILING_KEY_IS(10000000000000011);
    TILING_KEY_IS(10000000000000012);
    TILING_KEY_IS(10000000000100011);
    TILING_KEY_IS(10000000000100012);

    #if TILING_KEY_VAR == 10000000000000011
        // input BF16, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000000012
        // input BF16, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000100011
        // input BF16, cache_mode PA_BSND, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000100012
        // input BF16, cache_mode PA_NZ, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, bfloat16_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #endif
#endif

    // 半量化kv非量化
#if (ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_BF16)
    TILING_KEY_IS(10000000000000021);
    TILING_KEY_IS(10000000000000022);
    TILING_KEY_IS(10000000000010021);
    TILING_KEY_IS(10000000000010022);
    TILING_KEY_IS(10000000000001021);
    TILING_KEY_IS(10000000000001022);
    TILING_KEY_IS(10000000000100021);
    TILING_KEY_IS(10000000000100022);
    TILING_KEY_IS(10000000000101021);
    TILING_KEY_IS(10000000000101022);

    #if TILING_KEY_VAR == 10000000000000021
        // quant scenario MMQcQr量化, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000000022
        // quant scenario MMQcQr量化, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000010021
        // quant scenario MMQcQr量化, cache_mode PA_BSND, enable group_compute_opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, true, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000010022
        // quant scenario MMQcQr量化, cache_mode PA_NZ, enable group_compute_opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, true, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001021
        // quant scenario MMQcQr量化, cache_mode PA_BSND, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001022
        // quant scenario MMQcQr量化, cache_mode PA_NZ, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000100021
        // quant scenario MMQcQr量化, cache_mode PA_BSND, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000100022
        // quant scenario MMQcQr量化, cache_mode PA_NZ, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101021
        // quant scenario MMQcQr量化, cache_mode PA_BSND, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101022
        // quant scenario MMQcQr量化, cache_mode PA_NZ, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #endif 
#endif

    // 半量化kv量化
#if (ORIG_DTYPE_TOKEN_X == DT_BF16) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_INT8)
    TILING_KEY_IS(10000000000000121);
    TILING_KEY_IS(10000000000000122);
    TILING_KEY_IS(10000000000010121);
    TILING_KEY_IS(10000000000010122);
    TILING_KEY_IS(10000000000001121);
    TILING_KEY_IS(10000000000001122);
    TILING_KEY_IS(10000000000100121);
    TILING_KEY_IS(10000000000100122);
    TILING_KEY_IS(10000000000101121);
    TILING_KEY_IS(10000000000101122);

    #if TILING_KEY_VAR == 10000000000000121
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000000122
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000010121
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_BSND, enable group_compute_opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, false, true, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000010122
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_NZ, enable group_compute_opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, false, true, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001121
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_BSND, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001122
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_NZ, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000100121
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_BSND, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000100122
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_NZ, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101121
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_BSND, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101122
        // quant scenario MMQcQr量化+KVcache量化, cache_mode PA_NZ, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, bfloat16_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #endif
#endif

    // 全量化kv非量化
#if (ORIG_DTYPE_TOKEN_X == DT_INT8) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_BF16)
    TILING_KEY_IS(10000000000000221);
    TILING_KEY_IS(10000000000000222);
    TILING_KEY_IS(10000000000001221);
    TILING_KEY_IS(10000000000001222);
    TILING_KEY_IS(10000000000100221);
    TILING_KEY_IS(10000000000100222);
    TILING_KEY_IS(10000000000101221);
    TILING_KEY_IS(10000000000101222);

    #if TILING_KEY_VAR == 10000000000000221
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000000222
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001221
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_BSND, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001222
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_NZ, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000100221
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_BSND, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000100222
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_NZ, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101221
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_BSND, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101222
        // quant scenario MMCqCkvKr量化+MMQcQr量化, cache_mode PA_NZ, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, bfloat16_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #endif
#endif

    // 全量化kv量化
#if (ORIG_DTYPE_TOKEN_X == DT_INT8) && (ORIG_DTYPE_WEIGHT_UQ_QR == DT_INT8) && (ORIG_DTYPE_KV_CACHE == DT_INT8) && \
    (ORIG_DTYPE_KR_CACHE == DT_BF16)
    TILING_KEY_IS(10000000000000321);
    TILING_KEY_IS(10000000000000322);
    TILING_KEY_IS(10000000000001321);
    TILING_KEY_IS(10000000000001322);
    TILING_KEY_IS(10000000000100321);
    TILING_KEY_IS(10000000000100322);
    TILING_KEY_IS(10000000000101321);
    TILING_KEY_IS(10000000000101322);

    #if TILING_KEY_VAR == 10000000000000321
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_BSND
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000000322
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_NZ
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001321
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_BSND, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000001322
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_NZ, enable dequant opt
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::NON_EMPTY);
    #elif TILING_KEY_VAR == 10000000000100321
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_BSND, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000100322
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_NZ, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, false, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101321
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_BSND, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_BSND, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #elif TILING_KEY_VAR == 10000000000101322
        // quant scenario MMCqCkvKr量化+MMQcQr量化+KVcache量化, cache_mode PA_NZ, enable dequant opt, empty cache
        INVOKE_MLA_PROLOG_V2_NO_KFC_OP_IMPL(MlaPrologVecS1CubS2, int8_t, int8_t, int8_t,
                                         CACHE_MODE::PA_NZ, true, false, EMPTY_TENSOR_MODE::EMPTY_CACHE);
    #endif
#endif
}