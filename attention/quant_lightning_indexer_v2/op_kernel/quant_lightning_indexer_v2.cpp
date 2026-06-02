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
 * \file quant_lightning_indexer_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_lightning_indexer_v2_template_tiling_key.h"

#include "arch35/quant_lightning_indexer_v2_kernel.h"

using namespace QLIKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<QLIType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(QLITilingData, tiling_data_in, tiling);                                  \
        const QLITilingData *__restrict tiling_data = &tiling_data_in;                                       \
        op.Init(query, key, weights, queryScale, keyScale, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK,       \
        cmpResidualK, blockTable, outputIdxOffset, metadata,                                                 \
        sparseIndices, sparseValues, user, tiling_data, &tPipe);                                             \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer_v2_super(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                            __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                            __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                                            __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable,
                                                            __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata,
                                                            __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if (__CCE_AICORE__ == 310)
    if (ORIG_DTYPE_Q == DT_FLOAT8_E4M3FN) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, fp8_e4m3fn_t, fp8_e4m3fn_t, float, uint16_t, int32_t,
                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                float32_t, float32_t, float32_t);
    } else if (ORIG_DTYPE_Q == DT_HIFLOAT8) {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, hifloat8, hifloat8, float, uint16_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                bfloat16_t, float32_t, float32_t);
    } else {
        INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, int8_t, int8_t, float, uint16_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T),
                                half, half, int32_t);
    }
#else
    INVOKE_LI_NO_KFC_OP_IMPL(QLIPreload, int8_t, int8_t, float, uint16_t, int32_t,
                                PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
#endif
}

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void quant_lightning_indexer_v2(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w,
                                                      __gm__ uint8_t *qDeScale, __gm__ uint8_t *keyDeScale,
                                                      __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                                      __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK, 
                                                      __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blockTable,
                                                      __gm__ uint8_t *outputIdxOffset, __gm__ uint8_t *metadata,
                                                      __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                      __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    quant_lightning_indexer_v2_super<DT_Q, DT_K, DT_OUT, PAGE_ATTENTION, Q_LAYOUT_T, K_LAYOUT_T>(q, k, w, qDeScale, keyDeScale,
                                                                                                 cuSeqlensQ, cuSeqlensK,
                                                                                                 sequsedQ, sequsedK,
                                                                                                 cmpResidualK, blockTable,
                                                                                                 outputIdxOffset, metadata, 
                                                                                                 sparseIndices, sparseValues,
                                                                                                 workspace, tiling);
}