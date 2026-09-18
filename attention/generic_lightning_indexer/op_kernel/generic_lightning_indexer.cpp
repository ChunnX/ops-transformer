/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "generic_lightning_indexer_template_tiling_key.h"

#if (__CCE_AICORE__ == 310)
#include "arch35/generic_lightning_indexer_kernel_arch35.h"
#else
#include "arch22/generic_lightning_indexer_kernel_arch22.h"
#endif

using namespace GliKernel;

#define INVOKE_GLI_OP_IMPL(templateClass, ...) \
    do { \
        templateClass<GliType<__VA_ARGS__>> op; \
        GET_TILING_DATA_WITH_STRUCT(GenericLightningIndexerTilingData, tiling_data_in, tiling); \
        const GenericLightningIndexerTilingData *__restrict tiling_data = &tiling_data_in; \
        op.Init(query, key, weights, actualSeqQ, actualSeqK, cmpResidualK, blockTable, qDescale, kDescale, metadata, \
                sparseIndices, sparseValues, user, tiling_data, &tPipe); \
        op.Process(); \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int LAYOUT_Q, int LAYOUT_K, int SPARSE_MODE, int QUANT_MODE, int RETURN_VALUE,
          int HEAD_REDUCE, int EXPAND>
__global__ __aicore__ void generic_lightning_indexer(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                     __gm__ uint8_t *weights, __gm__ uint8_t *actualSeqQ,
                                                     __gm__ uint8_t *actualSeqK, __gm__ uint8_t *cmpResidualK,
                                                     __gm__ uint8_t *blockTable, __gm__ uint8_t *qDescale,
                                                     __gm__ uint8_t *kDescale, __gm__ uint8_t *metadata,
                                                     __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                     __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)SPARSE_MODE;
    (void)RETURN_VALUE;
    (void)metadata;

#if (__CCE_AICORE__ == 310)
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16) {
        INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, GLI_LAYOUT(LAYOUT_Q),
                           GLI_LAYOUT(LAYOUT_K), false, void, void, HEAD_REDUCE, EXPAND);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
        INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, half, half, int32_t, GLI_LAYOUT(LAYOUT_Q),
                           GLI_LAYOUT(LAYOUT_K), false, void, void, HEAD_REDUCE, EXPAND);
    } else if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN) {
        if constexpr (ORIG_DTYPE_WEIGHTS == DT_BF16) {
            if constexpr (QUANT_MODE == GLI_TPL_QUANT_MXFP8) {
                INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                                   GLI_LAYOUT(LAYOUT_Q), GLI_LAYOUT(LAYOUT_K), false, fp8_e8m0_t, bfloat16_t,
                                   HEAD_REDUCE, EXPAND);
            } else {
                INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                                   GLI_LAYOUT(LAYOUT_Q), GLI_LAYOUT(LAYOUT_K), false, float, bfloat16_t, HEAD_REDUCE,
                                   EXPAND);
            }
        } else {
            if constexpr (QUANT_MODE == GLI_TPL_QUANT_MXFP8) {
                INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                                   GLI_LAYOUT(LAYOUT_Q), GLI_LAYOUT(LAYOUT_K), false, fp8_e8m0_t, half, HEAD_REDUCE,
                                   EXPAND);
            } else {
                INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, fp8_e4m3fn_t, fp8_e4m3fn_t, int32_t,
                                   GLI_LAYOUT(LAYOUT_Q), GLI_LAYOUT(LAYOUT_K), false, float, half, HEAD_REDUCE, EXPAND);
            }
        }
    }
#else
    if constexpr (DT_Q == GLI_TPL_FP16 && DT_K == GLI_TPL_FP16) {
        INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, half, half, int32_t, GLI_LAYOUT(LAYOUT_Q),
                           GLI_LAYOUT(LAYOUT_K), false, void, void, HEAD_REDUCE, EXPAND);
    } else {
        INVOKE_GLI_OP_IMPL(GenericLightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, GLI_LAYOUT(LAYOUT_Q),
                           GLI_LAYOUT(LAYOUT_K), false, void, void, HEAD_REDUCE, EXPAND);
    }
#endif
}
