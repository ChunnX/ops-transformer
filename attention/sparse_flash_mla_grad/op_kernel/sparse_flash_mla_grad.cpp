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
 * \file sparse_flash_mla_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sparse_flash_mla_grad_template_tiling_key.h"
#include "arch22/sfag_basic.h"
#include "arch22/smlag_basic.h"
using namespace AscendC;

#define INVOKE_SMLAG_BASIC_IMPL(templateClass, ...)                                                  \
    do {                                                                                            \
        __gm__ uint8_t *user = GetUserWorkspace(workspace);                                         \
        GET_TILING_DATA_WITH_STRUCT(SparseFlashMlaGradTilingData, tiling_data_in, tiling_data); \
        const SparseFlashMlaGradTilingData *__restrict tilingData = &tiling_data_in;            \
        templateClass<SMLAG_BASIC::SMLAG_TYPE<SparseFlashMlaGradTilingData, __VA_ARGS__>> op;     \
        op.Process(query, ori_kv, cmp_kv, out, d_out, lse, cmp_sparse_indices,                      \
                   cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, cmp_residual_kv,             \
                   sinks, d_query, d_ori_kv, d_cmp_kv, d_sinks, cmp_softmax_l1_norm, user, tilingData);\
    } while (0)

template<int LAYOUT, int MODE>
__global__ __aicore__ void
sparse_flash_mla_grad(__gm__ uint8_t *query, __gm__ uint8_t *d_out, __gm__ uint8_t *out, __gm__ uint8_t *lse, 
                            __gm__ uint8_t *ori_kv, __gm__ uint8_t *cmp_kv,
                            __gm__ uint8_t *ori_sparse_indices, __gm__ uint8_t *cmp_sparse_indices, 
                            __gm__ uint8_t *cu_seqlens_q, __gm__ uint8_t *cu_seqlens_ori_kv, __gm__ uint8_t *cu_seqlens_cmp_kv, 
                            __gm__ uint8_t *seqused_q, __gm__ uint8_t *seqused_ori_kv, __gm__ uint8_t *seqused_cmp_kv,
                            __gm__ uint8_t *cmp_residual_kv, __gm__ uint8_t *ori_topk_length, __gm__ uint8_t *cmp_topk_length,
                            __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, 
                            __gm__ uint8_t *d_query, __gm__ uint8_t *d_ori_kv, __gm__ uint8_t *d_cmp_kv,
                            __gm__ uint8_t *d_sinks, __gm__ uint8_t *ori_softmax_l1_norm, __gm__ uint8_t *cmp_softmax_l1_norm,
                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16) {
        if (MODE == SMLAG_SCFA_MODE) {
            if (LAYOUT == SMLAG_LAYOUT_BSND) {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SelectedAttentionGradBasic, half, true);
            } else {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SelectedAttentionGradBasic, half, false);
            }
        } else {
            if (LAYOUT == SMLAG_LAYOUT_BSND) {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SparseFlashMlaGrad, half, true, MODE);
            } else {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SparseFlashMlaGrad, half, false, MODE);
            }
        }
    }
    if constexpr (ORIG_DTYPE_QUERY == DT_BF16) {
        if (MODE == SMLAG_SCFA_MODE) {
            if (LAYOUT == SMLAG_LAYOUT_BSND) {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SelectedAttentionGradBasic, bfloat16_t, true);
            } else {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SelectedAttentionGradBasic, bfloat16_t, false);
            }
        } else {
            if (LAYOUT == SMLAG_LAYOUT_BSND) {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SparseFlashMlaGrad, bfloat16_t, true, MODE);
            } else {
                INVOKE_SMLAG_BASIC_IMPL(SMLAG_BASIC::SparseFlashMlaGrad, bfloat16_t, false, MODE);
            }
        }
    }
}

