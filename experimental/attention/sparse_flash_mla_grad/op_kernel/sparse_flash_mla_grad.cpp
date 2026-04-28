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
 * \file sparse_flash_attention_grad.cpp
 * \brief
 */

#include "arch35/sparse_flash_attention_grad_template_tiling_key.h"
#include "arch35/sparse_flash_attention_grad_tiling_data_regbase.h"
#include "arch35/sparse_flash_attention_grad_entry_regbase.h"

#include "kernel_operator.h"

using namespace AscendC;
constexpr static uint32_t ND = 0;
constexpr static uint32_t NZ = 1;
constexpr static uint32_t TND = 3;


template <uint8_t inputDType, bool isTnd, uint16_t gTemplateType, uint16_t s2TemplateType, uint16_t dTemplateType, bool isOriKVExist, bool isCmpKVExist, bool isOriKVSparse, bool isCmpKVSparse, bool hasUsedSeqQ, bool hasUsedSeqOriKV, bool hasUsedSeqCmpKV, bool hasOriTopK, bool hasCmpTopK, bool isSink, bool deterministic>
__global__ __aicore__ void
sparse_flash_mla_grad(__gm__ uint8_t *query, __gm__ uint8_t *ori_kv, __gm__ uint8_t *cmp_kv,
                            __gm__ uint8_t *d_out, __gm__ uint8_t *out, __gm__ uint8_t *ori_sparse_indices, __gm__ uint8_t *cmp_sparse_indices,
                            __gm__ uint8_t *cu_seqlens_q, __gm__ uint8_t *cu_seqlens_ori_kv, __gm__ uint8_t *cu_seqlens_cmp_kv,
                            __gm__ uint8_t *seqused_q, __gm__ uint8_t *seqused_ori_kv, __gm__ uint8_t *seqused_cmp_kv, __gm__ uint8_t *cmp_residual_kv,
                            __gm__ uint8_t *ori_topk_length, __gm__ uint8_t *cmp_topk_length,
                            __gm__ uint8_t *softmax_lse, __gm__ uint8_t *sinks, __gm__ uint8_t *metadata,
                            __gm__ uint8_t *dq, __gm__ uint8_t *dori_kv, __gm__ uint8_t *dcmp_kv,
                            __gm__ uint8_t *dsinks, __gm__ uint8_t *ori_softmax_l1, __gm__ uint8_t *cmp_softmax_l1,
                            __gm__ uint8_t *workspace, __gm__ uint8_t *tiling_data)
{
    REGISTER_TILING_DEFAULT(optiling::sfmg::SparseFlashMlaGradTilingDataRegbase);
    RegbaseSFAG<inputDType, isTnd, gTemplateType, s2TemplateType, dTemplateType, isOriKVExist, isCmpKVExist, isOriKVSparse, isCmpKVSparse, hasUsedSeqQ, hasUsedSeqOriKV, hasUsedSeqCmpKV, hasOriTopK, hasCmpTopK, isSink, deterministic>(query, ori_kv, cmp_kv, d_out, out, ori_sparse_indices, cmp_sparse_indices, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_topk_length, cmp_topk_length, softmax_lse, sinks, metadata, dq, dori_kv, dcmp_kv, dsinks, ori_softmax_l1, cmp_softmax_l1, workspace, tiling_data);
}