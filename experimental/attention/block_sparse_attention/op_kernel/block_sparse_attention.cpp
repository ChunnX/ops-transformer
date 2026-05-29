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
 * \file block_sparse_attention.cpp
 * \brief TODO: 待实现
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "block_sparse_attention_template_tiling_key.h"
#include "block_sparse_attention_common.h"
#include "arch35/bsa_kernel.h"

using namespace AscendC;

template <uint32_t QKV_DTYPE, uint32_t LAYOUT_T, uint32_t KV_LAYOUT_T, uint32_t COMPRESS_MODE,
          bool RETURN_SOFTMAX_LSE>
__global__ __aicore__ void block_sparse_attention(
    __gm__ uint8_t *query,
    __gm__ uint8_t *key,
    __gm__ uint8_t *value,
    __gm__ uint8_t *qScale,
    __gm__ uint8_t *kScale,
    __gm__ uint8_t *vScale,
    __gm__ uint8_t *pScale,
    __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *cuSeqlensKv,
    __gm__ uint8_t *seqUsedQ,
    __gm__ uint8_t *seqUsedKv,
    __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseSeqLen,
    __gm__ uint8_t *blockTable,
    __gm__ uint8_t *attenMask,
    __gm__ uint8_t *metadata,
    __gm__ uint8_t *attentionOut,
    __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)query;
    (void)key;
    (void)value;
    (void)qScale;
    (void)kScale;
    (void)vScale;
    (void)pScale;
    (void)cuSeqlensQ;
    (void)cuSeqlensKv;
    (void)seqUsedQ;
    (void)seqUsedKv;
    (void)sparseIndices;
    (void)sparseSeqLen;
    (void)blockTable;
    (void)attenMask;
    (void)metadata;
    (void)attentionOut;
    (void)softmaxLse;
    (void)workspace;
    (void)tiling;
    // TODO: 待实现 - 根据 BSA tiling key dispatch 到 BSAKernel，接入 sparseIndices、scale 和 metadata。
}
