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
 * \file sparse_flash_mla.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "sparse_flash_mla_template_tiling_key.h"
#include "sparse_flash_mla_metadata.h"

using namespace AscendC;
using namespace optiling::detail;

template <int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE, int SPLIT_G>
__global__ __aicore__ void
sparse_flash_mla(__gm__ uint8_t *query, __gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                     __gm__ uint8_t *oriSparseIndices, __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                     __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                     __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedOriKV,
                     __gm__ uint8_t *seqUsedCmpKV, __gm__ uint8_t *cmpResidualKV,
                     __gm__ uint8_t *oriTopkLength, __gm__ uint8_t *cmpTopkLength,
                     __gm__ uint8_t *sinks, __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    return ;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
}