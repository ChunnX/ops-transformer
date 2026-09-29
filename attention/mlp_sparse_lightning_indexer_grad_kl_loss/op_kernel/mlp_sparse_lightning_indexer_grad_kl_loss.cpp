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
#include "mlp_sparse_lightning_indexer_grad_kl_loss_sync_only.h"

extern "C" __global__ __aicore__ void mlp_sparse_lightning_indexer_grad_kl_loss(
    GM_ADDR query, GM_ADDR key, GM_ADDR queryIndex, GM_ADDR keyIndex,
    GM_ADDR weight, GM_ADDR sparseIndices, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
    GM_ADDR queryRope, GM_ADDR keyRope, GM_ADDR mask,
    GM_ADDR actualSeqLengthsQuery, GM_ADDR actualSeqLengthsKey,
    GM_ADDR dQueryIndex, GM_ADDR dKeyIndex, GM_ADDR dWeight, GM_ADDR loss,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)query;
    (void)key;
    (void)queryIndex;
    (void)keyIndex;
    (void)weight;
    (void)sparseIndices;
    (void)softmaxMax;
    (void)softmaxSum;
    (void)queryRope;
    (void)keyRope;
    (void)mask;
    (void)actualSeqLengthsQuery;
    (void)actualSeqLengthsKey;
    (void)dQueryIndex;
    (void)dKeyIndex;
    (void)dWeight;
    (void)loss;
    (void)workspace;

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(MlpSparseLightningIndexerGradKLLossTilingData, tilingDataIn, tiling);
    MlpSliSyncOnly::DeterministicSyncKernel op;
    op.Init(&tilingDataIn);
    op.Process();
}
