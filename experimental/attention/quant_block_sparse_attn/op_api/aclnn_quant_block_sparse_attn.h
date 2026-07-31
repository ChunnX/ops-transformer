/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_QUANT_BLOCK_SPARSE_ATTN_H_
#define ACLNN_QUANT_BLOCK_SPARSE_ATTN_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnQuantBlockSparseAttnGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *qDescale,
    const aclTensor *kDescale, const aclTensor *vDescale, const aclTensor *pScale, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    const aclTensor *sparseIndices, const aclTensor *sparseSeqLen, const aclTensor *blockTableOptional,
    const aclTensor *attenMaskOptional, const aclTensor *metadataOptional, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    double softmaxScale, int64_t sparseQBlockSize, int64_t sparseKvBlockSize, int64_t paBlockStride, char *layoutKv,
    char *layoutQ, char *layoutSparseIndices, char *layoutOut, int64_t quantMode, int64_t maskMode,
    bool returnSoftmaxLse, const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnQuantBlockSparseAttn(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor,
                                                                             aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_QUANT_BLOCK_SPARSE_ATTN_H_
