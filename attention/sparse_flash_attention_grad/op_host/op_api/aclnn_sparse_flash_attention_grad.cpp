/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_sparse_flash_attention_grad.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"

#ifdef __cplusplus
extern "C" {
#endif

// 注意：不要把下面的定义放进匿名 namespace。匿名 namespace 会给它们内部链接，
// 与 extern "C" 要求的对外可见性冲突，aclnnSparseFlashAttentionGrad* 就不会被导出。
extern aclnnStatus aclnnInnerSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *sparseIndices, const aclTensor *dOut,
    const aclTensor *out, const aclTensor *softmaxMax, const aclTensor *softmaxSum, const aclTensor *valueOptional,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, double scaleValue, int64_t sparseBlockSize,
    char *layoutOptional, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool deterministic,
    const aclTensor *dQueryOut, const aclTensor *dKeyOut, const aclTensor *dValueOutOptional,
    const aclTensor *dQueryRopeOutOptional, const aclTensor *dKeyRopeOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparseFlashAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream);

aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *sparseIndices,
    const aclTensor *dOut, const aclTensor *out, const aclTensor *softmaxMax, const aclTensor *softmaxSum,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, double scaleValue, int64_t sparseBlockSize,
    char *layoutOptional, int64_t sparseMode, int64_t preTokens, int64_t nextTokens, bool deterministic,
    const aclTensor *dQueryOut, const aclTensor *dKeyOut, const aclTensor *dValueOut,
    const aclTensor *dQueryRopeOutOptional, const aclTensor *dKeyRopeOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    const bool kvMerge = (value == nullptr);
    const bool dValueAbsent = (dValueOut == nullptr) || dValueOut->IsEmpty();
    if (kvMerge != dValueAbsent) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return aclnnInnerSparseFlashAttentionGradGetWorkspaceSize(
        query, key, sparseIndices, dOut, out, softmaxMax, softmaxSum, value, actualSeqLengthsQueryOptional,
        actualSeqLengthsKvOptional, queryRopeOptional, keyRopeOptional, scaleValue, sparseBlockSize, layoutOptional,
        sparseMode, preTokens, nextTokens, deterministic, dQueryOut, dKeyOut, dValueOut, dQueryRopeOutOptional,
        dKeyRopeOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnSparseFlashAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    return aclnnInnerSparseFlashAttentionGrad(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
