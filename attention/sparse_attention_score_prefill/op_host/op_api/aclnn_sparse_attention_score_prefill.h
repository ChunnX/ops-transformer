/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

/*
 * 本文件声明 SparseAttentionScorePrefill 算子对外暴露的 aclnn C 接口。
 * 包含两阶段 API：GetWorkspaceSize（规划阶段）与算子执行接口（执行阶段）。
 */

#ifndef ACLNN_SPARSE_ATTENTION_SCORE_PREFILL_H_
#define ACLNN_SPARSE_ATTENTION_SCORE_PREFILL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScorePrefillGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockTable,
    const aclTensor *k2qRowPtr,
    const aclTensor *k2qQIndices,
    const aclTensor *k2qSlotIndices,
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    int64_t numKeyValueHeads,
    double scaleValue,
    int64_t blockSize,
    int64_t topK,
    int64_t innerPrecise,
    aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSparseAttentionScorePrefill(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_SPARSE_ATTENTION_SCORE_PREFILL_H_
