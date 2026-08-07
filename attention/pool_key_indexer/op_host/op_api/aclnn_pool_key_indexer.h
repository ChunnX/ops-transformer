/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_POOL_KEY_INDEXER_H
#define ACLNN_POOL_KEY_INDEXER_H

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Phase 1: compute workspace size and validate parameters.
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *poolKey,
    const aclTensor *weights,
    const aclTensor *poolTailK,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *qDescaleOptional,
    const aclTensor *kDescaleOptional,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t topk,
    int64_t poolSize,
    int64_t maskMode,
    int64_t quantMode,
    bool returnValue,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Phase 2: execute computation.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnPoolKeyIndexer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_POOL_KEY_INDEXER_H
