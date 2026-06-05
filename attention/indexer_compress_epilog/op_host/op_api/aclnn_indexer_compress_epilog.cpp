/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_indexer_compress_epilog.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnIndexerCompressEpilogGetWorkspaceSize(
    const aclTensor *indexerCompressCache,
    const aclTensor *indexerCompressCacheScale,
    const aclTensor *x,
    const aclTensor *slotMapping,
    int64_t quantMode,
    bool roundScale,
    float scale,
    const aclTensor *indexerCompressCacheOut,
    const aclTensor *indexerCompressCacheScaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L0_DFX(IndexerCompressEpilog, indexerCompressCache, indexerCompressCacheScale, x, slotMapping,
           quantMode, roundScale, scale, indexerCompressCacheOut, indexerCompressCacheScaleOut);

    if (indexerCompressCache == nullptr || indexerCompressCacheScale == nullptr ||
        x == nullptr || slotMapping == nullptr ||
        indexerCompressCacheOut == nullptr || indexerCompressCacheScaleOut == nullptr) {
        return ACLNN_ERR_PARAM_NULL;
    }

    OP_MAKE_EXECUTOR(
        "aclnnIndexerCompressEpilog",
        IndexerCompressEpilog, executor,
        indexerCompressCache, indexerCompressCacheScale,
        x, slotMapping,
        indexerCompressCacheOut, indexerCompressCacheScaleOut,
        quantMode, roundScale, scale);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexerCompressEpilog(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream)
{
    if (executor == nullptr) {
        return ACLNN_ERR_EXECUTOR_NULL;
    }
    if (executor->ExecuteAsync(workspace, workspaceSize, stream) != OP_RUN_SUCCESS) {
        return ACLNN_ERR_EXECUTOR_EXECUTE_FAIL;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
