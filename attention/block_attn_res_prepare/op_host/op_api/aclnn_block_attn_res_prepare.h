/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_BLOCK_ATTN_RES_PREPARE_H_
#define ACLNN_BLOCK_ATTN_RES_PREPARE_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates the workspace needed by BlockAttnResPrepare.
 *
 * @param blockResidual [IN] FP32 ND tensor with shape [T, N, D].
 * @param effectiveQuery [IN] FP32 ND tensor with shape [S, D].
 * @param validBlocks [IN] INT32 ND tensor with shape [1].
 * @param epsilon [IN] Positive FP32 numerical-stability constant.
 * @param softmaxMax [OUT] FP32 ND tensor with shape [S, T].
 * @param weightedOutput [OUT] FP32 ND tensor with shape [S, T, D].
 * @param softmaxSum [OUT] FP32 ND tensor with shape [S, T].
 * @param workspaceSize [OUT] Required device workspace size in bytes.
 * @param executor [OUT] Executor consumed by the second-stage API.
 * @return ACLNN_SUCCESS on success.
 */
ACLNN_API aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockResidual, const aclTensor *effectiveQuery, const aclTensor *validBlocks, float epsilon,
    aclTensor *softmaxMax, aclTensor *weightedOutput, aclTensor *softmaxSum, uint64_t *workspaceSize,
    aclOpExecutor **executor);

/**
 * @brief Runs BlockAttnResPrepare using the executor produced by the first-stage API.
 */
ACLNN_API aclnnStatus aclnnBlockAttnResPrepare(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_BLOCK_ATTN_RES_PREPARE_H_
