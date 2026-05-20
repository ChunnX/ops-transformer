/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef OP_API_INC_CAUSAL_CONV1D_H_
#define OP_API_INC_CAUSAL_CONV1D_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCausalConv1d的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnCausalConv1dGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *bias, aclTensor *convStates,
    const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *initialStateMode,
    const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId, int64_t runMode,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

/* @brief aclnnCausalConv1d的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
