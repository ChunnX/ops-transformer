/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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
    const aclTensor *x, const aclTensor *weight, aclTensor *convStatesRef, const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional, const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional, const aclTensor *numAcceptedTokensOptional, int64_t activationMode,
    int64_t nullBlockId, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

/* @brief aclnnCausalConv1d的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream);

// ============================================================================
// aclnnCausalConv1dV2 (方案A: 合并单接口, 兼容扩展)
// ============================================================================

/**
 * @brief aclnnCausalConv1dV2 的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnCausalConv1dV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, aclTensor *convStatesRef, const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional, const aclTensor *cacheIndicesOptional,
    const aclTensor *convStateIndicesOptional, const aclTensor *initialStateModeOptional,
    const aclTensor *initialStateIdxOptional, const aclTensor *numAcceptedTokensOptional,
    const aclTensor *numComputedTokensOptional, const aclTensor *blockIdxFirstScheduledTokenOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional, int64_t activationMode, int64_t nullBlockId,
    int64_t blockSizeToAlign, int64_t maxQueryLen, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

/* @brief aclnnCausalConv1dV2 的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnCausalConv1dV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
