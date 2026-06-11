/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_CAUSAL_CONV1D_FN_H_
#define OP_API_INC_CAUSAL_CONV1D_FN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnCausalConv1dFn (prefill / chunk-prefill) 的第一段接口，计算workspace大小。
 * @domain aclnn_ops_infer
 */
ACLNN_API aclnnStatus aclnnCausalConv1dFnGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *biasOptional, aclTensor *convStatesRef,
    const aclTensor *queryStartLocOptional, const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional, const aclTensor *initialStateIdxOptional,
    const aclTensor *numComputedTokensOptional, const aclTensor *blockIdxFirstScheduledTokenOptional,
    const aclTensor *blockIdxLastScheduledTokenOptional, int64_t activationMode, int64_t nullBlockId,
    int64_t blockSizeToAlign, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

/* @brief aclnnCausalConv1dFn 的第二段接口，用于执行计算。 */
ACLNN_API aclnnStatus aclnnCausalConv1dFn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
