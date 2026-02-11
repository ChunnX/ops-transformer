/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_rotary_position_embedding_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerRotaryPositionEmbeddingV2GetWorkspaceSize(const aclTensor* x, const aclTensor* cos,
                                                                     const aclTensor* sin, const aclTensor* rotate, int64_t mode,
                                                                     aclTensor* out, uint64_t* workspaceSize,
                                                                     aclOpExecutor** executor);

extern aclnnStatus aclnnInnerRotaryPositionEmbeddingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream);

aclnnStatus aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                           int64_t mode, const aclTensor* rotate, aclTensor* out,
                                                           uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnInnerRotaryPositionEmbeddingV2GetWorkspaceSize(x, cos, sin, rotate, mode, out, workspaceSize, executor);
}

aclnnStatus aclnnRotaryPositionEmbeddingV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    return aclnnInnerRotaryPositionEmbeddingV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
