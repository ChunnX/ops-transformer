/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_
#define OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Signatures mirror the built-in aclnnCumprod so the custom package drop-in
 * overrides it (torch_npu's torch.cumprod adapter calls these symbols).
 * The kernel computes the inclusive forward cumulative product; exclusive /
 * reverse semantics of the IR are fixed to false/false at this layer.
 * dtype support: DT_FLOAT / DT_FLOAT16 / DT_BF16 (custom AI Core kernel).
 */

ACLNN_API aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor *input, const aclScalar *dim,
                                                   const aclDataType dtype, aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   aclrtStream stream);

ACLNN_API aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor *input, const aclScalar *dim,
                                                          uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnInplaceCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif // OP_API_INC_LEVEL2_ACLNN_CUMPROD_H_
