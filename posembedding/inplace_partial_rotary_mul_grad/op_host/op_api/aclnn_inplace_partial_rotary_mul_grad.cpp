/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_inplace_partial_rotary_mul_grad.h"
#include "aclnnInner_inplace_partial_rotary_mul_grad.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnInplacePartialRotaryMulGradGetWorkspaceSize(const aclTensor *dyRef, const aclTensor *cos,
                                                             const aclTensor *sin, const aclTensor *xOptional,
                                                             int64_t rotaryMode, const aclIntArray *partialSlice,
                                                             aclTensor *dcosOut, aclTensor *dsinOut,
                                                             uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = aclnnInnerInplacePartialRotaryMulGradGetWorkspaceSize(const_cast<aclTensor *>(dyRef), cos, sin,
                                                                            xOptional, rotaryMode, partialSlice,
                                                                            dcosOut, dsinOut, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnInplacePartialRotaryMulGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerInplacePartialRotaryMulGrad(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
