/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MOE_TUTEL_COMBINE_X_H_
#define ACLNN_MOE_TUTEL_COMBINE_X_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// L2 public header for MoeTutelCombineX. The two declarations below are the
// CONTRACT callers were built against, copied VERBATIM from the op_build
// autogen header (build/autogen/aclnn_moe_tutel_combine_x.h): same parameter
// order, same names, same const — do not change them.
// =============================================================================

/* function: aclnnMoeTutelCombineXGetWorkspaceSize
 * parameters :
 * yGrad : required
 * gates : required
 * indices : required
 * locations : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeTutelCombineXGetWorkspaceSize(
    const aclTensor *yGrad, const aclTensor *gates, const aclTensor *indices, const aclTensor *locations,
    const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnMoeTutelCombineX
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeTutelCombineX(void *workspace, uint64_t workspaceSize,
                                                                         aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_MOE_TUTEL_COMBINE_X_H_
