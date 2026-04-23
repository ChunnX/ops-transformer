/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_token_permute.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeTokenPermute(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // 始终走 aclnnInnerMoeTokenPermute，与 answer 中自定义 kernel / api_desc 一致（勿在 Regbase 上改走 InitRoutingV2）。
    return aclnnInnerMoeTokenPermuteGetWorkspaceSize(
        tokens, indices, numOutTokens, paddedMode, permuteTokensOut, sortedIndicesOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeTokenPermute(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
