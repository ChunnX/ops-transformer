/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "cloud_causal_conv1d_fn.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(CloudCausalConv1dFn);

bool CloudCausalConv1dFn(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, aclTensor *convStates,
                       const aclTensor *queryStartLoc, const aclTensor *cacheIndices,
                       const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId, aclTensor *y,
                       aclOpExecutor *executor)
{
    L0_DFX(CloudCausalConv1dFn, x, weight, bias, convStates, queryStartLoc, cacheIndices, numAcceptedTokens,
           activationMode, padSlotId, y);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CloudCausalConv1dFn,
        OP_INPUT(x, weight, bias, convStates, queryStartLoc, cacheIndices, numAcceptedTokens),
        OP_OUTPUT(y),
        OP_ATTR(activationMode, padSlotId));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CloudCausalConv1dFn ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }
    return true;
}

} // namespace l0op
