/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <tuple>
#include "causal_conv1d.h"
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
OP_TYPE_REGISTER(CausalConv1d);

bool CausalConv1d(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, aclTensor *convStates,
                  const aclTensor *queryStartLoc, const aclTensor *cacheIndices, const aclTensor *initialStateMode,
                  const aclTensor *numAcceptedTokens, int64_t activationMode, int64_t padSlotId, int64_t runMode,
                  aclTensor *y, aclOpExecutor *executor)
{
    L0_DFX(CausalConv1d, x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
           activationMode, padSlotId, runMode, y, convStates);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        CausalConv1d,
        OP_INPUT(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens),
        OP_OUTPUT(convStates, y), OP_ATTR(activationMode, padSlotId, runMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CausalConv1d ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return false;
    }
    return true;
}

} // namespace l0op
