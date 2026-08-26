/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_unpermute_with_ep.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeTokenUnpermuteWithEp);

const aclTensor *MoeTokenUnpermuteWithEp(const aclTensor *permutedTokens, const aclTensor *sortedIndices,
                                         const aclTensor *probsOptional, int64_t numTopk,
                                         const aclIntArray *rangeOptional, bool paddedMode,
                                         const aclIntArray *restoreShapeOptional, const aclTensor *out,
                                         aclOpExecutor *executor)
{
    L0_DFX(MoeTokenUnpermuteWithEp, permutedTokens, sortedIndices, probsOptional, numTopk, rangeOptional, paddedMode,
           restoreShapeOptional);
    auto ret =
        ADD_TO_LAUNCHER_LIST_AICORE(MoeTokenUnpermuteWithEp, OP_INPUT(permutedTokens, sortedIndices, probsOptional),
                                    OP_OUTPUT(out), OP_ATTR(numTopk, rangeOptional, paddedMode, restoreShapeOptional));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }
    return out;
}
} // namespace l0op
