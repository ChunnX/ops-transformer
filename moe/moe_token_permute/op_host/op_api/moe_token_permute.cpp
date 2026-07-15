/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_permute.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeTokenPermute);

const std::array<const aclTensor *, 2> MoeTokenPermute(const aclTensor *tokens, const aclTensor *indices,
                                                       int64_t numOutTokens, bool paddedMode,
                                                       const aclTensor *permuteTokensOut,
                                                       const aclTensor *sortedIndicesOut, aclOpExecutor *executor)
{
    L0_DFX(MoeTokenPermute, tokens, indices, numOutTokens, paddedMode, permuteTokensOut, sortedIndicesOut);

    const std::array<const aclTensor *, 2> errorReturn = {nullptr, nullptr};
    auto permuteTokens =
        executor->AllocTensor(permuteTokensOut->GetViewShape(), permuteTokensOut->GetDataType(), Format::FORMAT_ND);
    auto sortedIndices =
        executor->AllocTensor(sortedIndicesOut->GetViewShape(), sortedIndicesOut->GetDataType(), Format::FORMAT_ND);
    if (permuteTokens == nullptr || sortedIndices == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MoeTokenPermute output allocation failed.");
        return errorReturn;
    }

    if (!(permuteTokens->IsEmpty() && sortedIndices->IsEmpty())) {
        auto ret =
            ADD_TO_LAUNCHER_LIST_AICORE(MoeTokenPermute, OP_INPUT(tokens, indices),
                                        OP_OUTPUT(permuteTokens, sortedIndices), OP_ATTR(numOutTokens, paddedMode));
        OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return errorReturn,
                                             "MoeTokenPermute ADD_TO_LAUNCHER_LIST_AICORE failed.");
    }

    return {permuteTokens, sortedIndices};
}
} // namespace l0op
