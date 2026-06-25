/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <tuple>
#include "moe_re_routing.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeReRouting);

std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>
MoeReRouting(const aclTensor *tokens, const aclTensor *expertTokenNumPerRank,
             const aclTensor *perTokenScales, const aclTensor *expertTopkWeight,
             int64_t expertTokenNumType, int64_t idxType,
             const aclTensor *permuteTokens, const aclTensor *permutePerTokenScales,
             const aclTensor *permuteTokenIdx, const aclTensor *expertTokenNum,
             const aclTensor *permuteTopkWeight, aclOpExecutor *executor)
{
    L0_DFX(MoeReRouting, tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight,
            expertTokenNumType, idxType, permuteTokens, permutePerTokenScales,
            permuteTokenIdx, expertTokenNum, permuteTopkWeight);

    auto permuteTokensOut = executor->AllocTensor(
        permuteTokens->GetViewShape(), permuteTokens->GetDataType(), Format::FORMAT_ND);
    auto permutePerTokenScalesOut = executor->AllocTensor(
        permutePerTokenScales->GetViewShape(), permutePerTokenScales->GetDataType(), Format::FORMAT_ND);
    auto permuteTokenIdxOut = executor->AllocTensor(
        permuteTokenIdx->GetViewShape(), permuteTokenIdx->GetDataType(), Format::FORMAT_ND);
    auto expertTokenNumOut = executor->AllocTensor(
        expertTokenNum->GetViewShape(), expertTokenNum->GetDataType(), Format::FORMAT_ND);
    if (permuteTokensOut == nullptr || permutePerTokenScalesOut == nullptr ||
        permuteTokenIdxOut == nullptr || expertTokenNumOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "alloc permuteTokensOut or permutePerTokenScalesOut or permuteTokenIdxOut "
                "or expertTokenNumOut tensor failed.");
        return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
            nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    aclTensor* permuteTopkWeightOut = nullptr;
    if (permuteTopkWeight != nullptr) {
        permuteTopkWeightOut = executor->AllocTensor(
            permuteTopkWeight->GetViewShape(), permuteTopkWeight->GetDataType(), Format::FORMAT_ND);
        if (permuteTopkWeightOut == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc permuteTopkWeightOut tensor failed.");
            return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
                nullptr, nullptr, nullptr, nullptr, nullptr);
        }
    }

    ADD_TO_LAUNCHER_LIST_AICORE(MoeReRouting,
        OP_INPUT(tokens, expertTokenNumPerRank, perTokenScales, expertTopkWeight),
        OP_OUTPUT(permuteTokensOut, permutePerTokenScalesOut, permuteTokenIdxOut,
                  expertTokenNumOut, permuteTopkWeightOut),
        OP_ATTR(expertTokenNumType, idxType));
    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
        permuteTokensOut, permutePerTokenScalesOut, permuteTokenIdxOut,
        expertTokenNumOut, permuteTopkWeightOut);
}

}  // namespace l0op
