/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file causal_conv1d_grad.cpp
 * @brief ACLNN L0 API 实现 - CausalConv1dGrad
 */

#include "causal_conv1d_grad.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(CausalConv1dGrad);

std::array<const aclTensor*, 3> CausalConv1dGrad(
    const aclTensor* gradY, const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
    aclOpExecutor* executor)
{
    L0_DFX(CausalConv1dGrad, gradY, x, weight, bias);

    // dx 同 gradY，dweight 同 weight，dbias 同 bias
    auto dx = executor->AllocTensor(gradY->GetViewShape(), gradY->GetDataType());
    if (dx == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for dx failed.");
        return {nullptr, nullptr, nullptr};
    }
    auto dweight = executor->AllocTensor(weight->GetViewShape(), weight->GetDataType());
    if (dweight == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for dweight failed.");
        return {nullptr, nullptr, nullptr};
    }
    auto dbias = executor->AllocTensor(bias->GetViewShape(), bias->GetDataType());
    if (dbias == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor for dbias failed.");
        return {nullptr, nullptr, nullptr};
    }

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CausalConv1dGrad,
        OP_INPUT(gradY, x, weight, bias),
        OP_OUTPUT(dx, dweight, dbias));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CausalConv1dGrad ADD_TO_LAUNCHER_LIST_AICORE failed with ret=%d.", static_cast<int>(ret));
        return {nullptr, nullptr, nullptr};
    }

    return {dx, dweight, dbias};
}

} // namespace l0op
