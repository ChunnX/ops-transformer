/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_backward.cpp
 * \brief ACLNN L0 API 实现 - CausalConv1dBackward
 */

#include "causal_conv1d_backward.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(CausalConv1dBackward);

bool CausalConv1dBackward(const aclTensor *gradY, const aclTensor *x, const aclTensor *weight,
                          const aclTensor *queryStartLoc, const char *activation,
                          aclTensor *gradX, aclTensor *gradWeight, aclOpExecutor *executor)
{
    L0_DFX(CausalConv1dBackward, gradY, x, weight, queryStartLoc, activation, gradX, gradWeight);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(CausalConv1dBackward,
        OP_INPUT(gradY, x, weight, queryStartLoc),
        OP_OUTPUT(gradX, gradWeight), OP_ATTR(activation));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CausalConv1dBackward ADD_TO_LAUNCHER_LIST_AICORE failed with ret=%d.",
                static_cast<int>(ret));
        return false;
    }
    return true;
}

} // namespace l0op
