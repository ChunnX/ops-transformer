/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "moe_token_permute_with_routing_map_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
static constexpr int64_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr int64_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

OP_TYPE_REGISTER(MoeTokenPermuteWithRoutingMapGrad);

const std::array<const aclTensor*, 2> MoeTokenPermuteWithRoutingMapGrad(
    const aclTensor* permutedTokenOutputGrad, const aclTensor* permutedProbsOutputGradOptional,
    const aclTensor* sortedIndices, const aclTensor* routingMapOptional, int64_t numExperts, int64_t tokensNum,
    bool dropAndPad, const aclTensor* tokensGradOut, const aclTensor* probsGradOut, aclOpExecutor* executor)
{
    L0_DFX(
        MoeTokenPermuteWithRoutingMapGrad, permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices,
        routingMapOptional, numExperts, tokensNum, dropAndPad);

    ADD_TO_LAUNCHER_LIST_AICORE(
        MoeTokenPermuteWithRoutingMapGrad,
        OP_INPUT(permutedTokenOutputGrad, permutedProbsOutputGradOptional, sortedIndices, routingMapOptional),
        OP_OUTPUT(tokensGradOut, probsGradOut), OP_ATTR(numExperts, tokensNum, dropAndPad));

    return std::array<const aclTensor*, 2>{tokensGradOut, probsGradOut};
}
} // namespace l0op