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
#include "moe_init_routing_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/aclnn_platform.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MoeInitRoutingV3);

std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>
MoeInitRoutingV3(const aclTensor *x, const aclTensor *expertIdx,
                 const aclTensor *scale, const aclTensor *offset,
                 const aclTensor *topk_weight, int64_t activeNum,
                 int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode,
                 int64_t expertTokensNumType, bool expertTokensNumFlag,
                 int64_t quantMode, const aclIntArray *activeExpertRange,
                 int64_t rowIdxType, const aclTensor *expandedX,
                 const aclTensor *expandedRowIdx,
                 const aclTensor *expertTokensCountOrCumsum,
                 const aclTensor *expandedScale,
                 const aclTensor *expanded_topk_weight,
                 aclOpExecutor *executor)
{
    L0_DFX(MoeInitRoutingV3, x, expertIdx, scale, offset, topk_weight,
           activeNum, expertCapacity, expertNum, dropPadMode,
           expertTokensNumType, expertTokensNumFlag, quantMode,
           activeExpertRange, rowIdxType, expandedX, expandedRowIdx,
           expertTokensCountOrCumsum, expandedScale, expanded_topk_weight);

    auto expandedXOut = executor->AllocTensor(expandedX->GetViewShape(), expandedX->GetDataType(),
                                              Format::FORMAT_ND);
    auto expandedRowIdxOut = executor->AllocTensor(expandedRowIdx->GetViewShape(), expandedRowIdx->GetDataType(),
                                                   Format::FORMAT_ND);
    auto expertTokensCountOrCumsumOut = executor->AllocTensor(expertTokensCountOrCumsum->GetViewShape(),
                                                              expertTokensCountOrCumsum->GetDataType(),
                                                              Format::FORMAT_ND);
    auto expandedScaleOut = executor->AllocTensor(expandedScale->GetViewShape(), expandedScale->GetDataType(),
                                                  Format::FORMAT_ND);
    if (expandedXOut == nullptr || expandedRowIdxOut == nullptr ||
        expertTokensCountOrCumsumOut == nullptr || expandedScaleOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "alloc expandedXOut or expandedRowIdxOut or "
                "expertTokensCountOrCumsumOut or expandedScaleOut tensor failed.");
        return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
            nullptr, nullptr, nullptr, nullptr, nullptr);
    }

    aclTensor* expandedTopkWeightOut = nullptr;
    const aclTensor* topkWeightInput = nullptr;
    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        topkWeightInput = topk_weight;
        if (expanded_topk_weight != nullptr) {
            expandedTopkWeightOut = executor->AllocTensor(expanded_topk_weight->GetViewShape(),
                                                          expanded_topk_weight->GetDataType(),
                                                          Format::FORMAT_ND);
            if (expandedTopkWeightOut == nullptr) {
                return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
                    nullptr, nullptr, nullptr, nullptr, nullptr);
            }
        }
    }
    ADD_TO_LAUNCHER_LIST_AICORE(
        MoeInitRoutingV3,
        OP_INPUT(x, expertIdx, scale, offset, topkWeightInput),
        OP_OUTPUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut,
                  expandedTopkWeightOut),
        OP_ATTR(activeNum, expertCapacity, expertNum, dropPadMode, expertTokensNumType, expertTokensNumFlag,
                quantMode, activeExpertRange, rowIdxType));
    return std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*, aclTensor*>(
        expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut,
        expandedScaleOut, expandedTopkWeightOut);
}

}  // namespace l0op
