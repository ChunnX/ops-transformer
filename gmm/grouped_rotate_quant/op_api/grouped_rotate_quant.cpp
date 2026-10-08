/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "grouped_rotate_quant.h"

#include <vector>
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GroupedRotateQuant);

namespace {
const aclTensorList *AllocOutputList(const aclTensorList *output, aclOpExecutor *executor)
{
    std::vector<const aclTensor *> tensors;
    tensors.reserve(output->Size());
    for (size_t i = 0; i < output->Size(); ++i) {
        auto tensor = executor->AllocTensor((*output)[i]->GetViewShape(), (*output)[i]->GetDataType(),
                                            op::Format::FORMAT_ND);
        if (tensor == nullptr) {
            return nullptr;
        }
        tensors.emplace_back(tensor);
    }
    return executor->AllocTensorList(tensors.data(), tensors.size());
}
} // namespace

GroupedRotateQuantResult GroupedRotateQuant(const grq::Params &p, aclOpExecutor *executor)
{
    L0_DFX(GroupedRotateQuant, p.x, p.rotation, p.alpha, p.groupList, p.splitItem, p.groupType,
           p.groupListType, p.axis, p.roundMode, p.scaleAlg, p.dstTypeMax, p.trans, p.tuningConfig);
    if (executor == nullptr || p.y == nullptr || p.scale == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GroupedRotateQuant: missing executor or output descriptors.");
        return {nullptr, nullptr};
    }
    // L2 validated these shapes. Separate M-group output sizes are caller metadata:
    // do not infer them by reading a device groupList in the host preparation path.
    auto y = AllocOutputList(p.y, executor);
    auto scale = AllocOutputList(p.scale, executor);
    if (y == nullptr || scale == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GroupedRotateQuant: output allocation failed.");
        return {nullptr, nullptr};
    }
    const int64_t dstType = static_cast<int64_t>((*p.y)[0]->GetDataType());
    const float dstTypeMax = static_cast<float>(p.dstTypeMax);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        GroupedRotateQuant, OP_INPUT(p.x, p.rotation, p.alpha, p.groupList), OP_OUTPUT(y, scale),
        OP_ATTR(p.splitItem, p.groupType, p.groupListType, p.axis, p.roundMode, p.scaleAlg,
                dstTypeMax, p.trans, dstType, p.tuningConfig));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GroupedRotateQuant: ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }
    return {y, scale};
}
} // namespace l0op
