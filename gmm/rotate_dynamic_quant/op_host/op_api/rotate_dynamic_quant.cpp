/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rotate_dynamic_quant.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

namespace l0op {

OP_TYPE_REGISTER(RotateDynamicQuant);

const std::tuple<const aclTensor *, const aclTensor *>
RotateDynamicQuant(
    const aclTensor *x, const aclTensor *rot,
    int64_t dstType,
    aclOpExecutor *executor) {

    L0_DFX(RotateDynamicQuant, x, rot);

    // 根据dst_type确定输出yOut的数据类型
    DataType yOutType = (dstType == 1) ? DataType::DT_INT32 : DataType::DT_INT8;
    Format format = Format::FORMAT_ND;

    // 获取输入x的shape
    auto xShape = x->GetViewShape();
    int64_t mDim = xShape.GetDim(0);
    int64_t nDim = xShape.GetDim(1);

    // yOut: shape与x一致 [M, N]，类型由dst_type决定
    auto yOutShape = op::Shape({mDim, nDim});
    auto yOut = executor->AllocTensor(yOutShape, yOutType, format);
    OP_CHECK_NULL(yOut, return std::make_tuple(nullptr, nullptr));

    // scaleOut: shape为[M]，类型固定为DT_FLOAT
    auto scaleOutShape = op::Shape({mDim});
    auto scaleOut = executor->AllocTensor(scaleOutShape, DataType::DT_FLOAT, format);
    OP_CHECK_NULL(scaleOut, return std::make_tuple(nullptr, nullptr));

    auto ret = INFER_SHAPE(RotateDynamicQuant, OP_INPUT(x, rot),
        OP_OUTPUT(yOut, scaleOut), OP_ATTR(dstType));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::make_tuple(nullptr, nullptr),
        "RotateDynamicQuant InferShape failed.");

    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(RotateDynamicQuant,
        OP_INPUT(x, rot),
        OP_OUTPUT(yOut, scaleOut), OP_ATTR(dstType));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return std::make_tuple(nullptr, nullptr),
        "RotateDynamicQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::make_tuple(yOut, scaleOut);
}

} // namespace l0op
