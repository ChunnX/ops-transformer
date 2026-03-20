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

const std::tuple<aclTensor *, aclTensor *>
RotateDynamicQuant(
    const aclTensor *x,
    const aclTensor *rot,
    int32_t dst_type,
    aclOpExecutor *executor) {

    L0_DFX(RotateDynamicQuant, x, rot);

    // 获取输入形状
    auto xShape = x->GetViewShape();
    int64_t M = xShape.GetDim(0);
    int64_t N = xShape.GetDim(1);

    // 根据属性 dst_type 确定输出数据类型
    DataType yOutType = (dst_type == 1) ? DataType::DT_INT32 : DataType::DT_INT8;
    DataType scaleOutType = DataType::DT_FLOAT;
    Format format = Format::FORMAT_ND;

    // 分配输出 tensor
    // yOut: [M, N]
    auto yOut = executor->AllocTensor(yOutType, format, format);
    yOut->SetViewShape(xShape);

    // scaleOut: [M]
    auto scaleOutShape = Shape({M});
    auto scaleOut = executor->AllocTensor(scaleOutType, format, format);
    scaleOut->SetViewShape(scaleOutShape);

    // InferShape
    auto ret = INFER_SHAPE(RotateDynamicQuant,
        OP_INPUT(x, rot),
        OP_OUTPUT(yOut, scaleOut),
        OP_ATTR(dst_type));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr),
        "RotateDynamicQuant InferShape failed.");

    // 添加到 launcher 列表
    auto ret1 = ADD_TO_LAUNCHER_LIST_AICORE(RotateDynamicQuant,
        OP_INPUT(x, rot),
        OP_OUTPUT(yOut, scaleOut),
        OP_ATTR(dst_type));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret1 != ACLNN_SUCCESS, return std::tuple(nullptr, nullptr),
        "RotateDynamicQuant ADD_TO_LAUNCHER_LIST_AICORE failed.");

    return std::make_tuple(yOut, scaleOut);
}

} // namespace l0op
