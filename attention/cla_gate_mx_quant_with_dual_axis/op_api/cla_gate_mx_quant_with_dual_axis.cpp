/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cla_gate_mx_quant_with_dual_axis.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ClaGateMxQuantWithDualAxis);

namespace {
constexpr int64_t ROW_BLOCK = 64;

op::Shape MakeRowDataShape(const op::Shape &xShape)
{
    op::Shape shape;
    shape.AppendDim(xShape.GetDim(0));
    shape.AppendDim(xShape.GetDim(1) * xShape.GetDim(2));
    return shape;
}

op::Shape MakeRowScaleShape(const op::Shape &xShape)
{
    op::Shape shape;
    shape.AppendDim(xShape.GetDim(0));
    shape.AppendDim((xShape.GetDim(1) * xShape.GetDim(2) + ROW_BLOCK - 1) / ROW_BLOCK);
    shape.AppendDim(2);
    return shape;
}

op::Shape MakeColScaleShape(const op::Shape &xShape)
{
    op::Shape shape;
    shape.AppendDim((xShape.GetDim(0) + ROW_BLOCK - 1) / ROW_BLOCK);
    shape.AppendDim(xShape.GetDim(1) * xShape.GetDim(2));
    shape.AppendDim(2);
    return shape;
}
} // namespace

std::tuple<aclTensor *, aclTensor *, aclTensor *, aclTensor *> ClaGateMxQuantWithDualAxis(
    const aclTensor *globalOut, const aclTensor *localOut, const aclTensor *globalGateLogits,
    const aclTensor *localGateLogits, const char *roundModeOptional, int64_t scaleAlg, int64_t dstType,
    int64_t layout, int64_t quantMode, aclOpExecutor *executor)
{
    L0_DFX(ClaGateMxQuantWithDualAxis, globalOut, localOut, globalGateLogits, localGateLogits, roundModeOptional,
           scaleAlg, dstType, layout, quantMode);

    auto xShape = globalOut->GetViewShape();
    auto rowDataShape = MakeRowDataShape(xShape);
    auto rowScaleShape = MakeRowScaleShape(xShape);
    op::Shape colDataShape;
    op::Shape colScaleShape;
    if (quantMode == 0) {
        colDataShape.AppendDim(0);
        colScaleShape.AppendDim(0);
    } else {
        colDataShape = rowDataShape;
        colScaleShape = MakeColScaleShape(xShape);
    }

    auto rowData = executor->AllocTensor(rowDataShape, op::DataType(dstType));
    auto rowScale = executor->AllocTensor(rowScaleShape, op::DataType::DT_FLOAT8_E8M0);
    auto colData = executor->AllocTensor(colDataShape, op::DataType(dstType));
    auto colScale = executor->AllocTensor(colScaleShape, op::DataType::DT_FLOAT8_E8M0);
    if (rowData == nullptr || rowScale == nullptr || colData == nullptr || colScale == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ClaGateMxQuantWithDualAxis alloc output tensor failed.");
        return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
    }

    // OP_ATTR order must match op_def attr declaration:
    // dst_type(0), scale_alg(1), round_mode(2), quant_mode(3), layout(4)
    const char *roundMode = (roundModeOptional != nullptr) ? roundModeOptional : "rint";
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ClaGateMxQuantWithDualAxis, OP_INPUT(globalOut, localOut, globalGateLogits, localGateLogits),
        OP_OUTPUT(rowData, rowScale, colData, colScale), OP_ATTR(dstType, scaleAlg, roundMode, quantMode, layout));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "ClaGateMxQuantWithDualAxis launch kernel failed.");
        return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
    }
    return std::make_tuple(rowData, rowScale, colData, colScale);
}

} // namespace l0op
