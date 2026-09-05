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
constexpr int64_t EXPECTED_S = 256;
constexpr int64_t EXPECTED_B = 4;
constexpr int64_t EXPECTED_N = 96;
constexpr int64_t EXPECTED_D = 256;
constexpr int64_t SPLIT_BLOCK_SIZE = 64;

op::Shape MakeRowDataShape()
{
    op::Shape shape;
    shape.AppendDim(EXPECTED_S * EXPECTED_B);
    shape.AppendDim(EXPECTED_N * EXPECTED_D);
    return shape;
}

op::Shape MakeRowScaleShape()
{
    op::Shape shape;
    shape.AppendDim(EXPECTED_S * EXPECTED_B);
    shape.AppendDim((EXPECTED_N * EXPECTED_D) / SPLIT_BLOCK_SIZE);
    shape.AppendDim(2);
    return shape;
}

op::Shape MakeColScaleShape()
{
    op::Shape shape;
    shape.AppendDim((EXPECTED_S * EXPECTED_B) / SPLIT_BLOCK_SIZE);
    shape.AppendDim(EXPECTED_N * EXPECTED_D);
    shape.AppendDim(2);
    return shape;
}
} // namespace

std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*> ClaGateMxQuantWithDualAxis(
    const aclTensor* globalOut, const aclTensor* localOut, const aclTensor* globalGateLogits,
    const aclTensor* localGateLogits, int64_t scaleAlg, int64_t dstType, aclOpExecutor* executor)
{
    L0_DFX(ClaGateMxQuantWithDualAxis, globalOut, localOut, globalGateLogits, localGateLogits, scaleAlg, dstType);

    auto rowDataShape = MakeRowDataShape();
    auto colDataShape = rowDataShape;
    auto rowScaleShape = MakeRowScaleShape();
    auto colScaleShape = MakeColScaleShape();

    auto rowData = executor->AllocTensor(rowDataShape, op::DataType(dstType));
    auto rowScale = executor->AllocTensor(rowScaleShape, op::DataType::DT_FLOAT8_E8M0);
    auto colData = executor->AllocTensor(colDataShape, op::DataType(dstType));
    auto colScale = executor->AllocTensor(colScaleShape, op::DataType::DT_FLOAT8_E8M0);
    if (rowData == nullptr || rowScale == nullptr || colData == nullptr || colScale == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ClaGateMxQuantWithDualAxis alloc output tensor failed.");
        return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
    }

    // OP_ATTR 顺序与 op_host ClaGateMxQuantWithDualAxis 的 attr 声明顺序保持一致。
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ClaGateMxQuantWithDualAxis,
                                           OP_INPUT(globalOut, localOut, globalGateLogits, localGateLogits),
                                           OP_OUTPUT(rowData, rowScale, colData, colScale),
                                           OP_ATTR(scaleAlg, dstType));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "ClaGateMxQuantWithDualAxis launch kernel failed.");
        return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
    }
    return std::make_tuple(rowData, rowScale, colData, colScale);
}

} // namespace l0op
