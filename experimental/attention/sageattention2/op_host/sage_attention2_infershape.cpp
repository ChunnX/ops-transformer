/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
bool UnknownRank(const gert::Shape &shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == -2;
}

ge::graphStatus InferShapeSageAttention2(gert::InferShapeContext *context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    const auto *q = context->GetInputShape(0);
    auto *out = context->GetOutputShape(0);
    auto *lse = context->GetOutputShape(1);
    if (!q || !out || !lse) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 3; ++i) {
        const auto *shape = context->GetInputShape(i);
        if (!shape || (!UnknownRank(*shape) && shape->GetDimNum() != 4)) {
            return ge::GRAPH_FAILED;
        }
    }
    // The GE operator consumes canonical HND tensors. NHD conversion belongs
    // to the public ACLNN/PyTorch wrapper, not to this graph-level operator.
    // Both outputs exist in the schema even when return_lse is false.
    *out = *q;
    lse->SetDimNum(3);
    for (size_t i = 0; i < 3; ++i) {
        lse->SetDim(i, UnknownRank(*q) ? -1 : q->GetDim(i));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSageAttention2(gert::InferDataTypeContext *context)
{
    if (!context) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDataType(0);
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
        context->GetInputDataType(1) != dtype || context->GetInputDataType(2) != dtype) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(SageAttention2).InferShape(InferShapeSageAttention2).InferDataType(InferDataTypeSageAttention2);
} // namespace ops
