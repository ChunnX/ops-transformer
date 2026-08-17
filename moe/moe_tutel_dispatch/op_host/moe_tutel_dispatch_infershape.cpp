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
 * \file moe_tutel_dispatch_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t MOE_TUTEL_DISPATCH_IN_X_IDX = 0;
constexpr size_t MOE_TUTEL_DISPATCH_IN_GATES_IDX = 1;
constexpr size_t MOE_TUTEL_DISPATCH_OUT_Y_IDX = 0;
constexpr size_t MOE_TUTEL_DISPATCH_ATTR_CAPACITY_IDX = 0;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t X_DIM_NUM = 2;
constexpr size_t GATES_DIM_NUM = 2;
constexpr size_t Y_DIM_NUM = 3;
} // namespace

// y = [gates.dim(0), capacity, x.dim(1)]
// 专家维取自 gates.dim(0)：同族 MoeTutelCombineX 的 tiling 用 y_grad.dim(0) 反推 batchSize，
// 并按 [batchSize, samples] 索引 indices/locations，故 y.dim(0) 必须等于 gates.dim(0)。
static ge::graphStatus InferShape4MoeTutelDispatch(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do InferShape4MoeTutelDispatch");

    const gert::Shape *xShape = context->GetInputShape(MOE_TUTEL_DISPATCH_IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *gatesShape = context->GetInputShape(MOE_TUTEL_DISPATCH_IN_GATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gatesShape);
    gert::Shape *yShape = context->GetOutputShape(MOE_TUTEL_DISPATCH_OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    OP_CHECK_IF(xShape->GetDimNum() != X_DIM_NUM,
                OP_LOGE(context, "x must be 2D, but got %zuD", xShape->GetDimNum()), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gatesShape->GetDimNum() != GATES_DIM_NUM,
                OP_LOGE(context, "gates must be 2D, but got %zuD", gatesShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *capacity = attrs->GetAttrPointer<int64_t>(MOE_TUTEL_DISPATCH_ATTR_CAPACITY_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, capacity);
    OP_CHECK_IF(*capacity < 0, OP_LOGE(context, "capacity must be non-negative, but got %ld", *capacity),
                return ge::GRAPH_FAILED);

    yShape->SetDimNum(Y_DIM_NUM);
    yShape->SetDim(DIM_0, gatesShape->GetDim(DIM_0));
    yShape->SetDim(DIM_1, *capacity);
    yShape->SetDim(DIM_2, xShape->GetDim(DIM_1));

    OP_LOGD(context, "End to do InferShape4MoeTutelDispatch");
    return ge::GRAPH_SUCCESS;
}

// InferDataType 仅图场景使用，交付在 op_graph/moe_tutel_dispatch_graph_plugin.cpp
IMPL_OP_INFERSHAPE(MoeTutelDispatch).InferShape(InferShape4MoeTutelDispatch);
} // namespace ops
