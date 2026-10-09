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
 * \file causal_conv1d_backward_infershape.cpp
 * \brief CausalConv1dBackward 算子形状推导实现
 *
 * grad_x.shape = x.shape（输入 1）
 * grad_weight.shape = weight.shape（输入 2）
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_GRAD_Y = 0;
static constexpr int64_t IDX_X = 1;
static constexpr int64_t IDX_WEIGHT = 2;

static ge::graphStatus InferShape4CausalConv1dBackward(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *gradXShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradXShape);
    *gradXShape = *xShape;

    const gert::Shape *weightShape = context->GetInputShape(IDX_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    gert::Shape *gradWeightShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradWeightShape);
    *gradWeightShape = *weightShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CausalConv1dBackward(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(IDX_GRAD_Y));
    context->SetOutputDataType(1, context->GetInputDataType(IDX_GRAD_Y));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CausalConv1dBackward)
    .InferShape(InferShape4CausalConv1dBackward)
    .InferDataType(InferDataType4CausalConv1dBackward);

} // namespace ops
