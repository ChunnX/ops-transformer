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
 * \file causal_conv1d_grad_infershape.cpp
 * \brief CausalConv1dGrad 算子形状推导实现
 *
 * dx.shape = grad_y.shape（输入 0）
 * dweight.shape = weight.shape（输入 2）
 * dbias.shape = bias.shape（输入 3）
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4CausalConv1dGrad(gert::InferShapeContext* context)
{
    const gert::Shape* gradYShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShape);
    const gert::Shape* weightShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
    const gert::Shape* biasShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasShape);

    gert::Shape* dxShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    gert::Shape* dweightShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, dweightShape);
    gert::Shape* dbiasShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, dbiasShape);

    *dxShape = *gradYShape;
    *dweightShape = *weightShape;
    *dbiasShape = *biasShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4CausalConv1dGrad(gert::InferDataTypeContext* context)
{
    auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    context->SetOutputDataType(2, dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CausalConv1dGrad)
    .InferShape(InferShape4CausalConv1dGrad)
    .InferDataType(InferDataType4CausalConv1dGrad);

} // namespace ops
