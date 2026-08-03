/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_rotary_pos_emb_grad_infershape.cpp
 * \brief apply_rotary_pos_emb_grad 形状推导
 * dx_q = shape(dy_q), dx_k = shape(dy_k)
 * dcos = shape(cos), dsin = shape(sin)
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t INPUT_DYQ_IDX = 0;
static constexpr size_t INPUT_DYK_IDX = 1;
static constexpr size_t INPUT_COS_IDX = 2;
static constexpr size_t INPUT_SIN_IDX = 3;
static constexpr size_t OUTPUT_DXQ_IDX = 0;
static constexpr size_t OUTPUT_DXK_IDX = 1;
static constexpr size_t OUTPUT_DCOS_IDX = 2;
static constexpr size_t OUTPUT_DSIN_IDX = 3;

static ge::graphStatus InferShapeForApplyRotaryPosEmbGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeForApplyRotaryPosEmbGrad.");
    const gert::Shape* dyQShape = context->GetInputShape(INPUT_DYQ_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyQShape);
    const gert::Shape* dyKShape = context->GetInputShape(INPUT_DYK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dyKShape);
    const gert::Shape* cosShape = context->GetInputShape(INPUT_COS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosShape);
    const gert::Shape* sinShape = context->GetInputShape(INPUT_SIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinShape);

    gert::Shape* dxQShape = context->GetOutputShape(OUTPUT_DXQ_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxQShape);
    gert::Shape* dxKShape = context->GetOutputShape(OUTPUT_DXK_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxKShape);
    gert::Shape* dcosShape = context->GetOutputShape(OUTPUT_DCOS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dcosShape);
    gert::Shape* dsinShape = context->GetOutputShape(OUTPUT_DSIN_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dsinShape);

    *dxQShape = *dyQShape;
    *dxKShape = *dyKShape;
    *dcosShape = *cosShape;
    *dsinShape = *sinShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForApplyRotaryPosEmbGrad.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataTypeForApplyRotaryPosEmbGrad(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_DXQ_IDX, context->GetInputDataType(INPUT_DYQ_IDX));
    context->SetOutputDataType(OUTPUT_DXK_IDX, context->GetInputDataType(INPUT_DYK_IDX));
    context->SetOutputDataType(OUTPUT_DCOS_IDX, context->GetInputDataType(INPUT_COS_IDX));
    context->SetOutputDataType(OUTPUT_DSIN_IDX, context->GetInputDataType(INPUT_SIN_IDX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ApplyRotaryPosEmbGrad)
    .InferShape(InferShapeForApplyRotaryPosEmbGrad)
    .InferDataType(InferDataTypeForApplyRotaryPosEmbGrad);
} // namespace ops
