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
 * \file tri_inverse_infershape.cpp
 * \brief
 */

#include "err/ops_err.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t CC_IN_INDEX = 0;
constexpr size_t CC_OUT_INDEX = 0;
constexpr size_t TRI_INVERSE_DIM_NUM = 3;
} // namespace

static ge::graphStatus InferShapeTriInverse(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TriInverse", "infer shape context is null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *inputShape = context->GetInputShape(CC_IN_INDEX);
    gert::Shape *outputShape = context->GetOutputShape(CC_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    OP_CHECK_IF(inputShape->GetDimNum() != TRI_INVERSE_DIM_NUM,
                OP_LOGE(context->GetNodeName(), "cc_in must be 3D, but got %zuD", inputShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTriInverse(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("TriInverse", "infer dtype context is null");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(CC_OUT_INDEX, context->GetInputDataType(CC_IN_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TriInverse)
    .InferShape(InferShapeTriInverse)
    .InferDataType(InferDataTypeTriInverse);
} // namespace ops
