/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lrn_infershape.cpp
 * \brief LRN infer shape / dtype. y = x (same 4-D NCHW shape, same dtype).
 */
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t X_IDX = 0;
constexpr uint32_t Y_IDX = 0;

static ge::graphStatus InferShapeLRN(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("LRN", "InferShapeContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *xShape = context->GetInputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape *yShape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    // y has the same 4-D NCHW shape as x (dim-by-dim copy)
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLRN(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("LRN", "InferDataTypeContext is nullptr.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(Y_IDX, context->GetInputDataType(X_IDX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LRN)
    .InferShape(InferShapeLRN)
    .InferDataType(InferDataTypeLRN);
} // namespace ops
