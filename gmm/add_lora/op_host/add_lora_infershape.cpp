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
 * \file add_lora_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_OUT_INDEX = 0;

static ge::graphStatus InferShapeAddLora(gert::InferShapeContext *context)
{
    const gert::Shape *yShape = context->GetInputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape *yOutShape = context->GetOutputShape(Y_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yOutShape);

    // y_out ≡ y（in-place：调用方将 y_out 绑定到 y 同地址），输出 shape = y 的前 2 维 [B, H3]。
    yOutShape->SetDimNum(2);
    yOutShape->SetDim(0, yShape->GetDim(0));
    yOutShape->SetDim(1, yShape->GetDim(1));

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAddLora(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(Y_OUT_INDEX, ge::DT_FLOAT16);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AddLora)
    .InferShape(InferShapeAddLora)
    .InferDataType(InferDataTypeAddLora);
} // namespace ops
