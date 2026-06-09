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
 * \file hc_post_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;
namespace ops {
const int32_t INPUT_IDX_X = 0;
const int32_t INPUT_IDX_RESIDUAL = 1;
const int32_t INPUT_IDX_POST = 2;
const int32_t INPUT_IDX_COMB = 3;
const int32_t INDEX_OUTPUT_Y = 0;


static ge::graphStatus InferShape4HcPost(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShape4HcPost.");

    const gert::Shape* xShape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape* residualShape = context->GetInputShape(INPUT_IDX_RESIDUAL);
    OP_CHECK_NULL_WITH_CONTEXT(context, residualShape);

    const gert::Shape* postShape = context->GetInputShape(INPUT_IDX_POST);
    OP_CHECK_NULL_WITH_CONTEXT(context, postShape);

    const gert::Shape* combShape = context->GetInputShape(INPUT_IDX_COMB);
    OP_CHECK_NULL_WITH_CONTEXT(context, combShape);

    auto yShape = context->GetOutputShape(INDEX_OUTPUT_Y);
    *yShape = *residualShape;

    OP_LOGI(context->GetNodeName(), "End to do InferShape4HcPost");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4HcPost(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "InferDtype4HcPost enter");
    const auto xDtype = context->GetInputDataType(INPUT_IDX_X);
    context->SetOutputDataType(INDEX_OUTPUT_Y, xDtype);
    OP_LOGI(context->GetNodeName(), "InferDtype4HcPost end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HcPost)
    .InferShape(InferShape4HcPost)
    .InferDataType(InferDtype4HcPost);
}  // namespace ops