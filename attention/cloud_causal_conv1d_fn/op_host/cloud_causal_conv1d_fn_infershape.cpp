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
 * \file cloud_causal_conv1d_fn_infershape.cpp
 * \brief
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr int64_t INPUT_X_INDEX = 0;
constexpr int64_t INPUT_WEIGHT_INDEX = 1;
constexpr int64_t INPUT_BIAS_INDEX = 2;
constexpr int64_t INPUT_CONV_STATES_INDEX = 3;
constexpr int64_t INPUT_QUERY_START_LOC_INDEX = 4;
constexpr int64_t INPUT_CACHE_INDICES_INDEX = 5;
constexpr int64_t INPUT_NUM_ACCEPTED_TOKEN_INDEX = 6;

constexpr int64_t OUTPUT_Y_INDEX = 0;

ge::graphStatus InferShape4CloudCausalConv1dFn(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do InferShape4CloudCausalConv1dFn");

    // inferShape y: same shape as input x
    auto inputXShape = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXShape);
    auto outputYShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputYShape);
    *outputYShape = *inputXShape;

    OP_LOGD(context, "End to do InferShape4CloudCausalConv1dFn");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4CloudCausalConv1dFn(gert::InferDataTypeContext *context)
{
    OP_LOGD(context, "Begin to do InferDataType4CloudCausalConv1dFn");

    // output y dtype: same as input x
    context->SetOutputDataType(OUTPUT_Y_INDEX, context->GetInputDataType(INPUT_X_INDEX));

    OP_LOGD(context, "End to do InferDataType4CloudCausalConv1dFn");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CloudCausalConv1dFn)
    .InferShape(InferShape4CloudCausalConv1dFn)
    .InferDataType(InferDataType4CloudCausalConv1dFn);

} // namespace ops
