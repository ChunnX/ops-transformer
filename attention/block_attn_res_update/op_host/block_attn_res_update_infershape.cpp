/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t PARTIAL_BLOCK_INDEX = 0L;
constexpr int64_t PARTIAL_DELTA_INDEX = 1L;
constexpr int64_t H_INDEX = 0L;
constexpr int64_t PARTIAL_BLOCKS_INDEX = 1L;
} // namespace

static ge::graphStatus InferShape4BlockAttnResUpdate(gert::InferShapeContext *context)
{
    const gert::Shape *partialBlockShape = context->GetInputShape(PARTIAL_BLOCK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialBlockShape);
    const gert::Shape *partialDeltaShape = context->GetInputShape(PARTIAL_DELTA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialDeltaShape);

    gert::Shape *hShape = context->GetOutputShape(H_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, hShape);
    gert::Shape *partialBlocksShape = context->GetOutputShape(PARTIAL_BLOCKS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialBlocksShape);

    *hShape = *partialDeltaShape;
    *partialBlocksShape = *partialBlockShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BlockAttnResUpdate(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(H_INDEX, context->GetInputDataType(PARTIAL_DELTA_INDEX));
    context->SetOutputDataType(PARTIAL_BLOCKS_INDEX, context->GetInputDataType(PARTIAL_BLOCK_INDEX));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockAttnResUpdate)
    .InferShape(InferShape4BlockAttnResUpdate)
    .InferDataType(InferDataType4BlockAttnResUpdate);
} // namespace ops
