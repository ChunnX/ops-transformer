/**
 * Copyright (c) 2026 fk2323
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fmm016_streaming_top_k_blocked_v2_infershape.cpp
 * \brief
 */
#include <register/op_impl_registry.h>
namespace ge {
static ge::graphStatus StreamingTopKBlockedShape(gert::InferShapeContext *context)
{
    const auto *q = context->GetInputShape(0);
    const int64_t *attr = context->GetAttrs()->GetInt(0);
    const int64_t topK = attr == nullptr ? 1 : *attr;
    gert::Shape shape({q->GetDim(0), q->GetDim(1), q->GetDim(2), topK});
    *context->GetOutputShape(0) = shape;
    *context->GetOutputShape(1) = shape;
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus StreamingTopKBlockedType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    context->SetOutputDataType(1, ge::DT_UINT8);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ge {
IMPL_OP_INFERSHAPE(Fmm016StreamingTopKBlockedV2)
    .InferShape(StreamingTopKBlockedShape)
    .InferDataType(StreamingTopKBlockedType);
}
