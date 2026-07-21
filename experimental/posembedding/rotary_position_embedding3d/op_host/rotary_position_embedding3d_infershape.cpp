/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ...
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRotaryPositionEmbedding3d(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    auto dimNum = inputShape->GetDimNum();
    outputShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        outputShape->SetDim(i, inputShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotaryPositionEmbedding3d).InferShape(InferShapeRotaryPositionEmbedding3d);
} // namespace ops
