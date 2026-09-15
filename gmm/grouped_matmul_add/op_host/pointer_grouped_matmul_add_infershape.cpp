/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t INDEX_Y_REFS = 3;

ge::graphStatus InferShapePointerGroupedMatmulAdd(gert::InferShapeContext *context)
{
    for (size_t index = 0;; ++index) {
        const auto *inputShape = context->GetDynamicInputShape(INDEX_Y_REFS, index);
        if (inputShape == nullptr) {
            break;
        }
        auto *outputShape = context->GetOutputShape(index);
        OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtypePointerGroupedMatmulAdd(gert::InferDataTypeContext *context)
{
    for (size_t index = 0; index < context->GetComputeNodeOutputNum(); ++index) {
        context->SetOutputDataType(index, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(PointerGroupedMatmulAdd)
    .InferShape(InferShapePointerGroupedMatmulAdd)
    .InferDataType(InferDtypePointerGroupedMatmulAdd);
} // namespace ops
