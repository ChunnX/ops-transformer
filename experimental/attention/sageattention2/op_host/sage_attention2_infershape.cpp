/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#include <string>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "sage_attention2_tiling.h"

namespace ops {
namespace {
constexpr const char *kOpName = "SageAttention2";
}

ge::graphStatus InferShapeSageAttention2(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferShape context is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *qShape = context->GetInputShape(optiling::SAGE_Q_INDEX);
    gert::Shape *outShape = context->GetOutputShape(optiling::SAGE_OUT_INDEX);
    gert::Shape *lseShape = context->GetOutputShape(optiling::SAGE_LSE_INDEX);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (qShape == nullptr || outShape == nullptr || lseShape == nullptr || attrs == nullptr) {
        OP_LOGE(kOpName, "InferShape received a nullptr shape or attrs");
        return ge::GRAPH_FAILED;
    }
    if (qShape->GetDimNum() != 4U) {
        OP_LOGE(kOpName, "query must be rank 4, got %zu", qShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }
    *outShape = *qShape;

    const char *layoutPtr = attrs->GetAttrPointer<char>(optiling::SAGE_LAYOUT_ATTR);
    const std::string layout = layoutPtr == nullptr ? "HND" : std::string(layoutPtr);
    const bool *returnLsePtr = attrs->GetAttrPointer<bool>(optiling::SAGE_RETURN_LSE_ATTR);
    const bool returnLse = returnLsePtr != nullptr && *returnLsePtr;
    if (!returnLse) {
        lseShape->SetDimNum(1U);
        lseShape->SetDim(0U, 0);
        return ge::GRAPH_SUCCESS;
    }
    if (layout != "HND" && layout != "NHD") {
        OP_LOGE(kOpName, "tensor_layout must be HND or NHD, got %s", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    const size_t headAxis = layout == "HND" ? 1U : 2U;
    const size_t seqAxis = layout == "HND" ? 2U : 1U;
    lseShape->SetDimNum(3U);
    lseShape->SetDim(0U, qShape->GetDim(0U));
    lseShape->SetDim(1U, qShape->GetDim(headAxis));
    lseShape->SetDim(2U, qShape->GetDim(seqAxis));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSageAttention2(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferDataType context is nullptr");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(optiling::SAGE_OUT_INDEX, context->GetInputDataType(optiling::SAGE_Q_INDEX));
    context->SetOutputDataType(optiling::SAGE_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(SageAttention2).InferShape(InferShapeSageAttention2).InferDataType(InferDataTypeSageAttention2);
} // namespace ops

