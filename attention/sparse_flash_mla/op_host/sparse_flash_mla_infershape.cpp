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
 * \file sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t RETURN_SOFTMAX_INDEX = 11;
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
constexpr int64_t EMPTY_LSE_SIZE = 0;
constexpr int64_t SOFTMAX_LSE_LAST_DIM = 1;
} // namespace

static ge::graphStatus InferShapeSparseFlashMla(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    if (queryShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *attentionOutShape = context->GetOutputShape(ATTN_OUT_INDEX);
    if (attentionOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    if (softmaxLseShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attr = context->GetAttrs();
    if (attr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(RETURN_SOFTMAX_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (returnSoftmaxLse) {
        *softmaxLseShape = *queryShape;
        auto lastDimIdx = softmaxLseShape->GetDimNum() - 1;
        softmaxLseShape->SetDim(lastDimIdx, SOFTMAX_LSE_LAST_DIM);
    } else {
        softmaxLseShape->SetDimNum(1);
        softmaxLseShape->SetDim(0, EMPTY_LSE_SIZE);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSparseFlashMla(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(ATTN_OUT_INDEX, inputDataType);
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashMla)
    .InferShape(InferShapeSparseFlashMla)
    .InferDataType(InferDataTypeSparseFlashMla);
} // namespace ops
