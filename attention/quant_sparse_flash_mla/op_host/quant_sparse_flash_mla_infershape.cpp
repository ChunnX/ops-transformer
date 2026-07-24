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
 * \file quant_sparse_flash_mla_infershape.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t RETURN_SOFTMAX_LSE_INDEX = 10;

ge::graphStatus InferShapeQuantSparseFlashMla(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantSparseFlashMla", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxLseShape);
    auto attr = context->GetAttrs();
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (returnSoftmaxLse) {
        *softmaxLseShape = *queryShape;
        auto lastDimIdx = softmaxLseShape->GetDimNum() - 1;
        softmaxLseShape->SetDim(lastDimIdx, 1);
    } else {
        softmaxLseShape->SetDimNum(1);
        softmaxLseShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantSparseFlashMla", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BF16); // 目前仅支持BF16
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantSparseFlashMla)
    .InferShape(InferShapeQuantSparseFlashMla)
    .InferDataType(InferDataTypeQuantSparseFlashMla);
} // namespace ops
