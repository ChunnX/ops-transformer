/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0
constexpr uint32_t LAYOUT_QUERY_ATTR_INDEX = 5;
constexpr uint32_t RETURN_SOFTMAX_LSE_INDEX = 7;
constexpr uint32_t ROPE_HEAD_DIM_ATTR_INDEX = 2;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;


ge::graphStatus InferShapeKvQuantSparseAttnSharedKv(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvQuantSparseAttnSharedKv", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(LAYOUT_QUERY_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    const int64_t ropeHeadDim = *attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);

    *attentionOutShape = *queryShape;
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    auto attr = context->GetAttrs();
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (returnSoftmaxLse) {
        // cl : 应该需要修改
        *softmaxLseShape = *queryShape;
        auto lastDimIdx = softmaxLseShape->GetDimNum() - 1;
        softmaxLseShape->SetDim(lastDimIdx, 1);
    } else {
        softmaxLseShape->SetDimNum(1);
        softmaxLseShape->SetDim(0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvQuantSparseAttnSharedKv(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("KvQuantSparseAttnSharedkv", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(KvQuantSparseAttnSharedkv).InferShape(InferShapeKvQuantSparseAttnSharedkv).InferDataType(InferDataTypeKvQuantSparseAttnSharedkv);
} // namespace ops