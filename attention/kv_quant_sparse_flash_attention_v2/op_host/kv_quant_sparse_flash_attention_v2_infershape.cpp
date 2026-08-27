/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_flash_attention_v2_infershape.cpp
 * \brief
 */

#include "kv_quant_sparse_flash_attention_v2_infershape.h"
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr uint32_t LAYOUT_KV_ATTR_INDEX = 5;
constexpr uint32_t RETURN_SOFTMAX_LSE_ATTR_INDEX = 13;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

ge::graphStatus InferShapeKvQuantSparseFlashAttentionV2(gert::InferShapeContext *context)
{
    auto ret = InferShapeKvQuantSparseFlashAttention(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(LAYOUT_KV_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    const bool *lseFlag = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
    bool returnSoftmaxLse = (lseFlag != nullptr) ? *lseFlag : false;

    gert::Shape *softmaxMaxShape = context->GetOutputShape(1);
    gert::Shape *softmaxSumShape = context->GetOutputShape(2);
    if (softmaxMaxShape == nullptr || softmaxSumShape == nullptr) {
        return GRAPH_SUCCESS;
    }

    if (returnSoftmaxLse) {
        if (queryShape->GetDimNum() == DIM_NUM_3) {
            if (inputLayoutKeyPtrStr == "PA_BSND") {
                softmaxMaxShape->SetDimNum(DIM_NUM_3);
                softmaxMaxShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_2));
                softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_2));

                softmaxSumShape->SetDimNum(DIM_NUM_3);
                softmaxSumShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_2));
                softmaxSumShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_2));
            } else {
                softmaxMaxShape->SetDimNum(DIM_NUM_3);
                softmaxMaxShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_1));
                softmaxMaxShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_1));

                softmaxSumShape->SetDimNum(DIM_NUM_3);
                softmaxSumShape->SetDim(DIM_INDEX_0, keyShape->GetDim(DIM_INDEX_1));
                softmaxSumShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
                softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / keyShape->GetDim(DIM_INDEX_1));
            }
        } else {
            softmaxMaxShape->SetDimNum(DIM_NUM_4);
            softmaxMaxShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxMaxShape->SetDim(DIM_INDEX_1, keyShape->GetDim(DIM_INDEX_2));
            softmaxMaxShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxMaxShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / keyShape->GetDim(DIM_INDEX_2));

            softmaxSumShape->SetDimNum(DIM_NUM_4);
            softmaxSumShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxSumShape->SetDim(DIM_INDEX_1, keyShape->GetDim(DIM_INDEX_2));
            softmaxSumShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxSumShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / keyShape->GetDim(DIM_INDEX_2));
        }
    } else {
        softmaxMaxShape->SetDimNum(DIM_NUM_1);
        softmaxMaxShape->SetDim(DIM_INDEX_0, 0);
        softmaxSumShape->SetDimNum(DIM_NUM_1);
        softmaxSumShape->SetDim(DIM_INDEX_0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvQuantSparseFlashAttentionV2(gert::InferDataTypeContext *context)
{
    auto ret = InferDataTypeKvQuantSparseFlashAttention(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvQuantSparseFlashAttentionV2)
    .InferShape(InferShapeKvQuantSparseFlashAttentionV2)
    .InferDataType(InferDataTypeKvQuantSparseFlashAttentionV2);

} // namespace ops
