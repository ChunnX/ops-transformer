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
 * \file mixed_quant_sparse_flash_mla_proto.cpp
 * \brief
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t ATTR_RETURN_SOFTMAX_LSE_INDEX = 11;

ge::graphStatus InferShapeMixedQuantSparseFlashMla(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("MixedQuantSparseFlashMla", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED)
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED)
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL(context, softmaxLseShape, return ge::GRAPH_FAILED)
    auto attr = context->GetAttrs();
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    int64_t kvHeadNum = GetKvHeadNum(kvShape, layoutKv);

    if (returnSoftmaxLse) {
        printf("[aclnnMixedQuantSparseFlashMla] returnSoftmaxLse 为True\n");
        if (queryShape->GetDimNum() == DIM_NUM_3) {
            printf("[aclnnMixedQuantSparseFlashMla] returnSoftmaxLse 为True  且进入 TND\n");
            softmaxLseShape->SetDimNum(DIM_NUM_3);
            softmaxLseShape->SetDim(DIM_INDEX_0, kvHeadNum);
            softmaxLseShape->SetDim(DIM_INDEX_1, queryShape->GetDim(DIM_INDEX_0));
            softmaxLseShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1) / kvHeadNum);
        int dimCount = softmaxLseShape->GetDimNum();
        printf("softmaxLseShape: dimNum=%d, shape=\[%lld, %lld, %lld\]\\n",
            dimCount,
            softmaxLseShape->GetDim(0),
            softmaxLseShape->GetDim(1),
            softmaxLseShape->GetDim(2));
        } else {
            softmaxLseShape->SetDimNum(DIM_NUM_4);
            softmaxLseShape->SetDim(DIM_INDEX_0, queryShape->GetDim(DIM_INDEX_0));
            softmaxLseShape->SetDim(DIM_INDEX_1, kvHeadNum);
            softmaxLseShape->SetDim(DIM_INDEX_2, queryShape->GetDim(DIM_INDEX_1));
            softmaxLseShape->SetDim(DIM_INDEX_3, queryShape->GetDim(DIM_INDEX_2) / kvHeadNum);
        }
    } else {
        printf("[aclnnMixedQuantSparseFlashMla] returnSoftmaxLse 为False\n");
        softmaxLseShape->SetDimNum(DIM_NUM_1);
        softmaxLseShape->SetDim(DIM_INDEX_0, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMixedQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("MixedQuantSparseFlashMla", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MixedQuantSparseFlashMla)
    .InferShape(InferShapeMixedQuantSparseFlashMla)
    .InferDataType(InferDataTypeMixedQuantSparseFlashMla);
} // namespace ops
