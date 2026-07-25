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
#include "mixed_quant_sparse_flash_mla_check.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INPUT_INDEX = 0;

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
    OPS_ERR_IF(attr == nullptr, OPS_LOG_E("MixedQuantSparseFlashMla", "attr is nullptr"),
               return ge::GRAPH_FAILED);
    const bool *returnSoftmaxLsePtr = attr->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;

    const gert::Shape *kvShape = context->GetInputShape(ORI_KV_INDEX);
    OPS_LOG_E_IF_NULL(context, kvShape, return ge::GRAPH_FAILED)
    const char *layoutQ = attr->GetStr(ATTR_LAYOUT_Q_INDEX);
    const char *layoutKv = attr->GetStr(ATTR_LAYOUT_KV_INDEX);
    std::string layoutQStr = (layoutQ != nullptr) ? std::string(layoutQ) : "BSND";
    std::string layoutKvStr = (layoutKv != nullptr) ? std::string(layoutKv) : "BSND";

    if (returnSoftmaxLse) {
        if (layoutQStr == "TND") {
            int64_t kvHeadNum;
            if (layoutKvStr == "PA_BBND") {
                kvHeadNum = kvShape->GetDim(DIM_IDX_TWO);
            } else {
                kvHeadNum = kvShape->GetDim(DIM_IDX_ONE);
            }
            OPS_ERR_IF(kvHeadNum <= 0,
                       OPS_LOG_E("MixedQuantSparseFlashMla", "kvHeadNum must be positive, but got %ld", kvHeadNum),
                       return ge::GRAPH_FAILED);
            softmaxLseShape->SetDimNum(DIM_NUM_THREE);
            softmaxLseShape->SetDim(DIM_IDX_ZERO, kvHeadNum);
            softmaxLseShape->SetDim(DIM_IDX_ONE, queryShape->GetDim(DIM_IDX_ZERO));
            softmaxLseShape->SetDim(DIM_IDX_TWO, queryShape->GetDim(DIM_IDX_ONE) / kvHeadNum);
        } else {
            int64_t kvHeadNum = kvShape->GetDim(DIM_IDX_TWO);
            OPS_ERR_IF(kvHeadNum <= 0,
                       OPS_LOG_E("MixedQuantSparseFlashMla", "kvHeadNum must be positive, but got %ld", kvHeadNum),
                       return ge::GRAPH_FAILED);
            softmaxLseShape->SetDimNum(DIM_NUM_FOUR);
            softmaxLseShape->SetDim(DIM_IDX_ZERO, queryShape->GetDim(DIM_IDX_ZERO));
            softmaxLseShape->SetDim(DIM_IDX_ONE, kvHeadNum);
            softmaxLseShape->SetDim(DIM_IDX_TWO, queryShape->GetDim(DIM_IDX_ONE));
            softmaxLseShape->SetDim(DIM_IDX_THREE, queryShape->GetDim(DIM_IDX_TWO) / kvHeadNum);
        }
    } else {
        softmaxLseShape->SetDimNum(DIM_NUM_ONE);
        softmaxLseShape->SetDim(DIM_IDX_ZERO, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMixedQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("MixedQuantSparseFlashMla", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(QUERY_INPUT_INDEX);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MixedQuantSparseFlashMla)
    .InferShape(InferShapeMixedQuantSparseFlashMla)
    .InferDataType(InferDataTypeMixedQuantSparseFlashMla);
} // namespace ops
