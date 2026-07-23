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
#include "error/ops_error.h"
#include "log/log.h"
#include "quant_sparse_flash_mla_check.h"

using namespace ge;

namespace ops {

ge::graphStatus InferShapeQuantSparseFlashMla(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("QuantSparseFlashMla", "InferShapeContext is nullptr"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(optiling::Q_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    gert::Shape *attentionOutShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, attentionOutShape, return ge::GRAPH_FAILED);
    *attentionOutShape = *queryShape;

    gert::Shape *softmaxLseShape = context->GetOutputShape(optiling::SOFTMAX_LSE_INDEX);
    OPS_LOG_E_IF_NULL(context, softmaxLseShape, return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E("QuantSparseFlashMla", "attrs is nullptr"),
               return ge::GRAPH_FAILED);
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(optiling::ATTR_RETURN_SOFTMAX_LSE_INDEX);
    bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;

    const gert::Shape *kvShape = context->GetInputShape(optiling::ORI_KV_INDEX);
    OPS_LOG_E_IF_NULL(context, kvShape, return ge::GRAPH_FAILED);
    const char *layoutQ = attrs->GetStr(optiling::ATTR_LAYOUT_Q_INDEX);
    const char *layoutKv = attrs->GetStr(optiling::ATTR_LAYOUT_KV_INDEX);
    std::string layoutQStr = (layoutQ != nullptr) ? std::string(layoutQ) : "BSND";
    std::string layoutKvStr = (layoutKv != nullptr) ? std::string(layoutKv) : "BSND";

    if (returnSoftmaxLse) {
        if (layoutQStr == "TND") {
            int64_t kvHeadNum;
            if (layoutKvStr == "PA_BBND") {
                kvHeadNum = kvShape->GetDim(optiling::DIM_IDX_TWO);
            } else {
                kvHeadNum = kvShape->GetDim(optiling::DIM_IDX_ONE);
            }
            OPS_ERR_IF(kvHeadNum <= 0,
                       OPS_LOG_E("QuantSparseFlashMla", "kvHeadNum must be positive, but got %ld", kvHeadNum),
                       return ge::GRAPH_FAILED);
            softmaxLseShape->SetDimNum(optiling::DIM_NUM_THREE);
            softmaxLseShape->SetDim(optiling::DIM_IDX_ZERO, kvHeadNum);
            softmaxLseShape->SetDim(optiling::DIM_IDX_ONE, queryShape->GetDim(optiling::DIM_IDX_ZERO));
            softmaxLseShape->SetDim(optiling::DIM_IDX_TWO, queryShape->GetDim(optiling::DIM_IDX_ONE) / kvHeadNum);
        } else {
            int64_t kvHeadNum = kvShape->GetDim(optiling::DIM_IDX_TWO);
            OPS_ERR_IF(kvHeadNum <= 0,
                       OPS_LOG_E("QuantSparseFlashMla", "kvHeadNum must be positive, but got %ld", kvHeadNum),
                       return ge::GRAPH_FAILED);
            softmaxLseShape->SetDimNum(optiling::DIM_NUM_FOUR);
            softmaxLseShape->SetDim(optiling::DIM_IDX_ZERO, queryShape->GetDim(optiling::DIM_IDX_ZERO));
            softmaxLseShape->SetDim(optiling::DIM_IDX_ONE, kvHeadNum);
            softmaxLseShape->SetDim(optiling::DIM_IDX_TWO, queryShape->GetDim(optiling::DIM_IDX_ONE));
            softmaxLseShape->SetDim(optiling::DIM_IDX_THREE, queryShape->GetDim(optiling::DIM_IDX_TWO) / kvHeadNum);
        }
    } else {
        softmaxLseShape->SetDimNum(optiling::DIM_NUM_ONE);
        softmaxLseShape->SetDim(optiling::DIM_IDX_ZERO, 0);
    }
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantSparseFlashMla(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("QuantSparseFlashMla", "InferDataTypeContext is nullptr"),
               return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, ge::DT_BF16); // 目前仅支持BF16
    context->SetOutputDataType(optiling::SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantSparseFlashMla)
    .InferShape(InferShapeQuantSparseFlashMla)
    .InferDataType(InferDataTypeQuantSparseFlashMla);
} // namespace ops
