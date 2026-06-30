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
 * \file quant_block_sparse_attn_infershape.cpp
 * \brief QuantBlockSparseAttn infer shape and data type.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "quant_block_sparse_attn_tiling.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";
} // namespace

ge::graphStatus InferShapeQuantBlockSparseAttn(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferShape: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(optiling::BSA_QUERY_INDEX);
    if (queryShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: query shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *attentionOutShape = context->GetOutputShape(optiling::BSA_ATTENTION_OUT_INDEX);
    if (attentionOutShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: attentionOut shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(kOpName, "InferShape: attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const std::string layoutQ = optiling::BSAGetStringAttr(attrs, optiling::BSA_LAYOUT_Q_ATTR_INDEX, "TND");
    if (layoutQ == "NTD" && queryShape->GetDimNum() == 3U) {
        attentionOutShape->SetDimNum(3);
        attentionOutShape->SetDim(0, queryShape->GetDim(1)); // T
        attentionOutShape->SetDim(1, queryShape->GetDim(0)); // N
        attentionOutShape->SetDim(2, queryShape->GetDim(2)); // D
    } else {
        *attentionOutShape = *queryShape;
    }

    gert::Shape *softmaxLseShape = context->GetOutputShape(optiling::BSA_SOFTMAX_LSE_INDEX);
    if (softmaxLseShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: softmaxLse shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(optiling::BSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    const bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (!returnSoftmaxLse) {
        softmaxLseShape->SetDimNum(1);
        softmaxLseShape->SetDim(0, 0);
        return ge::GRAPH_SUCCESS;
    }

    const size_t queryDimNum = queryShape->GetDimNum();
    if (queryDimNum == 4U) {
        softmaxLseShape->SetDimNum(3);
        softmaxLseShape->SetDim(0, queryShape->GetDim(0)); // B
        softmaxLseShape->SetDim(1, queryShape->GetDim(2)); // N1
        softmaxLseShape->SetDim(2, queryShape->GetDim(1)); // S1
    } else if (queryDimNum == 3U) {
        softmaxLseShape->SetDimNum(2);
        softmaxLseShape->SetDim(0, queryShape->GetDim(1)); // N1
        softmaxLseShape->SetDim(1, queryShape->GetDim(0)); // T
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, "query", std::to_string(queryDimNum) + "D", "3D or 4D");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeQuantBlockSparseAttn(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferDataType: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    ge::DataType attentionOutType = context->GetOutputDataType(optiling::BSA_ATTENTION_OUT_INDEX);
    if (attentionOutType != ge::DT_BF16) {
        attentionOutType = ge::DT_BF16;
    }
    context->SetOutputDataType(optiling::BSA_ATTENTION_OUT_INDEX, attentionOutType);
    context->SetOutputDataType(optiling::BSA_SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(QuantBlockSparseAttn)
    .InferShape(InferShapeQuantBlockSparseAttn)
    .InferDataType(InferDataTypeQuantBlockSparseAttn);
} // namespace ops
