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

#include <cstdint>
#include <limits>
#include <string>

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tiling_common/qbsa_op_contract.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr const char *kOpName = "QuantBlockSparseAttn";

struct QbsaQueryShapeFacts {
    int64_t tokenNum = 0;
    int64_t headNum = 0;
};

ge::graphStatus ParseQueryShape(const gert::Shape &queryShape, const std::string &layoutQ,
                                QbsaQueryShapeFacts &shapeFacts)
{
    if (queryShape.GetDimNum() != 3U || (layoutQ != "TND" && layoutQ != "NTD")) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "query",
                                                 std::to_string(queryShape.GetDimNum()) + "D with layout " + layoutQ,
                                                 "3D with layout TND or 3D with layout NTD");
        return ge::GRAPH_FAILED;
    }
    const size_t tokenAxis = layoutQ == "TND" ? 0U : 1U;
    const size_t headAxis = layoutQ == "TND" ? 1U : 0U;
    shapeFacts.tokenNum = queryShape.GetDim(tokenAxis);
    shapeFacts.headNum = queryShape.GetDim(headAxis);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferAttentionOutShape(const QbsaQueryShapeFacts &shapeFacts, const gert::Shape &valueShape,
                                       const std::string &layoutOut, gert::Shape &attentionOutShape)
{
    if (layoutOut != "TND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "layout_out", layoutOut,
                                              "Only the TND output contract is currently supported");
        return ge::GRAPH_FAILED;
    }
    if (valueShape.GetDimNum() != 4U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(kOpName, "value", std::to_string(valueShape.GetDimNum()) + "D",
                                                 "4D PA_BNBD [blockNum, kvHeadNum, blockSize, headDim]");
        return ge::GRAPH_FAILED;
    }
    attentionOutShape.SetDimNum(3U);
    attentionOutShape.SetDim(0U, shapeFacts.tokenNum);
    attentionOutShape.SetDim(1U, shapeFacts.headNum);
    attentionOutShape.SetDim(2U, valueShape.GetDim(3U));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferSoftmaxLseShape(const QbsaQueryShapeFacts &shapeFacts, uint32_t quantMode,
                                     gert::Shape &softmaxLseShape)
{
    softmaxLseShape.SetDimNum(2U);
    if (quantMode == optiling::QBSA_QUANT_MODE_FP8) {
        softmaxLseShape.SetDim(0U, shapeFacts.headNum);
        softmaxLseShape.SetDim(1U, shapeFacts.tokenNum);
        return ge::GRAPH_SUCCESS;
    }
    if (quantMode == optiling::QBSA_QUANT_MODE_MXFP8_FULL_QUANT) {
        softmaxLseShape.SetDim(0U, shapeFacts.tokenNum);
        softmaxLseShape.SetDim(1U, shapeFacts.headNum);
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quant_mode", std::to_string(quantMode),
                                          "No softmax_lse output contract is defined for this quant_mode");
    return ge::GRAPH_FAILED;
}
} // namespace

ge::graphStatus InferShapeQuantBlockSparseAttn(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferShape: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(optiling::QBSA_QUERY_INDEX);
    const gert::Shape *valueShape = context->GetInputShape(optiling::QBSA_VALUE_INDEX);
    if (queryShape == nullptr || valueShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: query or value shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *attentionOutShape = context->GetOutputShape(optiling::QBSA_ATTENTION_OUT_INDEX);
    if (attentionOutShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: attentionOut shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(kOpName, "InferShape: attrs is nullptr");
        return ge::GRAPH_FAILED;
    }
    const std::string layoutQ = optiling::QBSAGetStringAttr(attrs, optiling::QBSA_LAYOUT_Q_ATTR_INDEX, "TND");
    QbsaQueryShapeFacts queryShapeFacts;
    if (ParseQueryShape(*queryShape, layoutQ, queryShapeFacts) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const std::string layoutOut = optiling::QBSAGetStringAttr(attrs, optiling::QBSA_LAYOUT_OUT_ATTR_INDEX, "TND");
    if (InferAttentionOutShape(queryShapeFacts, *valueShape, layoutOut, *attentionOutShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *softmaxLseShape = context->GetOutputShape(optiling::QBSA_SOFTMAX_LSE_INDEX);
    if (softmaxLseShape == nullptr) {
        OP_LOGE(kOpName, "InferShape: softmaxLse shape is nullptr");
        return ge::GRAPH_FAILED;
    }
    const bool *returnSoftmaxLsePtr = attrs->GetAttrPointer<bool>(optiling::QBSA_RETURN_SOFTMAX_LSE_ATTR_INDEX);
    const bool returnSoftmaxLse = (returnSoftmaxLsePtr != nullptr) ? *returnSoftmaxLsePtr : false;
    if (!returnSoftmaxLse) {
        softmaxLseShape->SetDimNum(1U);
        softmaxLseShape->SetDim(0U, 0);
        return ge::GRAPH_SUCCESS;
    }
    const int64_t *quantModePtr = attrs->GetAttrPointer<int64_t>(optiling::QBSA_QUANT_MODE_ATTR_INDEX);
    if (quantModePtr != nullptr &&
        (*quantModePtr < 0 || *quantModePtr > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(kOpName, "quant_mode", std::to_string(*quantModePtr),
                                              "Must be representable as uint32_t");
        return ge::GRAPH_FAILED;
    }
    const uint32_t quantMode =
        quantModePtr == nullptr ? optiling::QBSA_QUANT_MODE_FP8 : static_cast<uint32_t>(*quantModePtr);
    return InferSoftmaxLseShape(queryShapeFacts, quantMode, *softmaxLseShape);
}

ge::graphStatus InferDataTypeQuantBlockSparseAttn(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "InferDataType: context is nullptr");
        return ge::GRAPH_FAILED;
    }
    ge::DataType attentionOutType = context->GetOutputDataType(optiling::QBSA_ATTENTION_OUT_INDEX);
    if (attentionOutType != ge::DT_BF16) {
        attentionOutType = ge::DT_BF16;
    }
    context->SetOutputDataType(optiling::QBSA_ATTENTION_OUT_INDEX, attentionOutType);
    context->SetOutputDataType(optiling::QBSA_SOFTMAX_LSE_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBlockSparseAttn)
    .InferShape(InferShapeQuantBlockSparseAttn)
    .InferDataType(InferDataTypeQuantBlockSparseAttn);
} // namespace ops
