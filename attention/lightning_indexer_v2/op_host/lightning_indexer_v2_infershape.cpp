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
 * \file lightning_indexer_v2_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;
constexpr int64_t TOPK_MIN = 1;
constexpr int64_t TOPK_MAX = 8192;

static ge::graphStatus InferShapeLightningIndexerV2(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("LightningIndexerV2", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);

    gert::Shape *sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseIndicesShape);
    gert::Shape *sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sparseValuesShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutQueryPtr);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputLayoutKeyPtr);
    const int64_t *selectedCount = attrs->GetInt(ATTR_TOPK_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, selectedCount);
    if (*selectedCount < TOPK_MIN || *selectedCount > TOPK_MAX) {
        const std::string actual = std::to_string(*selectedCount);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("LightningIndexerV2", "topk", actual.c_str(),
                                              "topk must be in [1, 8192]");
        return ge::GRAPH_FAILED;
    }
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
        OP_LOGE_FOR_INVALID_VALUE("LightningIndexerV2", "layout_q", inputLayoutQueryPtrStr.c_str(), "TND or BSND"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputLayoutKeyPtrStr != "TND" && inputLayoutKeyPtrStr != "BSND" && inputLayoutKeyPtrStr != "PA_BBND",
                OP_LOGE_FOR_INVALID_VALUE("LightningIndexerV2", "layout_k", inputLayoutKeyPtrStr.c_str(),
                                          "TND, BSND or PA_BBND"),
                return ge::GRAPH_FAILED);

    const uint32_t queryDimNum = inputLayoutQueryPtrStr == "BSND" ? 4U : 3U;
    const uint32_t keyDimNum = inputLayoutKeyPtrStr == "TND" ? 3U : 4U;
    OP_CHECK_IF(queryShape->GetDimNum() != queryDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM("LightningIndexerV2", "q", std::to_string(queryShape->GetDimNum()).c_str(),
                                             std::to_string(queryDimNum).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape->GetDimNum() != keyDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM("LightningIndexerV2", "k", std::to_string(keyShape->GetDimNum()).c_str(),
                                             std::to_string(keyDimNum).c_str()),
                return ge::GRAPH_FAILED);

    const uint32_t keyHeadDimIndex = inputLayoutKeyPtrStr == "TND" ? 1U : 2U;
    sparseIndicesShape->SetDimNum(queryDimNum);
    if (inputLayoutQueryPtrStr == "BSND") {
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));             // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1));             // 1:Dim S
        sparseIndicesShape->SetDim(2, keyShape->GetDim(keyHeadDimIndex)); // 2:Dim N
        sparseIndicesShape->SetDim(3, *selectedCount);                    // 3:Dim K
    } else {
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));             // 0:Dim T
        sparseIndicesShape->SetDim(1, keyShape->GetDim(keyHeadDimIndex)); // 1:Dim N
        sparseIndicesShape->SetDim(2, *selectedCount);                    // 2:Dim K
    }

    const int64_t *returnValue = attrs->GetInt(ATTR_RETURN_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnValue);
    bool returnValueFlag = *returnValue != 0;
    if (returnValueFlag) {
        *sparseValuesShape = *sparseIndicesShape;
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }
    const char *opName = context->GetNodeName();
    OP_LOGI(opName == nullptr ? "LightningIndexerV2" : opName, "LightningIndexerV2 InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerV2(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("LightningIndexerV2", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const char *opName = context->GetNodeName();
    opName = opName == nullptr ? "LightningIndexerV2" : opName;
    OP_LOGI(opName, "Enter LightningIndexerV2 InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = ge::DT_INT32;
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    OP_LOGI(opName, "LightningIndexerV2 InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerV2)
    .InferShape(InferShapeLightningIndexerV2)
    .InferDataType(InferDataTypeLightningIndexerV2);
} // namespace ops
