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
 * \file quant_lightning_indexer_v2_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 7;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr int64_t TOPK_MIN = 1;
constexpr int64_t TOPK_MAX = 8192;
constexpr uint32_t SPARSE_INDICES_OUTPUT_INDEX = 0;
constexpr uint32_t SPARSE_VALUES_OUTPUT_INDEX = 1;
constexpr uint32_t BSND_BATCH_AXIS = 0U;
constexpr uint32_t BSND_SEQ_AXIS = 1U;
constexpr uint32_t BSND_HEAD_AXIS = 2U;
constexpr uint32_t BSND_TOPK_AXIS = 3U;
constexpr uint32_t TND_TOKEN_AXIS = 0U;
constexpr uint32_t TND_HEAD_AXIS = 1U;
constexpr uint32_t TND_TOPK_AXIS = 2U;
constexpr char OP_NAME[] = "QuantLightningIndexerV2";
constexpr char LAYOUT_BSND[] = "BSND";
constexpr char LAYOUT_TND[] = "TND";
constexpr char LAYOUT_PA_BBND[] = "PA_BBND";

struct InferShapeParams {
    const gert::Shape *query = nullptr;
    const gert::Shape *key = nullptr;
    gert::Shape *sparseIndices = nullptr;
    gert::Shape *sparseValues = nullptr;
    std::string layoutQ;
    std::string layoutK;
    int64_t topk = 0;
    int64_t returnValue = 0;
};

static ge::graphStatus ParseInferShapeParams(gert::InferShapeContext *context, InferShapeParams &params)
{
    params.query = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, params.query);
    params.key = context->GetInputShape(KEY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, params.key);
    params.sparseIndices = context->GetOutputShape(SPARSE_INDICES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, params.sparseIndices);
    params.sparseValues = context->GetOutputShape(SPARSE_VALUES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, params.sparseValues);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *layoutQ = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutQ);
    const char *layoutK = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, layoutK);
    const int64_t *topk = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topk);
    const int64_t *returnValue = attrs->GetInt(ATTR_RETURN_VALUE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, returnValue);
    params.layoutQ = layoutQ;
    params.layoutK = layoutK;
    params.topk = *topk;
    params.returnValue = *returnValue;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ValidateInferShapeParams(const InferShapeParams &params)
{
    if (params.topk < TOPK_MIN || params.topk > TOPK_MAX) {
        const std::string actual = std::to_string(params.topk);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(OP_NAME, "topk", actual.c_str(), "topk must be in [1, 8192]");
        return ge::GRAPH_FAILED;
    }
    if (params.layoutQ != LAYOUT_TND && params.layoutQ != LAYOUT_BSND) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "layout_q", params.layoutQ.c_str(), "BSND or TND");
        return GRAPH_FAILED;
    }
    if (params.layoutK != LAYOUT_TND && params.layoutK != LAYOUT_BSND && params.layoutK != LAYOUT_PA_BBND) {
        OP_LOGE_FOR_INVALID_VALUE(OP_NAME, "layout_k", params.layoutK.c_str(), "BSND, TND or PA_BBND");
        return GRAPH_FAILED;
    }
    const uint32_t queryRank = params.layoutQ == LAYOUT_BSND ? DIM_NUM_4 : DIM_NUM_3;
    const uint32_t keyRank = params.layoutK == LAYOUT_TND ? DIM_NUM_3 : DIM_NUM_4;
    if (params.query->GetDimNum() != queryRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(OP_NAME, "q", std::to_string(params.query->GetDimNum()).c_str(),
                                     std::to_string(queryRank).c_str());
        return ge::GRAPH_FAILED;
    }
    if (params.key->GetDimNum() != keyRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(OP_NAME, "k", std::to_string(params.key->GetDimNum()).c_str(),
                                     std::to_string(keyRank).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void SetOutputShapes(const InferShapeParams &params)
{
    const uint32_t keyHeadAxis = params.layoutK == LAYOUT_TND ? TND_HEAD_AXIS : BSND_HEAD_AXIS;
    params.sparseIndices->SetDimNum(params.layoutQ == LAYOUT_BSND ? DIM_NUM_4 : DIM_NUM_3);
    params.sparseIndices->SetDim(TND_TOKEN_AXIS, params.query->GetDim(TND_TOKEN_AXIS));
    if (params.layoutQ == LAYOUT_BSND) {
        params.sparseIndices->SetDim(BSND_SEQ_AXIS, params.query->GetDim(BSND_SEQ_AXIS));
        params.sparseIndices->SetDim(BSND_HEAD_AXIS, params.key->GetDim(keyHeadAxis));
        params.sparseIndices->SetDim(BSND_TOPK_AXIS, params.topk);
    } else {
        params.sparseIndices->SetDim(TND_HEAD_AXIS, params.key->GetDim(keyHeadAxis));
        params.sparseIndices->SetDim(TND_TOPK_AXIS, params.topk);
    }
    if (params.returnValue != 0) {
        *params.sparseValues = *params.sparseIndices;
    } else {
        params.sparseValues->SetDimNum(1);
        params.sparseValues->SetDim(TND_TOKEN_AXIS, 0);
    }
}

static ge::graphStatus InferShapeQuantLightningIndexerV2(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE(OP_NAME, "context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    InferShapeParams params;
    if (ParseInferShapeParams(context, params) != ge::GRAPH_SUCCESS ||
        ValidateInferShapeParams(params) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetOutputShapes(params);
    const char *opName = context->GetNodeName();
    OP_LOGD(opName == nullptr ? OP_NAME : opName, "QuantLightningIndexerV2 InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantLightningIndexerV2(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("QuantLightningIndexerV2", "InferDataTypeContext context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    const char *opName = context->GetNodeName();
    opName = opName == nullptr ? "QuantLightningIndexerV2" : opName;
    OP_LOGD(opName, "Enter QuantLightningIndexerV2 InferDataType impl.");
    // default index data type is int32
    ge::DataType outputType = ge::DT_INT32;
    context->SetOutputDataType(0, outputType);
    OP_LOGD(opName, "QuantLightningIndexerV2 InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantLightningIndexerV2)
    .InferShape(InferShapeQuantLightningIndexerV2)
    .InferDataType(InferDataTypeQuantLightningIndexerV2);
} // namespace ops
