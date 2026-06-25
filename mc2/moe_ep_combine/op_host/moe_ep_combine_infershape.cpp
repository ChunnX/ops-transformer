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
 * \file moe_ep_combine_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace ops {

static constexpr size_t COMBINE_INPUT_CONTEXT_INDEX = 0;
static constexpr size_t COMBINE_INPUT_X_INDEX = 1;
static constexpr size_t COMBINE_INPUT_TOPK_IDX_INDEX = 2;
static constexpr size_t COMBINE_INPUT_RECV_SRC_METADATA_INDEX = 3;
static constexpr size_t COMBINE_INPUT_NUM_RECV_PER_EXPERT_INDEX = 4;
static constexpr size_t COMBINE_INPUT_TOPK_WEIGHTS_OPTIONAL_INDEX = 5;
static constexpr size_t COMBINE_INPUT_BIAS_0_INDEX = 6;
static constexpr size_t COMBINE_INPUT_BIAS_1_INDEX = 7;

static constexpr size_t COMBINE_OUTPUT_COMBINED_X_INDEX = 0;
static constexpr size_t COMBINE_OUTPUT_COMBINED_TOPK_WEIGHTS_OPTIONAL_INDEX = 1;

static constexpr size_t COMBINE_ATTR_EP_WORLD_SIZE_INDEX = 0;
static constexpr size_t COMBINE_ATTR_EP_RANK_ID_INDEX = 1;
static constexpr size_t COMBINE_ATTR_NUM_EXPERTS_INDEX = 2;
static constexpr size_t COMBINE_ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX = 3;
static constexpr size_t COMBINE_ATTR_CCL_BUFFER_SIZE_INDEX = 4;

static constexpr size_t DIM_TWO = 2UL;

static ge::graphStatus InferShapeMoeEpCombine(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape for MoeEpCombine.");

    const gert::Shape *xShape = context->GetInputShape(COMBINE_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);

    const gert::Shape *topkIdxShape = context->GetInputShape(COMBINE_INPUT_TOPK_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);

    int64_t numTokens = topkIdxShape->GetDim(0);
    int64_t hidden = xShape->GetDim(1);
    int64_t topK = topkIdxShape->GetDim(1);

    gert::Shape *combinedX = context->GetOutputShape(COMBINE_OUTPUT_COMBINED_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, combinedX);
    combinedX->SetDimNum(DIM_TWO);
    combinedX->SetDim(0U, numTokens);
    combinedX->SetDim(1U, hidden);

    const gert::Shape *topkWeightsShape = context->GetOptionalInputShape(COMBINE_INPUT_TOPK_WEIGHTS_OPTIONAL_INDEX);
    gert::Shape *combinedTopkWeights = context->GetOutputShape(COMBINE_OUTPUT_COMBINED_TOPK_WEIGHTS_OPTIONAL_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, combinedTopkWeights);
    if (topkWeightsShape != nullptr) {
        combinedTopkWeights->SetDimNum(DIM_TWO);
        combinedTopkWeights->SetDim(0U, numTokens);
        combinedTopkWeights->SetDim(1U, topK);
    } else {
        combinedTopkWeights->SetDimNum(0);
    }

    OP_LOGD(context->GetNodeName(), "InferShape for MoeEpCombine done.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeEpCombine(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(COMBINE_OUTPUT_COMBINED_X_INDEX, ge::DT_BF16);
    const ge::DataType topkWeightsDataType =
        context->GetOptionalInputDataType(COMBINE_INPUT_TOPK_WEIGHTS_OPTIONAL_INDEX);
    if (topkWeightsDataType != ge::DT_UNDEFINED) {
        context->SetOutputDataType(COMBINE_OUTPUT_COMBINED_TOPK_WEIGHTS_OPTIONAL_INDEX, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeEpCombine).InferShape(InferShapeMoeEpCombine).InferDataType(InferDataTypeMoeEpCombine);

} // namespace ops