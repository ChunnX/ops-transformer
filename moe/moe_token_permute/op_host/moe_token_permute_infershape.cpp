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
 * \file moe_token_permute_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/shape_util.h"


using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
static constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
static constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
static constexpr int64_t MX_QUANT_BLOCK_SIZE = 32;
static constexpr int64_t MXFP4_SCALE_BLOCK_SIZE = 64;

static ge::graphStatus InferShape4MoeTokenPermute(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteInfershape.");
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* indicesShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);

    gert::Shape* permuteXShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteXShape);
    gert::Shape* sortedIndices = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortedIndices);
    gert::Shape* expandedScaleShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedScaleShape);

    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t* quantModePtr = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, numOutTokensPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);

    int64_t sortedIndicesLen = NEG_ONE;
    sortedIndices->SetDimNum(DIM_ONE);
    if (sortedIndices->GetDim(0) < 0) {
        sortedIndices->SetDim(0U, NEG_ONE);
    } else {
        int64_t IndicesDimNnum = indicesShape->GetDimNum();
        if (IndicesDimNnum != DIM_TWO && IndicesDimNnum != DIM_ONE) {
            OP_LOGE(context->GetNodeName(), "The dim of indices should 1 or 2,but got %ld.", IndicesDimNnum);
            return ge::GRAPH_FAILED;
        }
        int64_t topK = (IndicesDimNnum == 1) ? 1 : indicesShape->GetDim(1);
        int64_t N = indicesShape->GetDim(0);
        sortedIndicesLen = (topK * N > 0) ? topK * N : NEG_ONE;
        sortedIndices->SetDim(0U, sortedIndicesLen);
    }

    *permuteXShape = *xShape;
    int64_t outTokens = NEG_ONE;
    if (sortedIndicesLen != NEG_ONE && !Ops::Base::IsUnknownRank(*xShape)) {
        outTokens = (*numOutTokensPtr <= 0) ? (*numOutTokensPtr) + sortedIndicesLen : *numOutTokensPtr;
        outTokens = std::min(outTokens, sortedIndicesLen);
        outTokens = std::max(outTokens, (int64_t)0);
        permuteXShape->SetDim(0, outTokens);
    }

    if (*quantModePtr == QUANT_MODE_MXFP8_E5M2 || *quantModePtr == QUANT_MODE_MXFP8_E4M3FN) {
        expandedScaleShape->SetDimNum(DIM_TWO);
        expandedScaleShape->SetDim(0, outTokens);
        int64_t hiddenSize = Ops::Base::IsUnknownRank(*xShape) ? NEG_ONE : xShape->GetDim(1);
        int64_t scaleCols = hiddenSize < 0 ? NEG_ONE : (hiddenSize + MX_QUANT_BLOCK_SIZE - 1) / MX_QUANT_BLOCK_SIZE;
        scaleCols = scaleCols < 0 ? NEG_ONE : ((scaleCols + 1) / 2) * 2;
        expandedScaleShape->SetDim(1, scaleCols);
    } else if (*quantModePtr == QUANT_MODE_MXFP4_E2M1) {
        expandedScaleShape->SetDimNum(DIM_THREE);
        expandedScaleShape->SetDim(0, outTokens);
        int64_t hiddenSize = Ops::Base::IsUnknownRank(*xShape) ? NEG_ONE : xShape->GetDim(1);
        int64_t scaleCols =
            hiddenSize < 0 ? NEG_ONE : (hiddenSize + MXFP4_SCALE_BLOCK_SIZE - 1) / MXFP4_SCALE_BLOCK_SIZE;
        expandedScaleShape->SetDim(1, scaleCols);
        expandedScaleShape->SetDim(2, 2);
    } else {
        expandedScaleShape->SetDimNum(DIM_ONE);
        expandedScaleShape->SetDim(0, 0);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeTokenPermute(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeTokenPermuteInferDataType.");
    auto xDtype = context->GetInputDataType(0);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* quantModePtr = attrs->GetAttrPointer<int64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantModePtr);
    if (*quantModePtr == QUANT_MODE_MXFP8_E5M2) {
        context->SetOutputDataType(0, ge::DT_FLOAT8_E5M2);
    } else if (*quantModePtr == QUANT_MODE_MXFP8_E4M3FN) {
        context->SetOutputDataType(0, ge::DT_FLOAT8_E4M3FN);
    } else if (*quantModePtr == QUANT_MODE_MXFP4_E2M1) {
        context->SetOutputDataType(0, ge::DT_FLOAT4_E2M1);
    } else {
        context->SetOutputDataType(0, xDtype);
    }
    context->SetOutputDataType(1, ge::DT_INT32);
    context->SetOutputDataType(
        2, *quantModePtr == QUANT_MODE_MXFP8_E5M2 || *quantModePtr == QUANT_MODE_MXFP8_E4M3FN ||
                   *quantModePtr == QUANT_MODE_MXFP4_E2M1
               ? ge::DT_FLOAT8_E8M0
               : ge::DT_FLOAT);
    OP_LOGD(context->GetNodeName(), "End to do MoeTokenPermuteInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenPermute).InferShape(InferShape4MoeTokenPermute).InferDataType(InferDataType4MoeTokenPermute);
} // namespace ops
