/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
    // INPUT
    constexpr uint32_t HIDDEN_STATES_INPUT_INDEX = 0;
    constexpr uint32_t WK_INPUT_INDEX = 1;
    constexpr uint32_t GATE_WEIGHT_INPUT_INDEX = 2;

    constexpr uint32_t STATE_CACHE_INPUT_INDEX = 3;

    constexpr uint32_t APE_INPUT_INDEX = 4;

    // INPUT(OPTION)
    constexpr uint32_t CACHE_BLOCK_TABLE_INPUT_INDEX = 5;

    constexpr uint32_t SEQ_LENS_INPUT_INDEX = 6;
    constexpr uint32_t SEQ_USED_INPUT_INDEX = 7;
    constexpr uint32_t START_POS_INPUT_INDEX = 8;

    // ATTR
    constexpr uint32_t CMP_RATIO_ATTR_INDEX = 0;
    constexpr uint32_t COFF_ATTR_INDEX = 1;
    constexpr uint32_t CACHE_MODE_ATTR_INDEX = 2;
    constexpr uint32_t STATE_CACHE_STRIDE_DIM0_ATTR_INDEX = 3;

    // OUTPUT
    constexpr uint32_t POOLED_KEY_OUTPUT_INDEX = 0;

    // ATTR DEFAULT VALUE
    constexpr uint32_t CMP_RATIO_VALUE = 4;
    constexpr uint32_t COFF_VALUE = 1;

struct KPoolProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t Sr { 0 };
    int64_t H { 0 };
    int64_t D { 0 };
};

// tmp
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_INDEX_0 = 0;
constexpr uint32_t DIM_INDEX_1 = 1;
constexpr uint32_t DIM_INDEX_2 = 2;
constexpr uint32_t DIM_INDEX_3 = 3;

ge::graphStatus GetKPoolShapeDim(const gert::InferShapeContext* context, KPoolProtoShapeParam &shapeParam)
{
    auto hiddenStatesShape = context->GetRequiredInputShape(HIDDEN_STATES_INPUT_INDEX);      // (B, S, H) | (T, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, hiddenStatesShape);
    auto wkShape = context->GetRequiredInputShape(WK_INPUT_INDEX);  // (coff * D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, wkShape);
    auto gateWeightShape = context->GetRequiredInputShape(GATE_WEIGHT_INPUT_INDEX);  // (coff * D, H)
    OP_CHECK_NULL_WITH_CONTEXT(context, gateWeightShape);
    // (block_num, block_size, 2 * coff * D) | (B, token_size, 2 * coff * D)
    auto stateCacheShape = context->GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, stateCacheShape);

    auto apeShape = context->GetRequiredInputShape(APE_INPUT_INDEX);    // (r, coff * D)
    OP_CHECK_NULL_WITH_CONTEXT(context, apeShape);

    // (B, sMax/block_size) | (B, )
    auto cacheBlockTableShape = context->GetOptionalInputShape(CACHE_BLOCK_TABLE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cacheBlockTableShape);
    if (hiddenStatesShape->GetDimNum() == DIM_NUM_2) {
        auto seqLensShape = context->GetOptionalInputShape(SEQ_LENS_INPUT_INDEX);    // (B+1,)
        OP_CHECK_NULL_WITH_CONTEXT(context, seqLensShape);
    }
    auto seqUsedShape = context->GetOptionalInputShape(SEQ_USED_INPUT_INDEX);    // (B,)
    OP_CHECK_NULL_WITH_CONTEXT(context, seqUsedShape);
    auto startPosShape = context->GetOptionalInputShape(START_POS_INPUT_INDEX);    // (B,)
    OP_CHECK_NULL_WITH_CONTEXT(context, startPosShape);

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMP_RATIO_VALUE;
    const int64_t *coffPtr = attr->GetAttrPointer<int64_t>(COFF_ATTR_INDEX);
    int64_t coff = (coffPtr != nullptr) ? *coffPtr : COFF_VALUE;

    if (hiddenStatesShape->GetDimNum() == DIM_NUM_3) {                // BS
        shapeParam.isBsMerge = false;
        shapeParam.B = hiddenStatesShape->GetDim(DIM_INDEX_0);
        shapeParam.S = hiddenStatesShape->GetDim(DIM_INDEX_1);
        shapeParam.Sr = (hiddenStatesShape->GetDim(DIM_INDEX_1) + cmpRatio - 1) / cmpRatio;
        shapeParam.H = hiddenStatesShape->GetDim(DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {                                                    // T
        shapeParam.isBsMerge = true;
        auto seqLensShape = context->GetOptionalInputShape(SEQ_LENS_INPUT_INDEX);
        shapeParam.Sr = std::min(hiddenStatesShape->GetDim(DIM_INDEX_0),
            hiddenStatesShape->GetDim(DIM_INDEX_0) / cmpRatio+ seqLensShape->GetDim(DIM_INDEX_0) - 1);
        shapeParam.T = hiddenStatesShape->GetDim(DIM_INDEX_0);
        shapeParam.H = hiddenStatesShape->GetDim(DIM_INDEX_1);
    }

    shapeParam.D = wkShape->GetDim(DIM_INDEX_0) / coff;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetKPoolShapeDim(const KPoolProtoShapeParam &shapeParam, gert::InferShapeContext* context)
{
    // query: (B, S, N, Hckv) | (T, N, Hckv)
    auto pooledKeyShape = context->GetOutputShape(POOLED_KEY_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pooledKeyShape);
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    // Set output shape
    if (!shapeParam.isBsMerge) {
        pooledKeyShape->SetDimNum(DIM_NUM_3);                   // (B, Sr, H)
        pooledKeyShape->SetDim(DIM_INDEX_0, shapeParam.B);
        pooledKeyShape->SetDim(DIM_INDEX_1, shapeParam.Sr);
        pooledKeyShape->SetDim(DIM_INDEX_2, shapeParam.D);
    } else {
        pooledKeyShape->SetDimNum(DIM_NUM_2);                   // (T, N, Hckv)
        pooledKeyShape->SetDim(DIM_INDEX_0, shapeParam.Sr);
        pooledKeyShape->SetDim(DIM_INDEX_1, shapeParam.D);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKPool(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter KPool inferDataType impl.");

    context->SetOutputDataType(POOLED_KEY_OUTPUT_INDEX, context->GetRequiredInputDataType(HIDDEN_STATES_INPUT_INDEX));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeKPool(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context->GetNodeName(), "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter KPool infershape impl.");

    KPoolProtoShapeParam shapeParam {};
    auto apiRet = GetKPoolShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS),  OP_LOGE(context->GetNodeName(), "Context get input shape failed"),
        return ge::GRAPH_FAILED);

    apiRet = SetKPoolShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"),
        return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KPool).InferShape(InferShapeKPool).InferDataType(InferDataTypeKPool);
}  // namespace ops