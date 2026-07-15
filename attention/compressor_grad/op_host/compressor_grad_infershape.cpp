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
    constexpr uint32_t CMPG_TOKEN_X_INPUT_INDEX = 0;
    constexpr uint32_t CMPG_WEIGHT_KV_INPUT_INDEX = 1;
    constexpr uint32_t CMPG_WEIGHT_WGATE_INPUT_INDEX = 2;
    constexpr uint32_t CMPG_D_CPM_KV_INPUT_INDEX = 3;
    constexpr uint32_t CMPG_SOFTMAX_SCORE_INPUT_INDEX = 4;
    constexpr uint32_t CMPG_KV_INPUT_INDEX = 5;

    // INPUT(OPTION)
    constexpr uint32_t CMPG_CU_SEQ_LEN_INPUT_INDEX = 6;
    constexpr uint32_t CMPG_SEQ_USED_INPUT_INDEX = 7;
    constexpr uint32_t CMPG_START_POS_INPUT_INDEX = 8;

    // ATTR
    constexpr uint32_t CMPG_CMP_RATIO_ATTR_INDEX = 0;
    constexpr uint32_t CMPG_COFF_ATTR_INDEX = 1;

    // OUTPUT
    constexpr uint32_t CMPG_D_X_OUTPUT_INDEX = 0;
    constexpr uint32_t CMPG_D_WKV_OUTPUT_INDEX = 1;
    constexpr uint32_t CMPG_D_WGATE_OUTPUT_INDEX = 2;
    constexpr uint32_t CMPG_D_APE_OUTPUT_INDEX = 3;

    // ATTR DEFAULT VALUE
    constexpr uint32_t CMPG_CMP_RATIO_VALUE = 4;
    constexpr uint32_t CMPG_COFF_VALUE = 1;

    // tmp
    constexpr uint32_t CMPG_DIM_NUM_1 = 1;
    constexpr uint32_t CMPG_DIM_NUM_2 = 2;
    constexpr uint32_t CMPG_DIM_NUM_3 = 3;
    constexpr uint32_t CMPG_DIM_NUM_4 = 4;
    constexpr uint32_t CMPG_DIM_INDEX_0 = 0;
    constexpr uint32_t CMPG_DIM_INDEX_1 = 1;
    constexpr uint32_t CMPG_DIM_INDEX_2 = 2;

struct CompressorGradProtoShapeParam {
    bool isBsMerge { false };
    int64_t B { 0 };
    int64_t T { 0 };
    int64_t S { 0 };
    int64_t H { 0 };
    int64_t D { 0 };
    int64_t coff { 1 };
};

ge::graphStatus GetCompressorGradShapeDim(const gert::InferShapeContext* context,
                                          CompressorGradProtoShapeParam &shapeParam)
{
    auto xShape = context->GetRequiredInputShape(CMPG_TOKEN_X_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto wkvShape = context->GetRequiredInputShape(CMPG_WEIGHT_KV_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wkvShape);
    auto wgateShape = context->GetRequiredInputShape(CMPG_WEIGHT_WGATE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wgateShape);
    auto dcpmKvShape = context->GetRequiredInputShape(CMPG_D_CPM_KV_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dcpmKvShape);
    auto kvShape = context->GetRequiredInputShape(CMPG_KV_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    auto softmaxScoreShape = context->GetRequiredInputShape(CMPG_SOFTMAX_SCORE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, softmaxScoreShape);

    if (xShape->GetDimNum() == CMPG_DIM_NUM_2) {
        auto cuSeqlensShape = context->GetOptionalInputShape(CMPG_CU_SEQ_LEN_INPUT_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqlensShape);
    }

    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMPG_CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMPG_CMP_RATIO_VALUE;
    const int64_t *coffPtr = attr->GetAttrPointer<int64_t>(CMPG_COFF_ATTR_INDEX);
    int64_t coff = (coffPtr != nullptr) ? *coffPtr : CMPG_COFF_VALUE;

    shapeParam.coff = coff;

    if (xShape->GetDimNum() == CMPG_DIM_NUM_3) {
        shapeParam.isBsMerge = false;
        shapeParam.B = xShape->GetDim(CMPG_DIM_INDEX_0);
        shapeParam.S = xShape->GetDim(CMPG_DIM_INDEX_1);
        shapeParam.H = xShape->GetDim(CMPG_DIM_INDEX_2);
        shapeParam.T = shapeParam.B * shapeParam.S;
    } else {
        shapeParam.isBsMerge = true;
        shapeParam.T = xShape->GetDim(CMPG_DIM_INDEX_0);
        shapeParam.H = xShape->GetDim(CMPG_DIM_INDEX_1);
        auto cuSeqlensShape = context->GetOptionalInputShape(CMPG_CU_SEQ_LEN_INPUT_INDEX);
        if (cuSeqlensShape != nullptr) {
            shapeParam.B = cuSeqlensShape->GetDim(CMPG_DIM_INDEX_0) - 1;
        }
    }

    shapeParam.D = wkvShape->GetDim(CMPG_DIM_INDEX_0) / coff;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetCompressorGradShapeDim(const CompressorGradProtoShapeParam &shapeParam,
                                          gert::InferShapeContext* context)
{
    auto xShape = context->GetRequiredInputShape(CMPG_TOKEN_X_INPUT_INDEX);
    auto wkvShape = context->GetRequiredInputShape(CMPG_WEIGHT_KV_INPUT_INDEX);

    // d_x: same shape and dtype as x
    auto dxShape = context->GetOutputShape(CMPG_D_X_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);
    dxShape->SetDimNum(xShape->GetDimNum());
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        dxShape->SetDim(i, xShape->GetDim(i));
    }

    // d_wkv: same shape as wkv
    auto dwkvShape = context->GetOutputShape(CMPG_D_WKV_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwkvShape);
    dwkvShape->SetDimNum(wkvShape->GetDimNum());
    for (size_t i = 0; i < wkvShape->GetDimNum(); ++i) {
        dwkvShape->SetDim(i, wkvShape->GetDim(i));
    }

    // d_wgate: same shape as wgate
    auto wgateShape = context->GetRequiredInputShape(CMPG_WEIGHT_WGATE_INPUT_INDEX);
    auto dwgateShape = context->GetOutputShape(CMPG_D_WGATE_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwgateShape);
    dwgateShape->SetDimNum(wgateShape->GetDimNum());
    for (size_t i = 0; i < wgateShape->GetDimNum(); ++i) {
        dwgateShape->SetDim(i, wgateShape->GetDim(i));
    }

    // d_ape: (cmp_ratio, coff * head_dim)
    auto attr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attr);
    const int64_t *cmpRatioPtr = attr->GetAttrPointer<int64_t>(CMPG_CMP_RATIO_ATTR_INDEX);
    int64_t cmpRatio = (cmpRatioPtr != nullptr) ? *cmpRatioPtr : CMPG_CMP_RATIO_VALUE;
    auto dapeShape = context->GetOutputShape(CMPG_D_APE_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dapeShape);
    dapeShape->SetDimNum(CMPG_DIM_NUM_2);
    dapeShape->SetDim(CMPG_DIM_INDEX_0, cmpRatio);
    dapeShape->SetDim(CMPG_DIM_INDEX_1, wkvShape->GetDim(CMPG_DIM_INDEX_0));

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeCompressorGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("CompressorGrad", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter CompressorGrad inferDataType impl.");

    // d_x, d_wkv, d_wgate: same dtype as x
    auto inputDtype = context->GetRequiredInputDataType(CMPG_TOKEN_X_INPUT_INDEX);
    context->SetOutputDataType(CMPG_D_X_OUTPUT_INDEX, inputDtype);
    context->SetOutputDataType(CMPG_D_WKV_OUTPUT_INDEX, inputDtype);
    context->SetOutputDataType(CMPG_D_WGATE_OUTPUT_INDEX, inputDtype);
    // d_ape: float32
    context->SetOutputDataType(CMPG_D_APE_OUTPUT_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeCompressorGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("CompressorGrad", "Context is nullptr."),
               return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter CompressorGrad infershape impl.");

    CompressorGradProtoShapeParam shapeParam {};
    auto apiRet = GetCompressorGradShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), "Context get input shape failed", return ge::GRAPH_FAILED);

    apiRet = SetCompressorGradShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), "Context set output shape failed", return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CompressorGrad).InferShape(InferShapeCompressorGrad).InferDataType(InferDataTypeCompressorGrad);
}  // namespace ops
