/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 */
#include "mla_prolog_v2_infershape.h"

using namespace ge;

namespace ops {

ge::graphStatus SetMlaPrologV2ShapeDim(const MlaPrologProtoShapeParam &shapeParam, gert::InferShapeContext *context)
{
    auto apiRet = SetMlaPrologShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "SetMlaPrologShapeDim failed"), return ge::GRAPH_FAILED);

    auto dequantScaleXShape = context->GetOptionalInputShape(DEQUANT_SCALE_X_INDEX);
    auto quantScaleCkvShape = context->GetOptionalInputShape(QUANT_SCALE_CKV_INDEX);
    bool isQuantCkv = (dequantScaleXShape != nullptr && quantScaleCkvShape != nullptr);

    auto dequantScaleQNopeShape = context->GetOutputShape(DEQUANT_SCALE_Q_NOPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dequantScaleQNopeShape);

    if (!isQuantCkv) {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_1);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, DIM_NUM_1);
    } else {
        dequantScaleQNopeShape->SetDimNum(DIM_NUM_3);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_0, shapeParam.isBsMerge ? shapeParam.T : shapeParam.B * shapeParam.S);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_1, shapeParam.N);
        dequantScaleQNopeShape->SetDim(DIM_INDEX_2, 1);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeMlaPrologV2(gert::InferShapeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV2 infershape impl.");

    MlaPrologProtoShapeParam shapeParam {};
    auto apiRet = GetMlaPrologShapeDim(context, shapeParam);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"), return ge::GRAPH_FAILED);

    apiRet = SetMlaPrologV2ShapeDim(shapeParam, context);
    OP_CHECK_IF((apiRet != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context set output shape failed"), return ge::GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMlaPrologV2(gert::InferDataTypeContext *context)
{
    OP_LOGI(context->GetNodeName(), "Enter MlaPrologV2 infershape impl.");

    OP_CHECK_IF((InferDataTypeMlaProlog(context) != GRAPH_SUCCESS), OP_LOGE(context->GetNodeName(), "Context get input shape failed"),
        return ge::GRAPH_FAILED);

    bool isQuantCkv = (context->GetInputDataType(TOKEN_X_INDEX) == ge::DT_INT8 &&
        context->GetOptionalInputDataType(QUANT_SCALE_CKV_INDEX) != ge::DT_UNDEFINED);

    context->SetOutputDataType(QUERY_INDEX, (isQuantCkv) ? ge::DT_INT8 : context->GetInputDataType(WEIGHT_UK_INDEX));
    context->SetOutputDataType(DEQUANT_SCALE_Q_NOPE_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MlaPrologV2).InferShape(InferShapeMlaPrologV2).InferDataType(InferDataTypeMlaPrologV2);
} // namespace ops
