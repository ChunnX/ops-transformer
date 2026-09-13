/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Shape/type inference for the Ascend 950 MXFP8 weight-NZ V3 path.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t X_INDEX = 0;
constexpr int64_t WEIGHT_SCALE_INDEX = 4;
constexpr int64_t QUANT_MODE_INDEX = 2;
constexpr int64_t QUANT_DTYPE_INDEX = 3;
constexpr int64_t TRANSPOSE_WEIGHT_INDEX = 4;
constexpr int64_t MX_QUANT_MODE = 2;
constexpr int64_t MX_BLOCK_SIZE = 64;
constexpr int64_t SWIGLU_SPLIT = 2;

ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto *xShape = context->GetInputShape(X_INDEX);
    const auto *weightScaleShape = context->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    auto *outputShape = context->GetOutputShape(0);
    auto *outputScaleShape = context->GetOutputShape(1);
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightScaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputScaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *quantMode = attrs->GetInt(QUANT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    OP_CHECK_IF(*quantMode != MX_QUANT_MODE,
                OP_LOGE(context->GetNodeName(), "V3 only supports quant_mode=2 for MXFP8 weight-NZ."),
                return GRAPH_FAILED);

    const int64_t *quantDtype = attrs->GetInt(QUANT_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantDtype);

    const int64_t m = xShape->GetDim(0);
    const bool *transposeWeight = attrs->GetBool(TRANSPOSE_WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, transposeWeight);
    const int64_t n = *transposeWeight ? weightScaleShape->GetDim(1)
                                       : weightScaleShape->GetDim(weightScaleShape->GetDimNum() - 2);
    const int64_t outputN = n < 0 ? n : n / SWIGLU_SPLIT;
    const int64_t outputScaleN = outputN < 0 ? outputN : Ops::Base::CeilDiv(outputN, MX_BLOCK_SIZE);

    outputShape->SetDimNum(2);
    outputShape->SetDim(0, m);
    outputShape->SetDim(1, outputN);
    outputScaleShape->SetDimNum(3);
    outputScaleShape->SetDim(0, m);
    outputScaleShape->SetDim(1, outputScaleN);
    outputScaleShape->SetDim(2, SWIGLU_SPLIT);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *quantMode = attrs->GetInt(QUANT_MODE_INDEX);
    const int64_t *quantDtype = attrs->GetInt(QUANT_DTYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    OP_CHECK_NULL_WITH_CONTEXT(context, quantDtype);
    OP_CHECK_IF(*quantMode != MX_QUANT_MODE,
                OP_LOGE(context->GetNodeName(), "V3 only supports quant_mode=2 for MXFP8 weight-NZ."),
                return GRAPH_FAILED);
    context->SetOutputDataType(0, static_cast<ge::DataType>(*quantDtype));
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    return GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(GroupedMatmulSwigluQuantV3)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
} // namespace ops
