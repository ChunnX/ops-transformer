/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "err/ops_err.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

namespace ops {
constexpr int64_t ROW_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t INDEX_ATTR_DST_TYPE = 0;
constexpr int64_t INDEX_ATTR_QUANT_MODE = 3;

ge::graphStatus InferShapeForClaGateMxQuantWithDualAxis(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *rowData = context->GetOutputShape(0);
    gert::Shape *rowScale = context->GetOutputShape(1);
    gert::Shape *colData = context->GetOutputShape(2);
    gert::Shape *colScale = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowScale);
    OP_CHECK_NULL_WITH_CONTEXT(context, colData);
    OP_CHECK_NULL_WITH_CONTEXT(context, colScale);
    if (xShape->GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "global_out must be 3D [T,N,D].");
        return ge::GRAPH_FAILED;
    }
    int64_t t = xShape->GetDim(0);
    int64_t n = xShape->GetDim(1);
    int64_t d = xShape->GetDim(2);
    if (t <= 0 || n <= 0 || d <= 0) {
        OP_LOGE(context->GetNodeName(), "global_out shape must be positive.");
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t *quantModePtr = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_QUANT_MODE);
    int64_t quantMode = quantModePtr != nullptr ? *quantModePtr : 1;
    int64_t k = n * d;
    *rowData = gert::Shape({t, k});
    *rowScale = gert::Shape({t, Ops::Base::CeilDiv(k, ROW_BLOCK), SCALE_LAST_DIM});
    if (quantMode == 0) {
        *colData = gert::Shape({0});
        *colScale = gert::Shape({0});
    } else {
        *colData = gert::Shape({t, k});
        *colScale = gert::Shape({Ops::Base::CeilDiv(t, ROW_BLOCK), k, SCALE_LAST_DIM});
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForClaGateMxQuantWithDualAxis(gert::InferDataTypeContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t *dstType = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_DST_TYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstType);
    ge::DataType outDtype = static_cast<ge::DataType>(*dstType);
    if (outDtype != ge::DT_FLOAT4_E2M1 && outDtype != ge::DT_FLOAT4_E1M2 && outDtype != ge::DT_FLOAT8_E4M3FN &&
        outDtype != ge::DT_FLOAT8_E5M2) {
        OP_LOGE(context->GetNodeName(), "dst_type must be one of float4_e2m1/e1m2/float8_e4m3fn/e5m2.");
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, outDtype);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(2, outDtype);
    context->SetOutputDataType(3, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ClaGateMxQuantWithDualAxis)
    .InferShape(InferShapeForClaGateMxQuantWithDualAxis)
    .InferDataType(InferDataTypeForClaGateMxQuantWithDualAxis);
} // namespace ops
