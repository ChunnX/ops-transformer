/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
#include "error_util.h"
#include "register/op_impl_registry.h"
namespace ops {
constexpr int64_t CLA_M = 256 * 4;             // S * B
constexpr int64_t CLA_K = 96 * 256;          // N * D
ge::graphStatus InferShapeForClaGateMxQuantWithDualAxis(gert::InferShapeContext* context)
{
    gert::Shape* rowData = context->GetOutputShape(0);
    gert::Shape* rowScale = context->GetOutputShape(1);
    gert::Shape* colData = context->GetOutputShape(2);
    gert::Shape* colScale = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowScale);
    OP_CHECK_NULL_WITH_CONTEXT(context, colData);
    OP_CHECK_NULL_WITH_CONTEXT(context, colScale);
    *rowData = gert::Shape({CLA_M, CLA_K});
    *colData = gert::Shape({CLA_M, CLA_K});
    *rowScale = gert::Shape({CLA_M, CLA_K / 64, 2});
    *colScale = gert::Shape({CLA_M / 64, CLA_K, 2});
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferDataTypeForClaGateMxQuantWithDualAxis(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(1, ge::DT_FLOAT8_E8M0);
    context->SetOutputDataType(2, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(3, ge::DT_FLOAT8_E8M0);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(ClaGateMxQuantWithDualAxis)
    .InferShape(InferShapeForClaGateMxQuantWithDualAxis)
    .InferDataType(InferDataTypeForClaGateMxQuantWithDualAxis);
} // namespace ops
