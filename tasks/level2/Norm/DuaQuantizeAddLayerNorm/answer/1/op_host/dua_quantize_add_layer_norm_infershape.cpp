/*!
 * \file dua_quantize_add_layer_norm_infershape.cpp
 * \brief
 */
#include "common/log.h"
#include "register/op_impl_registry.h"

static constexpr int IDX_X1 = 0;
static constexpr int IDX_X2 = 1;
static constexpr int IDX_GAMMA = 2;
static constexpr int IDX_BETA = 3;
static constexpr int IDX_BIAS = 4;
static constexpr int IDX_Y1 = 0;
static constexpr int IDX_Y2 = 1;
static constexpr int IDX_X = 2;

using namespace ge;

namespace ops {
static ge::graphStatus InferShape4DuaQuantizeAddLayerNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4DuaQuantizeAddLayerNorm");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(IDX_X1);
    const gert::Shape* x2Shape = context->GetInputShape(IDX_X2);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape* gammaShape = context->GetInputShape(IDX_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shape
    gert::Shape* y1Shape = context->GetOutputShape(IDX_Y1);
    gert::Shape* y2Shape = context->GetOutputShape(IDX_Y2);
    gert::Shape* xShape = context->GetOutputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, y1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y2Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *y1Shape = *x1Shape;
    *y2Shape = *x2Shape;
    *xShape = *x1Shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4DuaQuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4DuaQuantizeAddLayerNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4DuaQuantizeAddLayerNorm");
    context->SetOutputDataType(IDX_Y1, DT_INT8);
    context->SetOutputDataType(IDX_Y2, DT_INT8);
    context->SetOutputDataType(IDX_X, context->GetInputDataType(IDX_X1));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4DuaQuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DuaQuantizeAddLayerNorm)
    .InferShape(InferShape4DuaQuantizeAddLayerNorm)
    .InferDataType(InferDataType4DuaQuantizeAddLayerNorm);

} // namespace ops