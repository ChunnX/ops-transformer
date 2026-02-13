/*!
 * \file quantize_add_layer_norm_infershape.cpp
 * \brief
 */
#include "common/log.h"
#include "register/op_impl_registry.h"

static constexpr int X1_IDX = 0;
static constexpr int X2_IDX = 1;
static constexpr int GAMMA_IDX = 2;
static constexpr int BETA_IDX = 3;
static constexpr int BIAS_IDX = 4;
static constexpr int SCALES_IDX = 5;
static constexpr int ZERO_POINTS_IDX = 6;

static constexpr int Y_IDX = 0;
static constexpr int X_IDX = 1;

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4QuantizeAddLayerNorm(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4QuantizeAddLayerNorm");

    // get input shapes
    const gert::Shape* x1Shape = context->GetInputShape(X1_IDX);
    const gert::Shape* x2Shape = context->GetInputShape(X2_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    const gert::Shape* gammaShape = context->GetInputShape(GAMMA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);
    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(Y_IDX);
    gert::Shape* xShape = context->GetOutputShape(X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    *yShape = *x1Shape;
    *xShape = *x1Shape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4QuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4QuantizeAddLayerNorm(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4QuantizeAddLayerNorm");
    context->SetOutputDataType(Y_IDX, DT_INT8);
    context->SetOutputDataType(X_IDX, context->GetInputDataType(X1_IDX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4QuantizeAddLayerNorm");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantizeAddLayerNorm)
    .InferShape(InferShape4QuantizeAddLayerNorm)
    .InferDataType(InferDataType4QuantizeAddLayerNorm);
} // namespace ops