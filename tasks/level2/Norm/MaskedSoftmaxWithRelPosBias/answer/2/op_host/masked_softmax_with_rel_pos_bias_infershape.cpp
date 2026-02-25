#include "common/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShapeMaskedSoftmaxWithRelPosBias(gert::InferShapeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter MaskedSoftmaxWithRelPosBias infershape impl.");
  const gert::Shape* x_shape = context->GetInputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  gert::Shape* y_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  *y_shape = *x_shape;
  OP_LOGI(context->GetNodeName(), "MaskedSoftmaxWithRelPosBias infershape end.");
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMaskedSoftmaxWithRelPosBias(gert::InferDataTypeContext* context) {
  OP_LOGI(context->GetNodeName(), "Enter MaskedSoftmaxWithRelPosBias infershape impl.");
  auto dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, dtype);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaskedSoftmaxWithRelPosBias).InferShape(InferShapeMaskedSoftmaxWithRelPosBias).InferDataType(InferDataTypeMaskedSoftmaxWithRelPosBias);

}  // namespace ops