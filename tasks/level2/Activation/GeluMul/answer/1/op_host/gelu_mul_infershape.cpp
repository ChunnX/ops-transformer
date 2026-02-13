/*!
 * \file gelu_mul_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)         \
if ((ptr) == nullptr)                                    \
{                                                        \
    std::printf("nullptr error!");                       \
    return ge::GRAPH_SUCCESS;                            \
}                                                        \

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
}

namespace ops {

static constexpr int64_t IDX_0 = 0;

static constexpr size_t SIZE_2 = 2;

static ge::graphStatus InferShape4GeluMul(gert::InferShapeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferShape4GeluMul");

  // get input shapes
  auto xShape = context->GetInputShape(IDX_0);
  OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

  // get output shapes
  auto yShape = context->GetOutputShape(IDX_0);
  OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

  size_t xDimNum = xShape->GetDimNum();
  yShape->SetDimNum(xDimNum);

  *yShape = *xShape;
  yShape->SetDim(xDimNum - 1, xShape->GetDim(xDimNum - 1) / SIZE_2);

  OP_LOGD(context->GetNodeName(), "End to do InferShape4GeluMul");
  return GRAPH_SUCCESS;
}

static graphStatus InferDataType4GeluMul(gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4GeluMul");

  auto input_dtype = context->GetInputDataType(IDX_0);

  context->SetOutputDataType(IDX_0, input_dtype);

  OP_LOGD(context->GetNodeName(), "End to do InferDataType4GeluMul");

  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GeluMul).InferShape(InferShape4GeluMul).InferDataType(InferDataType4GeluMul);
}  // namespace ops