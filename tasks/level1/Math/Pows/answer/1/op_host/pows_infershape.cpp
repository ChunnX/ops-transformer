/*!
 * \file pows_infershape.cpp
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

static ge::graphStatus InferShape4Pows(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4Pows");

    // get input shapes
    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    // get output shapes
    auto yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    size_t xDimNum = xShape->GetDimNum();
    yShape->SetDimNum(xDimNum);

    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShape4Pows");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4Pows(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4Pows");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);

    OP_LOGD(context, "End to do InferDataType4Pows");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Pows).InferShape(InferShape4Pows).InferDataType(InferDataType4Pows);
} // namespace ops