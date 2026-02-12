#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static constexpr int64_t IDX_0 = 0;

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }

static ge::graphStatus InferShape4Logit(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4Logit");

    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    auto yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    size_t xDimNum = xShape->GetDimNum();
    yShape->SetDimNum(xDimNum);

    *yShape = *xShape;

    OP_LOGD(context, "End to do InferShape4Logit");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4Logit(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4Logit");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);

    OP_LOGD(context, "End to do InferDataType4Logit");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Logit).InferShape(InferShape4Logit).InferDataType(InferDataType4Logit);
} // namespace ops