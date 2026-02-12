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
#define unlikely(x) __builtin_expect((x), 0)

static ge::graphStatus InferShape4LogitGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4LogitGrad");

    auto xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    auto dxShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, dxShape);

    size_t xDimNum = xShape->GetDimNum();
    dxShape->SetDimNum(xDimNum);

    *dxShape = *xShape;

    OP_LOGD(context, "End to do InferShape4LogitGrad");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4LogitGrad(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4LogitGrad");

    auto input_dtype = context->GetInputDataType(IDX_0);

    context->SetOutputDataType(IDX_0, input_dtype);

    OP_LOGD(context, "End to do InferDataType4LogitGrad");

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LogitGrad).InferShape(InferShape4LogitGrad).InferDataType(InferDataType4LogitGrad);
} // namespace ops