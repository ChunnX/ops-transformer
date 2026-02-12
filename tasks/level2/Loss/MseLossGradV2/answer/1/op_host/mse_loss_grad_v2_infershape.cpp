#include "register/op_impl_registry.h"
#include "util/shape_util.h"
using namespace ge;
namespace {
constexpr uint32_t INPUT_PREDICT_IDX = 0;
constexpr uint32_t INPUT_LABEL_IDX = 1;
constexpr uint32_t INPUT_DOUT_IDX = 2;
constexpr uint32_t OUTPUT_LOSS_GRAD_IDX = 0;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2;
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
} // namespace

namespace ops {
static ge::graphStatus InferShapeForMseLossGrad(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "MseLossGrad Begin.");
    const gert::Shape* predictShape = context->GetInputShape(INPUT_PREDICT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictShape);
    const gert::Shape* labelShape = context->GetInputShape(INPUT_LABEL_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelShape);
    const gert::Shape* doutShape = context->GetInputShape(INPUT_DOUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, doutShape);

    gert::Shape* lossGradShape = context->GetOutputShape(OUTPUT_LOSS_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossGradShape);
    if (Ops::Base::IsUnknownRank(*predictShape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2,)");
        lossGradShape->SetDim(DIM_0, UNKNOWN_RANK_DIM_VALUE_);
        return ge::GRAPH_FAILED;
    }
    *lossGradShape = *predictShape;
    OP_LOGD(context->GetNodeName(), "InferShapeForMseLossGrad End.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForMseLossGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType predictDtype = context->GetInputDataType(INPUT_PREDICT_IDX);
    context->SetOutputDataType(OUTPUT_LOSS_GRAD_IDX, predictDtype);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MseLossGradV2).InferShape(InferShapeForMseLossGrad).InferDataType(InferDataTypeForMseLossGrad);
} // namespace ops