#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t INDEX_GRAD_Y = 0;
static constexpr size_t INDEX_GROUP_IDX = 1;

static constexpr size_t GRAD_Y_SHAPE_WITH_GROUP_IDX = 2;
static constexpr size_t GRAD_Y_SHAPE_NO_GROUP_IDX = 3;

static constexpr size_t GROUP_INDEX_SHAPE = 1;
static constexpr size_t GRAD_BIAS_SHAPE_SIZE = 2;

static constexpr size_t OTHER_SHAPE = -1;

static constexpr int64_t MAX_GROUP_NUM = 2048;

#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                       \
        std::printf("nullptr error!");            \
        return ge::GRAPH_FAILED;                  \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do {                                         \
        if (unlikely(condition)) {               \
            log;                                 \
            return_expr;                         \
        }                                        \
    } while (0)

static ge::graphStatus GroupedBiasAddGradInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do GroupedBiasAddGradInferShape.");

    const gert::Shape* gradYShape = context->GetInputShape(INDEX_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradYShape);

    auto gradYDimNum = gradYShape->GetDimNum();
    auto groupNum = gradYShape->GetDim(0);
    int64_t hNum = 0;

    if (gradYDimNum == 1 && groupNum == ge::UNKNOWN_DIM_NUM) {
        gert::Shape* gradBiasShape = context->GetOutputShape(0);
        gradBiasShape->SetDimNum(GRAD_BIAS_SHAPE_SIZE);
        gradBiasShape->SetDim(0, OTHER_SHAPE);
        gradBiasShape->SetDim(1, OTHER_SHAPE);
        return ge::GRAPH_SUCCESS;
    }

    const gert::Shape* groupIdxShape = context->GetOptionalInputShape(INDEX_GROUP_IDX);
    if (groupIdxShape == nullptr) {
        OP_CHECK_IF(
            gradYDimNum != GRAD_Y_SHAPE_NO_GROUP_IDX,
            OP_LOGE(
                context->GetNodeName(), "%s the grad_y of input should be 3D tensor when group_idx is null.",
                context->GetNodeName()),
            return ge::GRAPH_FAILED);
        hNum = gradYShape->GetDim(GRAD_Y_SHAPE_NO_GROUP_IDX - 1);
    } 

    gert::Shape* gradBiasShape = context->GetOutputShape(0);
    gradBiasShape->SetDimNum(GRAD_BIAS_SHAPE_SIZE);
    gradBiasShape->SetDim(0, groupNum);
    gradBiasShape->SetDim(1, hNum);

    OP_LOGD(context->GetNodeName(), "End to do GroupedBiasAddGradInferShape.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedBiasAddGrad).InferShape(GroupedBiasAddGradInferShape);
} // namespace ops