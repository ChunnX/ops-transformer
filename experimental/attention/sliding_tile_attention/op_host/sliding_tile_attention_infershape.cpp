#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeForSlidingTileAttention(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SlidingTileAttentionFai).InferShape(InferShapeForSlidingTileAttention);

} // namespace ops
