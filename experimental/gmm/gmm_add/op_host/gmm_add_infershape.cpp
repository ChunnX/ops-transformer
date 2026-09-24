#include "register/op_impl_registry.h"
#include "log/log.h"
namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* weightGradShape = context->GetInputShape(3);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *weightGradShape;
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GmmAdd).InferShape(InferShape);
}
