#include "register/op_impl_registry.h"
#include "log/log.h"
namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // M,K or K,M
    const gert::Shape* x_shape = context->GetInputShape(0);
    // K,N
    const gert::Shape* weight_shape = context->GetInputShape(1);
    const gert::Shape* group_list_shape = context->GetInputShape(2);
    // check dim
    if (x_shape->GetDimNum() != 2 || weight_shape->GetDimNum() != 2 || group_list_shape->GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "Dim of inputs not right");
        return ge::GRAPH_FAILED;
    }
    
    bool trans_a = *(context->GetAttrs()->GetAttrPointer<bool>(0));

    auto M = trans_a ? x_shape->GetDim(1) : x_shape->GetDim(0);
    auto N = weight_shape->GetDim(1);
    auto groupNum = group_list_shape->GetDim(0);

    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = gert::Shape({groupNum, M, N});
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GmmKDim).InferShape(InferShape);
}
