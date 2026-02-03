#include "common/log.h"
#include "register/op_impl_registry.h"
#include "common_dtype.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ForeachCommon(gert::InferShapeContext* context)
{
    uint32_t outputNumInferShape = context->GetComputeNodeOutputNum();
    const auto inputInfoInferShape = context->GetIrInputInstanceInfo(0);
    if (inputInfoInferShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input0 ", inputInfoInferShape->GetInstanceNum(), "not equal num of dynamic output0 ",
        outputNumInferShape);
    OP_CHECK_IF(
        inputInfoInferShape->GetInstanceNum() != outputNumInferShape,
        OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()), return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfoInferShape->GetInstanceNum(); i++) {
        auto xShape = context->GetDynamicInputShape(0, i);
        auto yShape = context->GetOutputShape(i);
        if ((xShape == nullptr) || (yShape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        *yShape = *xShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ForeachCommon(gert::InferDataTypeContext* context)
{
    uint32_t outputNumDataType = context->GetComputeNodeOutputNum();
    const auto inputInfoDataType = context->GetIrInputInstanceInfo(0);
    if (inputInfoDataType == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input0 ", inputInfoDataType->GetInstanceNum(), "not equal num of dynamic output0 ",
        outputNumDataType);
    OP_CHECK_IF(
        inputInfoDataType->GetInstanceNum() != outputNumDataType, OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()),
        return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfoDataType->GetInstanceNum(); i++) {
        auto xDtype = context->GetDynamicInputDataType(0, i);

        context->SetOutputDataType(i, xDtype);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ForeachLerpList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);
} // namespace ops