/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "grouped_rotate_quant_context.h"
#include "log/log.h"

namespace ops {
namespace {
using namespace optiling::grq;

ge::graphStatus InferShapeGroupedRotateQuant(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    InputParams params;
    ge::DataType dstType;
    std::string error;
    if (!ReadInputs(context, params, dstType, error) || !ValidateInputs(params, true, error)) {
        OP_LOGE(context, "%s", error.c_str());
        return ge::GRAPH_FAILED;
    }
    const auto yShapes = InferY(params);
    for (size_t irIndex = Y; irIndex <= SCALE; ++irIndex) {
        const auto *info = context->GetIrOutputInstanceInfo(irIndex);
        for (size_t i = 0; i < yShapes.size(); ++i) {
            auto *out = context->GetOutputShape(info->GetInstanceStart() + i);
            if (out == nullptr) {
                OP_LOGE(context, "missing output shape at list %zu, element %zu", irIndex, i);
                return ge::GRAPH_FAILED;
            }
            const auto shape = irIndex == Y ? yShapes[i] : ScaleShape(yShapes[i]);
            out->SetDimNum(shape.size());
            for (size_t dim = 0; dim < shape.size(); ++dim) {
                out->SetDim(dim, shape[dim]);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeGroupedRotateQuant(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    Attributes attrs;
    ge::DataType dstType;
    std::string error;
    if (!ReadAttributes(context, attrs, dstType, error)) {
        OP_LOGE(context, "%s", error.c_str());
        return ge::GRAPH_FAILED;
    }
    const auto *xInfo = context->GetIrInputInstanceInfo(X);
    if (xInfo == nullptr || xInfo->GetInstanceNum() == 0) {
        return ge::GRAPH_FAILED;
    }
    const auto inputType = context->GetInputDataType(xInfo->GetInstanceStart());
    if (inputType != ge::DT_FLOAT16 && inputType != ge::DT_BF16) {
        OP_LOGE(context, "x must be FLOAT16 or BFLOAT16");
        return ge::GRAPH_FAILED;
    }
    for (size_t irIndex = X; irIndex <= GROUP_LIST; ++irIndex) {
        const auto *info = context->GetIrInputInstanceInfo(irIndex);
        const auto count = info == nullptr ? 0 : info->GetInstanceNum();
        if ((irIndex <= ROTATION && (count == 0 || count > GroupedRotateQuant::MAX_GROUP_NUM)) ||
            (irIndex >= ALPHA && count > 1)) {
            return ge::GRAPH_FAILED;
        }
        const auto expected = irIndex == ALPHA ? ge::DT_BF16 : (irIndex == GROUP_LIST ? ge::DT_INT64 : inputType);
        for (size_t i = 0; i < count; ++i) {
            const auto actual = context->GetInputDataType(info->GetInstanceStart() + i);
            if (irIndex >= ALPHA && actual == ge::DT_UNDEFINED) {
                continue;
            }
            if (actual != expected) {
                OP_LOGE(context, "inconsistent input dtype at input %zu, element %zu", irIndex, i);
                return ge::GRAPH_FAILED;
            }
        }
    }
    const auto *yInfo = context->GetIrOutputInstanceInfo(Y);
    const auto *scaleInfo = context->GetIrOutputInstanceInfo(SCALE);
    if (yInfo == nullptr || scaleInfo == nullptr || yInfo->GetInstanceNum() == 0 ||
        yInfo->GetInstanceNum() > GroupedRotateQuant::MAX_GROUP_NUM ||
        yInfo->GetInstanceNum() != scaleInfo->GetInstanceNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < yInfo->GetInstanceNum(); ++i) {
        if (context->SetOutputDataType(yInfo->GetInstanceStart() + i, dstType) != ge::GRAPH_SUCCESS ||
            context->SetOutputDataType(scaleInfo->GetInstanceStart() + i, ge::DT_FLOAT8_E8M0) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_INFERSHAPE(GroupedRotateQuant)
    .InferShape(InferShapeGroupedRotateQuant)
    .InferDataType(InferDataTypeGroupedRotateQuant);
} // namespace ops
