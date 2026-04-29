/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include <set>
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INPUT_Q = 0U;
constexpr size_t INPUT_K = 1U;
constexpr size_t INPUT_V = 2U;
constexpr size_t OUTPUT_Y = 0U;
constexpr size_t BNSD_RANK = 4U;

static ge::graphStatus CheckSameShape(gert::InferShapeContext *context, const gert::Shape *lhs,
                                      const gert::Shape *rhs, const char *rhsName)
{
    if (lhs->GetDimNum() != rhs->GetDimNum()) {
        OP_LOGE(context, "q and %s rank mismatch, q rank %zu, %s rank %zu.", rhsName, lhs->GetDimNum(), rhsName,
                rhs->GetDimNum());
        return GRAPH_FAILED;
    }
    for (size_t i = 0U; i < lhs->GetDimNum(); ++i) {
        const int64_t lDim = lhs->GetDim(i);
        const int64_t rDim = rhs->GetDim(i);
        if (lDim > 0 && rDim > 0 && lDim != rDim) {
            OP_LOGE(context, "q and %s shape mismatch at dim %zu, q=%ld, %s=%ld.", rhsName, i, lDim, rhsName, rDim);
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus InferShapeSlidingTileAttention(gert::InferShapeContext *context)
{
    const gert::Shape *qShape = context->GetInputShape(INPUT_Q);
    const gert::Shape *kShape = context->GetInputShape(INPUT_K);
    const gert::Shape *vShape = context->GetInputShape(INPUT_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, vShape);

    if (qShape->GetDimNum() != BNSD_RANK) {
        OP_LOGE(context, "SlidingTileAttention only supports BNSD rank-4 input, got rank %zu.", qShape->GetDimNum());
        return GRAPH_FAILED;
    }
    if (CheckSameShape(context, qShape, kShape, "k") != GRAPH_SUCCESS ||
        CheckSameShape(context, qShape, vShape, "v") != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    gert::Shape *yShape = context->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(qShape->GetDimNum());
    for (size_t i = 0U; i < qShape->GetDimNum(); ++i) {
        yShape->SetDim(i, qShape->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeSlidingTileAttention(gert::InferDataTypeContext *context)
{
    const ge::DataType qDtype = context->GetInputDataType(INPUT_Q);
    const ge::DataType kDtype = context->GetInputDataType(INPUT_K);
    const ge::DataType vDtype = context->GetInputDataType(INPUT_V);
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_BF16};
    if (supportedDtypes.count(qDtype) == 0 || qDtype != kDtype || qDtype != vDtype) {
        OP_LOGE(context, "q/k/v should have the same dtype and only support float16 or bfloat16.");
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(OUTPUT_Y, qDtype);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SlidingTileAttention)
    .InferShape(InferShapeSlidingTileAttention)
    .InferDataType(InferDtypeSlidingTileAttention);
} // namespace ops
