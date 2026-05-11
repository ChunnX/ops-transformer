/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_sinkhorn_infershape.cpp
 * \brief mhc_sinkhorn_infershape
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"
#include "util/shape_util.h"

using namespace ge;
using namespace std;

namespace ops {
static constexpr size_t X_INDEX = 0;
static constexpr size_t Y_INDEX = 0;
static constexpr size_t NORM_INDEX = 1;
static constexpr size_t SUM_INDEX = 2;
static constexpr size_t DIM_TWO = 2;
static constexpr size_t DIM_THREE = 3;
static constexpr int64_t N_ALIGN = 8;
static constexpr int64_t DOUBLE_SIZE = 2;

static constexpr size_t INDEX_EPS = 0;
static constexpr size_t INDEX_NUM_ITERS = 1;
static constexpr size_t INDEX_OUT_FLAG = 2;

static constexpr size_t BSNN_DIMS = 4;
static constexpr size_t TNN_DIMS = 3;
static constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

static ge::graphStatus InferShape4MhcSinkhorn(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do MhcSinkhornInfershape.");

    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape* normShape = context->GetOutputShape(NORM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, normShape);

    gert::Shape* sumShape = context->GetOutputShape(SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);

    auto attrPtr = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrPtr);

    auto epsPtr = attrPtr->GetAttrPointer<float>(INDEX_EPS);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsPtr);

    auto numItersPtr = attrPtr->GetAttrPointer<int64_t>(INDEX_NUM_ITERS);
    OP_CHECK_NULL_WITH_CONTEXT(context, numItersPtr);

    auto outFlagPtr = attrPtr->GetAttrPointer<int64_t>(INDEX_OUT_FLAG);
    OP_CHECK_NULL_WITH_CONTEXT(context, outFlagPtr);

    int64_t numIters = *numItersPtr;
    int64_t outFlag = *outFlagPtr;

    OP_CHECK_IF((numIters <= 0),
        OP_LOGE(context->GetNodeName(), "num_iters should be greater than 0, but got %ld.", numIters),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((outFlag != 0) && (outFlag != 1),
        OP_LOGE(context->GetNodeName(), "out_flag should be 0 or 1, but got %ld.", outFlag),
        return ge::GRAPH_FAILED);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);

        normShape->SetDimNum(1);
        sumShape->SetDimNum(1);
        if (outFlag == 1) {
            normShape->SetDim(0, UNKNOWN_DIM_VALUE);
            sumShape->SetDim(0, UNKNOWN_DIM_VALUE);
        } else {
            normShape->SetDim(0, 1);
            sumShape->SetDim(0, 1);
        }

        OP_LOGD(context->GetNodeName(), "MhcSinkhorn infershape handles unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    size_t xDims = xShape->GetDimNum();
    OP_CHECK_IF((xDims != TNN_DIMS) && (xDims != BSNN_DIMS),
        OP_LOGE(context->GetNodeName(), "The dim of x should be 3 or 4, but got %lu", xDims),
        return ge::GRAPH_FAILED);

    yShape->SetDimNum(xDims);
    for (size_t i = 0; i < xDims; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    int64_t T = xShape->GetDim(0);
    int64_t n0 = xShape->GetDim(1);
    int64_t n1 = xShape->GetDim(DIM_TWO);

    if (xDims == BSNN_DIMS) {
        int64_t B = xShape->GetDim(0);
        int64_t S = xShape->GetDim(1);
        T = ((B < 0) || (S < 0)) ? UNKNOWN_DIM_VALUE : (B * S);
        n0 = xShape->GetDim(DIM_TWO);
        n1 = xShape->GetDim(DIM_THREE);
    }

    OP_CHECK_IF((n0 >= 0) && (n1 >= 0) && (n0 != n1),
        OP_LOGE(context->GetNodeName(), "input n0 is %ld, must be equal to n1 which is %ld.", n0, n1),
        return ge::GRAPH_FAILED);

    int64_t n = n0;
    if ((n < 0) && (n1 >= 0)) {
        n = n1;
    }

    normShape->SetDimNum(1);
    sumShape->SetDimNum(1);

    if (outFlag == 1) {
        if ((T < 0) || (n < 0)) {
            normShape->SetDim(0, UNKNOWN_DIM_VALUE);
            sumShape->SetDim(0, UNKNOWN_DIM_VALUE);
        } else {
            int64_t normSize = DOUBLE_SIZE * numIters * T * n * N_ALIGN;
            int64_t sumSize = DOUBLE_SIZE * numIters * T * N_ALIGN;
            normShape->SetDim(0, normSize);
            sumShape->SetDim(0, sumSize);
        }
    } else {
        normShape->SetDim(0, 1);
        sumShape->SetDim(0, 1);
    }

    OP_LOGD(context, "End to do MhcSinkhornInfershape.");
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDtype4MhcSinkhorn(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "MhcSinkhornInferDtype enter");

    const ge::DataType x = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, x);
    context->SetOutputDataType(NORM_INDEX, x);
    context->SetOutputDataType(SUM_INDEX, x);

    OP_LOGD(context->GetNodeName(), "MhcSinkhornInferDtype end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MhcSinkhorn)
    .InferShape(InferShape4MhcSinkhorn)
    .InferDataType(InferDtype4MhcSinkhorn);
} // namespace opsc