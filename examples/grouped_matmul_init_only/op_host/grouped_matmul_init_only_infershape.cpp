/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_init_only_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_X = 0;
static constexpr int64_t IDX_WEIGHT = 1;
static constexpr int64_t IDX_Y = 0;
static constexpr size_t DIM0 = 0;

static ge::graphStatus InferShapeGroupedMatmulInitOnly(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeGroupedMatmulInitOnly");

    // x: [M, K]
    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    // weight: [G, K, N]（末维为 N）
    const gert::Shape* weightShape = context->GetInputShape(IDX_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    // y.shape = [M, N]：M 取自 x 第 0 维，N 取自 weight 末维
    int64_t mDim = xShape->GetDim(DIM0);
    size_t weightDimNum = weightShape->GetDimNum();
    int64_t nDim = (weightDimNum > 0) ? weightShape->GetDim(weightDimNum - 1) : 0;

    yShape->SetDimNum(2);
    yShape->SetDim(0, mDim);
    yShape->SetDim(1, nDim);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeGroupedMatmulInitOnly");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatmulInitOnly).InferShape(InferShapeGroupedMatmulInitOnly);
} // namespace ops
