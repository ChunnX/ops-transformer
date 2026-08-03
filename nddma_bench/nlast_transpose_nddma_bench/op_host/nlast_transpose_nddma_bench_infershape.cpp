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
 * \file nlast_transpose_nddma_bench_infershape.cpp
 * \brief NlastTransposeNddmaBench InferShape/InferDataType
 *
 * permute [0,1,3,2,4]：输出 = 交换中间两轴 dim2/dim3，保留最内轴 dim4
 *   输入 (dim0, dim1, dim2, dim3, dim4) -> 输出 (dim0, dim1, dim3, dim2, dim4)
 * 输出 dtype 与输入 dtype 一致。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr int32_t X_INDEX = 0;
constexpr int32_t Y_INDEX = 0;
constexpr size_t RANK_5D = 5;
constexpr size_t AXIS_2 = 2;
constexpr size_t AXIS_3 = 3;

static ge::graphStatus InferShape4NlastTransposeNddmaBench(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Begin to do InferShape4NlastTransposeNddmaBench.");

    auto xShape = context->GetInputShape(X_INDEX);
    auto yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(xShape == nullptr || yShape == nullptr,
                OP_LOGE(context, "input/output shape is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetDimNum() != RANK_5D,
                OP_LOGE(context, "input x rank must be 5, got %zu", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // 拷贝输入 shape，然后交换中间两轴 (axis2, axis3)，保留 axis4
    *yShape = *xShape;
    yShape->SetDim(AXIS_2, xShape->GetDim(AXIS_3));
    yShape->SetDim(AXIS_3, xShape->GetDim(AXIS_2));

    OP_LOGI(context, "End to do InferShape4NlastTransposeNddmaBench.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4NlastTransposeNddmaBench(gert::InferDataTypeContext *context)
{
    OP_LOGI(context, "Begin to do InferDataType4NlastTransposeNddmaBench.");
    auto xDtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, xDtype);
    OP_LOGI(context, "End to do InferDataType4NlastTransposeNddmaBench.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NlastTransposeNddmaBench)
    .InferShape(ops::InferShape4NlastTransposeNddmaBench)
    .InferDataType(ops::InferDataType4NlastTransposeNddmaBench);

} // namespace ops
