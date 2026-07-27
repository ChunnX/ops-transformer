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
 * \file last_transpose_nddma_bench_infershape.cpp
 * \brief LastTransposeNddmaBench InferShape/InferDataType
 *
 * permute [0,1,3,2]：输出 = 输入交换最后两维
 *   输入 (dim0, dim1, dim2, dim3) -> 输出 (dim0, dim1, dim3, dim2)
 * 输出 dtype 与输入 dtype 一致。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr int32_t X_INDEX = 0;
constexpr int32_t Y_INDEX = 0;
constexpr size_t RANK_4D = 4;
constexpr size_t LAST_AXIS = 3;
constexpr size_t SECOND_LAST_AXIS = 2;

static ge::graphStatus InferShape4LastTransposeNddmaBench(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Begin to do InferShape4LastTransposeNddmaBench.");

    auto xShape = context->GetInputShape(X_INDEX);
    auto yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(xShape == nullptr || yShape == nullptr,
                OP_LOGE(context, "input/output shape is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetDimNum() != RANK_4D,
                OP_LOGE(context, "input x rank must be 4, got %zu", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // 拷贝输入 shape，然后交换最后两维
    *yShape = *xShape;
    yShape->SetDim(SECOND_LAST_AXIS, xShape->GetDim(LAST_AXIS));
    yShape->SetDim(LAST_AXIS, xShape->GetDim(SECOND_LAST_AXIS));

    OP_LOGI(context, "End to do InferShape4LastTransposeNddmaBench.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4LastTransposeNddmaBench(gert::InferDataTypeContext *context)
{
    OP_LOGI(context, "Begin to do InferDataType4LastTransposeNddmaBench.");
    auto xDtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, xDtype);
    OP_LOGI(context, "End to do InferDataType4LastTransposeNddmaBench.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LastTransposeNddmaBench)
    .InferShape(ops::InferShape4LastTransposeNddmaBench)
    .InferDataType(ops::InferDataType4LastTransposeNddmaBench);

} // namespace ops
