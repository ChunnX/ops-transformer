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
 * \file broadcast_nddma_bench_infershape.cpp
 * \brief BroadcastNddmaBench InferShape/InferDataType
 *
 * 广播输出恒为满 5D (dim0, dim1, dim2, dim3, dim4)：
 *   - dim0/dim1 取自输入 shape（广播轴从不折叠这两维）
 *   - dim2/dim3/dim4 取自 Attr（输入广播轴长度=1，无法从 input 恢复真实长度）
 * 输出 dtype 与输入 dtype 一致。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
constexpr int32_t X_INDEX = 0;
constexpr int32_t Y_INDEX = 0;
constexpr size_t RANK_5D = 5;
constexpr size_t MIN_RANK = 2;
constexpr size_t AXIS_0 = 0;
constexpr size_t AXIS_1 = 1;
constexpr size_t AXIS_2 = 2;
constexpr size_t AXIS_3 = 3;
constexpr size_t AXIS_4 = 4;

// OpDef Attr 声明顺序：is_mte3(0), scene(1), dim2(2), dim3(3), dim4(4)
constexpr size_t ATTR_DIM2_INDEX = 2;
constexpr size_t ATTR_DIM3_INDEX = 3;
constexpr size_t ATTR_DIM4_INDEX = 4;

static int64_t GetAttrI64(const gert::RuntimeAttrs *attrs, size_t index, int64_t defVal)
{
    if (attrs == nullptr) {
        return defVal;
    }
    const int64_t *ptr = attrs->GetAttrPointer<int64_t>(index);
    return (ptr != nullptr) ? *ptr : defVal;
}

static ge::graphStatus InferShape4BroadcastNddmaBench(gert::InferShapeContext *context)
{
    OP_LOGI(context, "Begin to do InferShape4BroadcastNddmaBench.");

    auto xShape = context->GetInputShape(X_INDEX);
    auto yShape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(xShape == nullptr || yShape == nullptr,
                OP_LOGE(context, "input/output shape is null"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape->GetDimNum() < MIN_RANK,
                OP_LOGE(context, "input x rank must be >= 2, got %zu", xShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    int64_t dim2 = GetAttrI64(attrs, ATTR_DIM2_INDEX, 0);
    int64_t dim3 = GetAttrI64(attrs, ATTR_DIM3_INDEX, 0);
    int64_t dim4 = GetAttrI64(attrs, ATTR_DIM4_INDEX, 0);

    // 输出恒满 5D：dim0/dim1 取自输入，dim2/dim3/dim4 取自 Attr
    yShape->SetDimNum(RANK_5D);
    yShape->SetDim(AXIS_0, xShape->GetDim(AXIS_0));
    yShape->SetDim(AXIS_1, xShape->GetDim(AXIS_1));
    yShape->SetDim(AXIS_2, dim2);
    yShape->SetDim(AXIS_3, dim3);
    yShape->SetDim(AXIS_4, dim4);

    OP_LOGI(context, "End to do InferShape4BroadcastNddmaBench.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4BroadcastNddmaBench(gert::InferDataTypeContext *context)
{
    OP_LOGI(context, "Begin to do InferDataType4BroadcastNddmaBench.");
    auto xDtype = context->GetInputDataType(X_INDEX);
    context->SetOutputDataType(Y_INDEX, xDtype);
    OP_LOGI(context, "End to do InferDataType4BroadcastNddmaBench.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BroadcastNddmaBench)
    .InferShape(ops::InferShape4BroadcastNddmaBench)
    .InferDataType(ops::InferDataType4BroadcastNddmaBench);

} // namespace ops
