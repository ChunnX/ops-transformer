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
 * \file moe_tutel_combine_x_infershape.cpp
 * \brief
 */

#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_Y_GRAD_IDX = 0;
constexpr size_t INPUT_GATES_IDX = 1;
constexpr size_t OUTPUT_X_GRAD_IDX = 0;
constexpr size_t Y_GRAD_DIM_NUM = 3;
constexpr size_t GATES_DIM_NUM = 2;
constexpr size_t Y_GRAD_H_IDX = 2;
constexpr size_t GATES_S_IDX = 1;
constexpr size_t X_GRAD_DIM_NUM = 2;
} // namespace

// x_grad = [gates.dim(1), y_grad.dim(2)] = [S, H]。
// InferShape 图模式与单算子直调共用，故留在 op_host；InferDataType 仅图场景使用，
// 已挪到 op_graph/moe_tutel_combine_x_graph_plugin.cpp。
static ge::graphStatus InferShape4MoeTutelCombineX(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4MoeTutelCombineX");

    const gert::Shape *yGradShape = context->GetInputShape(INPUT_Y_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yGradShape);
    const gert::Shape *gatesShape = context->GetInputShape(INPUT_GATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gatesShape);
    gert::Shape *xGradShape = context->GetOutputShape(OUTPUT_X_GRAD_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xGradShape);

    // -2（UNKNOWN_RANK）不支持：H 与 S 都要从具体轴上取，rank 未知时推不出输出。
    OP_CHECK_IF(yGradShape->GetDimNum() != Y_GRAD_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "y_grad",
                                                         std::to_string(yGradShape->GetDimNum()),
                                                         "y_grad must be a 3D tensor of [E, C, H]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gatesShape->GetDimNum() != GATES_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "gates",
                                                         std::to_string(gatesShape->GetDimNum()),
                                                         "gates must be a 2D tensor of [K, S]"),
                return ge::GRAPH_FAILED);

    xGradShape->SetDimNum(X_GRAD_DIM_NUM);
    xGradShape->SetDim(0, gatesShape->GetDim(GATES_S_IDX));
    xGradShape->SetDim(1, yGradShape->GetDim(Y_GRAD_H_IDX));

    OP_LOGD(context->GetNodeName(), "End to do InferShape4MoeTutelCombineX");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTutelCombineX).InferShape(InferShape4MoeTutelCombineX);
} // namespace ops
