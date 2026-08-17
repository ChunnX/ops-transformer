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
 * \file moe_tutel_combine_gates_graph_infer.cpp
 * \brief MoeTutelCombineGates InferDataType：gates_grad 跟随 x 的浮点 dtype。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 *
 * 注：REG_OP(MoeTutelCombineGates) 已在 common/include/op_graph/op_transformer_proto_extend.h 中
 *     定义并被 graph plugin 编入，本目录不再重复声明，避免同名 REG_OP 重定义。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t OUTPUT_GATES_GRAD_INDEX = 0;
} // namespace

static ge::graphStatus InferDataType4MoeTutelCombineGates(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4MoeTutelCombineGates");
    // x / y_grad / gates_grad 强制同一浮点 dtype，故输出直接跟随 x。
    context->SetOutputDataType(OUTPUT_GATES_GRAD_INDEX, context->GetInputDataType(INPUT_X_INDEX));
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4MoeTutelCombineGates");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeTutelCombineGates).InferDataType(InferDataType4MoeTutelCombineGates);
} // namespace ops
