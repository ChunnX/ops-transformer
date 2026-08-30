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
 * \file moe_tutel_combine_x_graph_plugin.cpp
 * \brief MoeTutelCombineX InferDataType：x_grad 与 y_grad 同 dtype。
 *        InferDataType 仅图场景使用，故落在 op_graph；InferShape 图与单算子共用，留在 op_host。
 *        文件名用 *_graph_plugin.cpp 而非 *_graph_infer.cpp：本仓 cmake/obj_func.cmake 的
 *        add_graph_plugin_sources() 只按 `*_graph_plugin*.cpp` 通配收集源文件，
 *        叫 *_graph_infer.cpp 不会被编进 graph plugin 库。
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_Y_GRAD_IDX = 0;
constexpr size_t OUTPUT_X_GRAD_IDX = 0;
} // namespace

static ge::graphStatus InferDataType4MoeTutelCombineX(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin InferDataType4MoeTutelCombineX");
    context->SetOutputDataType(OUTPUT_X_GRAD_IDX, context->GetInputDataType(INPUT_Y_GRAD_IDX));
    OP_LOGD(context->GetNodeName(), "End InferDataType4MoeTutelCombineX");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MoeTutelCombineX).InferDataType(InferDataType4MoeTutelCombineX);
} // namespace ops
