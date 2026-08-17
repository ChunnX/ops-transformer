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
 * \file moe_tutel_combine_gates_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_common/op_host/util/shape_util.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t INPUT_INDICES_INDEX = 2;
constexpr size_t OUTPUT_GATES_GRAD_INDEX = 0;
} // namespace

// gates_grad 与路由张量同形（[E, S]）。canndev 侧没有 InferShape，本形状由其 tiling / kernel 的
// taskNum = samples * y_grad.shape[0] 与官方 UT gen_data.py 的 [batchSize, samples] 造数共同确定。
static ge::graphStatus InferShape4MoeTutelCombineGates(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4MoeTutelCombineGates");

    const gert::Shape *indicesShape = context->GetInputShape(INPUT_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    gert::Shape *gatesGradShape = context->GetOutputShape(OUTPUT_GATES_GRAD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gatesGradShape);

    // def.cpp 声明了 DynamicRankSupportFlag(true)：indices 为 -2（UNKNOWN_RANK）时不能按普通 shape 拷贝，
    // 否则会把 -2 这个标记值当成真实维度传下去，推出非法输出形状。
    if (Ops::Base::IsUnknownRank(*indicesShape)) {
        Ops::Base::SetUnknownRank(*gatesGradShape);
        OP_LOGD(context->GetNodeName(), "indices is unknown rank, set gates_grad to unknown rank");
        return GRAPH_SUCCESS;
    }

    *gatesGradShape = *indicesShape;

    OP_LOGD(context->GetNodeName(), "End to do InferShape4MoeTutelCombineGates");
    return GRAPH_SUCCESS;
}

// InferDataType 仅图场景使用，已按交付件划分挪到
// op_graph/moe_tutel_combine_gates_graph_infer.cpp；此处只挂图与单算子共用的 InferShape。
IMPL_OP_INFERSHAPE(MoeTutelCombineGates).InferShape(InferShape4MoeTutelCombineGates);
} // namespace ops
