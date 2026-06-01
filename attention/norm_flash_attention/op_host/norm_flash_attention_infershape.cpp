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
 * \file norm_flash_attention_infershape.cpp
 * \brief 算子形状推导实现
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace gert;
using namespace ge;

namespace ops {

const constexpr int64_t QUERY_INDEX = 0;
const constexpr int64_t KEY_INDEX = 1;
const constexpr int64_t VALUE_INDEX = 2;

const constexpr int64_t ATTENTION_OUT_INDEX = 0;

static ge::graphStatus InferShapeNormFlashAttention(InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape NormFlashAttention");

    auto queryShape = context->GetDynamicInputShape(QUERY_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);

    auto attentionOutShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attentionOutShape);

    // output shape = query shape [B, S, H, D]
    auto dimNum = queryShape->GetDimNum();
    attentionOutShape->SetDimNum(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        attentionOutShape->SetDim(i, queryShape->GetDim(i));
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape NormFlashAttention");
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeNormFlashAttention(gert::InferDataTypeContext *context)
{
    auto queryType = context->GetInputDataType(QUERY_INDEX);
    context->SetOutputDataType(ATTENTION_OUT_INDEX, queryType);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NormFlashAttention)
    .InferShape(InferShapeNormFlashAttention)
    .InferDataType(InferDataTypeNormFlashAttention);
} // namespace ops
