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
 * \file block_attn_res_prepare_infershape.cpp
 * \brief Shape and dtype inference for BlockAttnResPrepare.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {

constexpr size_t BLOCK_RESIDUAL_INDEX = 0;
constexpr size_t EFFECTIVE_QUERY_INDEX = 1;
constexpr size_t SOFTMAX_MAX_INDEX = 0;
constexpr size_t WEIGHTED_OUTPUT_INDEX = 1;
constexpr size_t SOFTMAX_SUM_INDEX = 2;
constexpr const char *OP_NAME = "BlockAttnResPrepare";

ge::graphStatus InferShapeBlockAttnResPrepare(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "infer shape context is null"), return ge::GRAPH_FAILED);
    const gert::Shape *vShape = context->GetInputShape(BLOCK_RESIDUAL_INDEX);
    const gert::Shape *qShape = context->GetInputShape(EFFECTIVE_QUERY_INDEX);
    OP_CHECK_IF(vShape == nullptr || qShape == nullptr,
                OP_LOGE(context->GetNodeName(), "input shape is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape->GetDimNum() != 3 || qShape->GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "block_residual must be 3D and effective_query must be 2D"),
                return ge::GRAPH_FAILED);

    const int64_t totalT = vShape->GetDim(0);
    const int64_t totalD = vShape->GetDim(2);
    const int64_t totalS = qShape->GetDim(0);
    OP_CHECK_IF(qShape->GetDim(1) != totalD,
                OP_LOGE(context->GetNodeName(), "the last dimensions of block_residual and effective_query differ"),
                return ge::GRAPH_FAILED);

    gert::Shape *mShape = context->GetOutputShape(SOFTMAX_MAX_INDEX);
    gert::Shape *oShape = context->GetOutputShape(WEIGHTED_OUTPUT_INDEX);
    gert::Shape *lShape = context->GetOutputShape(SOFTMAX_SUM_INDEX);
    OP_CHECK_IF(mShape == nullptr || oShape == nullptr || lShape == nullptr,
                OP_LOGE(context->GetNodeName(), "output shape is null"), return ge::GRAPH_FAILED);

    *mShape = gert::Shape({totalS, totalT});
    *oShape = gert::Shape({totalS, totalT, totalD});
    *lShape = gert::Shape({totalS, totalT});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeBlockAttnResPrepare(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "infer data type context is null"), return ge::GRAPH_FAILED);
    const ge::DataType inputType = context->GetInputDataType(BLOCK_RESIDUAL_INDEX);
    context->SetOutputDataType(SOFTMAX_MAX_INDEX, inputType);
    context->SetOutputDataType(WEIGHTED_OUTPUT_INDEX, inputType);
    context->SetOutputDataType(SOFTMAX_SUM_INDEX, inputType);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(BlockAttnResPrepare)
    .InferShape(InferShapeBlockAttnResPrepare)
    .InferDataType(InferDataTypeBlockAttnResPrepare);

} // namespace ops
