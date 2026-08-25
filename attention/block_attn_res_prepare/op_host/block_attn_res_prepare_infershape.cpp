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

#include <string>

namespace ops {
namespace {

constexpr size_t BLOCK_RES_INDEX = 0;
constexpr size_t PSEUDO_QUERY_INDEX = 2;
constexpr size_t NUMERATOR_INDEX = 0;
constexpr size_t LOGIT_MAX_INDEX = 1;
constexpr size_t EXP_SUM_INDEX = 2;
constexpr size_t BLOCK_RES_RANK = 3;
constexpr size_t PSEUDO_QUERY_RANK = 2;
constexpr size_t T_DIM_INDEX = 0;
constexpr size_t D_DIM_INDEX = 2;
constexpr size_t S_DIM_INDEX = 0;
constexpr size_t PSEUDO_QUERY_D_DIM_INDEX = 1;
constexpr const char *OP_NAME = "BlockAttnResPrepare";

ge::graphStatus InferShapeBlockAttnResPrepare(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "infer shape context is null"), return ge::GRAPH_FAILED);
    const gert::Shape *blockResShape = context->GetInputShape(BLOCK_RES_INDEX);
    const gert::Shape *pseudoQueryShape = context->GetInputShape(PSEUDO_QUERY_INDEX);
    OP_CHECK_IF(blockResShape == nullptr || pseudoQueryShape == nullptr,
                OP_LOGE(context->GetNodeName(), "input shape is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockResShape->GetDimNum() != BLOCK_RES_RANK || pseudoQueryShape->GetDimNum() != PSEUDO_QUERY_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    context->GetNodeName(), "block_res, pseudo_query",
                    (std::to_string(blockResShape->GetDimNum()) + "D, " +
                     std::to_string(pseudoQueryShape->GetDimNum()) + "D")
                        .c_str(),
                    "block_res must be 3D and pseudo_query must be 2D"),
                return ge::GRAPH_FAILED);

    const int64_t totalT = blockResShape->GetDim(T_DIM_INDEX);
    const int64_t totalD = blockResShape->GetDim(D_DIM_INDEX);
    const int64_t totalS = pseudoQueryShape->GetDim(S_DIM_INDEX);
    OP_CHECK_IF(pseudoQueryShape->GetDim(PSEUDO_QUERY_D_DIM_INDEX) != totalD,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "block_res.shape[2], pseudo_query.shape[1]",
                    (std::to_string(totalD) + ", " +
                     std::to_string(pseudoQueryShape->GetDim(PSEUDO_QUERY_D_DIM_INDEX)))
                        .c_str(),
                    "block_res.shape[2] must equal pseudo_query.shape[1]"),
                return ge::GRAPH_FAILED);

    gert::Shape *numeratorShape = context->GetOutputShape(NUMERATOR_INDEX);
    gert::Shape *logitMaxShape = context->GetOutputShape(LOGIT_MAX_INDEX);
    gert::Shape *expSumShape = context->GetOutputShape(EXP_SUM_INDEX);
    OP_CHECK_IF(numeratorShape == nullptr || logitMaxShape == nullptr || expSumShape == nullptr,
                OP_LOGE(context->GetNodeName(), "output shape is null"), return ge::GRAPH_FAILED);

    *numeratorShape = gert::Shape({totalS, totalT, totalD});
    *logitMaxShape = gert::Shape({totalS, totalT});
    *expSumShape = gert::Shape({totalS, totalT});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeBlockAttnResPrepare(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "infer data type context is null"), return ge::GRAPH_FAILED);
    const ge::DataType inputType = context->GetInputDataType(BLOCK_RES_INDEX);
    context->SetOutputDataType(NUMERATOR_INDEX, inputType);
    context->SetOutputDataType(LOGIT_MAX_INDEX, inputType);
    context->SetOutputDataType(EXP_SUM_INDEX, inputType);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(BlockAttnResPrepare)
    .InferShape(InferShapeBlockAttnResPrepare)
    .InferDataType(InferDataTypeBlockAttnResPrepare);

} // namespace ops
