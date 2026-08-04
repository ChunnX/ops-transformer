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
 * \file msa_index_score_infershape.cpp
 * \brief MsaIndexScore InferShape / InferDataType。
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace {
constexpr uint32_t MSA_QUERY_INDEX = 0;
constexpr uint32_t MSA_BLOCK_TABLE_INDEX = 2;
constexpr uint32_t MSA_QUERY_DIM_NUM = 3;
constexpr uint32_t MSA_BLOCK_TABLE_DIM_NUM = 2;
constexpr int64_t MSA_SCORE_STRIDE_ALIGN = 16;

inline int64_t RoundUpTo(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}
} // namespace

static ge::graphStatus InferShapeMsaIndexScore(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MsaIndexScore", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter MsaIndexScore InferShape impl.");

    const gert::Shape *queryShape = context->GetInputShape(MSA_QUERY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    const gert::Shape *blockTableShape = context->GetInputShape(MSA_BLOCK_TABLE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTableShape);
    gert::Shape *scoreShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scoreShape);

    OP_CHECK_IF(queryShape->GetDimNum() != MSA_QUERY_DIM_NUM,
                OP_LOGE(context, "query must be TND with 3 dims, but got %zu.", queryShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockTableShape->GetDimNum() != MSA_BLOCK_TABLE_DIM_NUM,
                OP_LOGE(context, "block_table must have 2 dims, but got %zu.", blockTableShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    const int64_t totalQ = queryShape->GetDim(0);      // 0: T
    const int64_t numQHeads = queryShape->GetDim(1);   // 1: numQHeads
    const int64_t maxBlocks = blockTableShape->GetDim(1); // 1: maxBlocksPerBatch

    // maxpool 已经把一个 block 内的 blockSize 个 token 归约成 1 个分数，
    // 因此输出末维是 block 数（16 对齐）而不是 kv token 数。
    scoreShape->SetDimNum(MSA_QUERY_DIM_NUM);
    scoreShape->SetDim(0, numQHeads);                                       // 0: numQHeads
    scoreShape->SetDim(1, totalQ);                                          // 1: totalQ
    scoreShape->SetDim(2, RoundUpTo(maxBlocks, MSA_SCORE_STRIDE_ALIGN));    // 2: scoreBlockStride

    OP_LOGI(context->GetNodeName(), "MsaIndexScore InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMsaIndexScore(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("MsaIndexScore", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    // score 恒为 fp32：Cube 以 fp32 累加，下游 topk 直接消费 fp32。
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MsaIndexScore)
    .InferShape(InferShapeMsaIndexScore)
    .InferDataType(InferDataTypeMsaIndexScore);
} // namespace ops
