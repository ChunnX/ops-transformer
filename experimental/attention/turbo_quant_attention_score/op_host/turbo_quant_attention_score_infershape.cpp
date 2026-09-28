/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t IN_QUERY_ROTATED = 0;
constexpr size_t IN_QUERY_QJL = 1;
constexpr size_t IN_KEY_CACHE_IDX = 2;
constexpr size_t IN_KEY_CACHE_QJL = 3;
constexpr size_t IN_KEY_CACHE_NORM = 4;
constexpr size_t IN_KEY_CACHE_GAMMA = 5;
constexpr size_t IN_BLOCK_TABLE = 6;
constexpr size_t IN_SEQ_LENS = 7;
constexpr size_t OUT_SCORES = 0;
// Paged KV cache block size (product constraint), not UB align size.
constexpr int64_t PAGED_BLOCK_SIZE = 128;

bool KnownMismatch(int64_t lhs, int64_t rhs)
{
    return lhs >= 0 && rhs >= 0 && lhs != rhs;
}
} // namespace

static ge::graphStatus InferShapeTurboQuantAttentionScore(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "infer shape context is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *query = context->GetInputShape(IN_QUERY_ROTATED);
    OP_CHECK_NULL_WITH_CONTEXT(context, query);
    const gert::Shape *queryQjl = context->GetInputShape(IN_QUERY_QJL);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryQjl);
    const gert::Shape *idx = context->GetInputShape(IN_KEY_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx);
    const gert::Shape *qjl = context->GetInputShape(IN_KEY_CACHE_QJL);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjl);
    const gert::Shape *norm = context->GetInputShape(IN_KEY_CACHE_NORM);
    OP_CHECK_NULL_WITH_CONTEXT(context, norm);
    const gert::Shape *gamma = context->GetInputShape(IN_KEY_CACHE_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma);
    const gert::Shape *blockTable = context->GetInputShape(IN_BLOCK_TABLE);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTable);
    const gert::Shape *seqLens = context->GetInputShape(IN_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqLens);
    gert::Shape *output = context->GetOutputShape(OUT_SCORES);
    OP_CHECK_NULL_WITH_CONTEXT(context, output);

    OP_CHECK_IF(query->GetDimNum() != 3, OP_LOGE(context, "query_rotated rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryQjl->GetDimNum() != 3, OP_LOGE(context, "query_qjl rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idx->GetDimNum() != 4, OP_LOGE(context, "key_cache_idx rank must be 4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(qjl->GetDimNum() != 4, OP_LOGE(context, "key_cache_qjl rank must be 4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(norm->GetDimNum() != 3, OP_LOGE(context, "key_cache_norm rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gamma->GetDimNum() != 3, OP_LOGE(context, "key_cache_gamma rank must be 3"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockTable->GetDimNum() != 2, OP_LOGE(context, "block_table rank must be 2"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqLens->GetDimNum() != 1, OP_LOGE(context, "seq_lens rank must be 1"), return ge::GRAPH_FAILED);

    for (size_t i = 0; i < 3; ++i) {
        OP_CHECK_IF(KnownMismatch(query->GetDim(i), queryQjl->GetDim(i)),
                    OP_LOGE(context, "query_rotated/query_qjl dim %zu mismatch", i), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(KnownMismatch(idx->GetDim(0), qjl->GetDim(0)), OP_LOGE(context, "num_blocks mismatch idx/qjl"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(idx->GetDim(0), norm->GetDim(0)), OP_LOGE(context, "num_blocks mismatch idx/norm"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(idx->GetDim(0), gamma->GetDim(0)), OP_LOGE(context, "num_blocks mismatch idx/gamma"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(idx->GetDim(1), PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %ld", PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(qjl->GetDim(1), PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %ld", PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(norm->GetDim(1), PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %ld", PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(gamma->GetDim(1), PAGED_BLOCK_SIZE),
                OP_LOGE(context, "paged block_size must be %ld", PAGED_BLOCK_SIZE), return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(idx->GetDim(2), qjl->GetDim(2)), OP_LOGE(context, "num_kv_heads mismatch idx/qjl"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(idx->GetDim(2), norm->GetDim(2)), OP_LOGE(context, "num_kv_heads mismatch idx/norm"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(idx->GetDim(2), gamma->GetDim(2)), OP_LOGE(context, "num_kv_heads mismatch idx/gamma"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(KnownMismatch(blockTable->GetDim(0), seqLens->GetDim(0)),
                OP_LOGE(context, "block_table batch and seq_lens length mismatch"), return ge::GRAPH_FAILED);

    const int64_t numTokens = query->GetDim(0);
    const int64_t numQHeads = query->GetDim(1);
    const int64_t maxBlocks = blockTable->GetDim(1);
    int64_t maxKvLen = -1;
    if (maxBlocks >= 0) {
        maxKvLen = maxBlocks * PAGED_BLOCK_SIZE;
    }

    output->SetDimNum(3);
    output->SetDim(0, numTokens);
    output->SetDim(1, numQHeads);
    output->SetDim(2, maxKvLen);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTurboQuantAttentionScore(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "infer dtype context is nullptr"),
                return ge::GRAPH_FAILED);
    (void)context->GetInputDataType(IN_QUERY_ROTATED);
    context->SetOutputDataType(OUT_SCORES, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TurboQuantAttentionScore)
    .InferShape(InferShapeTurboQuantAttentionScore)
    .InferDataType(InferDataTypeTurboQuantAttentionScore);
} // namespace ops
