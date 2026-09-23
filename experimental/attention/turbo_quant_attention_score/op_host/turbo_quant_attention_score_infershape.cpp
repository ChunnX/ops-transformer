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
 * \file turbo_quant_attention_score_infershape.cpp
 * \brief TurboQuantAttentionScore shape/dtype 推导
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_ROTATED_INDEX = 0;
constexpr uint32_t QUERY_QJL_INDEX = 1;
constexpr uint32_t KEY_IDX_INDEX = 2;
constexpr uint32_t KEY_QJL_INDEX = 3;
constexpr uint32_t KEY_NORM_INDEX = 4;
constexpr uint32_t KEY_GAMMA_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
constexpr uint32_t SEQ_LENS_INDEX = 7;
constexpr uint32_t ATTN_SCORES_OUTPUT_INDEX = 0;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t QJL_BYTES = 16; // qjl_dim=128, 1 bit/channel -> 16 bytes
constexpr int64_t MAX_GROUP = 8;  // kernel UB 常驻 query 缓冲上限

static int64_t GetMseBits(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return -1;
    }
    const int64_t *pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    return (pBits != nullptr) ? *pBits : 3;
}

ge::graphStatus InferShapeTurboQuantAttentionScore(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape *qShape = context->GetInputShape(QUERY_ROTATED_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_IF(qShape->GetDimNum() != 3 || qShape->GetDim(2) != HEAD_DIM,
                OP_LOGE("TurboQuantAttentionScore",
                        "query_rotated must be a 3D tensor [num_q_tokens, num_q_heads, 128]."),
                return ge::GRAPH_FAILED);
    const gert::Shape *qqShape = context->GetInputShape(QUERY_QJL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qqShape);
    OP_CHECK_IF(qqShape->GetDimNum() != 3 || qqShape->GetDim(2) != HEAD_DIM,
                OP_LOGE("TurboQuantAttentionScore",
                        "query_qjl must be a 3D tensor [num_q_tokens, num_q_heads, 128]."),
                return ge::GRAPH_FAILED);

    const gert::Shape *idxShape = context->GetInputShape(KEY_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    OP_CHECK_IF(idxShape->GetDimNum() != 4 || idxShape->GetDim(1) != BLOCK_SIZE,
                OP_LOGE("TurboQuantAttentionScore",
                        "key_cache_idx must be a 4D tensor [num_blocks, 128, num_kv_heads, idx_bytes]."),
                return ge::GRAPH_FAILED);
    const gert::Shape *qjlShape = context->GetInputShape(KEY_QJL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);
    OP_CHECK_IF(qjlShape->GetDimNum() != 4 || qjlShape->GetDim(3) != QJL_BYTES,
                OP_LOGE("TurboQuantAttentionScore", "key_cache_qjl last dim must be 16 (128 x 1bit)."),
                return ge::GRAPH_FAILED);

    const int64_t mseBits = GetMseBits(context);
    OP_CHECK_IF(mseBits != 2 && mseBits != 3 && mseBits != 4,
                OP_LOGE("TurboQuantAttentionScore", "mse_bits only supports 2/3/4."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(idxShape->GetDim(3) != HEAD_DIM * mseBits / 8,
                OP_LOGE("TurboQuantAttentionScore", "key_cache_idx last dim must be 128*mse_bits/8."),
                return ge::GRAPH_FAILED);

    const int64_t numTokens = qShape->GetDim(0);
    const int64_t numQHeads = qShape->GetDim(1);
    const int64_t numKvHeads = idxShape->GetDim(2);
    OP_CHECK_IF(numKvHeads <= 0 || numKvHeads > 16 || numQHeads % numKvHeads != 0,
                OP_LOGE("TurboQuantAttentionScore",
                        "num_q_heads must be an integer multiple of num_kv_heads (num_kv_heads <= 16)."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numQHeads / numKvHeads > MAX_GROUP,
                OP_LOGE("TurboQuantAttentionScore", "GQA group size > 8 is not supported in the first version."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qqShape->GetDim(0) != numTokens || qqShape->GetDim(1) != numQHeads,
                OP_LOGE("TurboQuantAttentionScore", "query_qjl shape must equal query_rotated shape."),
                return ge::GRAPH_FAILED);

    const gert::Shape *btShape = context->GetInputShape(BLOCK_TABLE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, btShape);
    OP_CHECK_IF(btShape->GetDimNum() != 2,
                OP_LOGE("TurboQuantAttentionScore", "block_table must be 2D [max_batch, max_blocks_per_seq]."),
                return ge::GRAPH_FAILED);
    const gert::Shape *slShape = context->GetInputShape(SEQ_LENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, slShape);
    OP_CHECK_IF(slShape->GetDimNum() != 1 || slShape->GetDim(0) != btShape->GetDim(0),
                OP_LOGE("TurboQuantAttentionScore", "seq_lens must be 1D [max_batch]."),
                return ge::GRAPH_FAILED);

    // 输出第三维按静态信息推导：max_kv_len = max_blocks_per_seq * block_size
    gert::Shape *outShape = context->GetOutputShape(ATTN_SCORES_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    outShape->SetDimNum(3);
    outShape->SetDim(0, numTokens);
    outShape->SetDim(1, numQHeads);
    outShape->SetDim(2, btShape->GetDim(1) * BLOCK_SIZE);
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeTurboQuantAttentionScore(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TurboQuantAttentionScore", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(ATTN_SCORES_OUTPUT_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(TurboQuantAttentionScore)
    .InferShape(InferShapeTurboQuantAttentionScore)
    .InferDataType(InferDataTypeTurboQuantAttentionScore);
} // namespace ops
