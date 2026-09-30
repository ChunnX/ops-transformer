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
 * \file kv_cache_turbo_quant_infershape.cpp
 * \brief
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t KV_INPUT_INDEX = 0;
constexpr uint32_t ROT_INPUT_INDEX = 1;
constexpr uint32_t QJL_INPUT_INDEX = 2;
constexpr uint32_t IDX_OUTPUT_INDEX = 0;
constexpr uint32_t QJL_OUTPUT_INDEX = 1;
constexpr uint32_t NORM_OUTPUT_INDEX = 2;
constexpr uint32_t GAMMA_OUTPUT_INDEX = 3;
constexpr size_t ATTR_MSE_BITS_INDEX = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t KV_DIMS = 3;
constexpr int64_t MATRIX_DIMS = 2;
constexpr int64_t MIN_KV_HEADS = 4;
constexpr int64_t MAX_KV_HEADS = 32;
constexpr int64_t QJL_DIM_ALIGN = 8;

ge::graphStatus InferShapeKvCacheTurboQuant(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    const gert::Shape* kvShape = context->GetInputShape(KV_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    const gert::Shape* rotShape = context->GetInputShape(ROT_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rotShape);
    const gert::Shape* qjlShape = context->GetInputShape(QJL_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);

    // kv_vectors: [num_tokens, num_kv_heads, head_dim]
    OP_CHECK_IF(kvShape->GetDimNum() != KV_DIMS,
                OP_LOGE("KvCacheTurboQuant", "kv_vectors must be 3D [num_tokens, num_kv_heads, head_dim]."),
                return ge::GRAPH_FAILED);
    const int64_t numTokens = kvShape->GetDim(0);
    const int64_t numKvHeads = kvShape->GetDim(1);
    const int64_t headDim = kvShape->GetDim(2);
    OP_CHECK_IF(headDim != HEAD_DIM, OP_LOGE("KvCacheTurboQuant", "head_dim must be 128, got %ld.", headDim),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numKvHeads < MIN_KV_HEADS || numKvHeads > MAX_KV_HEADS,
                OP_LOGE("KvCacheTurboQuant", "num_kv_heads must be in [4, 32], got %ld.", numKvHeads),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numTokens <= 0, OP_LOGE("KvCacheTurboQuant", "num_tokens must be positive, got %ld.", numTokens),
                return ge::GRAPH_FAILED);

    // rotation_matrix: [head_dim, head_dim]
    OP_CHECK_IF(
        rotShape->GetDimNum() != MATRIX_DIMS || rotShape->GetDim(0) != HEAD_DIM || rotShape->GetDim(1) != HEAD_DIM,
        OP_LOGE("KvCacheTurboQuant", "rotation_matrix must be [128, 128]."), return ge::GRAPH_FAILED);

    // qjl_matrix: [qjl_dim, head_dim]
    OP_CHECK_IF(qjlShape->GetDimNum() != MATRIX_DIMS || qjlShape->GetDim(1) != HEAD_DIM,
                OP_LOGE("KvCacheTurboQuant", "qjl_matrix must be [qjl_dim, 128]."), return ge::GRAPH_FAILED);
    const int64_t qjlDim = qjlShape->GetDim(0);
    OP_CHECK_IF(qjlDim <= 0 || qjlDim > HEAD_DIM || (qjlDim % QJL_DIM_ALIGN) != 0,
                OP_LOGE("KvCacheTurboQuant", "qjl_dim must be in (0, 128] and multiple of 8, got %ld.", qjlDim),
                return ge::GRAPH_FAILED);

    // mse_bits attr: 2/3/4
    int64_t mseBits = 3;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* pBits = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS_INDEX);
    if (pBits != nullptr) {
        mseBits = *pBits;
    }
    // 8/9 are accepted as internal probe paths (not real bit widths); their
    // output shapes follow the 3-bit layout so probe tensors match
    OP_CHECK_IF(mseBits != 2 && mseBits != 3 && mseBits != 4 && mseBits != 8 && mseBits != 9,
                OP_LOGE("KvCacheTurboQuant", "mse_bits must be 2, 3 or 4, got %ld.", mseBits), return ge::GRAPH_FAILED);
    if (mseBits >= 8) {
        mseBits = 3;
    }

    gert::Shape* idxShape = context->GetOutputShape(IDX_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    gert::Shape* qjlOutShape = context->GetOutputShape(QJL_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlOutShape);
    gert::Shape* normShape = context->GetOutputShape(NORM_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, normShape);
    gert::Shape* gammaShape = context->GetOutputShape(GAMMA_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    // quant_idx: [num_tokens, num_kv_heads, ceil(head_dim * mse_bits / 8)] = head_dim/8 * mse_bits
    idxShape->SetDimNum(KV_DIMS);
    idxShape->SetDim(0, numTokens);
    idxShape->SetDim(1, numKvHeads);
    idxShape->SetDim(2, (HEAD_DIM / 8) * mseBits);

    // quant_qjl: [num_tokens, num_kv_heads, ceil(qjl_dim / 8)]
    qjlOutShape->SetDimNum(KV_DIMS);
    qjlOutShape->SetDim(0, numTokens);
    qjlOutShape->SetDim(1, numKvHeads);
    qjlOutShape->SetDim(2, qjlDim / QJL_DIM_ALIGN);

    // quant_norm / quant_gamma: [num_tokens, num_kv_heads]
    normShape->SetDimNum(2);
    normShape->SetDim(0, numTokens);
    normShape->SetDim(1, numKvHeads);
    *gammaShape = *normShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvCacheTurboQuant(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(IDX_OUTPUT_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(QJL_OUTPUT_INDEX, ge::DT_UINT8);
    context->SetOutputDataType(NORM_OUTPUT_INDEX, ge::DT_BF16);
    context->SetOutputDataType(GAMMA_OUTPUT_INDEX, ge::DT_BF16);
    return GRAPH_SUCCESS;
}

IMPL_OP(KvCacheTurboQuant).InferShape(InferShapeKvCacheTurboQuant).InferDataType(InferDataTypeKvCacheTurboQuant);
} // namespace ops
