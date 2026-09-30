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
 * \brief Shape inference: [T, H, 128] -> packed codes plus two per (token, head) scalars.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
namespace {
constexpr size_t IN_KV_VECTORS = 0;
constexpr size_t OUT_QUANT_IDX = 0;
constexpr size_t OUT_QUANT_QJL = 1;
constexpr size_t OUT_QUANT_NORM = 2;
constexpr size_t OUT_QUANT_GAMMA = 3;
constexpr size_t ATTR_MSE_BITS = 0;
constexpr int64_t HEAD_DIM = 128;

ge::graphStatus InferShapeKvCacheTurboQuant(gert::InferShapeContext *context)
{
    const gert::Shape *kv = context->GetInputShape(IN_KV_VECTORS);
    OP_CHECK_NULL_WITH_CONTEXT(context, kv);
    const auto rank = kv->GetDimNum();
    OP_CHECK_IF(rank != 3, OP_LOGE(context, "kv_vectors rank must be 3, got %zu", rank),
                return ge::GRAPH_FAILED);
    const int64_t numTokens = kv->GetDim(0);
    const int64_t numHeads = kv->GetDim(1);
    const int64_t headDim = kv->GetDim(2);
    OP_CHECK_IF(headDim != HEAD_DIM, OP_LOGE(context, "head_dim must be %ld, got %ld", HEAD_DIM, headDim),
                return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *mseBitsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MSE_BITS);
    const int64_t mseBits = (mseBitsPtr == nullptr) ? 3 : *mseBitsPtr;
    OP_CHECK_IF(mseBits != 2 && mseBits != 3 && mseBits != 4,
                OP_LOGE(context, "mse_bits must be 2, 3 or 4, got %ld", mseBits), return ge::GRAPH_FAILED);

    gert::Shape *idxShape = context->GetOutputShape(OUT_QUANT_IDX);
    gert::Shape *qjlShape = context->GetOutputShape(OUT_QUANT_QJL);
    gert::Shape *normShape = context->GetOutputShape(OUT_QUANT_NORM);
    gert::Shape *gammaShape = context->GetOutputShape(OUT_QUANT_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, normShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    idxShape->SetDimNum(3);
    idxShape->SetDim(0, numTokens);
    idxShape->SetDim(1, numHeads);
    idxShape->SetDim(2, headDim * mseBits / 8);

    qjlShape->SetDimNum(3);
    qjlShape->SetDim(0, numTokens);
    qjlShape->SetDim(1, numHeads);
    qjlShape->SetDim(2, headDim / 8);

    normShape->SetDimNum(2);
    normShape->SetDim(0, numTokens);
    normShape->SetDim(1, numHeads);

    gammaShape->SetDimNum(2);
    gammaShape->SetDim(0, numTokens);
    gammaShape->SetDim(1, numHeads);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvCacheTurboQuant(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(OUT_QUANT_IDX, ge::DT_UINT8);
    context->SetOutputDataType(OUT_QUANT_QJL, ge::DT_UINT8);
    // The formal interface emits bf16 metadata. The task book also allows fp16 metadata, so when the
    // caller asks for it the two scalars keep that dtype (an unspecified dtype keeps the default).
    ge::DataType metaDtype = ge::DT_BF16;
    const auto *normDesc = context->GetOutputDesc(OUT_QUANT_NORM);
    if (normDesc != nullptr && normDesc->GetDataType() == ge::DT_FLOAT16) {
        metaDtype = ge::DT_FLOAT16;
    }
    context->SetOutputDataType(OUT_QUANT_NORM, metaDtype);
    context->SetOutputDataType(OUT_QUANT_GAMMA, metaDtype);
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(KvCacheTurboQuant)
    .InferShape(InferShapeKvCacheTurboQuant)
    .InferDataType(InferDataTypeKvCacheTurboQuant);

} // namespace ops
