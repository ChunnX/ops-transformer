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
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr size_t kKvInput = 0;
constexpr size_t kQjlInput = 2;
constexpr size_t kIdxOutput = 0;
constexpr size_t kQjlOutput = 1;
constexpr size_t kNormOutput = 2;
constexpr size_t kGammaOutput = 3;
constexpr size_t kMseBitsAttr = 0;
constexpr int64_t kHeadDim = 128;

static int64_t PackedBytes(int64_t lanes, int64_t bits)
{
    if (lanes < 0 || bits < 0) {
        return -1;
    }
    return (lanes * bits + 7) / 8;
}

ge::graphStatus InferShapeKvCacheTurboQuant(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "InferShapeContext is nullptr"),
                return ge::GRAPH_FAILED);
    const gert::Shape *kv = context->GetInputShape(kKvInput);
    const gert::Shape *qjl = context->GetInputShape(kQjlInput);
    OP_CHECK_NULL_WITH_CONTEXT(context, kv);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjl);
    gert::Shape *idx = context->GetOutputShape(kIdxOutput);
    gert::Shape *qjlOut = context->GetOutputShape(kQjlOutput);
    gert::Shape *norm = context->GetOutputShape(kNormOutput);
    gert::Shape *gamma = context->GetOutputShape(kGammaOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlOut);
    OP_CHECK_NULL_WITH_CONTEXT(context, norm);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma);

    const auto *attrs = context->GetAttrs();
    int64_t bits = 3;
    if (attrs != nullptr) {
        const int64_t *bitsAttr = attrs->GetAttrPointer<int64_t>(kMseBitsAttr);
        if (bitsAttr != nullptr) {
            bits = *bitsAttr;
        }
    }

    const int64_t tokens = kv->GetDimNum() > 0 ? kv->GetDim(0) : -1;
    const int64_t heads = kv->GetDimNum() > 1 ? kv->GetDim(1) : -1;
    const int64_t qjlDim = qjl->GetDimNum() > 0 ? qjl->GetDim(0) : kHeadDim;

    idx->SetDimNum(3);
    idx->SetDim(0, tokens);
    idx->SetDim(1, heads);
    idx->SetDim(2, PackedBytes(kHeadDim, bits));
    qjlOut->SetDimNum(3);
    qjlOut->SetDim(0, tokens);
    qjlOut->SetDim(1, heads);
    qjlOut->SetDim(2, PackedBytes(qjlDim, 1));
    norm->SetDimNum(2);
    norm->SetDim(0, tokens);
    norm->SetDim(1, heads);
    gamma->SetDimNum(2);
    gamma->SetDim(0, tokens);
    gamma->SetDim(1, heads);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeKvCacheTurboQuant(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("KvCacheTurboQuant", "InferDataTypeContext is nullptr"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(kIdxOutput, ge::DT_UINT8);
    context->SetOutputDataType(kQjlOutput, ge::DT_UINT8);
    context->SetOutputDataType(kNormOutput, ge::DT_BF16);
    context->SetOutputDataType(kGammaOutput, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheTurboQuant)
    .InferShape(InferShapeKvCacheTurboQuant)
    .InferDataType(InferDataTypeKvCacheTurboQuant);
} // namespace ops
