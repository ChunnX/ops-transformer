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
 * \brief InferShape：idx / qjl 最后一维随 mse_bits 变化。
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {

namespace {
constexpr int64_t kHeadDim = 128;
} // namespace

static ge::graphStatus InferShapeKvCacheTurboQuant(gert::InferShapeContext *context)
{
    const gert::Shape *kvShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvShape);
    if (kvShape->GetDimNum() != 3 || kvShape->GetDim(2) != kHeadDim) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = kvShape->GetDim(0);
    const int64_t numKvHeads = kvShape->GetDim(1);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *mseBitsPtr = attrs->GetAttrPointer<int64_t>(0);
    const int64_t mseBits = (mseBitsPtr == nullptr) ? 3 : *mseBitsPtr;
    if (mseBits < 2 || mseBits > 4) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *idxShape = context->GetOutputShape(0);
    gert::Shape *qjlShape = context->GetOutputShape(1);
    gert::Shape *normShape = context->GetOutputShape(2);
    gert::Shape *gammaShape = context->GetOutputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, qjlShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, normShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaShape);

    idxShape->SetDimNum(3);
    idxShape->SetDim(0, numTokens);
    idxShape->SetDim(1, numKvHeads);
    idxShape->SetDim(2, (kHeadDim * mseBits + 7) / 8);

    qjlShape->SetDimNum(3);
    qjlShape->SetDim(0, numTokens);
    qjlShape->SetDim(1, numKvHeads);
    qjlShape->SetDim(2, (kHeadDim + 7) / 8);

    normShape->SetDimNum(2);
    normShape->SetDim(0, numTokens);
    normShape->SetDim(1, numKvHeads);

    *gammaShape = *normShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(KvCacheTurboQuant).InferShape(InferShapeKvCacheTurboQuant);

} // namespace ops
