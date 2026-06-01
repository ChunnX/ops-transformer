/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_KEY_CACHE_IDX = 2;
constexpr size_t INPUT_VALUE_CACHE_IDX = 3;
constexpr size_t INPUT_KEY_SCALE_CACHE_IDX = 6;
constexpr size_t OUTPUT_KEY_CACHE_IDX = 0;
constexpr size_t OUTPUT_VALUE_CACHE_IDX = 1;
constexpr size_t OUTPUT_KEY_SCALE_CACHE_IDX = 2;

static ge::graphStatus InferShapeScatterKvCacheWithKScale(gert::InferShapeContext *context)
{
    const gert::Shape *kcShape = context->GetInputShape(INPUT_KEY_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kcShape);
    gert::Shape *outKcShape = context->GetOutputShape(OUTPUT_KEY_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outKcShape);
    outKcShape->SetDimNum(kcShape->GetDimNum());
    for (size_t i = 0; i < kcShape->GetDimNum(); i++) {
        outKcShape->SetDim(i, kcShape->GetDim(i));
    }

    const gert::Shape *vcShape = context->GetInputShape(INPUT_VALUE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vcShape);
    gert::Shape *outVcShape = context->GetOutputShape(OUTPUT_VALUE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outVcShape);
    outVcShape->SetDimNum(vcShape->GetDimNum());
    for (size_t i = 0; i < vcShape->GetDimNum(); i++) {
        outVcShape->SetDim(i, vcShape->GetDim(i));
    }

    const gert::Shape *kscShape = context->GetInputShape(INPUT_KEY_SCALE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kscShape);
    gert::Shape *outKscShape = context->GetOutputShape(OUTPUT_KEY_SCALE_CACHE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outKscShape);
    outKscShape->SetDimNum(kscShape->GetDimNum());
    for (size_t i = 0; i < kscShape->GetDimNum(); i++) {
        outKscShape->SetDim(i, kscShape->GetDim(i));
    }

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterKvCacheWithKScale).InferShape(InferShapeScatterKvCacheWithKScale);
} // namespace ops
