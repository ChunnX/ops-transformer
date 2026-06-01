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
static ge::graphStatus InferShapeScatterKvCacheWithKScale(gert::InferShapeContext* context)
{
    // 本算子为 inplace 算子，输出 shape 与对应输入完全一致
    // 输入索引: 0=key, 1=value, 2=key_cache, 3=value_cache,
    //           4=slot_mapping, 5=key_scale, 6=key_scale_cache
    // 输出索引: 0=key_cache(out), 1=value_cache(out), 2=k_scale_cache(out)

    // 输出 0 (key_cache): shape = 输入 2 (key_cache) 的 shape
    const gert::Shape* kcShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, kcShape);
    gert::Shape* outKcShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outKcShape);
    outKcShape->SetDimNum(kcShape->GetDimNum());
    for (size_t i = 0; i < kcShape->GetDimNum(); i++) {
        outKcShape->SetDim(i, kcShape->GetDim(i));
    }

    // 输出 1 (value_cache): shape = 输入 3 (value_cache) 的 shape
    const gert::Shape* vcShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, vcShape);
    gert::Shape* outVcShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outVcShape);
    outVcShape->SetDimNum(vcShape->GetDimNum());
    for (size_t i = 0; i < vcShape->GetDimNum(); i++) {
        outVcShape->SetDim(i, vcShape->GetDim(i));
    }

    // 输出 2 (k_scale_cache): shape = 输入 6 (key_scale_cache) 的 shape
    const gert::Shape* kscShape = context->GetInputShape(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, kscShape);
    gert::Shape* outKscShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outKscShape);
    outKscShape->SetDimNum(kscShape->GetDimNum());
    for (size_t i = 0; i < kscShape->GetDimNum(); i++) {
        outKscShape->SetDim(i, kscShape->GetDim(i));
    }

    // Dtype 推导（inplace 算子，输出 dtype 与输入一致，框架自动处理）
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterKvCacheWithKScale).InferShape(InferShapeScatterKvCacheWithKScale);
} // namespace ops
