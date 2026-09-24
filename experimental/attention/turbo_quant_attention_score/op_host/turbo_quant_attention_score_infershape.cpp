/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *q = context->GetInputShape(0);
    const auto *lens = context->GetInputShape(7);
    auto *out = context->GetOutputShape(0);
    if (!q || !lens || !out || q->GetDimNum() != 3 || lens->GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    int64_t maxLen = -1;
    const auto *tensor = context->GetInputTensor(7);
    if (tensor && (tensor->GetPlacement() == gert::kOnHost || tensor->GetPlacement() == gert::kFollowing) &&
        tensor->GetData<int32_t>() && lens->GetDim(0) >= 0) {
        maxLen = 0;
        for (int64_t i = 0; i < lens->GetDim(0); ++i) {
            const int64_t length = context->GetInputDesc(7)->GetDataType() == ge::DT_INT32 ?
                                       tensor->GetData<int32_t>()[i] :
                                       tensor->GetData<int64_t>()[i];
            if (length < 0) {
                return ge::GRAPH_FAILED;
            }
            maxLen = std::max(maxLen, static_cast<int64_t>(length));
        }
    }
    *out = gert::Shape({q->GetDim(0), q->GetDim(1), maxLen});
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferType(gert::InferDataTypeContext *context)
{
    return context->SetOutputDataType(0, ge::DT_FLOAT);
}
IMPL_OP_INFERSHAPE(TurboQuantAttentionScore).InferShape(InferShape).InferDataType(InferType).InputsDataDependency({7});
} // namespace ops
