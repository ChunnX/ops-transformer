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
#include "flash_mla_shape.h"
namespace ops {
ge::graphStatus InferShapeFlashMla(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetAttrs() == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetOutputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto attrs = context->GetAttrs();
    auto heads = attrs->GetAttrPointer<int64_t>(0);
    auto layout = attrs->GetAttrPointer<char>(4);
    auto lseFlag = attrs->GetAttrPointer<bool>(9);
    std::string qLayout, outLayout;
    FlashMla::MlaAxes q;
    if (heads == nullptr || lseFlag == nullptr || !FlashMla::SplitMlaLayout(layout, qLayout, outLayout) ||
        !FlashMla::GetMlaAxes(*context->GetInputShape(0), qLayout, *heads, q)) {
        return ge::GRAPH_FAILED;
    }
    auto output = FlashMla::MlaOutputShape(q, outLayout);
    auto outShape = context->GetOutputShape(0);
    outShape->SetDimNum(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        outShape->SetDim(i, output[i]);
    }
    auto lse = context->GetOutputShape(1);
    if (!*lseFlag) {
        *lse = {0};
    } else if (qLayout == "TND") {
        *lse = {q.seq, q.heads, 1};
    } else {
        *lse = {q.batch, q.heads, q.seq, 1};
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferDataTypeFlashMla(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto type = context->GetInputDataType(0);
    if (context->SetOutputDataType(0, type == ge::DT_INT8 ? ge::DT_BF16 : type) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(1, ge::DT_FLOAT);
}
IMPL_OP_INFERSHAPE(FlashMla).InferShape(InferShapeFlashMla).InferDataType(InferDataTypeFlashMla);
} // namespace ops
