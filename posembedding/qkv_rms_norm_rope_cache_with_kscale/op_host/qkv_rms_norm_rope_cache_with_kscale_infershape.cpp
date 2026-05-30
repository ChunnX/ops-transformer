/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t kQkvIndex = 0;
constexpr size_t kKCacheIndex = 5;
constexpr size_t kVCacheIndex = 6;
constexpr size_t kKScaleCacheIndex = 7;
constexpr size_t kHeadNumsAttrIndex = 0;

constexpr size_t kQOutIndex = 0;
constexpr size_t kQScaleIndex = 1;
constexpr size_t kKCacheOutIndex = 2;
constexpr size_t kVCacheOutIndex = 3;
constexpr size_t kKScaleCacheOutIndex = 4;

constexpr int64_t kUnknownDim = -1;
} // namespace

ge::graphStatus InferShape4QkvRmsNormRopeCacheWithKScale(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(kHeadNumsAttrIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    if (headNums->GetSize() != 3) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qkvShape = context->GetInputShape(kQkvIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);
    const gert::Shape *kCacheShape = context->GetInputShape(kKCacheIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);
    const gert::Shape *vCacheShape = context->GetInputShape(kVCacheIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);
    const gert::Shape *kScaleCacheShape = context->GetInputShape(kKScaleCacheIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheShape);

    gert::Shape *qOutShape = context->GetOutputShape(kQOutIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutShape);
    gert::Shape *qScaleShape = context->GetOutputShape(kQScaleIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, qScaleShape);
    gert::Shape *kCacheOutShape = context->GetOutputShape(kKCacheOutIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheOutShape);
    gert::Shape *vCacheOutShape = context->GetOutputShape(kVCacheOutIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheOutShape);
    gert::Shape *kScaleCacheOutShape = context->GetOutputShape(kKScaleCacheOutIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheOutShape);

    const int64_t totalTokens = qkvShape->GetDimNum() >= 1 ? qkvShape->GetDim(0) : kUnknownDim;
    const int64_t qHeads = headNums->GetData()[0];
    const int64_t headDim = qkvShape->GetDimNum() >= 3 ? qkvShape->GetDim(2) : kUnknownDim;

    qOutShape->SetDimNum(3);
    qOutShape->SetDim(0, totalTokens);
    qOutShape->SetDim(1, qHeads);
    qOutShape->SetDim(2, headDim);

    qScaleShape->SetDimNum(2);
    qScaleShape->SetDim(0, totalTokens);
    qScaleShape->SetDim(1, qHeads);

    *kCacheOutShape = *kCacheShape;
    *vCacheOutShape = *vCacheShape;
    *kScaleCacheOutShape = *kScaleCacheShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4QkvRmsNormRopeCacheWithKScale(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(kQOutIndex, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(kQScaleIndex, ge::DT_FLOAT);
    context->SetOutputDataType(kKCacheOutIndex, context->GetInputDataType(kKCacheIndex));
    context->SetOutputDataType(kVCacheOutIndex, context->GetInputDataType(kVCacheIndex));
    context->SetOutputDataType(kKScaleCacheOutIndex, context->GetInputDataType(kKScaleCacheIndex));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QkvRmsNormRopeCacheWithKScale)
    .InferShape(InferShape4QkvRmsNormRopeCacheWithKScale)
    .InferDataType(InferDtype4QkvRmsNormRopeCacheWithKScale);

} // namespace ops
