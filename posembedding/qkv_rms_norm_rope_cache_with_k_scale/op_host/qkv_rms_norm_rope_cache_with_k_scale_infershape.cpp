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
constexpr size_t QKV_INDEX = 0;
constexpr size_t K_CACHE_INDEX = 5;
constexpr size_t V_CACHE_INDEX = 6;
constexpr size_t K_SCALE_CACHE_INDEX = 7;
constexpr size_t HEAD_NUMS_ATTR_INDEX = 0;

constexpr size_t Q_OUT_INDEX = 0;
constexpr size_t Q_SCALE_INDEX = 1;
constexpr size_t K_CACHE_OUT_INDEX = 2;
constexpr size_t V_CACHE_OUT_INDEX = 3;
constexpr size_t K_SCALE_CACHE_OUT_INDEX = 4;

constexpr int64_t UNKNOWN_DIM = -1;
} // namespace

ge::graphStatus InferShape4QkvRmsNormRopeCacheWithKScale(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(HEAD_NUMS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    if (headNums->GetSize() != 3) {
        return ge::GRAPH_FAILED;
    }
    const auto *headNumsData = headNums->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, headNumsData);
    for (size_t i = 0; i < 3; ++i) {
        if (headNumsData[i] <= 0) {
            OP_LOGE(context->GetNodeName(), "head_nums[%zu] must be greater than 0.", i);
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Shape *qkvShape = context->GetInputShape(QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);
    const gert::Shape *kCacheShape = context->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);
    const gert::Shape *vCacheShape = context->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);
    const gert::Shape *kScaleCacheShape = context->GetInputShape(K_SCALE_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheShape);

    gert::Shape *qOutShape = context->GetOutputShape(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutShape);
    gert::Shape *qScaleShape = context->GetOutputShape(Q_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qScaleShape);
    gert::Shape *kCacheOutShape = context->GetOutputShape(K_CACHE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheOutShape);
    gert::Shape *vCacheOutShape = context->GetOutputShape(V_CACHE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheOutShape);
    gert::Shape *kScaleCacheOutShape = context->GetOutputShape(K_SCALE_CACHE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheOutShape);

    const int64_t totalTokens = qkvShape->GetDimNum() >= 2 ? qkvShape->GetDim(1) : UNKNOWN_DIM;
    const int64_t qHeads = headNumsData[0];
    const int64_t headDim = qkvShape->GetDimNum() >= 3 ? qkvShape->GetDim(2) : UNKNOWN_DIM;

    qOutShape->SetDimNum(3);
    qOutShape->SetDim(0, qHeads);
    qOutShape->SetDim(1, totalTokens);
    qOutShape->SetDim(2, headDim);

    qScaleShape->SetDimNum(2);
    qScaleShape->SetDim(0, qHeads);
    qScaleShape->SetDim(1, totalTokens);

    *kCacheOutShape = *kCacheShape;
    *vCacheOutShape = *vCacheShape;
    *kScaleCacheOutShape = *kScaleCacheShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDtype4QkvRmsNormRopeCacheWithKScale(gert::InferDataTypeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(Q_OUT_INDEX, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(Q_SCALE_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(K_CACHE_OUT_INDEX, context->GetInputDataType(K_CACHE_INDEX));
    context->SetOutputDataType(V_CACHE_OUT_INDEX, context->GetInputDataType(V_CACHE_INDEX));
    context->SetOutputDataType(K_SCALE_CACHE_OUT_INDEX, context->GetInputDataType(K_SCALE_CACHE_INDEX));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QkvRmsNormRopeCacheWithKScale)
    .InferShape(InferShape4QkvRmsNormRopeCacheWithKScale)
    .InferDataType(InferDtype4QkvRmsNormRopeCacheWithKScale);

} // namespace ops
