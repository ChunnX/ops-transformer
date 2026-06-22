/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr uint64_t QKV_INDEX = 0;
constexpr uint64_t K_CACHE_INDEX = 5;
constexpr uint64_t V_CACHE_INDEX = 6;
constexpr uint64_t K_SCALE_CACHE_INDEX = 7;
constexpr uint64_t CU_SEQ_LEN_INDEX = 8;
constexpr uint64_t ACTUAL_SEQ_LENS_INDEX = 9;
constexpr uint64_t HEAD_NUMS_ATTR_INDEX = 0;

constexpr uint64_t Q_OUT_INDEX = 0;
constexpr uint64_t Q_SCALE_INDEX = 1;
constexpr uint64_t K_CACHE_OUT_INDEX = 2;
constexpr uint64_t V_CACHE_OUT_INDEX = 3;
constexpr uint64_t K_SCALE_CACHE_OUT_INDEX = 4;

constexpr uint64_t HEAD_NUMS_SIZE = 3;
constexpr uint64_t QKV_DIM_NUM = 3;
constexpr uint64_t CACHE_DIM_NUM = 4;
constexpr uint64_t SEQ_LEN_DIM_NUM = 1;

ge::graphStatus ValidateDimNum(gert::InferShapeContext *context, const char *tensorName, const gert::Shape *shape,
                               uint64_t expectedDimNum)
{
    const uint64_t actualDimNum = static_cast<uint64_t>(shape->GetDimNum());
    if (actualDimNum != expectedDimNum) {
        OP_LOGE(context, "%s dim num must be %llu, but got %llu.", tensorName, expectedDimNum, actualDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidatePositiveDims(gert::InferShapeContext *context, const char *tensorName, const gert::Shape *shape)
{
    const uint64_t dimNum = static_cast<uint64_t>(shape->GetDimNum());
    for (uint64_t i = 0; i < dimNum; ++i) {
        if (shape->GetDim(i) <= 0) {
            OP_LOGE(context, "%s dim[%llu] must be greater than 0, but got %ld.", tensorName, i,
                    shape->GetDim(i));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

ge::graphStatus InferShape4QkvRmsNormRopeCacheWithKScale(gert::InferShapeContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto headNums = attrs->GetListInt(HEAD_NUMS_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, headNums);
    const uint64_t headNumsSize = static_cast<uint64_t>(headNums->GetSize());
    if (headNumsSize != HEAD_NUMS_SIZE) {
        OP_LOGE(context, "head_nums size must be %llu, but got %llu.", HEAD_NUMS_SIZE, headNumsSize);
        return ge::GRAPH_FAILED;
    }
    const auto *headNumsData = headNums->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, headNumsData);
    for (uint64_t i = 0; i < HEAD_NUMS_SIZE; ++i) {
        if (headNumsData[i] <= 0) {
            OP_LOGE(context, "head_nums[%llu] must be greater than 0.", i);
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
    const gert::Shape *cuSeqLenShape = context->GetInputShape(CU_SEQ_LEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqLenShape);
    const gert::Shape *actualSeqLensShape = context->GetInputShape(ACTUAL_SEQ_LENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualSeqLensShape);

    if (ValidateDimNum(context, "qkv", qkvShape, QKV_DIM_NUM) != ge::GRAPH_SUCCESS ||
        ValidateDimNum(context, "k_cache", kCacheShape, CACHE_DIM_NUM) != ge::GRAPH_SUCCESS ||
        ValidateDimNum(context, "v_cache", vCacheShape, CACHE_DIM_NUM) != ge::GRAPH_SUCCESS ||
        ValidateDimNum(context, "k_scale_cache", kScaleCacheShape, CACHE_DIM_NUM) != ge::GRAPH_SUCCESS ||
        ValidateDimNum(context, "cu_seq_len", cuSeqLenShape, SEQ_LEN_DIM_NUM) != ge::GRAPH_SUCCESS ||
        ValidateDimNum(context, "actual_seq_lens", actualSeqLensShape, SEQ_LEN_DIM_NUM) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidatePositiveDims(context, "qkv", qkvShape) != ge::GRAPH_SUCCESS ||
        ValidatePositiveDims(context, "k_cache", kCacheShape) != ge::GRAPH_SUCCESS ||
        ValidatePositiveDims(context, "v_cache", vCacheShape) != ge::GRAPH_SUCCESS ||
        ValidatePositiveDims(context, "k_scale_cache", kScaleCacheShape) != ge::GRAPH_SUCCESS ||
        ValidatePositiveDims(context, "cu_seq_len", cuSeqLenShape) != ge::GRAPH_SUCCESS ||
        ValidatePositiveDims(context, "actual_seq_lens", actualSeqLensShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (cuSeqLenShape->GetDim(0) < 2 || actualSeqLensShape->GetDim(0) != cuSeqLenShape->GetDim(0) - 1) {
        OP_LOGE(context, "actual_seq_lens shape must be [cu_seq_len dim0 - 1], got actual=%ld cu_seq_len=%ld.",
                actualSeqLensShape->GetDim(0), cuSeqLenShape->GetDim(0));
        return ge::GRAPH_FAILED;
    }

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

    const int64_t totalTokens = qkvShape->GetDim(1);
    const int64_t qHeads = headNumsData[0];
    const int64_t headDim = qkvShape->GetDim(2);

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
    // q_out is always the quantized FP8 query output of this op, independent of the qkv input dtype.
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
