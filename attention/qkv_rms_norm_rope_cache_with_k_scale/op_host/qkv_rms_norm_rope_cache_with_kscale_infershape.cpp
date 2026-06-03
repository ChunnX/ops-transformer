/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
 * \brief Shape inference for QkvRmsNormRopeCacheWithKScale
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

namespace ops {
static constexpr size_t INDEX_IN_QKV = 0;
static constexpr size_t INDEX_IN_Q_GAMMA = 1;
static constexpr size_t INDEX_IN_K_GAMMA = 2;
static constexpr size_t INDEX_IN_COS_SIN = 3;
static constexpr size_t INDEX_IN_SLOT_MAPPING = 4;
static constexpr size_t INDEX_IN_K_CACHE = 5;
static constexpr size_t INDEX_IN_V_CACHE = 6;
static constexpr size_t INDEX_IN_K_SCALE_CACHE = 7;
static constexpr size_t INDEX_IN_CU_SEQ_LEN = 8;
static constexpr size_t INDEX_IN_ROTATION = 9;
static constexpr size_t INDEX_IN_V_SCALE = 10;

static constexpr size_t INDEX_OUT_Q_OUT = 0;
static constexpr size_t INDEX_OUT_Q_SCALE = 1;
static constexpr size_t INDEX_OUT_K_CACHE = 2;
static constexpr size_t INDEX_OUT_V_CACHE = 3;
static constexpr size_t INDEX_OUT_K_SCALE_CACHE = 4;

static constexpr size_t ATTR_HEAD_NUMS = 0;
static constexpr size_t ATTR_EPSILON = 1;

static ge::graphStatus InferShapeQkvRmsNormRopeCacheWithKscale(gert::InferShapeContext *context)
{
    const gert::Shape *qkvShape = context->GetInputShape(INDEX_IN_QKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);

    const gert::Shape *kCacheShape = context->GetInputShape(INDEX_IN_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);

    const gert::Shape *vCacheShape = context->GetInputShape(INDEX_IN_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);

    const gert::Shape *kScaleCacheShape = context->GetInputShape(INDEX_IN_K_SCALE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheShape);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const std::vector<int64_t> *headNums = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_NUMS);
    if (headNums == nullptr || headNums->size() != 3) {
        OP_LOGE("head_nums attribute is invalid, must have exactly 3 elements");
        return ge::GRAPH_FAILED;
    }

    int64_t Nq = (*headNums)[0];
    int64_t Nk = (*headNums)[1];
    int64_t Nv = (*headNums)[2];

    if (Nq <= 0 || Nk <= 0 || Nv <= 0) {
        OP_LOGE("head_nums values must be positive: Nq=%ld, Nk=%ld, Nv=%ld", Nq, Nk, Nv);
        return ge::GRAPH_FAILED;
    }

    if (Nv != Nk) {
        OP_LOGE("First phase only supports Nv == Nk, got Nv=%ld, Nk=%ld", Nv, Nk);
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> qkvDims = qkvShape->GetDims();
    if (qkvDims.size() != 3) {
        OP_LOGE("qkv must be 3D tensor, got %zu dimensions", qkvDims.size());
        return ge::GRAPH_FAILED;
    }

    int64_t T = qkvDims[0];
    int64_t Hqkv = qkvDims[1];
    int64_t D = qkvDims[2];

    if (Hqkv != Nq + Nk + Nv) {
        OP_LOGE("qkv shape[1]=%ld does not match Nq+Nk+Nv=%ld", Hqkv, Nq + Nk + Nv);
        return ge::GRAPH_FAILED;
    }

    if (D % 2 != 0) {
        OP_LOGE("head_dim D=%ld must be even for RoPE", D);
        return ge::GRAPH_FAILED;
    }

    if (D != 128) {
        OP_LOGE("First phase only supports D=128, got D=%ld", D);
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> kCacheDims = kCacheShape->GetDims();
    if (kCacheDims.size() != 4 || kCacheDims[1] != Nk) {
        OP_LOGE("k_cache shape invalid or head dim mismatch");
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> vCacheDims = vCacheShape->GetDims();
    if (vCacheDims.size() != 4 || vCacheDims[1] != Nv) {
        OP_LOGE("v_cache shape invalid or head dim mismatch");
        return ge::GRAPH_FAILED;
    }

    std::vector<int64_t> kScaleCacheDims = kScaleCacheShape->GetDims();
    if (kScaleCacheDims.size() != 4 || kScaleCacheDims[1] != Nk) {
        OP_LOGE("k_scale_cache shape invalid or head dim mismatch");
        return ge::GRAPH_FAILED;
    }

    gert::Shape *qOutShape = context->GetOutputShape(INDEX_OUT_Q_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutShape);
    qOutShape->SetDimNum(3);
    qOutShape->SetDim(0, T);
    qOutShape->SetDim(1, Nq);
    qOutShape->SetDim(2, D);

    gert::Shape *qScaleShape = context->GetOutputShape(INDEX_OUT_Q_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, qScaleShape);
    qScaleShape->SetDimNum(2);
    qScaleShape->SetDim(0, T);
    qScaleShape->SetDim(1, Nq);

    gert::Shape *kCacheOutShape = context->GetOutputShape(INDEX_OUT_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheOutShape);
    *kCacheOutShape = *kCacheShape;

    gert::Shape *vCacheOutShape = context->GetOutputShape(INDEX_OUT_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheOutShape);
    *vCacheOutShape = *vCacheShape;

    gert::Shape *kScaleCacheOutShape = context->GetOutputShape(INDEX_OUT_K_SCALE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheOutShape);
    *kScaleCacheOutShape = *kScaleCacheShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQkvRmsNormRopeCacheWithKscale(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(INDEX_OUT_Q_OUT, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUT_Q_SCALE, ge::DT_FLOAT);
    context->SetOutputDataType(INDEX_OUT_K_CACHE, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUT_V_CACHE, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUT_K_SCALE_CACHE, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QkvRmsNormRopeCacheWithKscale)
    .InferShape(InferShapeQkvRmsNormRopeCacheWithKscale)
    .InferDataType(InferDataTypeQkvRmsNormRopeCacheWithKscale);
} // namespace ops
