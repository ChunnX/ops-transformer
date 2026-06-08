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
 * \file qkv_rms_norm_rope_cache_with_kscale_infershape.cpp
 * \brief Shape inference for QkvRmsNormRopeCacheWithKScale
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

#include <vector>

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
static constexpr size_t INDEX_IN_Q_ROTATION = 9;
static constexpr size_t INDEX_IN_K_ROTATION = 10;
static constexpr size_t INDEX_IN_V_SCALE = 11;

static constexpr size_t INDEX_OUT_Q_OUT = 0;
static constexpr size_t INDEX_OUT_Q_SCALE = 1;
static constexpr size_t INDEX_OUT_K_CACHE = 2;
static constexpr size_t INDEX_OUT_V_CACHE = 3;
static constexpr size_t INDEX_OUT_K_SCALE_CACHE = 4;

static constexpr size_t ATTR_HEAD_NUMS = 0;

static bool IsShape(const gert::Shape *shape, const std::vector<int64_t> &expectedShape)
{
    if (shape == nullptr || shape->GetDimNum() != expectedShape.size()) {
        return false;
    }
    for (size_t i = 0; i < expectedShape.size(); ++i) {
        if (shape->GetDim(i) != expectedShape[i]) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus InferShapeQkvRmsNormRopeCacheWithKScale(gert::InferShapeContext *context)
{
    const gert::Shape *qkvShape = context->GetInputShape(INDEX_IN_QKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);

    const gert::Shape *qGammaShape = context->GetInputShape(INDEX_IN_Q_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, qGammaShape);

    const gert::Shape *kGammaShape = context->GetInputShape(INDEX_IN_K_GAMMA);
    OP_CHECK_NULL_WITH_CONTEXT(context, kGammaShape);

    const gert::Shape *cosSinShape = context->GetInputShape(INDEX_IN_COS_SIN);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosSinShape);

    const gert::Shape *slotMappingShape = context->GetInputShape(INDEX_IN_SLOT_MAPPING);
    OP_CHECK_NULL_WITH_CONTEXT(context, slotMappingShape);

    const gert::Shape *kCacheShape = context->GetInputShape(INDEX_IN_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);

    const gert::Shape *vCacheShape = context->GetInputShape(INDEX_IN_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);

    const gert::Shape *kScaleCacheShape = context->GetInputShape(INDEX_IN_K_SCALE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheShape);

    const gert::Shape *cuSeqLenShape = context->GetInputShape(INDEX_IN_CU_SEQ_LEN);
    OP_CHECK_NULL_WITH_CONTEXT(context, cuSeqLenShape);

    const gert::Shape *qRotationShape = context->GetInputShape(INDEX_IN_Q_ROTATION);
    OP_CHECK_NULL_WITH_CONTEXT(context, qRotationShape);

    const gert::Shape *kRotationShape = context->GetInputShape(INDEX_IN_K_ROTATION);
    OP_CHECK_NULL_WITH_CONTEXT(context, kRotationShape);

    const gert::Shape *vScaleShape = context->GetInputShape(INDEX_IN_V_SCALE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vScaleShape);

    const auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto *headNums = attrs->GetListInt(ATTR_HEAD_NUMS);
    if (headNums == nullptr || headNums->GetSize() != 3) {
        OP_LOGE(context->GetNodeName(), "head_nums attribute is invalid, must have exactly 3 elements");
        return ge::GRAPH_FAILED;
    }
    const int64_t *headNumsData = headNums->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context, headNumsData);

    int64_t Nq = headNumsData[0];
    int64_t Nk = headNumsData[1];
    int64_t Nv = headNumsData[2];

    if (Nq <= 0 || Nk <= 0 || Nv <= 0) {
        OP_LOGE(context->GetNodeName(), "head_nums values must be positive: Nq=%ld, Nk=%ld, Nv=%ld", Nq, Nk, Nv);
        return ge::GRAPH_FAILED;
    }

    if (Nv != Nk) {
        OP_LOGE(context->GetNodeName(), "First phase only supports Nv == Nk, got Nv=%ld, Nk=%ld", Nv, Nk);
        return ge::GRAPH_FAILED;
    }

    if (qkvShape->GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "qkv must be 3D tensor, got %zu dimensions", qkvShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t T = qkvShape->GetDim(0);
    int64_t Hqkv = qkvShape->GetDim(1);
    int64_t D = qkvShape->GetDim(2);

    if (Hqkv != Nq + Nk + Nv) {
        OP_LOGE(context->GetNodeName(), "qkv shape[1]=%ld does not match Nq+Nk+Nv=%ld", Hqkv, Nq + Nk + Nv);
        return ge::GRAPH_FAILED;
    }

    if (D % 2 != 0) {
        OP_LOGE(context->GetNodeName(), "head_dim D=%ld must be even for RoPE", D);
        return ge::GRAPH_FAILED;
    }

    if (D != 128) {
        OP_LOGE(context->GetNodeName(), "First phase only supports D=128, got D=%ld", D);
        return ge::GRAPH_FAILED;
    }

    if (!IsShape(qGammaShape, {D}) || !IsShape(kGammaShape, {D})) {
        OP_LOGE(context->GetNodeName(), "q_gamma and k_gamma must have shape [D]");
        return ge::GRAPH_FAILED;
    }

    if (cosSinShape->GetDimNum() != 2 || cosSinShape->GetDim(1) != D) {
        OP_LOGE(context->GetNodeName(), "cos_sin must have shape [MaxSeqLen, D]");
        return ge::GRAPH_FAILED;
    }

    if (!IsShape(slotMappingShape, {T})) {
        OP_LOGE(context->GetNodeName(), "slot_mapping must have shape [T]");
        return ge::GRAPH_FAILED;
    }

    if (cuSeqLenShape->GetDimNum() != 1 || cuSeqLenShape->GetDim(0) < 2) {
        OP_LOGE(context->GetNodeName(), "cu_seq_len must be 1D tensor with length >= 2");
        return ge::GRAPH_FAILED;
    }

    if (!IsShape(qRotationShape, {D, D}) || !IsShape(kRotationShape, {D, D})) {
        OP_LOGE(context->GetNodeName(), "q_rotation and k_rotation must have shape [D, D]");
        return ge::GRAPH_FAILED;
    }

    if (!IsShape(vScaleShape, {Nv})) {
        OP_LOGE(context->GetNodeName(), "v_scale must have shape [Nv]");
        return ge::GRAPH_FAILED;
    }

    if (kCacheShape->GetDimNum() != 4 || kCacheShape->GetDim(1) != Nk || kCacheShape->GetDim(3) != D) {
        OP_LOGE(context->GetNodeName(), "k_cache shape invalid or head dim mismatch");
        return ge::GRAPH_FAILED;
    }

    if (vCacheShape->GetDimNum() != 4 || vCacheShape->GetDim(1) != Nv || vCacheShape->GetDim(3) != D) {
        OP_LOGE(context->GetNodeName(), "v_cache shape invalid or head dim mismatch");
        return ge::GRAPH_FAILED;
    }

    if (kScaleCacheShape->GetDimNum() != 4 || kScaleCacheShape->GetDim(1) != Nk ||
        kScaleCacheShape->GetDim(3) != 1) {
        OP_LOGE(context->GetNodeName(), "k_scale_cache shape invalid or head dim mismatch");
        return ge::GRAPH_FAILED;
    }

    if (kCacheShape->GetDim(0) != vCacheShape->GetDim(0) || kCacheShape->GetDim(2) != vCacheShape->GetDim(2) ||
        kCacheShape->GetDim(0) != kScaleCacheShape->GetDim(0) ||
        kCacheShape->GetDim(2) != kScaleCacheShape->GetDim(2)) {
        OP_LOGE(context->GetNodeName(), "cache BlockNum/BlockSize dimensions must be consistent");
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

    gert::Shape *kCacheOutputShape = context->GetOutputShape(INDEX_OUT_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheOutputShape);
    *kCacheOutputShape = *kCacheShape;

    gert::Shape *vCacheOutputShape = context->GetOutputShape(INDEX_OUT_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheOutputShape);
    *vCacheOutputShape = *vCacheShape;

    gert::Shape *kScaleCacheOutputShape = context->GetOutputShape(INDEX_OUT_K_SCALE_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kScaleCacheOutputShape);
    *kScaleCacheOutputShape = *kScaleCacheShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQkvRmsNormRopeCacheWithKScale(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(INDEX_OUT_Q_OUT, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUT_Q_SCALE, ge::DT_FLOAT);
    context->SetOutputDataType(INDEX_OUT_K_CACHE, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUT_V_CACHE, ge::DT_FLOAT8_E4M3FN);
    context->SetOutputDataType(INDEX_OUT_K_SCALE_CACHE, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QkvRmsNormRopeCacheWithKScale)
    .InferShape(InferShapeQkvRmsNormRopeCacheWithKScale)
    .InferDataType(InferDataTypeQkvRmsNormRopeCacheWithKScale);
} // namespace ops
