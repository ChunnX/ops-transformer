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
 * \file und_gen_qkv_rms_norm_rope_cache_infershape.cpp
 * \brief 只做输出 shape/dtype 推导；dtype、shape 合法性与支持范围的完整校验统一在 tiling 侧
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
using namespace ge;
namespace ops {

constexpr size_t IN_UND_QKV = 0;
constexpr size_t IN_UND_WEIGHTS_Q = 1;
constexpr size_t IN_UND_WEIGHTS_K = 2;
constexpr size_t IN_COS_SIN_CACHE = 3;
constexpr size_t IN_K_CACHE = 4;
constexpr size_t IN_V_CACHE = 5;
constexpr size_t IN_GEN_QKV = 8;
constexpr size_t IN_GEN_WEIGHTS_Q = 9;
constexpr size_t IN_GEN_WEIGHTS_K = 10;

constexpr size_t OUT_Q = 0;
constexpr size_t OUT_K_CACHE = 1;
constexpr size_t OUT_V_CACHE = 2;

constexpr size_t ATTR_NUM_HEADS_Q = 0;

constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_NUM_ONE = 1;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

// D（headDim）在多个输入里都出现，任取一个能拿到具体值的即可：
// und_qkv/gen_qkv 的 [.., .., D]、四个 RMSNorm 权重的 [D]、cos_sin_cache 的 [max_pos, D]、
// k_cache/v_cache 的 [Bn, Bs, H, D]。按“必选输入优先”的顺序尝试，先命中先用。
struct HeadDimSource {
    size_t inputIdx;
    size_t dimIdx;
    size_t expectDimNum;
    bool optional;
};

constexpr HeadDimSource HEAD_DIM_SOURCES[] = {
    {IN_UND_QKV, DIM_TWO, DIM_NUM_THREE, false},
    {IN_COS_SIN_CACHE, DIM_ONE, DIM_NUM_TWO, false},
    {IN_UND_WEIGHTS_Q, DIM_ZERO, DIM_NUM_ONE, false},
    {IN_UND_WEIGHTS_K, DIM_ZERO, DIM_NUM_ONE, false},
    {IN_K_CACHE, DIM_THREE, DIM_NUM_FOUR, false},
    {IN_V_CACHE, DIM_THREE, DIM_NUM_FOUR, false},
    {IN_GEN_QKV, DIM_TWO, DIM_NUM_THREE, true},
    {IN_GEN_WEIGHTS_Q, DIM_ZERO, DIM_NUM_ONE, true},
    {IN_GEN_WEIGHTS_K, DIM_ZERO, DIM_NUM_ONE, true},
};

// 取某个输入的指定维；输入缺省、未知 rank、rank 不符或该维本身是 -1 时返回 -1
inline int64_t TryGetDim(gert::InferShapeContext* context, const HeadDimSource& src)
{
    const gert::Shape* shape =
        src.optional ? context->GetOptionalInputShape(src.inputIdx) : context->GetInputShape(src.inputIdx);
    if (shape == nullptr || Ops::Base::IsUnknownRank(*shape) || shape->GetDimNum() != src.expectDimNum) {
        return UNKNOWN_DIM_VALUE;
    }
    return shape->GetDim(src.dimIdx);
}

// 取 T = und_len + gen_len；任一段拿不到具体值时整体不可知
inline int64_t InferTotalTokens(const gert::Shape* undQkvShape, const gert::Shape* genQkvShape)
{
    if (Ops::Base::IsUnknownRank(*undQkvShape)) {
        return UNKNOWN_DIM_VALUE;
    }
    const int64_t undLen = undQkvShape->GetDim(DIM_ZERO);
    if (undLen == UNKNOWN_DIM_VALUE) {
        return UNKNOWN_DIM_VALUE;
    }
    if (genQkvShape == nullptr) {
        return undLen;
    }
    if (Ops::Base::IsUnknownRank(*genQkvShape)) {
        return UNKNOWN_DIM_VALUE;
    }
    const int64_t genLen = genQkvShape->GetDim(DIM_ZERO);
    return genLen == UNKNOWN_DIM_VALUE ? UNKNOWN_DIM_VALUE : undLen + genLen;
}

graphStatus InferShape4UndGenQkvRmsNormRopeCache(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4UndGenQkvRmsNormRopeCache.");

    const gert::Shape* undQkvShape = context->GetInputShape(IN_UND_QKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, undQkvShape);
    const gert::Shape* kCacheShape = context->GetInputShape(IN_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheShape);
    const gert::Shape* vCacheShape = context->GetInputShape(IN_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheShape);
    const gert::Shape* genQkvShape = context->GetOptionalInputShape(IN_GEN_QKV);

    gert::Shape* qOutShape = context->GetOutputShape(OUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qOutShape);
    gert::Shape* kCacheOutShape = context->GetOutputShape(OUT_K_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, kCacheOutShape);
    gert::Shape* vCacheOutShape = context->GetOutputShape(OUT_V_CACHE);
    OP_CHECK_NULL_WITH_CONTEXT(context, vCacheOutShape);

    // 非未知 rank 时 und_qkv/gen_qkv 必须是 3D，这是 T 的推导前提；其余合法性校验在 tiling
    OP_CHECK_IF(!Ops::Base::IsUnknownRank(*undQkvShape) && undQkvShape->GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(context, "und_qkv must be 3D tensor [und_len, N, D]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(genQkvShape != nullptr && !Ops::Base::IsUnknownRank(*genQkvShape) &&
                    genQkvShape->GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(context, "gen_qkv must be 3D tensor [gen_len, N, D]."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numHeadsQ = attrs->GetInt(ATTR_NUM_HEADS_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsQ);

    // q 恒为 3D [T, Hq, D]：Hq 直接来自属性，任何场景都可知；T、D 先按不可知铺底，再逐个尝试推导
    qOutShape->SetDimNum(DIM_NUM_THREE);
    qOutShape->SetDim(DIM_ZERO, UNKNOWN_DIM_VALUE);
    qOutShape->SetDim(DIM_ONE, *numHeadsQ);
    qOutShape->SetDim(DIM_TWO, UNKNOWN_DIM_VALUE);

    // T = und_len + gen_len
    qOutShape->SetDim(DIM_ZERO, InferTotalTokens(undQkvShape, genQkvShape));

    // D 从任一能给出具体值的输入取
    for (const auto& src : HEAD_DIM_SOURCES) {
        const int64_t headDim = TryGetDim(context, src);
        if (headDim != UNKNOWN_DIM_VALUE) {
            qOutShape->SetDim(DIM_TWO, headDim);
            break;
        }
    }

    // k_cache/v_cache 原地写入，shape 透传（输入为 -2 时输出同样是 -2）
    *kCacheOutShape = *kCacheShape;
    *vCacheOutShape = *vCacheShape;

    OP_LOGD(context, "End to do InferShape4UndGenQkvRmsNormRopeCache.");
    return GRAPH_SUCCESS;
}

graphStatus InferDtype4UndGenQkvRmsNormRopeCache(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "InferDtype4UndGenQkvRmsNormRopeCache enter");

    // dtype 合法性在 tiling 的 CheckDtypeValid 中校验，这里只做透传
    context->SetOutputDataType(OUT_Q, context->GetInputDataType(IN_UND_QKV));
    context->SetOutputDataType(OUT_K_CACHE, context->GetInputDataType(IN_K_CACHE));
    context->SetOutputDataType(OUT_V_CACHE, context->GetInputDataType(IN_V_CACHE));

    OP_LOGD(context, "InferDtype4UndGenQkvRmsNormRopeCache end");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UndGenQkvRmsNormRopeCache)
    .InferShape(InferShape4UndGenQkvRmsNormRopeCache)
    .InferDataType(InferDtype4UndGenQkvRmsNormRopeCache);
} // namespace ops
