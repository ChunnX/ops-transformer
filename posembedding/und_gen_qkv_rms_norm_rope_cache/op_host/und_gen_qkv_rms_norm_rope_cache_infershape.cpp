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
constexpr size_t IN_SLOT_MAPPING = 6;
constexpr size_t IN_POSITIONS = 7;
constexpr size_t IN_GEN_QKV = 8;
constexpr size_t IN_GEN_WEIGHTS_Q = 9;
constexpr size_t IN_GEN_WEIGHTS_K = 10;
constexpr size_t IN_CAT_INDICES = 11;

constexpr size_t OUT_Q = 0;
constexpr size_t OUT_K_CACHE = 1;
constexpr size_t OUT_V_CACHE = 2;

constexpr size_t ATTR_NUM_HEADS_Q = 0;
constexpr size_t ATTR_NUM_HEADS_K = 1;
constexpr size_t ATTR_NUM_HEADS_V = 2;

constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_THREE = 3;
constexpr size_t DIM_NUM_ONE = 1;
constexpr size_t DIM_NUM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;
constexpr size_t DIM_NUM_FOUR = 4;
constexpr int64_t UNKNOWN_DIM_VALUE = -1LL;

// 某个维度的一个候选来源：从第 inputIdx 个输入（rank 须为 expectDimNum）的第 dimIdx 维取值
struct DimSource {
    size_t inputIdx;
    size_t dimIdx;
    size_t expectDimNum;
    bool optional;
};

// D（headDim）在 9 个输入里都出现：und_qkv/gen_qkv 的 [.., .., D]、四个 RMSNorm 权重的 [D]、
// cos_sin_cache 的 [max_pos, D]、k_cache/v_cache 的 [Bn, Bs, H, D]
constexpr DimSource HEAD_DIM_SOURCES[] = {
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

// T 除了 und_len + gen_len 相加，还能从三个按 token 组织的输入直接读出
constexpr DimSource TOTAL_TOKEN_SOURCES[] = {
    {IN_POSITIONS, DIM_ONE, DIM_NUM_TWO, false},     // [3, T]
    {IN_SLOT_MAPPING, DIM_ZERO, DIM_NUM_ONE, false}, // [T]
    {IN_CAT_INDICES, DIM_ZERO, DIM_NUM_ONE, true},   // [T]
};

// Bn/Bs 是 k_cache 与 v_cache 共有的前两维（约束要求两者一致），任一可用即可
constexpr DimSource BLOCK_NUM_SOURCES[] = {
    {IN_K_CACHE, DIM_ZERO, DIM_NUM_FOUR, false},
    {IN_V_CACHE, DIM_ZERO, DIM_NUM_FOUR, false},
};

constexpr DimSource BLOCK_SIZE_SOURCES[] = {
    {IN_K_CACHE, DIM_ONE, DIM_NUM_FOUR, false},
    {IN_V_CACHE, DIM_ONE, DIM_NUM_FOUR, false},
};

// 取某个输入的指定维；输入缺省、未知 rank、rank 不符或该维本身是 -1 时返回 -1
inline int64_t TryGetDim(gert::InferShapeContext* context, const DimSource& src)
{
    const gert::Shape* shape =
        src.optional ? context->GetOptionalInputShape(src.inputIdx) : context->GetInputShape(src.inputIdx);
    if (shape == nullptr || Ops::Base::IsUnknownRank(*shape) || shape->GetDimNum() != src.expectDimNum) {
        return UNKNOWN_DIM_VALUE;
    }
    return shape->GetDim(src.dimIdx);
}

// 按给定顺序逐个来源尝试，先拿到具体值就返回；全都拿不到才返回 -1
template <size_t N>
inline int64_t InferDimFromSources(gert::InferShapeContext* context, const DimSource (&sources)[N])
{
    for (const auto& src : sources) {
        const int64_t dim = TryGetDim(context, src);
        if (dim != UNKNOWN_DIM_VALUE) {
            return dim;
        }
    }
    return UNKNOWN_DIM_VALUE;
}

// T = und_len + gen_len；任一段拿不到具体值时返回 -1，交由调用方回退到其他来源
inline int64_t InferTotalTokensByAdd(const gert::Shape* undQkvShape, const gert::Shape* genQkvShape)
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

// KV Cache 输出与输入同址，shape 为 [Bn, Bs, H, D]：H 由属性给出恒可知，Bn/Bs/D 能推则推
inline void SetCacheShape(gert::Shape* cacheShape, int64_t blockNum, int64_t blockSize, int64_t numHeads,
                          int64_t headDim)
{
    cacheShape->SetDimNum(DIM_NUM_FOUR);
    cacheShape->SetDim(DIM_ZERO, blockNum);
    cacheShape->SetDim(DIM_ONE, blockSize);
    cacheShape->SetDim(DIM_TWO, numHeads);
    cacheShape->SetDim(DIM_THREE, headDim);
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
    const int64_t* numHeadsK = attrs->GetInt(ATTR_NUM_HEADS_K);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsK);
    const int64_t* numHeadsV = attrs->GetInt(ATTR_NUM_HEADS_V);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsV);

    // 三个输出的 rank 恒定、头数恒由属性给出，其余维度能推则推、推不出才置 -1

    // T：优先按定义 und_len + gen_len 相加；相加不成立时退到按 token 组织的输入直接读
    int64_t total = InferTotalTokensByAdd(undQkvShape, genQkvShape);
    if (total == UNKNOWN_DIM_VALUE) {
        total = InferDimFromSources(context, TOTAL_TOKEN_SOURCES);
    }
    // D：9 个输入都带 D，取第一个能给出具体值的
    const int64_t headDim = InferDimFromSources(context, HEAD_DIM_SOURCES);

    // q: [T, Hq, D]
    qOutShape->SetDimNum(DIM_NUM_THREE);
    qOutShape->SetDim(DIM_ZERO, total);
    qOutShape->SetDim(DIM_ONE, *numHeadsQ);
    qOutShape->SetDim(DIM_TWO, headDim);

    // k_cache/v_cache 原地写入，输出与输入同址：[Bn, Bs, Hk/Hv, D]
    const int64_t blockNum = InferDimFromSources(context, BLOCK_NUM_SOURCES);
    const int64_t blockSize = InferDimFromSources(context, BLOCK_SIZE_SOURCES);
    SetCacheShape(kCacheOutShape, blockNum, blockSize, *numHeadsK, headDim);
    SetCacheShape(vCacheOutShape, blockNum, blockSize, *numHeadsV, headDim);

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
