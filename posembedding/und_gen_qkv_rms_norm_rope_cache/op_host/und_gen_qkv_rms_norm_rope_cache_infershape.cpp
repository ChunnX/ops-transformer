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
constexpr size_t IN_K_CACHE = 4;
constexpr size_t IN_V_CACHE = 5;
constexpr size_t IN_GEN_QKV = 8;

constexpr size_t OUT_Q = 0;
constexpr size_t OUT_K_CACHE = 1;
constexpr size_t OUT_V_CACHE = 2;

constexpr size_t ATTR_NUM_HEADS_Q = 0;

constexpr size_t DIM_ZERO = 0;
constexpr size_t DIM_ONE = 1;
constexpr size_t DIM_TWO = 2;
constexpr size_t DIM_NUM_THREE = 3;

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

    // 推导 q 需要 und_qkv 是 [und_len, N, D]，这条是推导前提，其余合法性校验在 tiling
    OP_CHECK_IF(undQkvShape->GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(context, "und_qkv must be 3D tensor [und_len, N, D]."), return ge::GRAPH_FAILED);
    int64_t total = undQkvShape->GetDim(DIM_ZERO);
    if (genQkvShape != nullptr) {
        OP_CHECK_IF(genQkvShape->GetDimNum() != DIM_NUM_THREE,
                    OP_LOGE(context, "gen_qkv must be 3D tensor [gen_len, N, D]."), return ge::GRAPH_FAILED);
        total += genQkvShape->GetDim(DIM_ZERO);
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numHeadsQ = attrs->GetInt(ATTR_NUM_HEADS_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, numHeadsQ);

    // q: [T, Hq, D]
    qOutShape->SetDimNum(DIM_NUM_THREE);
    qOutShape->SetDim(DIM_ZERO, total);
    qOutShape->SetDim(DIM_ONE, *numHeadsQ);
    qOutShape->SetDim(DIM_TWO, undQkvShape->GetDim(DIM_TWO));

    // k_cache/v_cache 原地写入，shape 透传
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
