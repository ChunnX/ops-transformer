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
 * \file ffn_worker_batching_infershape.cpp
 * \brief
 */

#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <limits>

using namespace ge;
namespace ops {
static constexpr int64_t EXPERT_NUM_ATTR = 0;
static constexpr int64_t MAX_OUT_SHAPE_ATTR = 1;
static constexpr int64_t TOKEN_DTYPE_ATTR = 2;

static constexpr int64_t Y_OUT = 0;
static constexpr int64_t GROUP_LIST_OUT = 1;
static constexpr int64_t SESSION_IDS_OUT = 2;
static constexpr int64_t MICRO_BATCH_IDS_OUT = 3;
static constexpr int64_t TOKEN_IDS_OUT = 4;
static constexpr int64_t EXPERT_OFFSETS_OUT = 5;
static constexpr int64_t DYNAMIC_SCALE_OUT = 6;
static constexpr int64_t ACTUAL_TOKEN_NUM_OUT = 7;

static constexpr int64_t TOKEN_KIND_ZERO = 0;
static constexpr int64_t TOKEN_KIND_ONE = 1;
static constexpr int64_t TOKEN_KIND_TWO = 2;

static constexpr int64_t NUM_FOUR = 4;
static constexpr uint32_t INDEX_ZERO = 0;
static constexpr uint32_t INDEX_ONE = 1;
static constexpr uint32_t INDEX_TWO = 2;
static constexpr uint32_t INDEX_THREE = 3;

static constexpr int64_t EVEN_ALIGN = 2;
static constexpr int64_t TOKEN_DTYPE_E5M2 = 3;
static constexpr int64_t TOKEN_DTYPE_E4M3 = 4;
static constexpr int64_t TOKEN_DTYPE_E2M1 = 5;
static constexpr int64_t MXFP_SCALE_GROUP = 32;

static graphStatus InferShape4FfnWorkerBatching(gert::InferShapeContext *context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t *expertNumPtr = attrs->GetAttrPointer<int64_t>(EXPERT_NUM_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertNumPtr);
    auto expertNum = *expertNumPtr;
    if (expertNum <= 0) {
        OP_LOGE(context->GetNodeName(), "expert_num must be positive");
        return ge::GRAPH_FAILED;
    }
    auto maxOutShapePtr = attrs->GetAttrPointer<gert::TypedContinuousVector<int64_t>>(MAX_OUT_SHAPE_ATTR);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxOutShapePtr);
    if (maxOutShapePtr->GetSize() != NUM_FOUR) {
        OP_LOGE(context->GetNodeName(), "attr max_out_shape len must be 4, but is [%d].",
                static_cast<int32_t>(maxOutShapePtr->GetSize()));
        return ge::GRAPH_FAILED;
    }

    const int64_t *maxOutShapeArray = reinterpret_cast<const int64_t *>(maxOutShapePtr->GetData());
    const int64_t A = maxOutShapeArray[INDEX_ZERO];
    const int64_t BS = maxOutShapeArray[INDEX_ONE];
    const int64_t K = maxOutShapeArray[INDEX_TWO];
    const int64_t H = maxOutShapeArray[INDEX_THREE];
    if (A <= 0 || BS <= 0 || K <= 0 || H <= 0) {
        OP_LOGE(context->GetNodeName(),
                "attr max_out_shape values must be greater than 0, but are [%ld, %ld, %ld, %ld].", A, BS, K, H);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(A > std::numeric_limits<int64_t>::max() / BS,
                OP_LOGE(context->GetNodeName(), "max_out_shape A*BS overflows int64: %ld*%ld", A, BS),
                return ge::GRAPH_FAILED);
    const int64_t aBs = A * BS;
    OP_CHECK_IF(aBs > std::numeric_limits<int64_t>::max() / K,
                OP_LOGE(context->GetNodeName(), "max_out_shape A*BS*K overflows int64: %ld*%ld*%ld", A, BS, K),
                return ge::GRAPH_FAILED);
    const int64_t Y = aBs * K;

    gert::Shape *yShape = context->GetOutputShape(Y_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    gert::Shape *groupList = context->GetOutputShape(GROUP_LIST_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupList);

    gert::Shape *sessionIds = context->GetOutputShape(SESSION_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, sessionIds);

    gert::Shape *microBatchIds = context->GetOutputShape(MICRO_BATCH_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, microBatchIds);

    gert::Shape *tokenIds = context->GetOutputShape(TOKEN_IDS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenIds);

    gert::Shape *expertOffsets = context->GetOutputShape(EXPERT_OFFSETS_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertOffsets);

    gert::Shape *dynamicScale = context->GetOutputShape(DYNAMIC_SCALE_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, dynamicScale);

    gert::Shape *actualTokenNum = context->GetOutputShape(ACTUAL_TOKEN_NUM_OUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, actualTokenNum);

    // GE/ACL 的 shape 以逻辑元素计数，即使 token_dtype=5，一个元素只有 4 bit，
    // 这里仍必须保留 H，不能改成 H/2。H/2 是 Torch FP4-x2 张量的字节存储宽度，
    // 由 Torch→aclTensor 的转换层处理；在此再除一次会使输出容量和 kernel 寻址相差一倍。
    // Y 使用静态最大容量而非设备上的有效 token 数，避免 shape 推导读取调度状态、
    // 引入流同步或让图编译依赖某一次运行的数据。有效前缀通过 actual_token_num 返回。
    *yShape = {Y, H};
    *groupList = {expertNum, 2};
    *sessionIds = {Y};
    *microBatchIds = {Y};
    *tokenIds = {Y};
    *expertOffsets = {Y};
    const int64_t *tokenDtypePtr = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    const int64_t tokenDtype = tokenDtypePtr == nullptr ? TOKEN_KIND_ZERO : *tokenDtypePtr;
    if (tokenDtype == TOKEN_DTYPE_E2M1 && H % EVEN_ALIGN != 0) {
        OP_LOGE(context->GetNodeName(), "float4_e2m1 requires even H, got %ld", H);
        return ge::GRAPH_FAILED;
    }
    if (tokenDtype >= TOKEN_DTYPE_E5M2 && tokenDtype <= TOKEN_DTYPE_E2M1) {
        // MX 的数据和 scale 成对搬运：每 32 个逻辑元素对应一个 E8M0 字节。
        // FP4 的分组仍按逻辑 H 计算，不按打包后的 H/2 字节计算；尾组不足 32 个
        // 元素也保留一个 scale。输出是紧凑二维布局，不包含输入记录的 512B padding。
        *dynamicScale = {Y, (H + MXFP_SCALE_GROUP - 1) / MXFP_SCALE_GROUP};
    } else {
        *dynamicScale = {Y};
    }
    *actualTokenNum = {1};

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType4FfnWorkerBatching(gert::InferDataTypeContext *context)
{
    // 唯一显式输入 schedule_context 是 INT8 字节容器，其 dtype 与 token 无关。
    // token 的真实设备地址藏在容器内，所以输出 dtype 必须由 token_dtype 属性决定，
    // 不能照抄输入 dtype。这里仅声明解释方式；gather 不做 FP16/BF16→MX 数值量化。
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *tokenDtypePtr = attrs->GetAttrPointer<int64_t>(TOKEN_DTYPE_ATTR);
    int64_t tokenDtype = TOKEN_KIND_ZERO;
    if (tokenDtypePtr != nullptr) {
        tokenDtype = *tokenDtypePtr;
    }

    if (tokenDtype == TOKEN_KIND_ZERO) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT16);
    } else if (tokenDtype == TOKEN_KIND_ONE) {
        context->SetOutputDataType(Y_OUT, ge::DT_BF16);
    } else if (tokenDtype == TOKEN_DTYPE_E5M2) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT8_E5M2);
    } else if (tokenDtype == TOKEN_DTYPE_E4M3) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT8_E4M3FN);
    } else if (tokenDtype == TOKEN_DTYPE_E2M1) {
        context->SetOutputDataType(Y_OUT, ge::DT_FLOAT4_E2M1);
    } else {
        context->SetOutputDataType(Y_OUT, ge::DT_INT8);
    }

    context->SetOutputDataType(GROUP_LIST_OUT, ge::DT_INT64);
    context->SetOutputDataType(SESSION_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(MICRO_BATCH_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(TOKEN_IDS_OUT, ge::DT_INT32);
    context->SetOutputDataType(EXPERT_OFFSETS_OUT, ge::DT_INT32);
    context->SetOutputDataType(DYNAMIC_SCALE_OUT, tokenDtype >= TOKEN_DTYPE_E5M2 && tokenDtype <= TOKEN_DTYPE_E2M1 ?
                                                      ge::DT_FLOAT8_E8M0 :
                                                      ge::DT_FLOAT);
    context->SetOutputDataType(ACTUAL_TOKEN_NUM_OUT, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FfnWorkerBatching)
    .InferShape(InferShape4FfnWorkerBatching)
    .InferDataType(InferDataType4FfnWorkerBatching);
} // namespace ops
