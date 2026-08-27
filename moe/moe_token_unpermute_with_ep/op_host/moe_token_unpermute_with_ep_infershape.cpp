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
 * \file moe_token_unpermute_with_ep_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {

static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_TOKENS = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_IDX = 1;
static constexpr int64_t UNPERMUTE_WITH_EP_INPUT_PROBS = 2;
static constexpr int64_t UNPERMUTE_WITH_EP_OUTPUT_TOKENS = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_TOPK = 0;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_RANGE = 1;

#if defined(MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950)
static constexpr size_t UNPERMUTE_WITH_EP_RANK_ONE = 1;
static constexpr size_t UNPERMUTE_WITH_EP_RANK_TWO = 2;
static constexpr size_t UNPERMUTE_WITH_EP_RANGE_SIZE = 2;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_PADDED_MODE = 2;
static constexpr int64_t UNPERMUTE_WITH_EP_ARRT_RESTORE_SHAPE = 3;
static constexpr int64_t UNPERMUTE_WITH_EP_MAX_TOP_K = 512;
static constexpr int64_t UNPERMUTE_WITH_EP_UNKNOWN_DIM = -1;
static constexpr int64_t UNPERMUTE_WITH_EP_UNKNOWN_RANK = -2;

static bool IsUnknownRank(const gert::Shape *shape)
{
    return shape->GetDimNum() == UNPERMUTE_WITH_EP_RANK_ONE && shape->GetDim(0) == UNPERMUTE_WITH_EP_UNKNOWN_RANK;
}

static void SetUnknownRank(gert::Shape *shape)
{
    shape->SetDimNum(0);
    shape->AppendDim(UNPERMUTE_WITH_EP_UNKNOWN_RANK);
}

static ge::graphStatus InferShapeForMoeTokenUnpermuteWithEp(gert::InferShapeContext *context)
{
    const gert::Shape *permutedInputsShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, permutedInputsShape);
    const gert::Shape *indicesShape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    const gert::Shape *probsShape = context->GetOptionalInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);
    gert::Shape *outShape = context->GetOutputShape(UNPERMUTE_WITH_EP_OUTPUT_TOKENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool *paddedMode = attrs->GetAttrPointer<bool>(UNPERMUTE_WITH_EP_ARRT_PADDED_MODE);
    if (paddedMode != nullptr && *paddedMode) {
        OP_LOGE(context->GetNodeName(), "padded_mode=true is not supported on Ascend 950.");
        return ge::GRAPH_FAILED;
    }
    auto restoreShape = attrs->GetAttrPointer<gert::ContinuousVector>(UNPERMUTE_WITH_EP_ARRT_RESTORE_SHAPE);
    if (restoreShape != nullptr && restoreShape->GetSize() != 0) {
        OP_LOGE(context->GetNodeName(), "non-empty restore_shape is not supported on Ascend 950.");
        return ge::GRAPH_FAILED;
    }
    auto rangePtr = attrs->GetAttrPointer<gert::ContinuousVector>(UNPERMUTE_WITH_EP_ARRT_RANGE);
    bool fallback = rangePtr == nullptr || rangePtr->GetSize() == 0;
    if (!fallback) {
        if (rangePtr->GetSize() != UNPERMUTE_WITH_EP_RANGE_SIZE) {
            OP_LOGE(context->GetNodeName(), "The size of range must be 0 or 2, but got %zu.", rangePtr->GetSize());
            return ge::GRAPH_FAILED;
        }
        const int64_t *range = static_cast<const int64_t *>(rangePtr->GetData());
        OP_CHECK_NULL_WITH_CONTEXT(context, range);
        fallback = range[0] == 0 && range[1] == 0;
    }

    int64_t effectiveTopK = 1;
    if (!fallback) {
        const int64_t *topK = attrs->GetAttrPointer<int64_t>(UNPERMUTE_WITH_EP_ARRT_TOPK);
        OP_CHECK_NULL_WITH_CONTEXT(context, topK);
        if (*topK < 1 || *topK > UNPERMUTE_WITH_EP_MAX_TOP_K) {
            OP_LOGE(context->GetNodeName(), "num_topk must be in [1, %ld] outside fallback mode, but got %ld.",
                    UNPERMUTE_WITH_EP_MAX_TOP_K, *topK);
            return ge::GRAPH_FAILED;
        }
        effectiveTopK = *topK;
    }

    if (IsUnknownRank(permutedInputsShape)) {
        SetUnknownRank(outShape);
        return ge::GRAPH_SUCCESS;
    }
    if (permutedInputsShape->GetDimNum() != UNPERMUTE_WITH_EP_RANK_TWO) {
        OP_LOGE(context->GetNodeName(), "permuted_tokens must be rank 2, but got %zu.",
                permutedInputsShape->GetDimNum());
        return ge::GRAPH_FAILED;
    }

    int64_t tokensNum = UNPERMUTE_WITH_EP_UNKNOWN_DIM;
    if (probsShape != nullptr) {
        if (!IsUnknownRank(probsShape)) {
            if (probsShape->GetDimNum() != UNPERMUTE_WITH_EP_RANK_TWO) {
                OP_LOGE(context->GetNodeName(), "probs must be rank 2, but got %zu.", probsShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
            tokensNum = probsShape->GetDim(0);
        }
    } else if (!IsUnknownRank(indicesShape)) {
        if (indicesShape->GetDimNum() != UNPERMUTE_WITH_EP_RANK_ONE) {
            OP_LOGE(context->GetNodeName(), "sorted_indices must be rank 1, but got %zu.", indicesShape->GetDimNum());
            return ge::GRAPH_FAILED;
        }
        const int64_t indicesLength = indicesShape->GetDim(0);
        if (indicesLength >= 0) {
            if (indicesLength % effectiveTopK != 0) {
                OP_LOGE(context->GetNodeName(), "sorted_indices length %ld must be divisible by effective TopK %ld.",
                        indicesLength, effectiveTopK);
                return ge::GRAPH_FAILED;
            }
            tokensNum = indicesLength / effectiveTopK;
        }
    }

    outShape->SetDimNum(UNPERMUTE_WITH_EP_RANK_TWO);
    outShape->SetDim(0, tokensNum);
    outShape->SetDim(1, permutedInputsShape->GetDim(1));
    return GRAPH_SUCCESS;
}
#else
static ge::graphStatus InferShapeForMoeTokenUnpermuteWithEp(gert::InferShapeContext *context)
{
    const gert::Shape *permuted_inputs_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_TOKENS);
    const gert::Shape *probs_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_PROBS);
    const int64_t *topk = context->GetAttrs()->GetAttrPointer<int64_t>(UNPERMUTE_WITH_EP_ARRT_TOPK);
    int64_t inputTopK = *topk;
    int64_t tokens_num;
    if (probs_shape == nullptr) {
        const gert::Shape *indices_shape = context->GetInputShape(UNPERMUTE_WITH_EP_INPUT_IDX);
        tokens_num = indices_shape->GetDim(0) / inputTopK;
    } else {
        tokens_num = probs_shape->GetDim(0);
    }

    gert::Shape *out_shape = context->GetOutputShape(UNPERMUTE_WITH_EP_OUTPUT_TOKENS);
    const int8_t out_dim_num = 2;
    out_shape->SetDimNum(out_dim_num);
    out_shape->SetDim(0, tokens_num);
    out_shape->SetDim(1, permuted_inputs_shape->GetDim(1));

    return GRAPH_SUCCESS;
}
#endif

static ge::graphStatus InferDataTypeForMoeTokenUnpermuteWithEp(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(UNPERMUTE_WITH_EP_INPUT_TOKENS));
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeTokenUnpermuteWithEp)
    .InferShape(InferShapeForMoeTokenUnpermuteWithEp)
    .InferDataType(InferDataTypeForMoeTokenUnpermuteWithEp);
} // namespace ops
