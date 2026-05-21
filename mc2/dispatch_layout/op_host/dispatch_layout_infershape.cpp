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
 * \file dispatch_layout_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "runtime/rt_external_base.h"
#include "platform/soc_spec.h"

using namespace ge;
namespace ops {

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;

static constexpr size_t INPUT_TOPKIDX_INDEX = 0;

static constexpr size_t OUTPUT_NUMTOKENSPERRANK_INDEX = 0;
static constexpr size_t OUTPUT_NUMTOKENSPEREXPERT_INDEX = 1;
static constexpr size_t OUTPUT_ISTOKENINRANK_INDEX = 2;
static constexpr size_t OUTPUT_NOTIFYSENDDATA_INDEX = 3;
static constexpr size_t OUTPUT_SENDTOKENIDXSMALL_INDEX = 4;

static constexpr size_t ATTR_NUM_TOKENS_INDEX = 0;
static constexpr size_t ATTR_NUM_RANKS_INDEX = 1;
static constexpr size_t ATTR_NUM_EXPERTS_INDEX = 2;
static constexpr size_t ATTR_NUM_TOPK_INDEX = 3;
static constexpr size_t ATTR_LOCAL_RANKSIZE_INDEX = 4;
static constexpr size_t ATTR_PER_ROUND_TOKENS_INDEX = 5;
static constexpr size_t ATTR_RANK_ID_INDEX = 6;

static constexpr size_t VERSION_SIZE = 32;

static ge::graphStatus InferShapeDispatchLayout(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeDispatchLayout.");

    const gert::Shape *topkIdxShape = context->GetInputShape(INPUT_TOPKIDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);

    gert::Shape *numTokensPerRankShape = context->GetOutputShape(OUTPUT_NUMTOKENSPERRANK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numTokensPerRankShape);
    gert::Shape *numTokensPerExpertShape = context->GetOutputShape(OUTPUT_NUMTOKENSPEREXPERT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numTokensPerExpertShape);
    gert::Shape *isTokenInRankShape = context->GetOutputShape(OUTPUT_ISTOKENINRANK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, isTokenInRankShape);
    gert::Shape *notifySendDataShape = context->GetOutputShape(OUTPUT_NOTIFYSENDDATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, notifySendDataShape);
    gert::Shape *sendTokenIdxSmallShape = context->GetOutputShape(OUTPUT_SENDTOKENIDXSMALL_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendTokenIdxSmallShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto numTokens = attrs->GetAttrPointer<int64_t>(ATTR_NUM_TOKENS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numTokens);

    const auto numRanks = attrs->GetAttrPointer<int64_t>(ATTR_NUM_RANKS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numRanks);

    const auto numExperts = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numExperts);

    const auto numTopk = attrs->GetAttrPointer<int64_t>(ATTR_NUM_TOPK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numTopk);

    const auto localRanksize = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANKSIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, localRanksize);

    const auto perRoundTokens = attrs->GetAttrPointer<int64_t>(ATTR_PER_ROUND_TOKENS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, perRoundTokens);

    const auto rankId = attrs->GetAttrPointer<int64_t>(ATTR_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rankId);

    OP_CHECK_IF(*numTokens <= 0, OP_LOGE(context->GetNodeName(), "numTokens must be greater than zero, but got"
        " numTokens: %ld.", *numTokens), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*numRanks <= 0, OP_LOGE(context->GetNodeName(), "numRanks must be greater than zero, but got"
        " numRanks: %ld.", *numRanks), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*numExperts <= 0, OP_LOGE(context->GetNodeName(), "numExperts must be greater than zero, but got"
        " numExperts: %ld.", *numExperts), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*numTopk <= 0, OP_LOGE(context->GetNodeName(), "numTopk must be greater than zero, but got"
        " numTopk: %ld.", *numTopk), return ge::GRAPH_FAILED);
    OP_CHECK_IF((*numExperts % *numRanks) != 0, OP_LOGE(context->GetNodeName(), "numExperts must be divisible by numRanks."
        " numExperts: %ld, numRanks: %ld.", *numExperts, *numRanks), return ge::GRAPH_FAILED);

    int64_t round = (*numTokens + *perRoundTokens - 1) / *perRoundTokens;
    int64_t numLocalExperts = *numExperts / *numRanks;

    numTokensPerRankShape->SetDimNum(DIM_ONE);
    numTokensPerRankShape->SetDim(0U, *numRanks);

    numTokensPerExpertShape->SetDimNum(DIM_ONE);
    numTokensPerExpertShape->SetDim(0U, round * *numExperts);

    isTokenInRankShape->SetDimNum(DIM_TWO);
    isTokenInRankShape->SetDim(0U, round * *perRoundTokens);
    isTokenInRankShape->SetDim(1U, *numRanks);

    notifySendDataShape->SetDimNum(DIM_ONE);
    notifySendDataShape->SetDim(0U, round * numLocalExperts);

    sendTokenIdxSmallShape->SetDimNum(DIM_ONE);
    sendTokenIdxSmallShape->SetDim(0U, *numTokens * *numTopk);

    OP_LOGD(context->GetNodeName(), "numTokensPerRank shape is [%ld] after infershape.", *numRanks);
    OP_LOGD(context->GetNodeName(), "numTokensPerExpert shape is [%ld] after infershape.", round * *numExperts);
    OP_LOGD(context->GetNodeName(), "isTokenInRank shape is [%ld, %ld] after infershape.", round * *perRoundTokens, *numRanks);
    OP_LOGD(context->GetNodeName(), "notifySendData shape is [%ld] after infershape.", round * numLocalExperts);
    OP_LOGD(context->GetNodeName(), "sendTokenIdxSmall shape is [%ld] after infershape.", *numTokens * *numTopk);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeDispatchLayout.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDispatchLayout(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeDispatchLayout.");

    context->SetOutputDataType(OUTPUT_NUMTOKENSPERRANK_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_NUMTOKENSPEREXPERT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_ISTOKENINRANK_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_NOTIFYSENDDATA_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_SENDTOKENIDXSMALL_INDEX, ge::DT_INT32);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeDispatchLayout.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DispatchLayout)
    .InferShape(InferShapeDispatchLayout)
    .InferDataType(InferDataTypeDispatchLayout);
}  // namespace ops