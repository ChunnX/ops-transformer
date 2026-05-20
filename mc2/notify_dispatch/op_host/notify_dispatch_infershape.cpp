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
 * \file notify_dispatch_infershape.cpp
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
static constexpr int64_t SEND_PER_GROUP = 3;

static constexpr size_t INPUT_CONTEXT_INDEX = 0;
static constexpr size_t INPUT_SEND_DATA_INDEX = 1;
static constexpr size_t INPUT_TOKEN_PER_EXPERT_DATA_INDEX = 2;

static constexpr size_t OUTPUT_SEND_DATA_OFFSET_INDEX = 0;
static constexpr size_t OUTPUT_RECV_DATA_INDEX = 1;
static constexpr size_t OUTPUT_RECV_COUNT_INDEX = 2;
static constexpr size_t OUTPUT_RECV_OFFSET_INDEX = 3;
static constexpr size_t OUTPUT_EXPERT_GLOBAL_OFFSET_INDEX = 4;
static constexpr size_t OUTPUT_SRCRANK_IN_EXPERT_OFFSET_INDEX = 5;
static constexpr size_t OUTPUT_R_IN_SRCRANK_OFFSET_INDEX = 6;
static constexpr size_t OUTPUT_TOTAL_RECV_TOKENS_INDEX = 7;
static constexpr size_t OUTPUT_MAX_BS_INDEX = 8;
static constexpr size_t OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX = 9;

static constexpr size_t ATTR_SEND_COUNT_INDEX = 0;
static constexpr size_t ATTR_NUM_TOKENS_INDEX = 1;
static constexpr size_t ATTR_EP_WORLD_SIZE_INDEX = 2;
static constexpr size_t ATTR_EP_RANK_ID_INDEX = 3;
static constexpr size_t ATTR_LOCAL_RANK_SIZE_INDEX = 4;
static constexpr size_t ATTR_LOCAL_RANK_ID_INDEX = 5;
static constexpr size_t ATTR_CCL_BUFFER_SIZE_INDEX = 6;
static constexpr size_t ATTR_ROUND_INDEX = 7;
static constexpr size_t ATTR_PER_ROUND_TOKENS_INDEX = 8;

static constexpr int64_t MAX_COMM_WORLD_SIZE = 384;

static ge::graphStatus InferShapeNotifyDispatch(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeNotifyDispatch.");

    const gert::Shape *contextShape = context->GetInputShape(INPUT_CONTEXT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, contextShape);
    const gert::Shape *sendDataShape = context->GetInputShape(INPUT_SEND_DATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendDataShape);
    const gert::Shape *tokenPerExpertDataShape = context->GetInputShape(INPUT_TOKEN_PER_EXPERT_DATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tokenPerExpertDataShape);

    gert::Shape *sendDataOffsetShape = context->GetOutputShape(OUTPUT_SEND_DATA_OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendDataOffsetShape);
    gert::Shape *recvDataShape = context->GetOutputShape(OUTPUT_RECV_DATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvDataShape);
    gert::Shape *recvCountShape = context->GetOutputShape(OUTPUT_RECV_COUNT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCountShape);
    gert::Shape *recvOffsetShape = context->GetOutputShape(OUTPUT_RECV_OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvOffsetShape);
    gert::Shape *expertGlobalOffsetShape = context->GetOutputShape(OUTPUT_EXPERT_GLOBAL_OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertGlobalOffsetShape);
    gert::Shape *srcrankInExpertOffsetShape = context->GetOutputShape(OUTPUT_SRCRANK_IN_EXPERT_OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, srcrankInExpertOffsetShape);
    gert::Shape *rInSrcrankOffsetShape = context->GetOutputShape(OUTPUT_R_IN_SRCRANK_OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, rInSrcrankOffsetShape);
    gert::Shape *totalRecvTokensShape = context->GetOutputShape(OUTPUT_TOTAL_RECV_TOKENS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, totalRecvTokensShape);
    gert::Shape *maxBsShape = context->GetOutputShape(OUTPUT_MAX_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, maxBsShape);
    gert::Shape *recvTokensPerExpertShape = context->GetOutputShape(OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvTokensPerExpertShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto sendCount = attrs->GetAttrPointer<int64_t>(ATTR_SEND_COUNT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCount);

    const auto numTokens = attrs->GetAttrPointer<int64_t>(ATTR_NUM_TOKENS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numTokens);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto cclBufferSize = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, cclBufferSize);

    const auto localRankSize = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, localRankSize);

    const auto localRankId = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, localRankId);

    const auto round = attrs->GetAttrPointer<int64_t>(ATTR_ROUND_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, round);

    const auto perRoundTokens = attrs->GetAttrPointer<int64_t>(ATTR_PER_ROUND_TOKENS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, perRoundTokens);

    OP_CHECK_IF(*sendCount <= 0, OP_LOGE(context->GetNodeName(), "sendCount must be greater than zero, but got"
        " sendCount: %ld.", *sendCount), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*epWorldSize <= 0 || *epWorldSize > MAX_COMM_WORLD_SIZE, OP_LOGE(context->GetNodeName(),
        "epWorldSize is invalid, only support (0, %ld], but got epWorldSize: %ld.", MAX_COMM_WORLD_SIZE,
        *epWorldSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*epRankId < 0 || *epRankId >= *epWorldSize, OP_LOGE(context->GetNodeName(),
        "epRankId is invalid, only support [0, %ld), but got epRankId: %ld.", *epWorldSize, *epRankId),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*cclBufferSize <= 0, OP_LOGE(context->GetNodeName(),
        "cclBufferSize must be greater than zero, but got cclBufferSize: %ld.", *cclBufferSize),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*round <= 0, OP_LOGE(context->GetNodeName(), "round must be greater than zero, but got"
        " round: %ld.", *round), return ge::GRAPH_FAILED);

    int64_t numExperts = *sendCount / SEND_PER_GROUP / *round;
    int64_t numLocalExperts = numExperts / *epWorldSize;

    sendDataOffsetShape->SetDimNum(DIM_ONE);
    sendDataOffsetShape->SetDim(0U, *round * numExperts);

    recvDataShape->SetDimNum(DIM_ONE);
    recvDataShape->SetDim(0U, *sendCount);

    recvCountShape->SetDimNum(DIM_ONE);
    recvCountShape->SetDim(0U, *round * numExperts);

    recvOffsetShape->SetDimNum(DIM_ONE);
    recvOffsetShape->SetDim(0U, *round * numExperts);

    expertGlobalOffsetShape->SetDimNum(DIM_ONE);
    expertGlobalOffsetShape->SetDim(0U, numLocalExperts);

    srcrankInExpertOffsetShape->SetDimNum(DIM_ONE);
    srcrankInExpertOffsetShape->SetDim(0U, numLocalExperts * *epWorldSize);

    rInSrcrankOffsetShape->SetDimNum(DIM_ONE);
    rInSrcrankOffsetShape->SetDim(0U, numLocalExperts * *epWorldSize * *round);

    totalRecvTokensShape->SetDimNum(DIM_ONE);
    totalRecvTokensShape->SetDim(0U, 1);

    maxBsShape->SetDimNum(DIM_ONE);
    maxBsShape->SetDim(0U, 1);

    recvTokensPerExpertShape->SetDimNum(DIM_ONE);
    recvTokensPerExpertShape->SetDim(0U, *round * numLocalExperts);

    OP_LOGD(context->GetNodeName(), "sendDataOffset shape is [%ld] after infershape.", *round * numExperts);
    OP_LOGD(context->GetNodeName(), "recvData shape is [%ld] after infershape.", *sendCount);
    OP_LOGD(context->GetNodeName(), "recvCount shape is [%ld] after infershape.", *round * numExperts);
    OP_LOGD(context->GetNodeName(), "recvOffset shape is [%ld] after infershape.", *round * numExperts);
    OP_LOGD(context->GetNodeName(), "expertGlobalOffset shape is [%ld] after infershape.", numLocalExperts);
    OP_LOGD(context->GetNodeName(), "srcrankInExpertOffset shape is [%ld] after infershape.",
            numLocalExperts * *epWorldSize);
    OP_LOGD(context->GetNodeName(), "rInSrcrankOffset shape is [%ld] after infershape.",
            numLocalExperts * *epWorldSize * *round);
    OP_LOGD(context->GetNodeName(), "totalRecvTokens shape is [1] after infershape.");
    OP_LOGD(context->GetNodeName(), "maxBs shape is [1] after infershape.");
    OP_LOGD(context->GetNodeName(), "recvTokensPerExpert shape is [%ld] after infershape.", *round * numLocalExperts);
    OP_LOGD(context->GetNodeName(), "End to do InferShapeNotifyDispatch.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNotifyDispatch(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeNotifyDispatch.");

    context->SetOutputDataType(OUTPUT_SEND_DATA_OFFSET_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_RECV_DATA_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_RECV_COUNT_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_RECV_OFFSET_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_EXPERT_GLOBAL_OFFSET_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_SRCRANK_IN_EXPERT_OFFSET_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_R_IN_SRCRANK_OFFSET_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_TOTAL_RECV_TOKENS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_MAX_BS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX, ge::DT_INT32);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeNotifyDispatch.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NotifyDispatch)
    .InferShape(InferShapeNotifyDispatch)
    .InferDataType(InferDataTypeNotifyDispatch);
}  // namespace ops