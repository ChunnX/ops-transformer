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
 * \file moe_ep_dispatch_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace ops {

static constexpr size_t DISPATCH_INPUT_CONTEXT_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_X_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_TOPK_IDX_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_TOPK_WEIGHTS_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_SCALES_INDEX = 4;
static constexpr size_t DISPATCH_INPUT_CACHED_SLOT_IDX_INDEX = 5;

static constexpr size_t DISPATCH_OUTPUT_NUM_RECV_PER_RANK_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_NUM_RECV_PER_EXPERT_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_DST_SLOT_IDX_INDEX = 2;

static constexpr size_t DISPATCH_ATTR_EP_WORLD_SIZE_INDEX = 0;
static constexpr size_t DISPATCH_ATTR_EP_RANK_ID_INDEX = 1;
static constexpr size_t DISPATCH_ATTR_NUM_EXPERTS_INDEX = 2;
static constexpr size_t DISPATCH_ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX = 3;
static constexpr size_t DISPATCH_ATTR_CCL_BUFFER_SIZE_INDEX = 4;
static constexpr size_t DISPATCH_ATTR_EXPERT_ALIGNMENT_INDEX = 5;
static constexpr size_t DISPATCH_ATTR_DO_CPU_SYNC_INDEX = 6;

static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;

static ge::graphStatus InferShapeMoeEpDispatch(gert::InferShapeContext *context) {
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape for MoeEpDispatch.");

    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *topkIdxShape = context->GetInputShape(DISPATCH_INPUT_TOPK_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);
    const auto numExperts = attrs->GetAttrPointer<int64_t>(DISPATCH_ATTR_NUM_EXPERTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numExperts);
    const auto numMaxTokensPerRank = attrs->GetAttrPointer<int64_t>(DISPATCH_ATTR_NUM_MAX_TOKENS_PER_RANK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numMaxTokensPerRank);

    int64_t numTokens = xShape->GetDim(0);
    int64_t hidden = xShape->GetDim(1);
    int64_t topK =  topkIdxShape->GetDim(1);
    int64_t ep = *epWorldSize;
    int64_t nmt = *numMaxTokensPerRank;
    int64_t numLocalExperts = *numExperts / ep;
    int64_t upperA = ep * nmt;

    gert::Shape *numRecvPerRank = context->GetOutputShape(DISPATCH_OUTPUT_NUM_RECV_PER_RANK_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numRecvPerRank);
    numRecvPerRank->SetDimNum(DIM_ONE);
    numRecvPerRank->SetDim(0U, ep);

    gert::Shape *numRecvPerExpert = context->GetOutputShape(DISPATCH_OUTPUT_NUM_RECV_PER_EXPERT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpert);
    numRecvPerExpert->SetDimNum(DIM_ONE);
    numRecvPerExpert->SetDim(0U, numLocalExperts);

    gert::Shape *dstSlotIdx = context->GetOutputShape(DISPATCH_OUTPUT_DST_SLOT_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dstSlotIdx);
    dstSlotIdx->SetDimNum(DIM_TWO);
    dstSlotIdx->SetDim(0U, numTokens);
    dstSlotIdx->SetDim(1U, topK);

    OP_LOGD(context->GetNodeName(), "InferShape for MoeEpDispatch done.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeEpDispatch(gert::InferDataTypeContext *context) {
    if (context == nullptr) { return ge::GRAPH_FAILED; }
    context->SetOutputDataType(DISPATCH_OUTPUT_NUM_RECV_PER_RANK_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_NUM_RECV_PER_EXPERT_INDEX, ge::DT_INT64);
    context->SetOutputDataType(DISPATCH_OUTPUT_DST_SLOT_IDX_INDEX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeEpDispatch)
    .InferShape(InferShapeMoeEpDispatch)
    .InferDataType(InferDataTypeMoeEpDispatch);

}  // namespace ops