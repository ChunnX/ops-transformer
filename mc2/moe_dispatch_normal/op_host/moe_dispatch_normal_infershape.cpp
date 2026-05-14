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
 * \file moe_dispatch_normal_infershape.cpp
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

// static constexpr size_t DISPATCH_NORMAL_INPUT_CONTEXT_INDEX = 0;    // TODO
static constexpr size_t DISPATCH_NORMAL_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_NORMAL_INPUT_TOPK_IDS_INDEX = 1;
static constexpr size_t DISPATCH_NORMAL_INPUT_SEND_OFFSETS_INDEX = 2;
static constexpr size_t DISPATCH_NORMAL_INPUT_SEND_TOKENIDX_INDEX = 3;
static constexpr size_t DISPATCH_NORMAL_INPUT_RECV_OFFSET_INDEX = 4;
static constexpr size_t DISPATCH_NORMAL_INPUT_RECV_COUNT_INDEX = 5;

static constexpr size_t DISPATCH_NORMAL_OUTPUT_RECV_X_INDEX = 0;
static constexpr size_t DISPATCH_NORMAL_OUTPUT_RECV_X_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_NORMAL_OUTPUT_ASSIST_INFO_INDEX = 2;
static constexpr size_t DISPATCH_NORMAL_OUTPUT_WAIT_RECV_COST_INDEX = 3;

static constexpr size_t DISPATCH_NORMAL_ATTR_GROUP_EP_INDEX = 0;
static constexpr size_t DISPATCH_NORMAL_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_NORMAL_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_NORMAL_ATTR_GROUP_TP_INDEX = 3;
static constexpr size_t DISPATCH_NORMAL_ATTR_TP_WORLD_SIZE_INDEX = 4;
static constexpr size_t DISPATCH_NORMAL_ATTR_TP_RANK_ID_INDEX = 5;
static constexpr size_t DISPATCH_NORMAL_ATTR_MOE_EXPERT_NUM_INDEX = 6;
static constexpr size_t DISPATCH_NORMAL_ATTR_QUANT_MODE_INDEX = 7;
static constexpr size_t DISPATCH_NORMAL_ATTR_GLOBAL_BS_INDEX = 8;

static ge::graphStatus InferShapeMoeDispatchNormal(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDispatchNormal.");

    const gert::Shape *xShape = context->GetInputShape(DISPATCH_NORMAL_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *topkIdxShape = context->GetInputShape(DISPATCH_NORMAL_INPUT_TOPK_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);
    const gert::Shape *sendOffsetShape = context->GetInputShape(DISPATCH_NORMAL_INPUT_SEND_OFFSETS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendOffsetShape);
    const gert::Shape *sendTokenIdxShape = context->GetInputShape(DISPATCH_NORMAL_INPUT_SEND_TOKENIDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendTokenIdxShape);
    const gert::Shape *recvOffsetShape = context->GetInputShape(DISPATCH_NORMAL_INPUT_RECV_OFFSET_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvOffsetShape);
    const gert::Shape *recvCountShape = context->GetInputShape(DISPATCH_NORMAL_INPUT_RECV_COUNT_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCountShape);

    gert::Shape *recvXShape = context->GetOutputShape(DISPATCH_NORMAL_OUTPUT_RECV_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvXShape);
    gert::Shape *recvXScalesShape = context->GetOutputShape(DISPATCH_NORMAL_OUTPUT_RECV_X_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvXScalesShape);
    gert::Shape *assistInfoForCombineShape = context->GetOutputShape(DISPATCH_NORMAL_OUTPUT_ASSIST_INFO_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, assistInfoForCombineShape);
    gert::Shape *waitRecvCostStatsShape = context->GetOutputShape(DISPATCH_NORMAL_OUTPUT_WAIT_RECV_COST_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, waitRecvCostStatsShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_NORMAL_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);
    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_NORMAL_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);
    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_NORMAL_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_NORMAL_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_NORMAL_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);

    OP_CHECK_IF((*epRankId < 0) || (*epRankId >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "epRankId shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, epRankId: %ld.", *epWorldSize, *epRankId), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*globalBs < 0, OP_LOGE(context->GetNodeName(), 
        "globalBS is invalid, should be positive, but got %ld.", *globalBs), return ge::GRAPH_FAILED);

    int64_t bs = xShape->GetDim(0);
    int64_t h = xShape->GetDim(1);
    int64_t bsTmp = topkIdxShape->GetDim(0);
    int64_t k = topkIdxShape->GetDim(1);

    OP_CHECK_IF((bs <= 0) || (h <= 0) || (bsTmp <= 0) || (k <= 0),
        OP_LOGE(context->GetNodeName(), "Input shape of xShape or input shape of topkIdxShape is incorrect, "
        "xShape [%ld, %ld], topkIdxShape [%ld, %ld]", bs, h, bsTmp, k),
        return ge::GRAPH_FAILED);

    // TODO 输出shape推导
    // auto realA = ((*tpWorldSize == 0) ? a : (a * *tpWorldSize));
    // recvXShape->SetDimNum(DIM_TWO);
    // recvXShape->SetDim(0U, realA);
    // recvXShape->SetDim(1U, h);

    // recvXScalesShape->SetDimNum(DIM_ONE);
    // recvXScalesShape->SetDim(0U, realA);

    // assistInfoShape->SetDimNum(DIM_ONE);
    // assistInfoShape->SetDim(0U, a * ASSIST_INFO_NUM_PER_A);

    // waitRecvCostStatsShape->SetDimNum(DIM_ONE);
    // waitRecvCostStatsShape->SetDim(0U, *moeExpertNum / *epWorldSize);

    // OP_LOGD(context->GetNodeName(), "y shape is [%ld, %ld] after infershape.", bs, h);
    // OP_LOGD(context->GetNodeName(), "End to do InferShapeMegaMoe.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDispatchNormal(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDispatchNormal.");

    auto xDtype = context->GetInputDataType(DISPATCH_NORMAL_INPUT_X_INDEX);
    // TODO 输出Dtype
    // context->SetOutputDataType(DISPATCH_NORMAL_OUTPUT_RECV_X_INDEX, xDtype);
    // context->SetOutputDataType(DISPATCH_NORMAL_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT32);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDispatchNormal.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeDispatchNormal)
    .InferShape(InferShapeMoeDispatchNormal)
    .InferDataType(InferDataTypeMoeDispatchNormal);
}  // namespace ops