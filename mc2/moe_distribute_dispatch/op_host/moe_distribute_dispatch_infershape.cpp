/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_infer.cpp
 * \brief
 */
#include "runtime_util.h"
#include "mc2_log.h"
#include "platform/platform_info.h"
using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1UL;
static constexpr size_t DIM_TWO = 2UL;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t RANK_NUM_PER_NODE = 8;

static constexpr size_t DISPATCH_INPUT_X_INDEX = 0;
static constexpr size_t DISPATCH_INPUT_EXPERT_IDX_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_SCALES_IDX_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_X_INDEX = 0;
static constexpr size_t DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX = 1;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_IDX_INDEX = 2;
static constexpr size_t DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX = 3;
static constexpr size_t DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX = 4;
static constexpr size_t DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX = 5;
static constexpr size_t DISPATCH_OUTPUT_EXPAND_SCALES = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX = 1;
static constexpr size_t DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX = 2;
static constexpr size_t DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX = 3;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX = 5;
static constexpr size_t DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX = 6;
static constexpr size_t DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX = 7;
static constexpr size_t DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX = 9;
static constexpr size_t DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX = 10;
static constexpr size_t DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX = 11;


static bool IsPlatform910B(const gert::InferShapeContext *context) {
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    GE_ASSERT_SUCCESS(fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info));
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info)
        != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Cannot get platform info!");
        return false;
    }
    static std::set<std::string> supported_soc = {"Ascend910B"};
    OP_LOGD(context->GetNodeName(), "Get soc version: %s", optional_info.soc_version.c_str());
    return supported_soc.count(platform_info.str_info.short_soc_version) > 0;
}

static ge::graphStatus InferShapeMoeDistributeDispatch(gert::InferShapeContext *context)
{
    if (context == nullptr){
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeMoeDistributeDispatch.");
    // 获取输入shape
    const gert::Shape *xShape = context->GetInputShape(DISPATCH_INPUT_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape *expertIdsShape = context->GetInputShape(DISPATCH_INPUT_EXPERT_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertIdsShape);
    const gert::Shape *expertScalesShape = context->GetOptionalInputShape(DISPATCH_INPUT_EXPERT_SCALES_IDX_INDEX);

    gert::Shape *expandXShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_X_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandXShape);
    gert::Shape *dynamicScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dynamicScalesShape);
    gert::Shape *expandIdxShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_IDX_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandIdxShape);
    gert::Shape *expertTokenNumsShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertTokenNumsShape);
    gert::Shape *epRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRecvCountShape);
    gert::Shape *tpRecvCountShape = context->GetOutputShape(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRecvCountShape);
    gert::Shape *expandScalesShape = context->GetOutputShape(DISPATCH_OUTPUT_EXPAND_SCALES);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expandScalesShape);

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const auto epWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSize);

    const auto epRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epRankId);

    const auto moeExpertNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_MOE_EXPERT_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, moeExpertNum);

    const auto tpWorldSize = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_WORLD_SIZE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpWorldSize);

    const auto tpRankId = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_TP_RANK_ID_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, tpRankId);

    const auto expertShardType = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_EXPERT_SHARD_TYPE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, expertShardType);

    const auto sharedExpertRankNum = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_SHARED_EXPERT_RANK_NUM_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sharedExpertRankNum);

    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);

    const auto globalBs = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_GLOBAL_BS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, globalBs);
    OP_CHECK_IF((*epRankId < 0) || (*epRankId >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "epRankId shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, epRankId: %ld.", *epWorldSize, *epRankId), return ge::GRAPH_FAILED);
    OP_CHECK_IF((*sharedExpertRankNum < 0) || (*sharedExpertRankNum >= *epWorldSize),
        OP_LOGE(context->GetNodeName(), "sharedExpertRankNum shoule be in [0, epWorldSize), but got"
        " epWorldSize: %ld, sharedExpertRankNum: %ld.", *epWorldSize, *sharedExpertRankNum), return ge::GRAPH_FAILED);
    int64_t bs = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t h = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);
    int64_t bsTmp = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(0);
    int64_t k = expertIdsShape->GetDimNum() == 1U ? NEG_ONE : expertIdsShape->GetDim(1);

    OP_CHECK_IF((bs <= 0) || (h <= 0) || (bsTmp <= 0) || (k <= 0),
        OP_LOGE(context->GetNodeName(), "Input shape of xShape or input shape of expertIdsShape is incorrect, "
        "xShape [%ld, %ld], expertIdsShape [%ld, %ld]", bs, h, bsTmp, k),
        return ge::GRAPH_FAILED);

    int64_t a;
    int64_t localExpertNum;
    int64_t localMoeExpertNum;
    int64_t globalBsReal = (*globalBs == 0) ? (bs * *epWorldSize) : *globalBs;
    OP_CHECK_IF(globalBsReal < 0, OP_LOGE(context->GetNodeName(), "real global bs should be larger than 0"
        " but got %ld.", globalBsReal), return ge::GRAPH_FAILED);

    int64_t moeRankNum = *epWorldSize - *sharedExpertRankNum;
    OP_CHECK_IF(moeRankNum <= 0, OP_LOGE(context->GetNodeName(), "moeRankNum(epWorldSize - sharedExpertRankNum)"
        " should be larger than 0, but got %ld.", moeRankNum), return ge::GRAPH_FAILED);
    localMoeExpertNum = *moeExpertNum / moeRankNum;
    if (*expertShardType == 0) {
        if (*epRankId < *sharedExpertRankNum) {
            localExpertNum = 1;
            a = globalBsReal / *sharedExpertRankNum;
        } else {
            localExpertNum = localMoeExpertNum;
            a = globalBsReal * std::min(localExpertNum, k);
        }
    } else {
        if (*epRankId >= (*epWorldSize - *sharedExpertRankNum)) {
            localExpertNum = 1;
            a = globalBsReal / *sharedExpertRankNum;
        } else {
            localExpertNum = localMoeExpertNum;
            a = globalBsReal * std::min(localExpertNum, k);
        }
    }

    expandXShape->SetDimNum(DIM_TWO);
    auto realA = (*tpWorldSize == 0) ? a : a * *tpWorldSize;
    expandXShape->SetDim(0U, realA);
    expandXShape->SetDim(1U, h);
    OP_LOGD(context->GetNodeName(), "expandx shape is :%s after infershape.",
        Ops::Base::ToString(*expandXShape).c_str());

    dynamicScalesShape->SetDimNum(DIM_ONE);
    dynamicScalesShape->SetDim(0U, realA);
    OP_LOGD(context->GetNodeName(), "dynamicScalesShape shape is :%s after infershape.",
        Ops::Base::ToString(*dynamicScalesShape).c_str());

    expandIdxShape->SetDimNum(DIM_ONE);
    expandIdxShape->SetDim(0U, bs * k);
    OP_LOGD(context->GetNodeName(), "expandIdxShape shape is :%s after infershape.",
        Ops::Base::ToString(*expandIdxShape).c_str());

    expertTokenNumsShape->SetDimNum(DIM_ONE);
    expertTokenNumsShape->SetDim(0U, localExpertNum);
    OP_LOGD(context->GetNodeName(), "expertTokenNumsShape shape is :%s after infershape.",
        Ops::Base::ToString(*expertTokenNumsShape).c_str());

    epRecvCountShape->SetDimNum(DIM_ONE);
    if (IsPlatform910B(context)) {
        epRecvCountShape->SetDim(0U, *epWorldSize * localExpertNum + globalBsReal * 2 * k * (*epWorldSize) / RANK_NUM_PER_NODE); // 2：globalbs * 2kn memory size, to support different bs in ranks
    } else {
        if (*tpWorldSize == DIM_TWO)  {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum * (*tpWorldSize));
        } else {
            epRecvCountShape->SetDim(0U, (*epWorldSize) * localExpertNum);
        }
    }
    OP_LOGD(context->GetNodeName(), "epRecvCountShape shape is :%s after infershape.",
        Ops::Base::ToString(*epRecvCountShape).c_str());

    tpRecvCountShape->SetDimNum(DIM_ONE);
    tpRecvCountShape->SetDim(0U, *tpWorldSize);
    OP_LOGD(context->GetNodeName(), "tpRecvCountShape shape is :%s after infershape.",
        Ops::Base::ToString(*tpRecvCountShape).c_str());

    expandScalesShape->SetDimNum(DIM_ONE);
    expandScalesShape->SetDim(0U, 0);
    if (expertScalesShape != nullptr) {
        expandScalesShape->SetDim(0U, a);
    }
    OP_LOGD(context->GetNodeName(), "expandScalesShape shape is :%s after infershape.",
        Ops::Base::ToString(*expandScalesShape).c_str());

    OP_LOGD(context->GetNodeName(), "End to do InferShapeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMoeDistributeDispatch(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMoeDistributeDispatch.");
    auto xDtype = context->GetInputDataType(DISPATCH_INPUT_X_INDEX);
    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto quantMode = attrs->GetAttrPointer<int64_t>(DISPATCH_INPUT_ATTR_QUANT_MODE_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, quantMode);
    const auto scalesType = context->GetOptionalInputDataType(DISPATCH_INPUT_SCALES_IDX_INDEX);
    bool quantFlag = (scalesType != ge::DT_UNDEFINED) ? true : false;
    OP_LOGD(context->GetNodeName(), "quantFlag id %d.", quantFlag);
    if (quantFlag || (*quantMode != 0)) {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, ge::DT_INT8);
    } else {
        context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_X_INDEX, xDtype);
    }
    context->SetOutputDataType(DISPATCH_OUTPUT_DYNAMIC_SCALES_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPAND_IDX_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_EXPERT_TOKEN_NUMS_INDEX, ge::DT_INT64);
    context->SetOutputDataType(DISPATCH_OUTPUT_EP_RECV_COUNTS_INDEX, ge::DT_INT32);
    context->SetOutputDataType(DISPATCH_OUTPUT_TP_RECV_COUNTS_INDEX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeMoeDistributeDispatch.");
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_INFERSHAPE(MoeDistributeDispatch)
    .InferShape(InferShapeMoeDistributeDispatch)
    .InferDataType(InferDataTypeMoeDistributeDispatch);
}  // namespace ops