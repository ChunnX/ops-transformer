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
 * \file engram_fetch_tiling.cpp
 * \brief host侧tiling实现
 */

#include <string>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/engram_fetch_tiling_data.h"
// #include "../../op_kernel/engram_fetch_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
constexpr uint32_t BUFFER_HANDLE_INDEX = 0U;
constexpr uint32_t INDICES_INDEX = 1U;
constexpr uint32_t FETCHED_INDEX = 0U;

constexpr uint32_t ATTR_HIDDEN_SIZE_INDEX = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK_INDEX = 1U;
constexpr uint32_t ATTR_GROUP_INDEX = 2U;
constexpr uint32_t ATTR_OUTPUT_DTYPE_INDEX = 3U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
constexpr uint32_t AIV_TYPE = 3;

constexpr uint32_t BF16_BYTES = 2U;
constexpr uint32_t FP16_BYTES = 2U;
constexpr uint32_t FP32_BYTES = 4U;

static ge::graphStatus CheckInputDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto bufferHandleDesc = context->GetInputDesc(BUFFER_HANDLE_INDEX);
    OP_TILING_CHECK(bufferHandleDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "bufferHandle"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(bufferHandleDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "bufferHandle",
            Ops::Base::ToString(bufferHandleDesc->GetDataType()).c_str(), "INT32"),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    OP_TILING_CHECK(indicesDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "indices"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(indicesDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "indices",
            Ops::Base::ToString(indicesDesc->GetDataType()).c_str(), "INT32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    OP_TILING_CHECK(fetchedDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((fetchedDesc->GetDataType() != ge::DT_BF16) &&
                    (fetchedDesc->GetDataType() != ge::DT_FLOAT16) &&
                    (fetchedDesc->GetDataType() != ge::DT_FLOAT),
        OP_LOGE_FOR_INVALID_DTYPE(nodeName, "fetched",
            Ops::Base::ToString(fetchedDesc->GetDataType()).c_str(), "BF16/FP16/FP32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto hiddenSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(hiddenSizePtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "hidden_size"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*hiddenSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*numEntriesPerRankPtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
            std::to_string(*numEntriesPerRankPtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    OP_TILING_CHECK(groupPtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(strnlen(groupPtr, 128) == 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "group", "empty", "non-empty string"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape(const gert::TilingContext *context, uint32_t &numTokens)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *indicesShape = context->GetInputShape(INDICES_INDEX);
    OP_TILING_CHECK(indicesShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "indices"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(indicesShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "indices",
            std::to_string(indicesShape->GetStorageShape().GetDimNum()).c_str(), "1"),
        return ge::GRAPH_FAILED);
    numTokens = indicesShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numTokens == 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "indices",
            (std::string("dim0=") + std::to_string(numTokens)).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputShape(const gert::TilingContext *context, uint32_t numTokens,
                                         uint32_t hiddenDim)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *fetchedShape = context->GetOutputShape(FETCHED_INDEX);
    OP_TILING_CHECK(fetchedShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "fetched",
            std::to_string(fetchedShape->GetStorageShape().GetDimNum()).c_str(), "2"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "fetched",
            (std::string("dim0=") + std::to_string(fetchedShape->GetStorageShape().GetDim(0))).c_str(),
            "dim0 must equal numTokens"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDim(1) != hiddenDim,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "fetched",
            (std::string("dim1=") + std::to_string(fetchedShape->GetStorageShape().GetDim(1))).c_str(),
            "dim1 must equal hiddenDim"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchTilingData &tilingData,
                                      uint32_t numTokens)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    tilingData.engramFetchInfo.numTokens = numTokens;

    auto hiddenSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_HIDDEN_SIZE_INDEX);
    tilingData.engramFetchInfo.hiddenDim = static_cast<uint32_t>(*hiddenSizePtr);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    tilingData.engramFetchInfo.numEntriesPerRank = static_cast<uint32_t>(*numEntriesPerRankPtr);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    OP_TILING_CHECK(fetchedDesc == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    if (fetchedDtype == ge::DT_BF16 || fetchedDtype == ge::DT_FLOAT16) {
        tilingData.engramFetchInfo.hiddenBytes = BF16_BYTES;
    } else {
        tilingData.engramFetchInfo.hiddenBytes = FP32_BYTES;
    }

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    context->SetScheduleMode(1);
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    OP_LOGD(nodeName, "SetTilingData: numTokens=%u, hiddenDim=%u, numEntriesPerRank=%u, hiddenBytes=%u",
            tilingData.engramFetchInfo.numTokens, tilingData.engramFetchInfo.hiddenDim,
            tilingData.engramFetchInfo.numEntriesPerRank, tilingData.engramFetchInfo.hiddenBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetHcommCfg(const gert::TilingContext *context, EngramFetchTilingData *tilingData,
                                    const std::string &group)
{
    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "group is %s in engram_fetch.", group.c_str());
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, OP_TYPE_ALL_TO_ALL,
                                                 "AlltoAll=level0:fullmesh;level1:pairwise");
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "EngramFetch set HCCL success.");
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext *context)
{
    // const char *nodeName = context->GetNodeName();
    // uint32_t ARCH_TYPE = static_cast<int>(2);
    // const uint64_t tilingKey = GET_TPL_TILING_KEY(ARCH_TYPE);  // 模板参数tilingkey配置
    context->SetTilingKey(0);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EngramFetchTilingFunc(gert::TilingContext *context)
{
    OP_LOGD("EngramFetch", "tiling start.");
    OP_TILING_CHECK(context == nullptr,
        OP_LOGE("engram_fetch_tiling", "failed to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
        OP_LOGE("engram_fetch_tiling", "failed to get nodeName."), return ge::GRAPH_FAILED);

    EngramFetchTilingData *tilingData = context->GetTilingData<EngramFetchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    // 1. 校验
    OP_TILING_CHECK(CheckInputDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check output dtype failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attrs failed."), return ge::GRAPH_FAILED);

    uint32_t numTokens = 0;
    OP_TILING_CHECK(CheckInputShape(context, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input shape failed."), return ge::GRAPH_FAILED);

    auto hiddenSizePtr = context->GetAttrs()->GetAttrPointer<int32_t>(ATTR_HIDDEN_SIZE_INDEX);
    uint32_t hiddenDim = static_cast<uint32_t>(*hiddenSizePtr);
    OP_TILING_CHECK(CheckOutputShape(context, numTokens, hiddenDim) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check output shape failed."), return ge::GRAPH_FAILED);

    // 2. 设置
    auto groupPtr = context->GetAttrs()->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    std::string group = std::string(groupPtr);

    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetTilingData failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetHcommCfg(context, tilingData, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHcommCfg failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetWorkSpace failed."), return ge::GRAPH_FAILED);
    SetTilingKey(context);

    OP_LOGD(nodeName, "EngramFetch tiling end.");
    return ge::GRAPH_SUCCESS;
}

struct EngramFetchCompileInfo {};

ge::graphStatus TilingParseForEngramFetch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetch)
    .Tiling(EngramFetchTilingFunc)
    .TilingParse<EngramFetchCompileInfo>(TilingParseForEngramFetch);

} // namespace optiling