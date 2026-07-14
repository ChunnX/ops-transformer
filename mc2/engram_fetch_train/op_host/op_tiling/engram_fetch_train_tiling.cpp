/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
* \file engram_fetch_train_tiling.cpp
* \brief host侧tiling实现 — EngramFetchTrain（训练态前向）
*        校验 2 input + 6 output（dtype/shape/format）
*        rankSize 从 sendCountsOut dim0 获取
*        totalRecv 从 recvLocalEntryOut dim0 获取
*        workspace: sdispls + rdispls + sortedIndices + tokenIdxInRank
*                 + recvIndices + localData + recvData + flagScratch + 16MB
*/

#include <string>
#include <climits>
#include <cstdint>
#include <algorithm>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/engram_fetch_train_tiling_data.h"
#include "../../op_kernel/engram_fetch_train_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace MC2Tiling {
constexpr uint32_t IN_COMM_CONTEXT = 0U;
constexpr uint32_t IN_INDICES = 1U;
constexpr uint32_t OUT_FETCHED = 0U;
constexpr uint32_t OUT_PERM = 1U;
constexpr uint32_t OUT_SEND_COUNTS = 2U;
constexpr uint32_t OUT_RECV_COUNTS = 3U;
constexpr uint32_t OUT_RECV_LOCAL_ENTRY = 4U;
constexpr uint32_t OUT_NUM_RECV = 5U;

constexpr uint32_t ATTR_HIDDEN_SIZE = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK = 1U;
constexpr uint32_t ATTR_LOCAL_STORAGE_ADDR = 2U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr int32_t HIDDEN_SIZE_ALIGN = 128;
constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;

static const std::vector<ge::DataType> OUTPUT_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static bool IsContains(const std::vector<ge::DataType> &list, ge::DataType value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

static int64_t CeilDiv(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

static int64_t AlignTo(int64_t x, int64_t y)
{
    return CeilDiv(x, y) * y;
}

static void PrintEngramFetchTrainTilingData(const EngramFetchTrainTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== EngramFetchTrainTilingData ==========");
    OP_LOGD(nodeName, "numTokens is %ld", tilingData->numTokens);
    OP_LOGD(nodeName, "numEntriesPerRank is %d", tilingData->numEntriesPerRank);
    OP_LOGD(nodeName, "hiddenDim is %ld", tilingData->hiddenDim);
    OP_LOGD(nodeName, "hiddenBytes is %ld", tilingData->hiddenBytes);
    OP_LOGD(nodeName, "aivNum is %u", tilingData->aivNum);
    OP_LOGD(nodeName, "ubSize is %lu", tilingData->ubSize);
    OP_LOGD(nodeName, "rankSize is %u", tilingData->rankSize);
    OP_LOGD(nodeName, "numMaxTokensPerRank is %ld", tilingData->numMaxTokensPerRank);
    OP_LOGD(nodeName, "totalRecv is %ld", tilingData->totalRecv);
    OP_LOGD(nodeName, "localStorageAddr is %ld", tilingData->localStorageAddr);
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
{
    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    auto indicesDesc = context->GetInputDesc(IN_INDICES);
    auto fetchedDesc = context->GetOutputDesc(OUT_FETCHED);
    auto permOutDesc = context->GetOutputDesc(OUT_PERM);
    auto sendCountsOutDesc = context->GetOutputDesc(OUT_SEND_COUNTS);
    auto recvCountsOutDesc = context->GetOutputDesc(OUT_RECV_COUNTS);
    auto recvLocalEntryOutDesc = context->GetOutputDesc(OUT_RECV_LOCAL_ENTRY);
    auto numRecvOutDesc = context->GetOutputDesc(OUT_NUM_RECV);

    OP_CHECK_NULL_WITH_CONTEXT(context, commContextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, permOutDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sendCountsOutDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvCountsOutDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvLocalEntryOutDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvOutDesc);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commContextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "commContext",
            Ops::Base::ToString(commContextDesc->GetDataType()).c_str(),
            "The dtype of commContext must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(IN_INDICES);
    OP_TILING_CHECK(indicesDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "indices",
            Ops::Base::ToString(indicesDesc->GetDataType()).c_str(),
            "The dtype of indices must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(OUT_FETCHED);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, fetchedDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "fetched",
            Ops::Base::ToString(fetchedDtype).c_str(),
            "The dtype of fetched must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
        return ge::GRAPH_FAILED);

    auto permOutDesc = context->GetOutputDesc(OUT_PERM);
    OP_TILING_CHECK(permOutDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "permOut",
            Ops::Base::ToString(permOutDesc->GetDataType()).c_str(),
            "The dtype of permOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto sendCountsOutDesc = context->GetOutputDesc(OUT_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsOutDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "sendCountsOut",
            Ops::Base::ToString(sendCountsOutDesc->GetDataType()).c_str(),
            "The dtype of sendCountsOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto recvCountsOutDesc = context->GetOutputDesc(OUT_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsOutDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "recvCountsOut",
            Ops::Base::ToString(recvCountsOutDesc->GetDataType()).c_str(),
            "The dtype of recvCountsOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto recvLocalEntryOutDesc = context->GetOutputDesc(OUT_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryOutDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "recvLocalEntryOut",
            Ops::Base::ToString(recvLocalEntryOutDesc->GetDataType()).c_str(),
            "The dtype of recvLocalEntryOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto numRecvOutDesc = context->GetOutputDesc(OUT_NUM_RECV);
    OP_TILING_CHECK(numRecvOutDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "numRecvOut",
            Ops::Base::ToString(numRecvOutDesc->GetDataType()).c_str(),
            "The dtype of numRecvOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, int64_t &numTokens,
                                      uint32_t &rankSize, int64_t &totalRecv)
{
    const char *nodeName = context->GetNodeName();

    const gert::StorageShape *commContextShape = context->GetInputShape(IN_COMM_CONTEXT);
    OP_TILING_CHECK(commContextShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commContextShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "commContext",
            (std::to_string(commContextShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of commContext must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commContextShape->GetStorageShape().GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "commContext",
            (std::string("dim0=") + std::to_string(commContextShape->GetStorageShape().GetDim(0))).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *indicesShape = context->GetInputShape(IN_INDICES);
    OP_TILING_CHECK(indicesShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "indices"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(indicesShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "indices",
            (std::to_string(indicesShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of indices must be 1D."),
        return ge::GRAPH_FAILED);
    numTokens = indicesShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numTokens < 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "indices",
            (std::string("dim0=") + std::to_string(numTokens)).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *fetchedShape = context->GetOutputShape(OUT_FETCHED);
    OP_TILING_CHECK(fetchedShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "fetched",
            (std::to_string(fetchedShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of fetched must be 2D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "fetched",
            (std::string("dim0=") + std::to_string(fetchedShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal numTokens=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *permOutShape = context->GetOutputShape(OUT_PERM);
    OP_TILING_CHECK(permOutShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "permOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "permOut",
            (std::to_string(permOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of permOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permOutShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "permOut",
            (std::string("dim0=") + std::to_string(permOutShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal numTokens=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *sendCountsOutShape = context->GetOutputShape(OUT_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsOutShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCountsOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "sendCountsOut",
            (std::to_string(sendCountsOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of sendCountsOut must be 1D."),
        return ge::GRAPH_FAILED);
    rankSize = static_cast<uint32_t>(sendCountsOutShape->GetStorageShape().GetDim(0));

    const gert::StorageShape *recvCountsOutShape = context->GetOutputShape(OUT_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsOutShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvCountsOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "recvCountsOut",
            (std::to_string(recvCountsOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of recvCountsOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsOutShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(rankSize),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "recvCountsOut",
            (std::string("dim0=") + std::to_string(recvCountsOutShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal sendCountsOut dim0=") + std::to_string(rankSize)).c_str()),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *recvLocalEntryOutShape = context->GetOutputShape(OUT_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryOutShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvLocalEntryOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvLocalEntryOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "recvLocalEntryOut",
            (std::to_string(recvLocalEntryOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of recvLocalEntryOut must be 1D."),
        return ge::GRAPH_FAILED);
    totalRecv = recvLocalEntryOutShape->GetStorageShape().GetDim(0);

    const gert::StorageShape *numRecvOutShape = context->GetOutputShape(OUT_NUM_RECV);
    OP_TILING_CHECK(numRecvOutShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "numRecvOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRecvOutShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "numRecvOut",
            (std::to_string(numRecvOutShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of numRecvOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRecvOutShape->GetStorageShape().GetDim(0) != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "numRecvOut",
            (std::string("dim0=") + std::to_string(numRecvOutShape->GetStorageShape().GetDim(0))).c_str(),
            "dim0 must be 1."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    ge::Format commContextFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(commContextDesc->GetStorageFormat()));
    OP_TILING_CHECK(commContextFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "commContext",
            Ops::Base::ToString(commContextFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(IN_INDICES);
    ge::Format indicesFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(indicesDesc->GetStorageFormat()));
    OP_TILING_CHECK(indicesFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "indices",
            Ops::Base::ToString(indicesFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(OUT_FETCHED);
    ge::Format fetchedFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(fetchedDesc->GetStorageFormat()));
    OP_TILING_CHECK(fetchedFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "fetched",
            Ops::Base::ToString(fetchedFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto permOutDesc = context->GetOutputDesc(OUT_PERM);
    ge::Format permOutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(permOutDesc->GetStorageFormat()));
    OP_TILING_CHECK(permOutFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "permOut",
            Ops::Base::ToString(permOutFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto sendCountsOutDesc = context->GetOutputDesc(OUT_SEND_COUNTS);
    ge::Format sendCountsOutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(sendCountsOutDesc->GetStorageFormat()));
    OP_TILING_CHECK(sendCountsOutFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "sendCountsOut",
            Ops::Base::ToString(sendCountsOutFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto recvCountsOutDesc = context->GetOutputDesc(OUT_RECV_COUNTS);
    ge::Format recvCountsOutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(recvCountsOutDesc->GetStorageFormat()));
    OP_TILING_CHECK(recvCountsOutFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "recvCountsOut",
            Ops::Base::ToString(recvCountsOutFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto recvLocalEntryOutDesc = context->GetOutputDesc(OUT_RECV_LOCAL_ENTRY);
    ge::Format recvLocalEntryOutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(recvLocalEntryOutDesc->GetStorageFormat()));
    OP_TILING_CHECK(recvLocalEntryOutFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "recvLocalEntryOut",
            Ops::Base::ToString(recvLocalEntryOutFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto numRecvOutDesc = context->GetOutputDesc(OUT_NUM_RECV);
    ge::Format numRecvOutFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(numRecvOutDesc->GetStorageFormat()));
    OP_TILING_CHECK(numRecvOutFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "numRecvOut",
            Ops::Base::ToString(numRecvOutFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckEngramFetchTrain(const gert::TilingContext *context, int64_t &numTokens,
                                                   uint32_t &rankSize, int64_t &totalRecv)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDim(context, numTokens, rankSize, totalRecv) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params shape is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorFormat(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params format is invalid."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE);
    OP_TILING_CHECK(hiddenSizePtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "hidden_size"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*hiddenSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*hiddenSizePtr % HIDDEN_SIZE_ALIGN != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(),
            (std::string("must be ") + std::to_string(HIDDEN_SIZE_ALIGN) + "-aligned").c_str()),
        return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*numEntriesPerRankPtr < 0 || *numEntriesPerRankPtr > INT32_MAX,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
            std::to_string(*numEntriesPerRankPtr).c_str(), "[0, INT32_MAX]"),
        return ge::GRAPH_FAILED);

    auto localStorageAddrPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_STORAGE_ADDR);
    OP_TILING_CHECK(localStorageAddrPtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "local_storage_addr"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*localStorageAddrPtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "local_storage_addr",
            std::to_string(*localStorageAddrPtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetPlatformInfo(gert::TilingContext *context, EngramFetchTrainTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschNumBlocks(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchTrainTilingData &tilingData,
                                     int64_t numTokens, uint32_t rankSize, int64_t totalRecv)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;
    tilingData.rankSize = rankSize;
    tilingData.totalRecv = totalRecv;

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE);
    tilingData.hiddenDim = *hiddenSizePtr;

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK);
    tilingData.numEntriesPerRank = static_cast<int32_t>(*numEntriesPerRankPtr);

    auto fetchedDesc = context->GetOutputDesc(OUT_FETCHED);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    int64_t hiddenDim = tilingData.hiddenDim;
    int64_t bytesPerElem = ge::GetSizeByDataType(fetchedDtype);
    OP_TILING_CHECK(hiddenDim > INT64_MAX / bytesPerElem,
        OP_LOGE(nodeName, "hiddenBytes overflow: hiddenDim=%ld * bytesPerElem=%ld exceeds INT64_MAX",
            hiddenDim, bytesPerElem),
        return ge::GRAPH_FAILED);
    tilingData.hiddenBytes = hiddenDim * bytesPerElem;

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData.ubSize = ubSize;

    tilingData.numMaxTokensPerRank = (rankSize > 0) ? (totalRecv / static_cast<int64_t>(rankSize)) : 0;

    auto localStorageAddrPtr = attrs->GetAttrPointer<int64_t>(ATTR_LOCAL_STORAGE_ADDR);
    tilingData.localStorageAddr = *localStorageAddrPtr;

    OP_LOGD(nodeName, "SetTilingData: numTokens=%ld, hiddenDim=%ld, numEntriesPerRank=%d, hiddenBytes=%ld, "
            "ubSize=%lu, rankSize=%u, numMaxTokensPerRank=%ld, totalRecv=%ld, localStorageAddr=%ld",
            tilingData.numTokens, tilingData.hiddenDim,
            tilingData.numEntriesPerRank, tilingData.hiddenBytes, tilingData.ubSize,
            tilingData.rankSize, tilingData.numMaxTokensPerRank, tilingData.totalRecv,
            tilingData.localStorageAddr);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ENGRAM_FETCH_TRAIN_DEFAULT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in engram_fetch_train.", tilingKey);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context,
                                    const EngramFetchTrainTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    int64_t numRanks = static_cast<int64_t>(tilingData.rankSize);
    int64_t numTokens = tilingData.numTokens;
    int64_t hiddenBytes = tilingData.hiddenBytes;
    int64_t totalRecv = tilingData.totalRecv;

    OP_TILING_CHECK(numTokens > 0 && hiddenBytes > INT64_MAX / numTokens,
        OP_LOGE(nodeName, "workspace overflow: numTokens=%ld, hiddenBytes=%ld", numTokens, hiddenBytes),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(totalRecv > 0 && hiddenBytes > INT64_MAX / totalRecv,
        OP_LOGE(nodeName, "workspace overflow: totalRecv=%ld, hiddenBytes=%ld", totalRecv, hiddenBytes),
        return ge::GRAPH_FAILED);

    int64_t wsSdispls = numRanks * static_cast<int64_t>(sizeof(int64_t));
    int64_t wsRdispls = numRanks * static_cast<int64_t>(sizeof(int64_t));
    int64_t wsSortedIndices = numTokens * static_cast<int64_t>(sizeof(int32_t));
    int64_t wsTokenIdxRank = numTokens * static_cast<int64_t>(sizeof(int32_t));
    int64_t wsRecvIndices = totalRecv * static_cast<int64_t>(sizeof(int32_t));
    int64_t wsLocalData = totalRecv * hiddenBytes;
    int64_t wsRecvData = numTokens * hiddenBytes;
    int64_t wsFlagScratch = 32;

    int64_t wsTotal = wsSdispls + wsRdispls + wsSortedIndices + wsTokenIdxRank
                    + wsRecvIndices + wsLocalData + wsRecvData + wsFlagScratch;
    wsTotal = AlignTo(wsTotal, BUFFER_ALIGNMENT);
    wsTotal += SYSTEM_NEED_WORKSPACE;

    workSpaces[0] = static_cast<size_t>(wsTotal);
    OP_LOGD(nodeName, "training workspace: sdispls=%ld, rdispls=%ld, sortedIndices=%ld, tokenIdxInRank=%ld, "
            "recvIndices=%ld, localData=%ld, recvData=%ld, flagScratch=%ld, total=%zu",
            wsSdispls, wsRdispls, wsSortedIndices, wsTokenIdxRank,
            wsRecvIndices, wsLocalData, wsRecvData, wsFlagScratch, workSpaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EngramFetchTrainTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr,
        OP_LOGE("engram_fetch_train_tiling", "failed to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter EngramFetchTrain tiling func.");

    EngramFetchTrainTilingData *tilingData = context->GetTilingData<EngramFetchTrainTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    int64_t numTokens = 0;
    uint32_t rankSize = 0;
    int64_t totalRecv = 0;
    OP_TILING_CHECK(TilingCheckEngramFetchTrain(context, numTokens, rankSize, totalRecv) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check input/output failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attrs failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set platform info failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens, rankSize, totalRecv) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set tiling data failed."), return ge::GRAPH_FAILED);

    SetTilingKey(context);

    OP_TILING_CHECK(SetWorkSpace(context, *tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set workspace failed."), return ge::GRAPH_FAILED);

    PrintEngramFetchTrainTilingData(tilingData, nodeName);
    OP_LOGI(nodeName, "EngramFetchTrain tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetchTrain)
    .Tiling(EngramFetchTrainTilingFunc);
} // namespace MC2Tiling
