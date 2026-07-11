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
* \file engram_fetch_grad_tiling.cpp
* \brief host侧tiling实现 — EngramFetchGrad
*        校验 6 input + 3 output（dtype/shape/format）
*        hiddenBytes 按 fp32 计算（反向 FP32 聚合）
*        rankSize 从 sendCounts dim0 获取
*        workspace: max(a2a_grad, scatter_add) + 16MB
*/

#include <string>
#include <climits>
#include <cstdint>
#include <algorithm>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/engram_fetch_grad_tiling_data.h"
#include "../../op_kernel/engram_fetch_grad_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace MC2Tiling {
constexpr uint32_t IN_COMM_CONTEXT = 0U;
constexpr uint32_t IN_GRAD_FETCHED = 1U;
constexpr uint32_t IN_PERM = 2U;
constexpr uint32_t IN_SEND_COUNTS = 3U;
constexpr uint32_t IN_RECV_COUNTS = 4U;
constexpr uint32_t IN_RECV_LOCAL_ENTRY = 5U;
constexpr uint32_t OUT_GRAD_UNIQUE = 0U;
constexpr uint32_t OUT_UNIQUE_LOCAL_ENTRY = 1U;
constexpr uint32_t OUT_NUM_UNIQUE = 2U;

constexpr uint32_t ATTR_HIDDEN_SIZE = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK = 1U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr int32_t HIDDEN_SIZE_ALIGN = 128;
constexpr int64_t BUFFER_ALIGNMENT = 2 * 1024 * 1024;

static const std::vector<ge::DataType> GRAD_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

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

static void PrintEngramFetchGradTilingData(const EngramFetchGradTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== EngramFetchGradTilingData ==========");
    OP_LOGD(nodeName, "numTokens is %ld", tilingData->numTokens);
    OP_LOGD(nodeName, "numEntriesPerRank is %d", tilingData->numEntriesPerRank);
    OP_LOGD(nodeName, "hiddenDim is %ld", tilingData->hiddenDim);
    OP_LOGD(nodeName, "hiddenBytes is %ld", tilingData->hiddenBytes);
    OP_LOGD(nodeName, "aivNum is %u", tilingData->aivNum);
}

static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
{
    auto commContextDesc = context->GetInputDesc(IN_COMM_CONTEXT);
    auto gradDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    auto permDesc = context->GetInputDesc(IN_PERM);
    auto sendCountsDesc = context->GetInputDesc(IN_SEND_COUNTS);
    auto recvCountsDesc = context->GetInputDesc(IN_RECV_COUNTS);
    auto recvLocalEntryDesc = context->GetInputDesc(IN_RECV_LOCAL_ENTRY);
    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);

    OP_CHECK_NULL_WITH_CONTEXT(context, commContextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, permDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, sendCountsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvCountsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvLocalEntryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradUniqueDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, uniqueLocalEntryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, numUniqueDesc);

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

    auto gradDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    OP_TILING_CHECK(!IsContains(GRAD_DTYPE_LIST, gradDesc->GetDataType()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "gradFetched",
            Ops::Base::ToString(gradDesc->GetDataType()).c_str(),
            "The dtype of gradFetched must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
        return ge::GRAPH_FAILED);

    auto permDesc = context->GetInputDesc(IN_PERM);
    OP_TILING_CHECK(permDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "perm",
            Ops::Base::ToString(permDesc->GetDataType()).c_str(),
            "The dtype of perm must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto sendCountsDesc = context->GetInputDesc(IN_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "sendCounts",
            Ops::Base::ToString(sendCountsDesc->GetDataType()).c_str(),
            "The dtype of sendCounts must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto recvCountsDesc = context->GetInputDesc(IN_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "recvCounts",
            Ops::Base::ToString(recvCountsDesc->GetDataType()).c_str(),
            "The dtype of recvCounts must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto recvLocalEntryDesc = context->GetInputDesc(IN_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "recvLocalEntry",
            Ops::Base::ToString(recvLocalEntryDesc->GetDataType()).c_str(),
            "The dtype of recvLocalEntry must be DT_INT32."),
        return ge::GRAPH_FAILED);

    // gradUniqueOut: fp32（反向 FP32 聚合）
    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(gradUniqueDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "gradUniqueOut",
            Ops::Base::ToString(gradUniqueDesc->GetDataType()).c_str(),
            "The dtype of gradUniqueOut must be DT_FLOAT."),
        return ge::GRAPH_FAILED);

    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(uniqueLocalEntryDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "uniqueLocalEntryOut",
            Ops::Base::ToString(uniqueLocalEntryDesc->GetDataType()).c_str(),
            "The dtype of uniqueLocalEntryOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);
    OP_TILING_CHECK(numUniqueDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "numUniqueOut",
            Ops::Base::ToString(numUniqueDesc->GetDataType()).c_str(),
            "The dtype of numUniqueOut must be DT_INT32."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, int64_t &numTokens)
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

    // gradFetched: 2D (T, H)
    const gert::StorageShape *gradShape = context->GetInputShape(IN_GRAD_FETCHED);
    OP_TILING_CHECK(gradShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradFetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gradShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "gradFetched",
            (std::to_string(gradShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradFetched must be 2D."),
        return ge::GRAPH_FAILED);
    numTokens = gradShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(numTokens < 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "gradFetched",
            (std::string("dim0=") + std::to_string(numTokens)).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);

    // perm: 1D (T,)
    const gert::StorageShape *permShape = context->GetInputShape(IN_PERM);
    OP_TILING_CHECK(permShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "perm"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "perm",
            (std::to_string(permShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of perm must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(permShape->GetStorageShape().GetDim(0) != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "perm",
            (std::string("dim0=") + std::to_string(permShape->GetStorageShape().GetDim(0))).c_str(),
            (std::string("dim0 must equal gradFetched dim0=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    // sendCounts: 1D (W,)
    const gert::StorageShape *sendCountsShape = context->GetInputShape(IN_SEND_COUNTS);
    OP_TILING_CHECK(sendCountsShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "sendCounts"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(sendCountsShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "sendCounts",
            (std::to_string(sendCountsShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of sendCounts must be 1D."),
        return ge::GRAPH_FAILED);

    // recvCounts: 1D (W,)
    const gert::StorageShape *recvCountsShape = context->GetInputShape(IN_RECV_COUNTS);
    OP_TILING_CHECK(recvCountsShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvCounts"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvCountsShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "recvCounts",
            (std::to_string(recvCountsShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of recvCounts must be 1D."),
        return ge::GRAPH_FAILED);

    // recvLocalEntry: 1D (R,)
    const gert::StorageShape *recvLocalEntryShape = context->GetInputShape(IN_RECV_LOCAL_ENTRY);
    OP_TILING_CHECK(recvLocalEntryShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "recvLocalEntry"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "recvLocalEntry",
            (std::to_string(recvLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of recvLocalEntry must be 1D."),
        return ge::GRAPH_FAILED);

    // gradUniqueOut: 2D
    const gert::StorageShape *gradUniqueShape = context->GetOutputShape(OUT_GRAD_UNIQUE);
    OP_TILING_CHECK(gradUniqueShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "gradUniqueOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gradUniqueShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "gradUniqueOut",
            (std::to_string(gradUniqueShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of gradUniqueOut must be 2D."),
        return ge::GRAPH_FAILED);

    // uniqueLocalEntryOut: 1D
    const gert::StorageShape *uniqueLocalEntryShape = context->GetOutputShape(OUT_UNIQUE_LOCAL_ENTRY);
    OP_TILING_CHECK(uniqueLocalEntryShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "uniqueLocalEntryOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(uniqueLocalEntryShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "uniqueLocalEntryOut",
            (std::to_string(uniqueLocalEntryShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of uniqueLocalEntryOut must be 1D."),
        return ge::GRAPH_FAILED);

    // numUniqueOut: 1D, dim0 == 1
    const gert::StorageShape *numUniqueShape = context->GetOutputShape(OUT_NUM_UNIQUE);
    OP_TILING_CHECK(numUniqueShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "numUniqueOut"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numUniqueShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "numUniqueOut",
            (std::to_string(numUniqueShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of numUniqueOut must be 1D."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numUniqueShape->GetStorageShape().GetDim(0) != 1,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "numUniqueOut",
            (std::string("dim0=") + std::to_string(numUniqueShape->GetStorageShape().GetDim(0))).c_str(),
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

    auto gradFetchedDesc = context->GetInputDesc(IN_GRAD_FETCHED);
    ge::Format gradFetchedFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(gradFetchedDesc->GetStorageFormat()));
    OP_TILING_CHECK(gradFetchedFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "gradFetched",
            Ops::Base::ToString(gradFetchedFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto permDesc = context->GetInputDesc(IN_PERM);
    ge::Format permFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(permDesc->GetStorageFormat()));
    OP_TILING_CHECK(permFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "perm",
            Ops::Base::ToString(permFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto sendCountsDesc = context->GetInputDesc(IN_SEND_COUNTS);
    ge::Format sendCountsFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(sendCountsDesc->GetStorageFormat()));
    OP_TILING_CHECK(sendCountsFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "sendCounts",
            Ops::Base::ToString(sendCountsFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto recvCountsDesc = context->GetInputDesc(IN_RECV_COUNTS);
    ge::Format recvCountsFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(recvCountsDesc->GetStorageFormat()));
    OP_TILING_CHECK(recvCountsFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "recvCounts",
            Ops::Base::ToString(recvCountsFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto recvLocalEntryDesc = context->GetInputDesc(IN_RECV_LOCAL_ENTRY);
    ge::Format recvLocalEntryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(recvLocalEntryDesc->GetStorageFormat()));
    OP_TILING_CHECK(recvLocalEntryFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "recvLocalEntry",
            Ops::Base::ToString(recvLocalEntryFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto gradUniqueDesc = context->GetOutputDesc(OUT_GRAD_UNIQUE);
    ge::Format gradUniqueFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(gradUniqueDesc->GetStorageFormat()));
    OP_TILING_CHECK(gradUniqueFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "gradUniqueOut",
            Ops::Base::ToString(gradUniqueFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto uniqueLocalEntryDesc = context->GetOutputDesc(OUT_UNIQUE_LOCAL_ENTRY);
    ge::Format uniqueLocalEntryFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(uniqueLocalEntryDesc->GetStorageFormat()));
    OP_TILING_CHECK(uniqueLocalEntryFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "uniqueLocalEntryOut",
            Ops::Base::ToString(uniqueLocalEntryFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto numUniqueDesc = context->GetOutputDesc(OUT_NUM_UNIQUE);
    ge::Format numUniqueFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(numUniqueDesc->GetStorageFormat()));
    OP_TILING_CHECK(numUniqueFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "numUniqueOut",
            Ops::Base::ToString(numUniqueFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingCheckEngramFetchGrad(const gert::TilingContext *context, int64_t &numTokens)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckTensorPtrNullptr(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params check nullptr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDataType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "params dataType is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckTensorDim(context, numTokens) != ge::GRAPH_SUCCESS,
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

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetPlatformInfo(gert::TilingContext *context, EngramFetchGradTilingData &tilingData)
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

static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchGradTilingData &tilingData,
                                     int64_t numTokens)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE);
    tilingData.hiddenDim = *hiddenSizePtr;

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK);
    tilingData.numEntriesPerRank = static_cast<int32_t>(*numEntriesPerRankPtr);

    // hiddenBytes 按 fp32 算（反向 FP32 聚合）
    int64_t bytesPerElem = ge::GetSizeByDataType(ge::DT_FLOAT);
    OP_TILING_CHECK(tilingData.hiddenDim > INT64_MAX / bytesPerElem,
        OP_LOGE(nodeName, "hiddenBytes overflow: hiddenDim=%ld * bytesPerElem=%ld exceeds INT64_MAX",
            tilingData.hiddenDim, bytesPerElem),
        return ge::GRAPH_FAILED);
    tilingData.hiddenBytes = tilingData.hiddenDim * bytesPerElem;

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    tilingData.ubSize = ubSize;

    OP_LOGD(nodeName, "SetTilingData: numTokens=%ld, hiddenDim=%ld, numEntriesPerRank=%d, hiddenBytes=%ld, "
            "ubSize=%lu",
            tilingData.numTokens, tilingData.hiddenDim,
            tilingData.numEntriesPerRank, tilingData.hiddenBytes, tilingData.ubSize);
    return ge::GRAPH_SUCCESS;
}

static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ENGRAM_FETCH_GRAD_DEFAULT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in engram_fetch_grad.", tilingKey);
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context,
                                    const EngramFetchGradTilingData &tilingData, int64_t numTokens)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    // a2a grad_sorted: send(T) + recv(R_max ≈ T)
    int64_t wsGrad = numTokens * tilingData.hiddenBytes * 2;
    // scatter-add 中间结果
    int64_t wsScatter = numTokens * tilingData.hiddenBytes;
    int64_t wsTotal = std::max(wsGrad, wsScatter);
    wsTotal = AlignTo(wsTotal, BUFFER_ALIGNMENT);
    wsTotal += SYSTEM_NEED_WORKSPACE;

    workSpaces[0] = static_cast<size_t>(wsTotal);
    OP_LOGD(nodeName, "workspace: wsGrad=%ld, wsScatter=%ld, total=%zu",
            wsGrad, wsScatter, workSpaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus EngramFetchGradTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr,
        OP_LOGE("engram_fetch_grad_tiling", "failed to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter EngramFetchGrad tiling func.");

    EngramFetchGradTilingData *tilingData = context->GetTilingData<EngramFetchGradTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    int64_t numTokens = 0;
    OP_TILING_CHECK(TilingCheckEngramFetchGrad(context, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check input/output failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attrs failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set platform info failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set tiling data failed."), return ge::GRAPH_FAILED);

    SetTilingKey(context);

    OP_TILING_CHECK(SetWorkSpace(context, *tilingData, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set workspace failed."), return ge::GRAPH_FAILED);

    PrintEngramFetchGradTilingData(tilingData, nodeName);
    OP_LOGI(nodeName, "EngramFetchGrad tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetchGrad)
    .Tiling(EngramFetchGradTilingFunc);
} // namespace MC2Tiling
