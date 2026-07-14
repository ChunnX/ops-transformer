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
* \file engram_fetch_tiling.cpp
* \brief host侧tiling实现
*/

#include <string>
#include <climits>
#include <cstdint>
#include <algorithm>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../../op_kernel/engram_fetch_tiling_data.h"
#include "../../../op_kernel/engram_fetch_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace MC2Tiling {
constexpr uint32_t COMM_CONTEXT_INDEX = 0U;
constexpr uint32_t INDICES_INDEX = 1U;
constexpr uint32_t FETCHED_INDEX = 0U;

constexpr uint32_t ATTR_HIDDEN_SIZE_INDEX = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK_INDEX = 1U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr int32_t HIDDEN_SIZE_ALIGN = 128;

static const std::vector<ge::DataType> OUTPUT_DTYPE_LIST = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static bool IsContains(const std::vector<ge::DataType> &list, ge::DataType value)
{
    return std::find(list.begin(), list.end(), value) != list.end();
}

/**
 * @brief 打印tiling数据
 */
static void PrintEngramFetchTilingData(const EngramFetchTilingData *tilingData, const char *nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);

    OP_LOGD(nodeName, "========== EngramFetchTilingData ==========");
    OP_LOGD(nodeName, "numTokens is %ld", tilingData->numTokens);
    OP_LOGD(nodeName, "numEntriesPerRank is %d", tilingData->numEntriesPerRank);
    OP_LOGD(nodeName, "hiddenDim is %ld", tilingData->hiddenDim);
    OP_LOGD(nodeName, "hiddenBytes is %ld", tilingData->hiddenBytes);
    OP_LOGD(nodeName, "aivNum is %u", tilingData->aivNum);
}

/**
 * @brief 校验tensor指针非空
 */
static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
{
    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, commContextDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedDesc);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor数据类型
 */
static ge::graphStatus CheckTensorDataType(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    OP_TILING_CHECK(commContextDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "commContext",
            Ops::Base::ToString(commContextDesc->GetDataType()).c_str(),
            "The dtype of commContext must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    OP_TILING_CHECK(indicesDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "indices",
            Ops::Base::ToString(indicesDesc->GetDataType()).c_str(),
            "The dtype of indices must be DT_INT32."),
        return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, fetchedDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "fetched",
            Ops::Base::ToString(fetchedDtype).c_str(),
            "The dtype of fetched must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor维度
 */
static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, int64_t &numTokens)
{
    const char *nodeName = context->GetNodeName();
    // input dim check
    const gert::StorageShape *commContextShape = context->GetInputShape(COMM_CONTEXT_INDEX);
    OP_TILING_CHECK(commContextShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "commContext"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(commContextShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "commContext",
            (std::to_string(commContextShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of commContext must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t commContextDim0 = commContextShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "commContext dim0 = %ld", commContextDim0);
    OP_TILING_CHECK(commContextDim0 <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "commContext",
            (std::string("dim0=") + std::to_string(commContextDim0)).c_str(), "> 0"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *indicesShape = context->GetInputShape(INDICES_INDEX);
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
    OP_LOGD(nodeName, "indices dim0 (numTokens) = %ld", numTokens);

    // output dim check
    const gert::StorageShape *fetchedShape = context->GetOutputShape(FETCHED_INDEX);
    OP_TILING_CHECK(fetchedShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "fetched"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(fetchedShape->GetStorageShape().GetDimNum() != DIM_TWO,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "fetched",
            (std::to_string(fetchedShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of fetched must be 2D."),
        return ge::GRAPH_FAILED);
    const int64_t fetchedDim0 = fetchedShape->GetStorageShape().GetDim(0);
    const int64_t fetchedDim1 = fetchedShape->GetStorageShape().GetDim(1);
    OP_LOGD(nodeName, "fetched dim0 = %ld", fetchedDim0);
    OP_LOGD(nodeName, "fetched dim1 = %ld", fetchedDim1);
    OP_TILING_CHECK(fetchedDim0 != numTokens,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "fetched",
            (std::string("dim0=") + std::to_string(fetchedDim0)).c_str(),
            (std::string("dim0 must equal numTokens=") + std::to_string(numTokens)).c_str()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验tensor格式
 */
static ge::graphStatus CheckTensorFormat(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    auto commContextDesc = context->GetInputDesc(COMM_CONTEXT_INDEX);
    ge::Format commContextFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(commContextDesc->GetStorageFormat()));
    OP_TILING_CHECK(commContextFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "commContext",
            Ops::Base::ToString(commContextFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    ge::Format indicesFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(indicesDesc->GetStorageFormat()));
    OP_TILING_CHECK(indicesFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "indices",
            Ops::Base::ToString(indicesFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    ge::Format fetchedFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(fetchedDesc->GetStorageFormat()));
    OP_TILING_CHECK(fetchedFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "fetched",
            Ops::Base::ToString(fetchedFormat).c_str(), "ND"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验所有tensor（指针、数据类型、维度、格式）
 */
static ge::graphStatus TilingCheckEngramFetch(const gert::TilingContext *context, int64_t &numTokens)
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

/**
 * @brief 校验属性指针非空
 */
static ge::graphStatus CheckAttrPtrNullptr(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(hiddenSizePtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "hidden_size"), return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验属性参数值
 */
static ge::graphStatus CheckAttrParams(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    const gert::StorageShape *fetchedShape = context->GetOutputShape(FETCHED_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, fetchedShape);

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(*hiddenSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*hiddenSizePtr % HIDDEN_SIZE_ALIGN != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(),
            (std::string("must be ") + std::to_string(HIDDEN_SIZE_ALIGN) + "-aligned").c_str()),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "hidden_size is %ld", *hiddenSizePtr);

    // fetched dim1 must equal hidden_size attr
    int64_t hiddenDim = *hiddenSizePtr;
    const int64_t fetchedDim1 = fetchedShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(fetchedDim1 != hiddenDim,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "fetched",
            (std::string("dim1=") + std::to_string(fetchedDim1)).c_str(),
            (std::string("dim1 must equal hidden_size attr(") + std::to_string(hiddenDim) + ").").c_str()),
        return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(*numEntriesPerRankPtr < 0 || *numEntriesPerRankPtr > INT32_MAX,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
            std::to_string(*numEntriesPerRankPtr).c_str(), "[0, INT32_MAX]"),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "num_entries_per_rank is %ld", *numEntriesPerRankPtr);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 校验所有属性
 */
static ge::graphStatus CheckAttrs(const gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    OP_TILING_CHECK(CheckAttrPtrNullptr(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "attr params check nullptr failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrParams(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attr params failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置平台信息
 */
static ge::graphStatus SetPlatformInfo(gert::TilingContext *context, EngramFetchTilingData &tilingData)
{
    const char *nodeName = context->GetNodeName();

    auto platformInfo = context->GetPlatformInfo();
    OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling数据
 */
static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchTilingData &tilingData,
                                     int64_t numTokens)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;

    auto hiddenSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HIDDEN_SIZE_INDEX);
    tilingData.hiddenDim = *hiddenSizePtr;

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    tilingData.numEntriesPerRank = static_cast<int32_t>(*numEntriesPerRankPtr);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
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

    OP_LOGD(nodeName, "SetTilingData: numTokens=%ld, hiddenDim=%ld, numEntriesPerRank=%d, hiddenBytes=%ld, ubSize=%lu",
            tilingData.numTokens, tilingData.hiddenDim,
            tilingData.numEntriesPerRank, tilingData.hiddenBytes, tilingData.ubSize);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling key
 */
static void SetTilingKey(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    const uint64_t tilingKey = GET_TPL_TILING_KEY(ENGRAM_FETCH_DEFAULT_MODE);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey is [%lu] in engram_fetch.", tilingKey);
}

/**
 * @brief 设置workspace大小
 */
static ge::graphStatus SetWorkSpace(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief engram_fetch算子的tiling函数
 */
static ge::graphStatus EngramFetchTilingFunc(gert::TilingContext *context)
{
    OP_TILING_CHECK(context == nullptr,
        OP_LOGE("engram_fetch_tiling", "failed to get tiling context."), return ge::GRAPH_FAILED);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "nodeName"), return ge::GRAPH_FAILED);

    OP_LOGI(nodeName, "Enter EngramFetch tiling check func.");

    EngramFetchTilingData *tilingData = context->GetTilingData<EngramFetchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return ge::GRAPH_FAILED);

    // 1. tensor check (ptr + dtype + shape + format)
    int64_t numTokens = 0;
    OP_TILING_CHECK(TilingCheckEngramFetch(context, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check input/output failed."), return ge::GRAPH_FAILED);

    // 2. attr check
    OP_TILING_CHECK(CheckAttrs(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "check attrs failed."), return ge::GRAPH_FAILED);

    // 3. platform info
    OP_TILING_CHECK(SetPlatformInfo(context, *tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set platform info failed."), return ge::GRAPH_FAILED);

    // 4. set tiling data
    OP_TILING_CHECK(SetTilingData(context, *tilingData, numTokens) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set tiling data failed."), return ge::GRAPH_FAILED);

    // 5. set tiling key
    SetTilingKey(context);

    // 6. set workspace
    OP_TILING_CHECK(SetWorkSpace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "set workspace failed."), return ge::GRAPH_FAILED);

    // 7. print info
    PrintEngramFetchTilingData(tilingData, nodeName);
    OP_LOGI(nodeName, "EngramFetch tiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetch)
    .Tiling(EngramFetchTilingFunc);
} // namespace MC2Tiling
