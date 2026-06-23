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
#include <climits>
#include <algorithm>
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/engram_fetch_tiling_data.h"
#include "../../op_kernel/engram_fetch_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {
constexpr uint32_t BUFFER_HANDLE_INDEX = 0U;
constexpr uint32_t INDICES_INDEX = 1U;
constexpr uint32_t FETCHED_INDEX = 0U;

constexpr uint32_t ATTR_HIDDEN_SIZE_INDEX = 0U;
constexpr uint32_t ATTR_NUM_ENTRIES_PER_RANK_INDEX = 1U;
constexpr uint32_t ATTR_OUTPUT_DTYPE_INDEX = 2U;

constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;

constexpr uint32_t BF16_BYTES = 2U;
constexpr uint32_t FP16_BYTES = 2U;
constexpr uint32_t FP32_BYTES = 4U;

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
    OP_LOGD(nodeName, "numTokens is %u", tilingData->numTokens);
    OP_LOGD(nodeName, "numEntriesPerRank is %u", tilingData->numEntriesPerRank);
    OP_LOGD(nodeName, "hiddenDim is %u", tilingData->hiddenDim);
    OP_LOGD(nodeName, "hiddenBytes is %u", tilingData->hiddenBytes);
    OP_LOGD(nodeName, "aivNum is %u", tilingData->aivNum);
}

/**
 * @brief 校验tensor指针非空
 */
static ge::graphStatus CheckTensorPtrNullptr(const gert::TilingContext *context)
{
    auto bufferHandleDesc = context->GetInputDesc(BUFFER_HANDLE_INDEX);
    auto indicesDesc = context->GetInputDesc(INDICES_INDEX);
    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, bufferHandleDesc);
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

    auto bufferHandleDesc = context->GetInputDesc(BUFFER_HANDLE_INDEX);
    OP_TILING_CHECK(bufferHandleDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "bufferHandle",
            Ops::Base::ToString(bufferHandleDesc->GetDataType()).c_str(),
            "The dtype of bufferHandle must be DT_INT32."),
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
static ge::graphStatus CheckTensorDim(const gert::TilingContext *context, uint32_t &numTokens)
{
    const char *nodeName = context->GetNodeName();
    // input dim check
    const gert::StorageShape *bufferHandleShape = context->GetInputShape(BUFFER_HANDLE_INDEX);
    OP_TILING_CHECK(bufferHandleShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "bufferHandle"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(bufferHandleShape->GetStorageShape().GetDimNum() != DIM_ONE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(nodeName, "bufferHandle",
            (std::to_string(bufferHandleShape->GetStorageShape().GetDimNum()) + "D").c_str(),
            "The shape dim of bufferHandle must be 1D."),
        return ge::GRAPH_FAILED);
    int64_t bufferHandleDim0 = bufferHandleShape->GetStorageShape().GetDim(0);
    OP_LOGD(nodeName, "bufferHandle dim0 = %ld", bufferHandleDim0);

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
    OP_LOGD(nodeName, "indices dim0 (numTokens) = %u", numTokens);

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
    OP_TILING_CHECK(fetchedDim0 != static_cast<int64_t>(numTokens),
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

    auto bufferHandleDesc = context->GetInputDesc(BUFFER_HANDLE_INDEX);
    ge::Format bufferHandleFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(bufferHandleDesc->GetStorageFormat()));
    OP_TILING_CHECK(bufferHandleFormat != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "bufferHandle",
            Ops::Base::ToString(bufferHandleFormat).c_str(), "ND"),
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
static ge::graphStatus TilingCheckEngramFetch(const gert::TilingContext *context, uint32_t &numTokens)
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

    auto hiddenSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(hiddenSizePtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "hidden_size"), return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(numEntriesPerRankPtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "num_entries_per_rank"), return ge::GRAPH_FAILED);

    auto outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_OUTPUT_DTYPE_INDEX);
    OP_TILING_CHECK(outputDtypePtr == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(nodeName, "output_dtype"), return ge::GRAPH_FAILED);

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

    auto hiddenSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_HIDDEN_SIZE_INDEX);
    OP_TILING_CHECK(*hiddenSizePtr <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(), "> 0"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*hiddenSizePtr % HIDDEN_SIZE_ALIGN != 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "hidden_size",
            std::to_string(*hiddenSizePtr).c_str(),
            (std::string("must be ") + std::to_string(HIDDEN_SIZE_ALIGN) + "-aligned").c_str()),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "hidden_size is %d", *hiddenSizePtr);

    // fetched dim1 must equal hidden_size attr
    uint32_t hiddenDim = static_cast<uint32_t>(*hiddenSizePtr);
    const int64_t fetchedDim1 = fetchedShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(fetchedDim1 != static_cast<int64_t>(hiddenDim),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "fetched",
            (std::string("dim1=") + std::to_string(fetchedDim1)).c_str(),
            (std::string("dim1 must equal hidden_size attr(") + std::to_string(hiddenDim) + ").").c_str()),
        return ge::GRAPH_FAILED);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    OP_TILING_CHECK(*numEntriesPerRankPtr < 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "num_entries_per_rank",
            std::to_string(*numEntriesPerRankPtr).c_str(), ">= 0"),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "num_entries_per_rank is %ld", *numEntriesPerRankPtr);

    auto outputDtypePtr = attrs->GetAttrPointer<int64_t>(ATTR_OUTPUT_DTYPE_INDEX);
    ge::DataType outputDtype = static_cast<ge::DataType>(*outputDtypePtr);
    OP_TILING_CHECK(!IsContains(OUTPUT_DTYPE_LIST, outputDtype),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "output_dtype",
            Ops::Base::ToString(outputDtype).c_str(),
            "The value of output_dtype must be DT_BF16, DT_FLOAT16 or DT_FLOAT."),
        return ge::GRAPH_FAILED);
    OP_LOGD(nodeName, "output_dtype is %ld", *outputDtypePtr);

    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    OP_TILING_CHECK(fetchedDtype != outputDtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(nodeName, "fetched",
            Ops::Base::ToString(fetchedDtype).c_str(),
            (std::string("fetched dtype must match output_dtype attr(") +
             Ops::Base::ToString(outputDtype).c_str() + ")").c_str()),
        return ge::GRAPH_FAILED);

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
    uint32_t numBlocks = ascendcPlatform.CalcTschNumBlocks(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData.aivNum = aivNum;
    OP_LOGD(nodeName, "aicNum=%u, aivNum=%u, numBlocks=%u", aicNum, aivNum, numBlocks);

    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 设置tiling数据
 */
static ge::graphStatus SetTilingData(gert::TilingContext *context, EngramFetchTilingData &tilingData,
                                      uint32_t numTokens)
{
    const char *nodeName = context->GetNodeName();
    auto attrs = context->GetAttrs();

    tilingData.numTokens = numTokens;

    auto hiddenSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_HIDDEN_SIZE_INDEX);
    tilingData.hiddenDim = static_cast<uint32_t>(*hiddenSizePtr);

    auto numEntriesPerRankPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_ENTRIES_PER_RANK_INDEX);
    tilingData.numEntriesPerRank = static_cast<uint32_t>(*numEntriesPerRankPtr);

    auto fetchedDesc = context->GetOutputDesc(FETCHED_INDEX);
    ge::DataType fetchedDtype = fetchedDesc->GetDataType();
    uint32_t hiddenDim = tilingData.hiddenDim;
    if (fetchedDtype == ge::DT_BF16 || fetchedDtype == ge::DT_FLOAT16) {
        tilingData.hiddenBytes = hiddenDim * BF16_BYTES;
    } else {
        tilingData.hiddenBytes = hiddenDim * FP32_BYTES;
    }

    OP_LOGD(nodeName, "SetTilingData: numTokens=%u, hiddenDim=%u, numEntriesPerRank=%u, hiddenBytes=%u",
            tilingData.numTokens, tilingData.hiddenDim,
            tilingData.numEntriesPerRank, tilingData.hiddenBytes);
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
    uint32_t numTokens = 0;
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

struct EngramFetchCompileInfo {};

static ge::graphStatus TilingParseForEngramFetch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetch)
    .Tiling(EngramFetchTilingFunc)
    .TilingParse<EngramFetchCompileInfo>(TilingParseForEngramFetch);

} // namespace optiling
