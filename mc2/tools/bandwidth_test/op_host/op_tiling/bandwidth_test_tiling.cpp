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
 * \file bandwidth_test_tiling.cpp
 * \brief
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

#include "../../op_kernel/bandwidth_test_tiling.h"
#include "../../op_kernel/bandwidth_test_tiling_key.h"
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_templates_registry.h"

using namespace BandwidthTestTiling;
using namespace AscendC;
using namespace ge;

namespace optiling {

// ============ 输入输出索引常量 ============
constexpr uint32_t INPUT_X_INDEX = 1U;            // 输入x的索引
constexpr uint32_t INPUT_DST_RANK_ID_INDEX = 2U;  // 输入dst_rank_id的索引
constexpr uint32_t OUTPUT_Y_INDEX = 0U;           // 输出y的索引
constexpr uint32_t OUTPUT_RECEIVE_CNT_INDEX = 1U; // 输出receive_cnt的索引

// ============ 属性索引常量 ============
constexpr uint32_t ATTR_GROUP_INDEX = 0U;      // 通信组名称
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U; // 世界大小
constexpr uint32_t ATTR_MAX_BS_INDEX = 2U;     // 最大batch size
constexpr uint32_t ATTR_MODE_INDEX = 3U;       // 执行模式
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4U;
constexpr uint32_t ATTR_COMM_ALG_INDEX = 5U;
constexpr uint32_t ATTR_AIV_NUM_INDEX = 6U;    // AIV核数量
constexpr uint32_t ATTR_BATCH_TOKENS_INDEX = 7U;
constexpr uint32_t ATTR_SHARED_JETTY_INDEX = 8U;
constexpr uint32_t ATTR_SHARED_JETTY_NUM_INDEX = 9U;

// ============ 其他常量 ============
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U; // 系统预留workspace 16MB
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U; // AlltoAll操作类型
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 512UL;

// ============ 参数范围约束 ============
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MIN_WORLD_SIZE = 1;

// ============ Tiling Key ============
constexpr uint64_t TILING_KEY_BASE_A5 = 2ULL; // A5架构base key

static bool IsUrma(const gert::TilingContext *context)
{
    return std::strcmp(context->GetAttrs()->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX), "urma") == 0;
}

static bool IsSharedJetty(const gert::TilingContext *context)
{
    auto shared = context->GetAttrs()->GetAttrPointer<bool>(ATTR_SHARED_JETTY_INDEX);
    return IsUrma(context) && shared != nullptr && *shared;
}

// 打印TilingData调试信息
static void PrintTilingDataInfo(const char *nodeName, BandwidthTestTilingData &tilingData)
{
    OP_LOGD(nodeName, "data_size is %d.", tilingData.dataSize);
    OP_LOGD(nodeName, "world_size is %d.", tilingData.worldSize);
    OP_LOGD(nodeName, "max_bs is %d.", tilingData.maxBs);
    OP_LOGD(nodeName, "mode is %d.", tilingData.mode);
    OP_LOGD(nodeName, "aiv_num is %d.", tilingData.aivNum);
    OP_LOGD(nodeName, "shared_jetty_num is %d.", tilingData.sharedJettyNum);
    OP_LOGD(nodeName, "total_ub_size is %lu.", tilingData.totalUbSize);
    OP_LOGD(nodeName, "batch_buffer_bytes is %u.", tilingData.batchBufferBytes);
    OP_LOGD(nodeName, "batch_tokens is %u.", tilingData.batchTokens);
    OP_LOGD(nodeName, "total_win_size is %lu.", tilingData.totalWinSize);
}

// 校验属性指针非空
static ge::graphStatus CheckAttrPointersNotNull(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);

    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "group"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "worldSize"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxBsPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "maxBs"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(modePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "mode"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验属性值范围合法性
static ge::graphStatus CheckAttrValuesValid(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);

    OP_TILING_CHECK((strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
                        (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
                    OP_LOGE(nodeName, "group name is invalid, length should be in (0, %zu), but got length=%zu.",
                            MAX_GROUP_NAME_LENGTH, strnlen(groupPtr, MAX_GROUP_NAME_LENGTH)),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (*worldSizePtr < MIN_WORLD_SIZE || *worldSizePtr > 1024),
        OP_LOGE(nodeName, "worldSize must be in [1, 1024], got %ld.", *worldSizePtr),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*maxBsPtr <= 0 || *maxBsPtr > INT32_MAX / *worldSizePtr),
                    OP_LOGE(nodeName, "maxBs must be in [1, INT32_MAX/worldSize], got %ld.", *maxBsPtr),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(*modePtr != 0 && *modePtr != 1 && *modePtr != 3,
                    OP_LOGE(nodeName, "mode must be 0, 1 or 3, got %ld.", *modePtr),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取属性并填充TilingData
static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               BandwidthTestTilingData &tilingData, std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "attrs"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrPointersNotNull(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "attr pointers", "null", "not null"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrValuesValid(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check attr values valid failed."), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);

    group = std::string(groupPtr);
    tilingData.worldSize = static_cast<int32_t>(*worldSizePtr);
    tilingData.maxBs = static_cast<int32_t>(*maxBsPtr);
    tilingData.mode = static_cast<int32_t>(*modePtr);

    auto commAlg = attrs->GetAttrPointer<char>(ATTR_COMM_ALG_INDEX);
    OP_TILING_CHECK(commAlg == nullptr ||
                        (std::strcmp(commAlg, "") != 0 && std::strcmp(commAlg, "mte") != 0 &&
                         std::strcmp(commAlg, "urma") != 0),
                    OP_LOGE(nodeName, "comm_alg must be mte or urma."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData.mode == 3 && std::strcmp(commAlg, "urma") != 0,
                    OP_LOGE(nodeName, "mode=3 requires comm_alg=urma."), return ge::GRAPH_FAILED);

    auto shared = attrs->GetAttrPointer<bool>(ATTR_SHARED_JETTY_INDEX);
    auto jettyNum = attrs->GetAttrPointer<int64_t>(ATTR_SHARED_JETTY_NUM_INDEX);
    OP_TILING_CHECK(shared == nullptr || jettyNum == nullptr || *jettyNum < 1 ||
                        *jettyNum > URMA_MAX_SHARED_JETTYS,
                    OP_LOGE(nodeName, "shared_jetty_num must be in [1, %u].", URMA_MAX_SHARED_JETTYS),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*shared && std::strcmp(commAlg, "urma") != 0,
                    OP_LOGE(nodeName, "shared_jetty requires comm_alg=urma."), return ge::GRAPH_FAILED);
    tilingData.sharedJettyNum = static_cast<int32_t>(*jettyNum);

    OP_LOGD(nodeName, "group = %s", groupPtr);
    OP_LOGD(nodeName, "worldSize = %ld", *worldSizePtr);
    OP_LOGD(nodeName, "maxBs = %ld", *maxBsPtr);
    OP_LOGD(nodeName, "mode = %ld", *modePtr);

    return ge::GRAPH_SUCCESS;
}

// 校验输入tensor shape
// x: [bs, data_size], dst_rank_id: [bs]
static ge::graphStatus CheckInputTensorShape(gert::TilingContext *context, const char *nodeName,
                                             BandwidthTestTilingData &tilingData)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "x"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "x", (std::to_string(xStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(xDim0 <= 0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "x dim0 (bs)", std::to_string(xDim0).c_str(), "positive"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        xDim1 <= 0,
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "x dim1 (data_size)", std::to_string(xDim1).c_str(), "positive"),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xDim0 != tilingData.maxBs || xDim1 >= 32768 || xDim1 % 16 != 0,
                    OP_LOGE(nodeName, "Require bs == max_bs and hidden in [16, 32768), divisible by 16."),
                    return ge::GRAPH_FAILED);
    tilingData.dataSize = static_cast<int32_t>(xDim1);

    const gert::StorageShape *dstRankIdStorageShape = context->GetInputShape(INPUT_DST_RANK_ID_INDEX);
    OP_TILING_CHECK(dstRankIdStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "dst_rank_id"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dstRankIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "dst_rank_id",
                        (std::to_string(dstRankIdStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);

    const int64_t dstRankIdDim0 = dstRankIdStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(dstRankIdDim0 != xDim0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "dst_rank_id",
                                                          std::to_string(dstRankIdDim0).c_str(),
                                                          "The dim0 of dst_rank_id must be the same as that of x"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 校验输出tensor shape
// y: [maxBs * worldSize, dataSize], receive_cnt: [worldSize]
static ge::graphStatus CheckOutputTensorShape(gert::TilingContext *context, const char *nodeName,
                                              BandwidthTestTilingData &tilingData)
{
    int64_t worldSize = static_cast<int64_t>(tilingData.worldSize);
    int64_t maxBs = static_cast<int64_t>(tilingData.maxBs);
    int64_t dataSize = static_cast<int64_t>(tilingData.dataSize);

    const gert::StorageShape *yStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "y"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            nodeName, "y", (std::to_string(yStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    const int64_t expectedYDim0 = maxBs * worldSize;

    OP_TILING_CHECK(yDim0 < expectedYDim0,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "y dim0", std::to_string(yDim0).c_str(),
                                              (std::string(">= ") + std::to_string(expectedYDim0)).c_str()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDim1 != dataSize,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "y dim1", std::to_string(yDim1).c_str(),
                                              ("=" + std::to_string(dataSize)).c_str()),
                    return ge::GRAPH_FAILED);

    const gert::StorageShape *receiveCntStorageShape = context->GetOutputShape(OUTPUT_RECEIVE_CNT_INDEX);
    OP_TILING_CHECK(receiveCntStorageShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "receive_cnt"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(receiveCntStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(
                        nodeName, "receive_cnt",
                        (std::to_string(receiveCntStorageShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
                    return ge::GRAPH_FAILED);

    const int64_t receiveCntDim0 = receiveCntStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(receiveCntDim0 != worldSize,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "receive_cnt",
                                                          std::to_string(receiveCntDim0).c_str(),
                                                          "The dim0 of receive_cnt must be the same as worldSize"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
                                        BandwidthTestTilingData &tilingData)
{
    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorShape(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check output tensor shape failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 根据NPU架构计算TilingKey
static uint64_t CalTilingKey(const gert::TilingContext *context)
{
    uint64_t archTag = TILING_KEY_BASE_A5;
    uint64_t commMode = IsUrma(context) ? TILINGKEY_TPL_URMA : TILINGKEY_TPL_MTE;
    uint64_t tilingKey = GET_TPL_TILING_KEY(archTag, commMode, IsSharedJetty(context) ? 1 : 0);
    return tilingKey;
}

// 设置Workspace大小
static ge::graphStatus SetWorkSpace(gert::TilingContext *context, BandwidthTestTilingData *tilingData,
                                    const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE;
    if (tilingData->mode != 3) {
        workSpaces[0] += static_cast<size_t>(WORKSPACE_ELEMENT_OFFSET * aivNum * aivNum);
    }

    if (IsSharedJetty(context) && tilingData->mode != 3) {
        uint64_t countStride = (static_cast<uint64_t>(tilingData->worldSize) + 7) / 8 * 8;
        workSpaces[0] += static_cast<uint64_t>(tilingData->sharedJettyNum) * countStride * sizeof(uint32_t);
    }
    return ge::GRAPH_SUCCESS;
}

// 检查HCCL窗口大小是否满足通信需求
// 需要的空间: world_size * max_bs * data_size * 2 * 2 (双缓冲 + 预留)
static ge::graphStatus CheckWinSize(const gert::TilingContext *context, BandwidthTestTilingData *tilingData,
                                    const char *nodeName)
{
    if (IsUrma(context)) {
        auto bufferSize = context->GetAttrs()->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
        OP_TILING_CHECK(bufferSize == nullptr || *bufferSize <= 0,
                        OP_LOGE(nodeName, "ccl_buffer_size must describe the context-owned Win."),
                        return ge::GRAPH_FAILED);
        uint64_t rowBytes = (static_cast<uint64_t>(tilingData->dataSize) * 2 + 511) / 512 * 512;
        uint64_t rankBytes = static_cast<uint64_t>(tilingData->maxBs) * rowBytes;
        uint64_t worldSize = tilingData->worldSize;
        // Reuse the MTE state area and its reserved max_bs staging block in each data half.
        uint64_t required = MB_SIZE + 2 * (worldSize + 1) * rankBytes;
        if (tilingData->mode == 3) {
            // One read-only BS block and one disjoint receive block, with compact token rows.
            required = MB_SIZE + 2 * static_cast<uint64_t>(tilingData->maxBs) * tilingData->dataSize * 2;
        }
        OP_TILING_CHECK(static_cast<uint64_t>(*bufferSize) < required,
                        OP_LOGE(nodeName, "Context Win too small: need %lu bytes, got %ld.", required, *bufferSize),
                        return ge::GRAPH_FAILED);
        tilingData->totalWinSize = static_cast<uint64_t>(*bufferSize) - MB_SIZE;
        return ge::GRAPH_SUCCESS;
    }
    uint64_t hcclBufferSize = 0;
    uint64_t maxWindowSize = 0;

    OP_TILING_CHECK(mc2tiling::GetEpWinSize(context, nodeName, hcclBufferSize, maxWindowSize, ATTR_GROUP_INDEX,
                                            false) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get WinSize failed"), return ge::GRAPH_FAILED);

    uint64_t dataSize = static_cast<uint64_t>(tilingData->dataSize);
    uint64_t worldSize = static_cast<uint64_t>(tilingData->worldSize);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->maxBs);
    uint64_t actualSize = worldSize * maxBs * dataSize * 2UL * 2UL;

    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName,
                "HCCL_BUFFSIZE is too SMALL, maxBs=%lu, dataSize=%lu, worldSize=%lu, "
                "actualSize=%luMB, HCCL_BUFFSIZE=%luMB.",
                maxBs, dataSize, worldSize, actualSize / MB_SIZE + 1UL, hcclBufferSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }

    tilingData->totalWinSize = maxWindowSize;
    OP_LOGD(nodeName, "WindowSize = %lu", maxWindowSize);

    return ge::GRAPH_SUCCESS;
}

// Tiling校验主函数
static ge::graphStatus BandwidthTestTilingCheckAttr(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    BandwidthTestTilingData *tilingData = context->GetTilingData<BandwidthTestTilingData>();
    std::string group = "";

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, group) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check winsize failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Tiling主入口函数
// 执行流程: 参数校验 -> 窗口检查 -> 设置通信配置 -> 填充TilingData
static ge::graphStatus BandwidthTestTilingFuncImpl(gert::TilingContext *context)
{
    uint32_t batchMode = 1U;
    context->SetScheduleMode(batchMode);

    const char *nodeName = context->GetNodeName();
    BandwidthTestTilingData *tilingData = context->GetTilingData<BandwidthTestTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter BandwidthTest tiling check func.");

    OP_TILING_CHECK(BandwidthTestTilingCheckAttr(context) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "BandwidthTest tiling check failed."), return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey(context);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t platformAivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t platformUbSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize);
    // Ascend950 has 256 KiB UB; the system's 8 KiB reservation is already excluded here.
    constexpr uint64_t ubSize = 248 * 1024;

    auto attrs = context->GetAttrs();
    auto aivNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_AIV_NUM_INDEX);
    uint32_t aivNum = platformAivNum;
    OP_TILING_CHECK(aivNumPtr == nullptr || (*aivNumPtr != -1 && (*aivNumPtr <= 0 || *aivNumPtr > platformAivNum)),
                    OP_LOGE(nodeName, "aiv_num must be -1 or in [1, platform AIV count]."),
                    return ge::GRAPH_FAILED);
    if (*aivNumPtr > 0) {
        aivNum = static_cast<uint32_t>(*aivNumPtr);
    }

    if (IsSharedJetty(context)) {
        OP_TILING_CHECK(static_cast<uint32_t>(tilingData->sharedJettyNum) > aivNum,
                        OP_LOGE(nodeName, "shared_jetty_num=%d exceeds selected AIV count=%u.",
                                tilingData->sharedJettyNum, aivNum), return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK(aivNum == 0, OP_LOGE(nodeName, "Platform has no AIV cores."), return ge::GRAPH_FAILED);
    auto alignUp = [](uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; };
    // Shared send/status work belongs to the jetty-owning cores; receive waiting still uses every launched AIV.
    uint64_t waitRanksPerCore = (static_cast<uint64_t>(tilingData->worldSize) + aivNum - 1) / aivNum;
    uint64_t statusCoreNum = IsSharedJetty(context) ? tilingData->sharedJettyNum : aivNum;
    uint64_t statusRanksPerCore =
        (static_cast<uint64_t>(tilingData->worldSize) + statusCoreNum - 1) / statusCoreNum;
    uint64_t statusBytes = alignUp(statusRanksPerCore, 8) * 32;
    uint64_t rowBytes = static_cast<uint64_t>(tilingData->dataSize) * 2;
    uint64_t requiredUb = 0;
    tilingData->batchBufferBytes = 0;
    auto batchTokensPtr = attrs->GetAttrPointer<int64_t>(ATTR_BATCH_TOKENS_INDEX);
    OP_TILING_CHECK(batchTokensPtr == nullptr || *batchTokensPtr < 0,
                    OP_LOGE(nodeName, "batch_tokens must be 0 (auto) or positive."), return ge::GRAPH_FAILED);
    tilingData->batchTokens = 0;
    bool directWin = tilingData->mode == 3;
    if (IsUrma(context)) {
        // Direct Win streaming only needs WQEs; business modes also stage one row and a chunk of IDs.
        uint64_t countBytes =
            IsSharedJetty(context) ? alignUp(static_cast<uint64_t>(tilingData->worldSize) * 4, 32) : 0;
        uint64_t fixedBytes = directWin ? 0 : 32 + statusBytes + rowBytes + countBytes;
        uint64_t idBytesPerToken = directWin ? 0 : sizeof(int32_t);
        uint64_t bytesPerToken = idBytesPerToken + URMA_WQE_BYTES;
        OP_TILING_CHECK(fixedBytes > ubSize || ubSize - fixedBytes < 32 + URMA_WQE_BYTES,
                        OP_LOGE(nodeName,
                                "Insufficient UB for URMA batch: max_bs=%d, hidden=%d, world_size=%d, "
                                "aiv_num=%u, platform_ub=%lu, usable_ub=%lu, fixed_bytes=%lu.",
                                tilingData->maxBs, tilingData->dataSize, tilingData->worldSize, aivNum,
                                platformUbSize, ubSize, fixedBytes), return ge::GRAPH_FAILED);
        uint64_t availableBytes = ubSize - fixedBytes;
        uint64_t maxBatchTokens = availableBytes / bytesPerToken;
        // Rank IDs need one aligned allocation, not eight-token granularity for commits.
        if (maxBatchTokens * URMA_WQE_BYTES + alignUp(maxBatchTokens * idBytesPerToken, 32) > availableBytes) {
            --maxBatchTokens;
        }
        maxBatchTokens = std::min(maxBatchTokens, static_cast<uint64_t>(URMA_MAX_BATCH_TOKENS));
        OP_TILING_CHECK(static_cast<uint64_t>(*batchTokensPtr) > maxBatchTokens,
                        OP_LOGE(nodeName, "batch_tokens=%ld exceeds maximum=%lu (usable_ub=%lu, fixed_bytes=%lu).",
                                *batchTokensPtr, maxBatchTokens, ubSize, fixedBytes), return ge::GRAPH_FAILED);
        uint64_t batchTokens = *batchTokensPtr == 0 ? maxBatchTokens : static_cast<uint64_t>(*batchTokensPtr);
        tilingData->batchTokens = static_cast<uint32_t>(batchTokens);
        tilingData->batchBufferBytes = batchTokens * URMA_WQE_BYTES;
        requiredUb = fixedBytes + batchTokens * URMA_WQE_BYTES + alignUp(batchTokens * idBytesPerToken, 32);
        OP_LOGI(nodeName, "URMA batch_tokens: requested=%ld, selected=%lu, maximum=%lu.",
                *batchTokensPtr, batchTokens, maxBatchTokens);
    } else {
        OP_TILING_CHECK(*batchTokensPtr != 0,
                        OP_LOGE(nodeName, "batch_tokens is only supported for urma; use 0 for mte."),
                        return ge::GRAPH_FAILED);
        uint64_t dstRankBytes = alignUp(static_cast<uint64_t>(tilingData->maxBs) * 4, 32);
        uint64_t dstRankBufferBytes =
            std::max(dstRankBytes, alignUp(static_cast<uint64_t>(tilingData->maxBs) * 2, 256));
        uint64_t initBytes = 32 + statusBytes + dstRankBufferBytes + 3 * std::max(rowBytes * 2, dstRankBytes);
        uint64_t queueCount = initBytes + rowBytes * 2 > 170 * 1024 ? 1 : 2;
        requiredUb = initBytes + queueCount * rowBytes;
    }
    uint64_t waitBytes = std::max(alignUp(waitRanksPerCore, 8) * 32, uint64_t(256));
    uint64_t gatherBytes = alignUp(std::max(static_cast<uint64_t>(tilingData->worldSize) * 4,
                                          alignUp(waitRanksPerCore * 4, 32)), 32);
    uint64_t receiveBytes = waitBytes + gatherBytes + 64 + static_cast<uint64_t>(tilingData->worldSize) * 32;
    OP_TILING_CHECK(requiredUb > ubSize || (!directWin && receiveBytes + rowBytes * 2 > ubSize),
                    OP_LOGE(nodeName, "Insufficient UB for BandwidthTest send or receive buffers."),
                    return ge::GRAPH_FAILED);
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    context->SetBlockDim(numBlocks);
    tilingData->totalUbSize = ubSize;
    tilingData->aivNum = aivNum;

    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %u, ubSize = %lu", numBlocks, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

struct BandwidthTestCompileInfo {};

static ge::graphStatus TilingParseForBandwidthTest(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BandwidthTest)
    .Tiling(BandwidthTestTilingFuncImpl)
    .TilingParse<BandwidthTestCompileInfo>(TilingParseForBandwidthTest);

} // namespace optiling
