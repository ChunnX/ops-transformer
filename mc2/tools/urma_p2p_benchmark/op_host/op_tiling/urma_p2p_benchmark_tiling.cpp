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
 * \file urma_p2p_benchmark_tiling.cpp
 * \brief URMA点对点通信基准测试算子的Tiling实现
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <string>
#include <vector>

#include "../../op_kernel/urma_p2p_benchmark_tiling.h"
#include "../../op_kernel/urma_p2p_benchmark_tiling_key.h"
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

using namespace UrmaP2pBenchmarkTiling;
using namespace AscendC;
using namespace ge;

namespace {

// Mc2CcTiling内部结构体，用于设置protocal字段
struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[8];
    uint8_t protocal;
    uint8_t communicationEngine;
    uint8_t srcDataType;
    uint8_t dstDataType;
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};

constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8U;

} // namespace anonymous

namespace optiling {

// ============ 输入输出索引常量 ============
// dst_rank_id通过属性传递，不再作为输入tensor
constexpr uint32_t INPUT_X_INDEX = 0U;              // 输入x的索引
constexpr uint32_t OUTPUT_Y_INDEX = 0U;             // 输出y的索引
constexpr uint32_t OUTPUT_RECEIVE_CNT_INDEX = 1U;   // 输出receive_cnt的索引
constexpr uint32_t OUTPUT_PERFORMANCE_INFO_INDEX = 2U;  // 输出performance_info的索引

// ============ 属性索引常量 ============
constexpr uint32_t ATTR_GROUP_INDEX = 0U;           // 通信组名称
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1U;      // 世界大小
constexpr uint32_t ATTR_MAX_BS_INDEX = 2U;          // 最大batch size
constexpr uint32_t ATTR_MODE_INDEX = 3U;            // 执行模式: 0=overhead, 1=latency, 2=bandwidth
constexpr uint32_t ATTR_HCCL_BUFFER_INDEX = 4U;     // HCCL缓冲区大小
constexpr uint32_t ATTR_DST_RANK_ID_INDEX = 5U;     // 点对点目标rank ID
constexpr uint32_t ATTR_AIV_NUM_INDEX = 6U;         // AIV核数量

// ============ 其他常量 ============
constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;  // 系统预留workspace 16MB
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t WORKSPACE_ELEMENT_OFFSET = 512UL;

// ============ 参数范围约束 ============
const size_t MAX_GROUP_NAME_LENGTH = 128UL;
const int64_t MIN_WORLD_SIZE = 1;

// ============ Tiling Key ============
// URMA模式固定，只有A5架构组合
constexpr uint64_t TILING_KEY_BASE_A5 = 2ULL;  // A5架构base key

// 打印TilingData调试信息
static void PrintTilingDataInfo(const char *nodeName, UrmaP2pBenchmarkTilingData &tilingData)
{
    OP_LOGD(nodeName, "data_size is %d.", tilingData.data_size);
    OP_LOGD(nodeName, "world_size is %d.", tilingData.world_size);
    OP_LOGD(nodeName, "max_bs is %d.", tilingData.max_bs);
    OP_LOGD(nodeName, "mode is %d.", tilingData.mode);
    OP_LOGD(nodeName, "dst_rank_id is %d.", tilingData.dst_rank_id);
    OP_LOGD(nodeName, "aiv_num is %d.", tilingData.aiv_num);
    OP_LOGD(nodeName, "timing_slots is %d.", tilingData.timing_slots);
    OP_LOGD(nodeName, "total_ub_size is %lu.", tilingData.total_ub_size);
    OP_LOGD(nodeName, "total_win_size is %lu.", tilingData.total_win_size);
}

// 校验属性指针非空
static ge::graphStatus CheckAttrPointersNotNull(gert::TilingContext *context, const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);
    auto hcclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_HCCL_BUFFER_INDEX);
    auto dstRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_DST_RANK_ID_INDEX);

    OP_TILING_CHECK(groupPtr == nullptr,
        OP_LOGE(nodeName, "groupPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(worldSizePtr == nullptr,
        OP_LOGE(nodeName, "worldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxBsPtr == nullptr,
        OP_LOGE(nodeName, "maxBsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(modePtr == nullptr,
        OP_LOGE(nodeName, "modePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hcclBufferSizePtr == nullptr,
        OP_LOGE(nodeName, "hcclBufferSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(dstRankIdPtr == nullptr,
        OP_LOGE(nodeName, "dstRankIdPtr is null."), return ge::GRAPH_FAILED);

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
    auto dstRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_DST_RANK_ID_INDEX);

    // 校验group名称长度
    OP_TILING_CHECK((strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == 0) ||
        (strnlen(groupPtr, MAX_GROUP_NAME_LENGTH) == MAX_GROUP_NAME_LENGTH),
        OP_LOGE(nodeName, "group name is invalid, length should be in (0, %zu), but got length=%zu.",
        MAX_GROUP_NAME_LENGTH, strnlen(groupPtr, MAX_GROUP_NAME_LENGTH)), return ge::GRAPH_FAILED);

    // 校验worldSize正数
    OP_TILING_CHECK((*worldSizePtr < MIN_WORLD_SIZE),
        OP_LOGE(nodeName, "worldSize is invalid, should be positive, but got worldSize=%ld.", *worldSizePtr),
                return ge::GRAPH_FAILED);

    // 校验maxBs正数
    OP_TILING_CHECK((*maxBsPtr <= 0),
        OP_LOGE(nodeName, "maxBs is invalid, should be positive, but got maxBs=%ld.", *maxBsPtr),
                return ge::GRAPH_FAILED);

    // 校验mode范围: 0=overhead+latency, 1=bandwidth
    OP_TILING_CHECK((*modePtr < 0) || (*modePtr > 1),
        OP_LOGE(nodeName, "mode is invalid, should be 0(overhead+latency) or 1(bandwidth), but got mode=%ld.",
        *modePtr), return ge::GRAPH_FAILED);

    // 校验dst_rank_id范围: -1表示全部rank, 0~worldSize-1表示指定rank
    OP_TILING_CHECK((*dstRankIdPtr < -1) || (*dstRankIdPtr >= *worldSizePtr),
        OP_LOGE(nodeName, "dstRankId is invalid, should be in [-1, worldSize-1], but got dstRankId=%ld, worldSize=%ld.",
        *dstRankIdPtr, *worldSizePtr), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 获取属性并填充TilingData
static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
    UrmaP2pBenchmarkTilingData &tilingData, std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrPointersNotNull(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr pointers not null failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckAttrValuesValid(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr values valid failed."), return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto worldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX);
    auto maxBsPtr = attrs->GetAttrPointer<int64_t>(ATTR_MAX_BS_INDEX);
    auto modePtr = attrs->GetAttrPointer<int64_t>(ATTR_MODE_INDEX);
    auto dstRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_DST_RANK_ID_INDEX);

    group = std::string(groupPtr);
    tilingData.world_size = static_cast<int32_t>(*worldSizePtr);
    tilingData.max_bs = static_cast<int32_t>(*maxBsPtr);
    tilingData.mode = static_cast<int32_t>(*modePtr);
    tilingData.dst_rank_id = static_cast<int32_t>(*dstRankIdPtr);
    tilingData.timing_slots = 12;  // 9个timing slot + 3个padding, 12是4的倍数满足DataCopy对齐

    OP_LOGD(nodeName, "group = %s", groupPtr);
    OP_LOGD(nodeName, "worldSize = %ld", *worldSizePtr);
    OP_LOGD(nodeName, "maxBs = %ld", *maxBsPtr);
    OP_LOGD(nodeName, "mode = %ld", *modePtr);
    OP_LOGD(nodeName, "dstRankId = %ld", *dstRankIdPtr);

    return ge::GRAPH_SUCCESS;
}

// 校验输入tensor shape
// x: [bs, data_size]
static ge::graphStatus CheckInputTensorShape(gert::TilingContext *context, const char *nodeName,
    UrmaP2pBenchmarkTilingData &tilingData)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr,
        OP_LOGE(nodeName, "xStorageShape is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "x dims is invalid, should be 2, but got %zu.",
        xStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(xDim0 <= 0,
        OP_LOGE(nodeName, "x dim0 (bs) is invalid, should be positive, but got %ld.", xDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim1 <= 0,
        OP_LOGE(nodeName, "x dim1 (data_size) is invalid, should be positive, but got %ld.", xDim1),
                return ge::GRAPH_FAILED);

    tilingData.max_bs = static_cast<int32_t>(xDim0);
    tilingData.data_size = static_cast<int32_t>(xDim1);

    return ge::GRAPH_SUCCESS;
}

// 校验输出tensor shape
// y: [max_bs * world_size, data_size], receive_cnt: [world_size]
static ge::graphStatus CheckOutputTensorShape(gert::TilingContext *context, const char *nodeName,
    UrmaP2pBenchmarkTilingData &tilingData)
{
    int64_t worldSize = static_cast<int64_t>(tilingData.world_size);
    int64_t maxBs = static_cast<int64_t>(tilingData.max_bs);
    int64_t dataSize = static_cast<int64_t>(tilingData.data_size);

    const gert::StorageShape *yStorageShape = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_TILING_CHECK(yStorageShape == nullptr,
        OP_LOGE(nodeName, "yStorageShape is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "y dims is invalid, should be 2, but got %zu.",
        yStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    const int64_t yDim0 = yStorageShape->GetStorageShape().GetDim(0);
    const int64_t yDim1 = yStorageShape->GetStorageShape().GetDim(1);
    const int64_t expectedYDim0 = maxBs * worldSize;

    OP_TILING_CHECK(yDim0 < expectedYDim0,
        OP_LOGE(nodeName, "y dim0 should be >= maxBs * worldSize, but got y dim0=%ld, expected=%ld.",
        yDim0, expectedYDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDim1 != dataSize,
        OP_LOGE(nodeName, "y dim1 should equal to x dim1, but got y dim1=%ld, x dim1=%ld.",
        yDim1, dataSize), return ge::GRAPH_FAILED);

    const gert::StorageShape *receiveCntStorageShape = context->GetOutputShape(OUTPUT_RECEIVE_CNT_INDEX);
    OP_TILING_CHECK(receiveCntStorageShape == nullptr,
        OP_LOGE(nodeName, "receiveCntStorageShape is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(receiveCntStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "receive_cnt dims is invalid, should be 1, but got %zu.",
        receiveCntStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    const int64_t receiveCntDim0 = receiveCntStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(receiveCntDim0 != worldSize,
        OP_LOGE(nodeName, "receive_cnt dim0 should equal to worldSize, but got receive_cnt dim0=%ld, worldSize=%ld.",
        receiveCntDim0, worldSize), return ge::GRAPH_FAILED);

    // 校验performance_info: 1维, dim0等于timing_slots(8, 含1个DataCopy对齐padding), dtype int64
    const gert::StorageShape *performanceInfoStorageShape = context->GetOutputShape(OUTPUT_PERFORMANCE_INFO_INDEX);
    OP_TILING_CHECK(performanceInfoStorageShape == nullptr,
        OP_LOGE(nodeName, "performanceInfoStorageShape is nullptr."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(performanceInfoStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "performance_info dims is invalid, should be 1, but got %zu.",
        performanceInfoStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);

    const int64_t performanceInfoDim0 = performanceInfoStorageShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(performanceInfoDim0 != tilingData.timing_slots,
        OP_LOGE(nodeName, "performance_info dim0 should equal to timing_slots(%d), but got performance_info dim0=%ld.",
        tilingData.timing_slots, performanceInfoDim0), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorShape(gert::TilingContext *context, const char *nodeName,
    UrmaP2pBenchmarkTilingData &tilingData)
{
    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckOutputTensorShape(context, nodeName, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check output tensor shape failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 根据NPU架构计算TilingKey
// URMA模式固定，只有URMA+A5的组合key
static uint64_t CalTilingKey(const char *nodeName)
{
    uint64_t archTag = TILING_KEY_BASE_A5;
    uint8_t commModeByte = TILINGKEY_TPL_URAMA;  // 固定为URMA模式
    uint64_t tilingKey = GET_TPL_TILING_KEY(commModeByte, archTag);
    OP_LOGI(nodeName, "tilingKey is %ld", tilingKey);
    return tilingKey;
}

// 设置Workspace大小
static ge::graphStatus SetWorkSpace(gert::TilingContext *context, UrmaP2pBenchmarkTilingData *tilingData,
    const char *nodeName)
{
    auto attrs = context->GetAttrs();
    int64_t hcclBufferSize = *(attrs->GetAttrPointer<int64_t>(ATTR_HCCL_BUFFER_INDEX));
    int64_t worldSize = *(attrs->GetAttrPointer<int64_t>(ATTR_WORLD_SIZE_INDEX));

    uint64_t dataSize = static_cast<uint64_t>(worldSize * hcclBufferSize);
    OP_LOGD(nodeName, "dataSize = %lu", dataSize);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + dataSize + 2UL * 1024UL * 1024UL * 1024UL;

    return ge::GRAPH_SUCCESS;
}

// 检查HCCL窗口大小是否满足通信需求
// 需要的空间: world_size * max_bs * data_size * 2 * 2 (双缓冲 + 预留)
static ge::graphStatus CheckWinSize(const gert::TilingContext *context, UrmaP2pBenchmarkTilingData *tilingData,
    const char *nodeName)
{
    uint64_t hcclBufferSize = 0;
    uint64_t maxWindowSize = 0;

    OP_TILING_CHECK(mc2tiling::GetEpWinSize(
        context, nodeName, hcclBufferSize, maxWindowSize, ATTR_GROUP_INDEX, false) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get WinSize failed"), return ge::GRAPH_FAILED);

    uint64_t dataSize = static_cast<uint64_t>(tilingData->data_size);
    uint64_t worldSize = static_cast<uint64_t>(tilingData->world_size);
    uint64_t maxBs = static_cast<uint64_t>(tilingData->max_bs);
    uint64_t actualSize = worldSize * maxBs * dataSize * 2UL * 2UL;

    if (actualSize > maxWindowSize) {
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, maxBs=%lu, dataSize=%lu, worldSize=%lu, "
            "actualSize=%luMB, HCCL_BUFFSIZE=%luMB.",
            maxBs, dataSize, worldSize, actualSize / MB_SIZE + 1UL, hcclBufferSize / MB_SIZE);
        return ge::GRAPH_FAILED;
    }

    tilingData->total_win_size = maxWindowSize;
    OP_LOGD(nodeName, "WindowSize = %lu", maxWindowSize);

    return ge::GRAPH_SUCCESS;
}

// 设置SQE级HcclCombinOpParam URMA通信配置
// 创建Mc2CcTilingConfig，配置AlltoAll算法和AIV引擎，设置protocal=1表示URMA
static ge::graphStatus SetHcommCfg(UrmaP2pBenchmarkTilingData *tilingData, const std::string &group,
    const char *nodeName)
{
    OP_LOGD(nodeName, "SetHcommCfg group = %s", group.c_str());
    uint32_t opType = OP_TYPE_ALL_TO_ALL;
    std::string algConfigStr = "AlltoAll=level0:fullmesh;level1:pairwise";

    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType, algConfigStr);
    mc2CcTilingConfig.SetCommEngine(mc2tiling::AIV_ENGINE);  // AIV引擎(=3)，SQE级URMA通信
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2InitTiling GetTiling failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling) != 0,
        OP_LOGE(nodeName, "mc2CcTilingConfig mc2CcTiling GetTiling failed"), return ge::GRAPH_FAILED);
    reinterpret_cast<Mc2CcTilingInner *>(&tilingData->mc2CcTiling)->protocal = 1; // 0: UB-MEM, 1: URMA

    return ge::GRAPH_SUCCESS;
}

// Tiling校验主函数
static ge::graphStatus UrmaP2pBenchmarkTilingCheckAttr(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    UrmaP2pBenchmarkTilingData *tilingData = context->GetTilingData<UrmaP2pBenchmarkTilingData>();
    std::string group = "";

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData, group) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckTensorShape(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check tensor shape failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckWinSize(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check win size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, tilingData, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Tiling主入口函数
// 执行流程: 参数校验 -> 窗口检查 -> 设置通信配置 -> 填充TilingData
static ge::graphStatus UrmaP2pBenchmarkTilingFuncImpl(gert::TilingContext *context)
{
    uint32_t batchMode = 1U;
    auto ret = context->SetScheduleMode(batchMode);

    const char *nodeName = context->GetNodeName();
    UrmaP2pBenchmarkTilingData *tilingData = context->GetTilingData<UrmaP2pBenchmarkTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter UrmaP2pBenchmark tiling check func.");

    OP_TILING_CHECK(UrmaP2pBenchmarkTilingCheckAttr(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "UrmaP2pBenchmark tiling check failed."), return ge::GRAPH_FAILED);

    // URMA模式固定，直接计算TilingKey
    uint64_t tilingKey = CalTilingKey(nodeName);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t platformAivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto attrs = context->GetAttrs();
    auto aivNumPtr = attrs->GetAttrPointer<int64_t>(ATTR_AIV_NUM_INDEX);
    uint32_t aivNum = platformAivNum;
    if (aivNumPtr != nullptr && *aivNumPtr <= platformAivNum) {
        aivNum = static_cast<uint32_t>(*aivNumPtr);
        OP_LOGD(nodeName, "User set aiv_num = %ld", *aivNumPtr);
    }

    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);

    context->SetBlockDim(numBlocks);
    tilingData->total_ub_size = ubSize;
    tilingData->aiv_num = aivNum;

    OP_LOGD(nodeName, "numBlocks = %u, aivNum = %u, ubSize = %lu", numBlocks, aivNum, ubSize);

    // 设置SQE级HcclCombinOpParam URMA通信配置
    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    std::string group = std::string(groupPtr);
    OP_TILING_CHECK(SetHcommCfg(tilingData, group, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "SetHcommCfg failed."), return ge::GRAPH_FAILED);

    PrintTilingDataInfo(nodeName, *tilingData);

    return ge::GRAPH_SUCCESS;
}

struct UrmaP2pBenchmarkCompileInfo {};

static ge::graphStatus TilingParseForUrmaP2pBenchmark(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UrmaP2pBenchmark)
    .Tiling(UrmaP2pBenchmarkTilingFuncImpl)
    .TilingParse<UrmaP2pBenchmarkCompileInfo>(TilingParseForUrmaP2pBenchmark);

} // namespace optiling