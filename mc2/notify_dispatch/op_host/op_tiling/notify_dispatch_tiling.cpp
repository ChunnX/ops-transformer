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
 * \file notify_dispatch_tiling.cpp
 * \brief
 */
#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "../../op_kernel/notify_dispatch_tiling.h"
#include "../../op_kernel/notify_dispatch_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace {
constexpr size_t ONE_DIM = 1UL;

constexpr uint32_t INPUT_CONTEXT_INDEX = 0;
constexpr uint32_t INPUT_SEND_DATA_INDEX = 1;
constexpr uint32_t INPUT_TOKEN_PER_EXPERT_INDEX = 2;

constexpr uint32_t OUTPUT_SEND_DATA_OFFSET_INDEX = 0;
constexpr uint32_t OUTPUT_RECV_DATA_INDEX = 1;
constexpr uint32_t OUTPUT_RECV_OFFSET_RECV_OFFSET_INDEX = 2;
constexpr uint32_t OUTPUT_RECV_OFFSET_RECV_COUNT_INDEX = 3;
constexpr uint32_t OUTPUT_EXPERT_OUTPUT_GLOBAL_INDEX = 4;
constexpr uint32_t OUTPUT_EXPERT_OUTPUT_SRCRANK_INDEX = 5;
constexpr uint32_t OUTPUT_EXPERT_OUTPUT_R_IN_SRCRANK_INDEX = 6;
constexpr uint32_t OUTPUT_TOTAL_RECV_TOKENS_INDEX = 7;
constexpr uint32_t OUTPUT_MAX_BS_INDEX = 8;
constexpr uint32_t OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX = 9;
constexpr uint32_t OUTPUT_EXT_HANDLE_DATA_INDEX = 10;

constexpr uint32_t ATTR_SEND_COUNT_INDEX = 0;
constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 1;
constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 2;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 3;
constexpr uint32_t ATTR_LOCAL_RANK_SIZE_INDEX = 4;
constexpr uint32_t ATTR_LOCAL_RANK_ID_INDEX = 5;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 6;
constexpr uint32_t ATTR_ROUND_INDEX = 7;
constexpr uint32_t ATTR_PER_ROUND_TOKENS_INDEX = 8;

const int64_t MAX_COMM_WORLD_SIZE = 384;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;
constexpr int32_t HCCL_BUFFER_SIZE_DEFAULT = 200 * 1024 * 1024;  // Bytes
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;

constexpr static int ALL_TO_ALL_CORE_NUM = 32;
}  // namespace

namespace optiling {
static void PrintTilingDataInfo(const char *nodeName, NotifyDispatchTilingData &tilingData)
{
    OP_LOGI(nodeName,
            "NotifyDispatch tiling: rankSize=%u rankId=%u localRankSize=%u localRankId=%u "
            "sendCount=%u numTokens=%u round=%u perRoundTokens=%u aivNum=%u totalUbSize=%lu totalWinSize=%lu",
            tilingData.notifyDispatchInfo.rankSize, tilingData.notifyDispatchInfo.rankId,
            tilingData.notifyDispatchInfo.localRankSize, tilingData.notifyDispatchInfo.localRankId,
            tilingData.notifyDispatchInfo.sendCount, tilingData.notifyDispatchInfo.numTokens,
            tilingData.notifyDispatchInfo.round, tilingData.notifyDispatchInfo.perRoundTokens,
            tilingData.notifyDispatchInfo.aivNum, tilingData.notifyDispatchInfo.totalUbSize,
            tilingData.notifyDispatchInfo.totalWinSize);
}

static ge::graphStatus CheckMc2Context(gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(INPUT_CONTEXT_INDEX);
    OP_TILING_CHECK(contextStorageShape == nullptr, OP_LOGE(nodeName, "contextShape is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
                    OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %lu.",
                            contextStorageShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);

    auto contextDesc = context->GetInputDesc(INPUT_CONTEXT_INDEX);
    OP_TILING_CHECK(contextDesc == nullptr, OP_LOGE(nodeName, "contextDesc is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(contextDesc->GetDataType() != ge::DT_INT32,
                    OP_LOGE(nodeName, "context dataType is invalid, dataType should be int32."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(contextDesc->GetStorageFormat())) ==
                        ge::FORMAT_FRACTAL_NZ,
                    OP_LOGE(nodeName, "context format is invalid."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAttrAndSetTilingData(gert::TilingContext *context, const char *nodeName,
                                               NotifyDispatchTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto sendCountPtr = attrs->GetAttrPointer<int32_t>(ATTR_SEND_COUNT_INDEX);
    auto numTokenPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_TOKENS_INDEX);
    auto epWorldSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int32_t>(ATTR_EP_RANK_ID_INDEX);
    auto localRankSizePtr = attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_RANK_SIZE_INDEX);
    auto localRankIdPtr = attrs->GetAttrPointer<int32_t>(ATTR_LOCAL_RANK_ID_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto roundPtr = attrs->GetAttrPointer<int32_t>(ATTR_ROUND_INDEX);
    auto perRoundTokensPtr = attrs->GetAttrPointer<int32_t>(ATTR_PER_ROUND_TOKENS_INDEX);

    OP_TILING_CHECK(sendCountPtr == nullptr, OP_LOGE(nodeName, "sendCountPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numTokenPtr == nullptr, OP_LOGE(nodeName, "numTokenPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankSizePtr == nullptr, OP_LOGE(nodeName, "localRankSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankIdPtr == nullptr, OP_LOGE(nodeName, "localRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(roundPtr == nullptr, OP_LOGE(nodeName, "roundPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(perRoundTokensPtr == nullptr, OP_LOGE(nodeName, "perRoundTokensPtr is null."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*cclBufferSizePtr <= 0,
                    OP_LOGE(nodeName, "cclBufferSize is invalid, but got cclBufferSize=%ld.", *cclBufferSizePtr),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*epWorldSizePtr <= 0) || (*epWorldSizePtr > MAX_COMM_WORLD_SIZE),
                    OP_LOGE(nodeName, "epWorldSize is invalid, only support (0, %ld], but got epWorldSize=%d.",
                            MAX_COMM_WORLD_SIZE, *epWorldSizePtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*epRankIdPtr < 0) || (*epRankIdPtr >= *epWorldSizePtr),
        OP_LOGE(nodeName, "epRankId is invalid, only support [0, %d), but got epRankId=%d.", *epWorldSizePtr,
                *epRankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*sendCountPtr <= 0),
                    OP_LOGE(nodeName, "sendCount is invalid, only support > 0, but got sendCount=%d.", *sendCountPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (*numTokenPtr < 0),
        OP_LOGE(nodeName, "numTokenPtr is invalid, only support >= 0, but got numTokenPtr=%d.", *numTokenPtr),
        return ge::GRAPH_FAILED);

    tilingData.notifyDispatchInfo.rankSize = static_cast<uint32_t>(*epWorldSizePtr);
    tilingData.notifyDispatchInfo.rankId = static_cast<uint32_t>(*epRankIdPtr);
    tilingData.notifyDispatchInfo.localRankSize = static_cast<uint32_t>(*localRankSizePtr);
    tilingData.notifyDispatchInfo.localRankId = static_cast<uint32_t>(*localRankIdPtr);
    tilingData.notifyDispatchInfo.sendCount = static_cast<uint32_t>(*sendCountPtr);
    tilingData.notifyDispatchInfo.numTokens = static_cast<uint32_t>(*numTokenPtr);
    tilingData.notifyDispatchInfo.round = static_cast<uint32_t>(*roundPtr);
    tilingData.notifyDispatchInfo.perRoundTokens = static_cast<uint32_t>(*perRoundTokensPtr);
    tilingData.notifyDispatchInfo.totalWinSize = static_cast<uint64_t>(*cclBufferSizePtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + KERNEL_USE_WORKSPACE + KERNEL_A2_ARG_SIZE;
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDataType(gert::TilingContext *context, const char *nodeName)
{
    auto sendData = context->GetInputDesc(INPUT_SEND_DATA_INDEX);
    OP_TILING_CHECK(sendData == nullptr, OP_LOGE(nodeName, "sendData is null."), return false);
    OP_TILING_CHECK(
        (sendData->GetDataType() != ge::DT_BF16) && (sendData->GetDataType() != ge::DT_FLOAT16) &&
            (sendData->GetDataType() != ge::DT_FLOAT) && (sendData->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "sendData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(sendData->GetDataType())),
        return false);
    uint64_t dataSize;
    if ((sendData->GetDataType() == ge::DT_BF16) || (sendData->GetDataType() == ge::DT_FLOAT16)) {
        dataSize = 2;
    } else {
        dataSize = 4;
    }
    auto tokenPerExpertData = context->GetInputDesc(INPUT_TOKEN_PER_EXPERT_INDEX);
    OP_TILING_CHECK(tokenPerExpertData == nullptr, OP_LOGE(nodeName, "tokenPerExpertData is null."), return false);
    OP_TILING_CHECK(
        (tokenPerExpertData->GetDataType() != ge::DT_BF16) && (tokenPerExpertData->GetDataType() != ge::DT_FLOAT16) &&
            (tokenPerExpertData->GetDataType() != ge::DT_FLOAT) && (tokenPerExpertData->GetDataType() != ge::DT_INT32),
        OP_LOGE(
            nodeName,
            "tokenPerExpertData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
            static_cast<ge::DataType>(tokenPerExpertData->GetDataType())),
        return false);

    auto sendDataOffset = context->GetOutputDesc(OUTPUT_SEND_DATA_OFFSET_INDEX);
    OP_TILING_CHECK(sendDataOffset == nullptr, OP_LOGE(nodeName, "sendDataOffset is null."), return false);
    OP_TILING_CHECK(
        (sendDataOffset->GetDataType() != ge::DT_BF16) && (sendDataOffset->GetDataType() != ge::DT_FLOAT16) &&
            (sendDataOffset->GetDataType() != ge::DT_FLOAT) && (sendDataOffset->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "sendDataOffset datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(sendDataOffset->GetDataType())),
        return false);

    auto recvData = context->GetOutputDesc(OUTPUT_RECV_DATA_INDEX);
    OP_TILING_CHECK(recvData == nullptr, OP_LOGE(nodeName, "recvData is null."), return false);
    OP_TILING_CHECK(
        (recvData->GetDataType() != ge::DT_BF16) && (recvData->GetDataType() != ge::DT_FLOAT16) &&
            (recvData->GetDataType() != ge::DT_FLOAT) && (recvData->GetDataType() != ge::DT_INT32),
        OP_LOGE(nodeName,
                "recvData datatype is invalid, datatype should be bf16 or float16 or float or int, but is %d.",
                static_cast<ge::DataType>(recvData->GetDataType())),
        return false);

    auto recvOffset = context->GetOutputDesc(OUTPUT_RECV_OFFSET_RECV_OFFSET_INDEX);
    OP_TILING_CHECK(recvOffset == nullptr, OP_LOGE(nodeName, "recv_offset[0] is null."), return false);
    OP_TILING_CHECK((recvOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "recv_offset[0] datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(recvOffset->GetDataType())),
                    return false);

    auto recvCount = context->GetOutputDesc(OUTPUT_RECV_OFFSET_RECV_COUNT_INDEX);
    OP_TILING_CHECK(recvCount == nullptr, OP_LOGE(nodeName, "recv_offset[1] is null."), return false);
    OP_TILING_CHECK((recvCount->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "recv_offset[1] datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(recvCount->GetDataType())),
                    return false);

    auto expertGlobalOffset = context->GetOutputDesc(OUTPUT_EXPERT_OUTPUT_GLOBAL_INDEX);
    OP_TILING_CHECK(expertGlobalOffset == nullptr, OP_LOGE(nodeName, "expert_output_offset[0] is null."), return false);
    OP_TILING_CHECK((expertGlobalOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "expert_output_offset[0] datatype is invalid, but is %d.",
                            static_cast<ge::DataType>(expertGlobalOffset->GetDataType())),
                    return false);

    auto srcrankInExpertOffset = context->GetOutputDesc(OUTPUT_EXPERT_OUTPUT_SRCRANK_INDEX);
    OP_TILING_CHECK(srcrankInExpertOffset == nullptr, OP_LOGE(nodeName, "expert_output_offset[1] is null."),
                    return false);
    OP_TILING_CHECK((srcrankInExpertOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "expert_output_offset[1] datatype is invalid, but is %d.",
                            static_cast<ge::DataType>(srcrankInExpertOffset->GetDataType())),
                    return false);

    auto rInSrcrankOffset = context->GetOutputDesc(OUTPUT_EXPERT_OUTPUT_R_IN_SRCRANK_INDEX);
    OP_TILING_CHECK(rInSrcrankOffset == nullptr, OP_LOGE(nodeName, "expert_output_offset[2] is null."), return false);
    OP_TILING_CHECK((rInSrcrankOffset->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "expert_output_offset[2] datatype is invalid, but is %d.",
                            static_cast<ge::DataType>(rInSrcrankOffset->GetDataType())),
                    return false);

    auto totalRecvTokens = context->GetOutputDesc(OUTPUT_TOTAL_RECV_TOKENS_INDEX);
    OP_TILING_CHECK(totalRecvTokens == nullptr, OP_LOGE(nodeName, "totalRecvTokens is null."), return false);
    OP_TILING_CHECK((totalRecvTokens->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "totalRecvTokens datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(totalRecvTokens->GetDataType())),
                    return false);

    auto maxBs = context->GetOutputDesc(OUTPUT_MAX_BS_INDEX);
    OP_TILING_CHECK(maxBs == nullptr, OP_LOGE(nodeName, "maxBs is null."), return false);
    OP_TILING_CHECK((maxBs->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "maxBs datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(maxBs->GetDataType())),
                    return false);

    auto recvTokensPerExpert = context->GetOutputDesc(OUTPUT_RECV_TOKENS_PER_EXPERT_INDEX);
    OP_TILING_CHECK(recvTokensPerExpert == nullptr, OP_LOGE(nodeName, "recvTokensPerExpert is null."), return false);
    OP_TILING_CHECK((recvTokensPerExpert->GetDataType() != ge::DT_INT32),
                    OP_LOGE(nodeName, "recvTokensPerExpert datatype is invalid, datatype should be int, but is %d.",
                            static_cast<ge::DataType>(recvTokensPerExpert->GetDataType())),
                    return false);

    auto extHandleData = context->GetOutputDesc(OUTPUT_EXT_HANDLE_DATA_INDEX);
    if (extHandleData != nullptr) {
        OP_TILING_CHECK((extHandleData->GetDataType() != ge::DT_INT32),
                        OP_LOGE(nodeName, "ext_handle_data datatype is invalid, but is %d.",
                                static_cast<ge::DataType>(extHandleData->GetDataType())),
                        return false);
    }

    // ccl_buffer_size is total raw bytes (1MB state + data). Ping-pong uses each half of the data region.
    NotifyDispatchTilingData *tilingData = context->GetTilingData<NotifyDispatchTilingData>();
    constexpr uint64_t EP_HCCL_STATE_SIZE = 1024UL * 1024UL;
    uint64_t maxWindowSize = tilingData->notifyDispatchInfo.totalWinSize;
    uint64_t dataWinSize = maxWindowSize - EP_HCCL_STATE_SIZE;
    uint64_t halfWinSize = dataWinSize / 2;
    uint64_t actualPerHalf =
        static_cast<uint64_t>(dataSize) * tilingData->notifyDispatchInfo.sendCount + 2 * 1024 * 1024;  // flag + payload
    if (actualPerHalf > halfWinSize) {
        OP_LOGE(nodeName,
                "ccl_buffer_size is too SMALL, need %luMB per ping-pong half (data region %luMB = ccl_buffer_size - 1MB "
                "state), but payload+flag need %luMB.",
                halfWinSize / MB_SIZE, dataWinSize / MB_SIZE, actualPerHalf / MB_SIZE + 1UL);
        return false;
    }
    return true;
}

static ge::graphStatus TilingCheckTensor(gert::TilingContext *context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OP_LOGE(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchTilingFuncImpl(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    NotifyDispatchTilingData *tilingData = context->GetTilingData<NotifyDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter NotifyDispatch tiling check func.");

    OP_TILING_CHECK(CheckMc2Context(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Check context failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OP_LOGE(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    const bool isA2 = (socVersion == "Ascend910B");
    const bool isA5 = (socVersion == "Ascend950");
    uint32_t archTag = TILINGKEY_TPL_A3;
    if (isA5) {
        archTag = TILINGKEY_TPL_A5;
    } else if (isA2) {
        archTag = TILINGKEY_TPL_A2;
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(archTag);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey=%lu socVersion=%s isA2=%d isA5=%d", tilingKey, socVersion.c_str(),
            static_cast<int>(isA2), static_cast<int>(isA5));

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    blockDim = aivNum;
    context->SetBlockDim(blockDim);
    tilingData->notifyDispatchInfo.totalUbSize = ubSize;
    tilingData->notifyDispatchInfo.aivNum = aivNum;
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus NotifyDispatchTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = NotifyDispatchTilingFuncImpl(context);
    return ret;
}

struct NotifyDispatchCompileInfo {};
ge::graphStatus TilingParseForNotifyDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NotifyDispatch)
    .Tiling(NotifyDispatchTilingFunc)
    .TilingParse<NotifyDispatchCompileInfo>(TilingParseForNotifyDispatch);
}  // namespace optiling
