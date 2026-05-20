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
 * \file dispatch_layout_tiling.cpp
 * \brief dispatch layout tiling implementation
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../op_kernel/dispatch_layout_tiling.h"
#include "graph/utils/type_utils.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace Mc2Tiling;
using namespace ge;
using namespace AscendC;

namespace {
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 4U * 1024UL * 1024UL;
constexpr uint64_t NOTIFY_DISPATCH_WIN_OFFSET = 204U * 1024UL * 1024UL;
constexpr uint32_t INPUT_TOPK_IDX_INDEX = 0;

constexpr uint32_t OUTPUT_NUM_TOKEN_PER_RANK_INDEX = 0;
constexpr uint32_t OUTPUT_NUM_TOKEN_PER_EXPERT_INDEX = 1;
constexpr uint32_t OUTPUT_IS_TOKEN_IN_RANK_INDEX = 2;
constexpr uint32_t OUTPUT_NOTIFY_SEND_DATA_INDEX = 3;

constexpr uint32_t ATTR_NUM_TOKENS_INDEX = 0;
constexpr uint32_t ATTR_NUM_RANKS_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_TOPK_INDEX = 3;
constexpr uint32_t ATTR_LOCAL_RANKSIZE_INDEX = 4;

const int64_t MAX_COMM_WORLD_SIZE = 384;
const int64_t MAX_MOE_EXPERTS_NUM = 512;
const int64_t MAX_LOCAL_RANKSIZE = 8;
const int64_t A2_MAX_BATCH_SIZE = 4096;
const int64_t A3_MAX_BATCH_SIZE = 8000;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t KERNEL_USE_WORKSPACE = 1 * 1024 * 1024;
constexpr uint32_t KERNEL_A2_ARG_SIZE = 1 * 1024 * 1024;

constexpr static int TILING_KEY_INT = 23;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t K_MAX = 16;
} // namespace

namespace optiling {

static void PrintTilingDataInfo(const char *nodeName, const DispatchLayoutTilingData &tilingData)
{
    OPS_LOG_D(nodeName, "numToken is %u.", tilingData.dispatchLayoutInfo.numTokens);
    OPS_LOG_D(nodeName, "numRanks is %u.", tilingData.dispatchLayoutInfo.numRanks);
    OPS_LOG_D(nodeName, "numExperts is %u.", tilingData.dispatchLayoutInfo.numExperts);
    OPS_LOG_D(nodeName, "numTopk is %u.", tilingData.dispatchLayoutInfo.numTopk);
    OPS_LOG_D(nodeName, "localRankSize is %u.", tilingData.dispatchLayoutInfo.localRankSize);
    OPS_LOG_D(nodeName, "totalUbSize is %lu.", tilingData.dispatchLayoutInfo.totalUbSize);
}

static bool CheckIfA2MultiMachine(const gert::TilingContext &context, const char *nodeName,
                                  const DispatchLayoutTilingData &tilingData)
{
    fe::PlatFormInfos *platformInfoPtr = context.GetPlatformInfo();
    OP_TILING_CHECK(platformInfoPtr == nullptr, OPS_LOG_E(nodeName, "platformInfoPtr is nullptr."), return false);
    fe::PlatFormInfos &platformInfo = *platformInfoPtr;

    std::string socVersion;
    (void)platformInfo.GetPlatformResWithLock("version", "Short_SoC_version", socVersion);

    uint32_t numRanks = tilingData.dispatchLayoutInfo.numRanks;
    uint32_t localRankSize = tilingData.dispatchLayoutInfo.localRankSize;

    if (socVersion == "Ascend910B" && numRanks > localRankSize) {
        return true;
    }
    return false;
}

static ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext &context, const char *nodeName,
                                                DispatchLayoutTilingData &tilingData)
{
    auto attrs = context.GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OPS_LOG_E(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto numTokensPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_TOKENS_INDEX));
    auto numRanksPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_RANKS_INDEX));
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_EXPERTS_INDEX));
    auto numTopkPtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_NUM_TOPK_INDEX));
    auto localRankSizePtr = attrs->GetAttrPointer<int64_t>(static_cast<int>(ATTR_LOCAL_RANKSIZE_INDEX));

    OP_TILING_CHECK(numTokensPtr == nullptr, OPS_LOG_E(nodeName, "numTokensPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRanksPtr == nullptr, OPS_LOG_E(nodeName, "numRanksPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OPS_LOG_E(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numTopkPtr == nullptr, OPS_LOG_E(nodeName, "numTopkPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(localRankSizePtr == nullptr, OPS_LOG_E(nodeName, "localRankSizePtr is null."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK((*numRanksPtr <= 0) || (*numRanksPtr > MAX_COMM_WORLD_SIZE),
                    OPS_LOG_E(nodeName, "rankSize is invalid, only support (0, %ld], but got rankSize=%ld.",
                            MAX_COMM_WORLD_SIZE, *numRanksPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numExpertsPtr <= 0) || (*numExpertsPtr > MAX_MOE_EXPERTS_NUM),
                    OPS_LOG_E(nodeName, "numExperts is invalid, only support (0, %ld], but got numExperts=%ld.",
                            MAX_MOE_EXPERTS_NUM, *numExpertsPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numExpertsPtr % *numRanksPtr) != 0,
                    OPS_LOG_E(nodeName, "numExperts (%ld) not divisible by numRanks (%ld).",
                            *numExpertsPtr, *numRanksPtr),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numTopkPtr <= 0) || (*numTopkPtr > K_MAX),
                    OPS_LOG_E(nodeName, "numTopkPtr is invalid, only support (0, %u], but got numTopk=%ld.", K_MAX, *numTopkPtr),
                    return ge::GRAPH_FAILED);

    tilingData.dispatchLayoutInfo.numRanks = static_cast<uint32_t>(*numRanksPtr);
    tilingData.dispatchLayoutInfo.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    tilingData.dispatchLayoutInfo.numTopk = static_cast<uint32_t>(*numTopkPtr);
    tilingData.dispatchLayoutInfo.localRankSize = static_cast<uint32_t>(*localRankSizePtr);

    if (CheckIfA2MultiMachine(context, nodeName, tilingData)) {
        OP_TILING_CHECK((*localRankSizePtr <= 0) || (*localRankSizePtr > MAX_LOCAL_RANKSIZE),
                    OPS_LOG_E(nodeName, "localRankSizePtr is invalid, only support (0, %ld], but got localRankSize=%ld.",
                            MAX_LOCAL_RANKSIZE, *localRankSizePtr),
                    return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*numRanksPtr % *localRankSizePtr != 0),
                    OPS_LOG_E(nodeName,
                            "localRankSizePtr isn't an aliquot of numRanks, numRanks=%ld, but got localRankSize=%ld.",
                            *numRanksPtr, *localRankSizePtr),
                    return ge::GRAPH_FAILED);
        OP_TILING_CHECK((*numTokensPtr <= 0) || (*numTokensPtr > A2_MAX_BATCH_SIZE),
            OPS_LOG_E(nodeName, "tokenNum is invalid, only support (0, %ld], but got tokenNum=%ld.",
                    A2_MAX_BATCH_SIZE, *numTokensPtr),
            return ge::GRAPH_FAILED);
    } else {
        OP_TILING_CHECK((*numTokensPtr <= 0) || (*numTokensPtr > A3_MAX_BATCH_SIZE),
            OPS_LOG_E(nodeName, "tokenNum is invalid, only support (0, %ld], but got tokenNum=%ld.",
                    A3_MAX_BATCH_SIZE, *numTokensPtr),
            return ge::GRAPH_FAILED);
    }
    tilingData.dispatchLayoutInfo.numTokens = static_cast<uint32_t>(*numTokensPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext &context, const char *nodeName)
{
    size_t *workSpaces = context.GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OPS_LOG_E(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + KERNEL_USE_WORKSPACE + KERNEL_A2_ARG_SIZE;
    return ge::GRAPH_SUCCESS;
}

static bool CheckTensorDataType(const gert::TilingContext &context, const char *nodeName)
{
    auto topkIdx = context.GetInputDesc(INPUT_TOPK_IDX_INDEX);
    auto numTokensPerRank = context.GetOutputDesc(OUTPUT_NUM_TOKEN_PER_RANK_INDEX);
    auto numTokensPerExpert = context.GetOutputDesc(OUTPUT_NUM_TOKEN_PER_EXPERT_INDEX);
    auto isTokenInRank = context.GetOutputDesc(OUTPUT_IS_TOKEN_IN_RANK_INDEX);
    auto notifySendData = context.GetOutputDesc(OUTPUT_NOTIFY_SEND_DATA_INDEX);

    OP_TILING_CHECK(topkIdx == nullptr, OPS_LOG_E(nodeName, "topkIdx is null."), return false);
    OP_TILING_CHECK(numTokensPerRank == nullptr, OPS_LOG_E(nodeName, "numTokensPerRank is null."), return false);
    OP_TILING_CHECK(numTokensPerExpert == nullptr, OPS_LOG_E(nodeName, "numTokensPerExpert is null."), return false);
    OP_TILING_CHECK(isTokenInRank == nullptr, OPS_LOG_E(nodeName, "isTokenInRank is null."), return false);
    OP_TILING_CHECK(notifySendData == nullptr, OPS_LOG_E(nodeName, "notifySendData is null."), return false);

    OP_TILING_CHECK((topkIdx->GetDataType() != ge::DT_INT64),
                    OPS_LOG_E(nodeName, "topkIdx datatype is invalid, datatype should be int64, but is %d.",
                            static_cast<ge::DataType>(topkIdx->GetDataType())),
                    return false);
    OP_TILING_CHECK((numTokensPerRank->GetDataType() != ge::DT_INT32),
                    OPS_LOG_E(nodeName, "numTokensPerRank datatype is invalid, datatype should be int32, but is %d.",
                            static_cast<ge::DataType>(numTokensPerRank->GetDataType())),
                    return false);
    OP_TILING_CHECK((numTokensPerExpert->GetDataType() != ge::DT_INT32),
                    OPS_LOG_E(nodeName, "numTokensPerExpert datatype is invalid, datatype should be int32, but is %d.",
                            static_cast<ge::DataType>(numTokensPerExpert->GetDataType())),
                    return false);
    OP_TILING_CHECK((isTokenInRank->GetDataType() != ge::DT_INT32),
                    OPS_LOG_E(nodeName, "isTokenInRank datatype is invalid, datatype should be int32, but is %d.",
                            static_cast<ge::DataType>(isTokenInRank->GetDataType())),
                    return false);
    OP_TILING_CHECK((notifySendData->GetDataType() != ge::DT_INT32),
                    OPS_LOG_E(nodeName, "notifySendData datatype is invalid, datatype should be int32, but is %d.",
                            static_cast<ge::DataType>(notifySendData->GetDataType())),
                    return false);

    return true;
}

static bool CheckTensorShape(const gert::TilingContext &context, const char *nodeName)
{
    const gert::StorageShape *topkIdxStorageShape = context.GetInputShape(INPUT_TOPK_IDX_INDEX);
    OP_TILING_CHECK(topkIdxStorageShape == nullptr, OPS_LOG_E(nodeName, "topkIdxStorageShape is null."), return false);

    OP_TILING_CHECK((topkIdxStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS),
                    OPS_LOG_E(nodeName, "topkIdx must be 2-dimension, but get %lu dim.",
                            topkIdxStorageShape->GetStorageShape().GetDimNum()),
                    return false);

    return true;
}

static ge::graphStatus TilingCheckTensor(const gert::TilingContext &context, const char *nodeName)
{
    OP_TILING_CHECK(!CheckTensorDataType(context, nodeName), OPS_LOG_E(nodeName, "params dataType is invalid."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckTensorShape(context, nodeName), OPS_LOG_E(nodeName, "params shape is invalid."),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchLayoutTilingFuncImpl(gert::TilingContext &context)
{
    const char *nodeName = context.GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OPS_LOG_E("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);
    DispatchLayoutTilingData *tilingData = context.GetTilingData<DispatchLayoutTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OPS_LOG_E(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);
    OPS_LOG_I(nodeName, "Enter DispatchLayout tiling check func.");

    OP_TILING_CHECK(GetAttrAndSetTilingData(context, nodeName, *tilingData) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Get attr and set tiling data failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(TilingCheckTensor(context, nodeName) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Tiling check param failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, nodeName) != ge::GRAPH_SUCCESS,
                    OPS_LOG_E(nodeName, "Tiling set workspace failed."), return ge::GRAPH_FAILED);

    int tilingKey = TILING_KEY_INT;
    if (CheckIfA2MultiMachine(context, nodeName, *tilingData)) {
        tilingKey = tilingKey + 100;
    }
    context.SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context.GetPlatformInfo());
    uint32_t blockDim;
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    blockDim = aivNum;
    context.SetBlockDim(blockDim);
    tilingData->dispatchLayoutInfo.totalUbSize = ubSize;
    OPS_LOG_D(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);
    PrintTilingDataInfo(nodeName, *tilingData);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchLayoutTilingFunc(gert::TilingContext *context)
{
    ge::graphStatus ret = DispatchLayoutTilingFuncImpl(*context);
    return ret;
}

struct DispatchLayoutCompileInfo {};
ge::graphStatus TilingParseForDispatchLayout(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchLayout)
    .Tiling(DispatchLayoutTilingFunc)
    .TilingParse<DispatchLayoutCompileInfo>(TilingParseForDispatchLayout);
} // namespace optiling