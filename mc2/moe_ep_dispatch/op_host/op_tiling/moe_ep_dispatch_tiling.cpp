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
 * \file moe_ep_dispatch_tiling.cpp
 * \brief
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_dispatch_tiling.h"
#include "../../op_kernel/moe_ep_dispatch_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t X_INDEX = 1U;
constexpr uint32_t TOPK_IDX_INDEX = 2U;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 3U;
constexpr uint32_t SCALES_INDEX = 4U;
constexpr uint32_t CACHED_SLOT_IDX_INDEX = 5U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;
constexpr uint32_t ATTR_EXPERT_ALIGNMENT_INDEX = 5;
constexpr uint32_t ATTR_DO_CPU_SYNC_INDEX = 6;
constexpr uint32_t ATTR_HOST_PINNED_COUNTER_ADDR_INDEX = 7;

constexpr uint32_t ONE_DIM = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr int64_t MAX_EP_WORLD_SIZE = 768;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t METADATA_DTYPE_SIZE = 4UL;   // sizeof(int32)=sizeof(float)=4
constexpr int64_t H_MAX = 8192;
constexpr int64_t K_MAX = 32;

constexpr uint64_t INIT_TILINGKEY = 10000;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpDispatchInfo &info)
{
    OP_LOGD(nodeName, "epWorldSize is %u.", info.cfg.epWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", info.cfg.epRankId);
    OP_LOGD(nodeName, "numExperts is %u.", info.cfg.numExperts);
    OP_LOGD(nodeName, "numLocalExperts is %u.", info.cfg.numLocalExperts);
    OP_LOGD(nodeName, "numTokens is %u.", info.cfg.numTokens);
    OP_LOGD(nodeName, "hidden is %u.", info.cfg.hidden);
    OP_LOGD(nodeName, "topK is %u.", info.cfg.topK);
    OP_LOGD(nodeName, "numMaxTokensPerRank is %u.", info.cfg.numMaxTokensPerRank);
    OP_LOGD(nodeName, "perSlotBytes is %u.", info.cfg.perSlotBytes);
    OP_LOGD(nodeName, "expertAlignment is %u.", info.cfg.expertAlignment);
    OP_LOGD(nodeName, "doCpuSync is %u.", info.doCpuSync);
    OP_LOGD(nodeName, "isCached is %u.", info.isCached);
    OP_LOGD(nodeName, "networkMode is %u.", info.networkMode);
    OP_LOGD(nodeName, "numScaleupRanks is %u.", info.numScaleupRanks);
    OP_LOGD(nodeName, "numScaleoutRanks is %u.", info.numScaleoutRanks);
    OP_LOGD(nodeName, "numAivStage1 is %u.", info.numAivStage1);
    OP_LOGD(nodeName, "numAivStage2 is %u.", info.numAivStage2);
    OP_LOGD(nodeName, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName, "hostPinnedCounterAddr is %lu.", info.hostPinnedCounterAddr);
    OP_LOGD(nodeName, "totalWinSizeEp is %lu.", info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalUbSize is %lu.", info.totalUbSize);
}

static ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context, const char *nodeName,
                                               MoeEpDispatchInfo &info)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    const gert::StorageShape *xStorageShape = context->GetInputShape(X_INDEX);
    const gert::StorageShape *topkIdxStorageShape = context->GetInputShape(TOPK_IDX_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, contextStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxStorageShape);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != ONE_DIM,
        OP_LOGE(nodeName, "contextShape dims must be 1, but current dim num is %lu.",
            contextStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "xShape dims must be 2, but current dim num is %lu.",
            xStorageShape->GetStorageShape().GetDimNum()), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkIdxStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkIdxShape dims must be 2."), return ge::GRAPH_FAILED);

    const int64_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int64_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    const int64_t topkDim0 = topkIdxStorageShape->GetStorageShape().GetDim(0);
    const int64_t topkDim1 = topkIdxStorageShape->GetStorageShape().GetDim(1);
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(info.cfg.numMaxTokensPerRank);
    OP_TILING_CHECK((xDim0 <= 0) || (xDim0 > numMaxTokensPerRank), OP_LOGE(nodeName,
        "xDim0(numTokens) is invalid, should be in [1, %ld], but got %ld.", numMaxTokensPerRank, xDim0),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((xDim1 <= 0) || (xDim1 > H_MAX), OP_LOGE(nodeName, "xDim1(hidden) is invalid, "
        "should be in (0, %ld], but got %ld.", H_MAX, xDim1), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xDim0 != topkDim0, OP_LOGE(nodeName, "topkIdx dim0 must equal x dim0, got %ld vs %ld.",
        topkDim0, xDim0), return ge::GRAPH_FAILED);
    OP_TILING_CHECK((topkDim1 <= 0) || (topkDim1 > K_MAX), OP_LOGE(nodeName, "topkIdx dim1(topK) is invalid, "
        "should be in (0, %ld], got %ld.", K_MAX, topkDim1), return ge::GRAPH_FAILED);

    info.cfg.numTokens = static_cast<uint32_t>(xDim0);
    info.cfg.hidden = static_cast<uint32_t>(xDim1);
    info.cfg.topK = static_cast<uint32_t>(topkDim1);

    bool cached = (context->GetOptionalInputShape(CACHED_SLOT_IDX_INDEX) != nullptr);
    if (!cached) {
        const gert::StorageShape *topkWeightsStorageShape = context->GetOptionalInputShape(TOPK_WEIGHTS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsStorageShape);
        OP_TILING_CHECK(topkWeightsStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "topkWeightsShape dims must be 2."), return ge::GRAPH_FAILED);
        const int64_t weightsDim0 = topkWeightsStorageShape->GetStorageShape().GetDim(0);
        const int64_t weightsDim1 = topkWeightsStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(xDim0 != weightsDim0, OP_LOGE(nodeName, "topkWeights dim0 must equal x dim0, "
            "got %ld vs %ld.", weightsDim0, xDim0), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(topkDim1 != weightsDim1, OP_LOGE(nodeName, "topkWeights dim1 must equal topkIdx dim1, "
            "got %ld vs %ld.", weightsDim1, topkDim1), return ge::GRAPH_FAILED);
    } else {
        const gert::StorageShape *cachedSlotStorageShape = context->GetOptionalInputShape(CACHED_SLOT_IDX_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, cachedSlotStorageShape);
        OP_TILING_CHECK(cachedSlotStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "cachedSlotIdxShape dims must be 2."), return ge::GRAPH_FAILED);
        const int64_t cachedDim0 = cachedSlotStorageShape->GetStorageShape().GetDim(0);
        const int64_t cachedDim1 = cachedSlotStorageShape->GetStorageShape().GetDim(1);
        OP_TILING_CHECK(xDim0 != cachedDim0, OP_LOGE(nodeName, "cachedSlotIdx dim0 must equal x dim0, "
            "got %ld vs %ld.", cachedDim0, xDim0), return ge::GRAPH_FAILED);
        OP_TILING_CHECK((cachedDim1 <= 0) || (cachedDim1 > K_MAX), OP_LOGE(nodeName,
            "cachedSlotIdx dim1(topK) is invalid, should be in (0, %ld], got %ld.", K_MAX, cachedDim1),
            return ge::GRAPH_FAILED);
        // info.cfg.topK = static_cast<uint32_t>(cachedDim1);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, const char *nodeName,
                                         MoeEpDispatchInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto expertAlignmentPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_ALIGNMENT_INDEX);
    auto doCpuSyncPtr = attrs->GetAttrPointer<bool>(ATTR_DO_CPU_SYNC_INDEX);
    auto hpCounterAddrPtr = attrs->GetAttrPointer<int64_t>(ATTR_HOST_PINNED_COUNTER_ADDR_INDEX);

    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nmtPtr == nullptr, OP_LOGE(nodeName, "numMaxTokensPerRankPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertAlignmentPtr == nullptr, OP_LOGE(nodeName, "expertAlignmentPtr is null."),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(doCpuSyncPtr == nullptr, OP_LOGE(nodeName, "doCpuSyncPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(hpCounterAddrPtr == nullptr, OP_LOGE(nodeName, "hpCounterAddrPtr is null."), return ge::GRAPH_FAILED);

    int64_t epWorldSize = *epWorldSizePtr;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE), OP_LOGE(nodeName,
        "epWorldSize is invalid, should be in [%ld, %ld], got %ld.", MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize), OP_LOGE(nodeName,
        "epRankId is invalid, should be in [0, %ld), got %ld.", epWorldSize, *epRankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numExpertsPtr <= 0) || (*numExpertsPtr % epWorldSize != 0), OP_LOGE(nodeName,
        "numExperts must be positive and divisible by epWorldSize, got %ld.", *numExpertsPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*nmtPtr <= 0), OP_LOGE(nodeName, "numMaxTokensPerRank must be positive, got %ld.", *nmtPtr),
        return ge::GRAPH_FAILED);

    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);
    info.cfg.expertAlignment = static_cast<uint32_t>(*expertAlignmentPtr);

    bool cached = (context->GetOptionalInputShape(CACHED_SLOT_IDX_INDEX) != nullptr);
    bool doCpuSync = *doCpuSyncPtr;
    OP_TILING_CHECK(cached && doCpuSync, OP_LOGE(nodeName, "doCpuSync and cached can't be true at the same time."),
        return ge::GRAPH_FAILED);

    info.doCpuSync = (doCpuSync && !cached) ? 1 : 0;
    info.isCached = cached ? 1 : 0;
    info.hostPinnedCounterAddr = static_cast<uint64_t>(*hpCounterAddrPtr);

    // info.cfg.perSlotBytes = static_cast<uint32_t>(
    //     ((static_cast<uint64_t>(info.cfg.hidden) * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN
    //     + 80UL + 16UL);
    uint32_t hAlign32 = ((info.cfg.hidden * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t kAlign32 = ((info.cfg.topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    info.cfg.perSlotBytes = static_cast<uint32_t>((hAlign32 + kAlign32 * 2 + UB_ALIGN));
    OP_LOGD(nodeName, "perSlotBytes = %u (hidden=%u)", info.cfg.perSlotBytes, info.cfg.hidden);

    return ge::GRAPH_SUCCESS;
}

static uint64_t CalcDispatchWorkspace(const MoeEpDispatchInfo &info)
{
    uint64_t epWorldSize = static_cast<uint64_t>(info.cfg.epWorldSize);
    uint64_t nmt = static_cast<uint64_t>(info.cfg.numMaxTokensPerRank);
    uint64_t perSlotBytes = static_cast<uint64_t>(info.cfg.perSlotBytes);
    uint64_t moeExpertNumPerRank = static_cast<uint64_t>(info.cfg.numLocalExperts);
    uint64_t counterBytes = (epWorldSize * sizeof(int32_t) + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN;
    uint64_t sendCntPerRankBytes = (2 * epWorldSize * sizeof(int32_t) + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN;
    uint64_t sendCntPerExpertBytes = (moeExpertNumPerRank * epWorldSize * sizeof(int32_t) + UB_ALIGN -1) / UB_ALIGN * UB_ALIGN;

    uint64_t sendCntBytes = counterBytes + sendCntPerRankBytes + sendCntPerExpertBytes;
    uint64_t globalABytes = UB_ALIGN;
    uint64_t stashBytes = epWorldSize * nmt * perSlotBytes;
    return SYSTEM_NEED_WORKSPACE + counterBytes + globalABytes + stashBytes;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeEpDispatchInfo &info,
                                     const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    uint64_t epWorldSize = static_cast<uint64_t>(info.cfg.epWorldSize);
    uint64_t nmt = static_cast<uint64_t>(info.cfg.numMaxTokensPerRank);
    uint64_t perSlotBytes = static_cast<uint64_t>(info.cfg.perSlotBytes);

    uint64_t winNeedDirect = epWorldSize * nmt * perSlotBytes;
    uint64_t winNeed = winNeedDirect;
    if (epWorldSize > 128) {
        uint64_t numScaleup = 128UL;
        uint64_t numScaleout = (epWorldSize + 127UL) / 128UL;
        uint64_t scaleupRecv = numScaleup * (numScaleout * nmt) * perSlotBytes;
        uint64_t scaleoutSend = 1UL * nmt * perSlotBytes;
        uint64_t scaleoutRecv = numScaleout * (nmt + 32UL) * perSlotBytes;
        winNeed = scaleupRecv + scaleoutSend + scaleoutRecv;
    }
    OP_TILING_CHECK(winNeed > maxWindowSize,
        OP_LOGE(nodeName, "HCCL_BUFFSIZE is too SMALL, need %luMB, available %luMB.",
            (winNeed / MB_SIZE) + 1UL, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    info.totalWinSizeEp = maxWindowSize;
    OP_LOGD(nodeName, "windowSize = %lu", maxWindowSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context, const MoeEpDispatchInfo &info,
                                      const char *nodeName)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = CalcDispatchWorkspace(info);
    return ge::GRAPH_SUCCESS;
}

static uint64_t CalTilingKey(const uint32_t doCpuSync, const uint32_t isCached, const uint32_t networkMode)
{
    return GET_TPL_TILING_KEY(doCpuSync, isCached, networkMode);
}

static ge::graphStatus MoeEpDispatchTilingFunc(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."),
        return ge::GRAPH_FAILED);

    MoeEpDispatchTilingData *tilingData = context->GetTilingData<MoeEpDispatchTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter MoeEpDispatch tiling func.");

    MoeEpDispatchInfo &info = tilingData->moeEpDispatchInfo;

    OP_TILING_CHECK(CheckAttrParams(context, nodeName, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    uint32_t epWorldSize = info.cfg.epWorldSize;
    info.networkMode = (epWorldSize > 128) ? 1 : 0;
    if (epWorldSize > 128) {
        info.numScaleupRanks = 128;
        info.numScaleoutRanks = (epWorldSize + 127) / 128;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t totalAiv = ascendcPlatform.GetCoreNumAiv();
        info.numAivStage1 = totalAiv * 5 / 8;
        info.numAivStage2 = totalAiv * 2 / 8;
    } else {
        info.numScaleupRanks = epWorldSize;
        info.numScaleoutRanks = 1;
        info.numAivStage1 = 0;
        info.numAivStage2 = 0;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;
    context->SetBlockDim(blockDim);
    context->SetScheduleMode(1U);
    OP_LOGD(nodeName, "blockDim=%u, aivNum=%u, ubSize=%lu", blockDim, aivNum, ubSize);

    OP_TILING_CHECK(CheckWinSize(context, info, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check window size failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SetWorkSpace(context, info, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Set workspace failed."), return ge::GRAPH_FAILED);

    uint64_t tilingKey = CalTilingKey(info.doCpuSync, info.isCached, info.networkMode);
    OP_LOGD(nodeName, "tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);

    PrintTilingDataInfo(nodeName, info);

    return ge::GRAPH_SUCCESS;
}

struct MoeEpDispatchCompileInfo {};
ge::graphStatus TilingParseForMoeEpDispatch(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatch)
    .Tiling(MoeEpDispatchTilingFunc)
    .TilingParse<MoeEpDispatchCompileInfo>(TilingParseForMoeEpDispatch);
}  // namespace optiling