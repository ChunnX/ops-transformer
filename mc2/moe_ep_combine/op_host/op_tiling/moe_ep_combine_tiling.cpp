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
 * \file moe_ep_combine_tiling.cpp
 * \brief
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "mc2_tiling_utils.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "../../op_kernel/moe_ep_combine_tiling.h"
#include "../../op_kernel/moe_ep_combine_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t RECVX_INDEX = 1U;
constexpr uint32_t TOPK_IDX_INDEX = 2U;
constexpr uint32_t RECV_SRC_METADATA_INDEX = 3U;
constexpr uint32_t NUM_RECV_PER_EXPERT_INDEX = 4U;
constexpr uint32_t TOPK_WEIGHTS_INDEX = 5U;
constexpr uint32_t BIAS_0_INDEX = 6U;
constexpr uint32_t BIAS_1_INDEX = 7U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;

constexpr uint32_t ONE_DIMS = 1U;
constexpr uint32_t TWO_DIMS = 2U;
constexpr int64_t MAX_EP_WORLD_SIZE = 128;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t COMM_ALIGN = 512UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t METADATA_DTYPE_SIZE = 4UL;
constexpr int64_t H_MIN = 1;
constexpr int64_t H_MAX = 8192;
constexpr int64_t K_MAX = 32;
constexpr int64_t META_INNER_DIM = 4;

static void PrintTilingDataInfo(const char *nodeName, const MoeEpCombineInfo &info)
{
    OP_LOGD(nodeName, "epWorldSize=%u, epRankId=%u, numExperts=%u, numLocalExperts=%u", info.cfg.epWorldSize,
            info.cfg.epRankId, info.cfg.numExperts, info.cfg.numLocalExperts);
    OP_LOGD(nodeName, "numTokens=%u, hidden=%u, topK=%u, numMaxTokensPerRank=%u", info.cfg.numTokens, info.cfg.hidden,
            info.cfg.topK, info.cfg.numMaxTokensPerRank);
    OP_LOGD(nodeName, "perSlotBytes=%u, hasTopkWeights=%u, aivNum=%u", info.cfg.perSlotBytes, info.hasTopkWeights,
            info.aivNum);
    OP_LOGD(nodeName, "totalWinSizeEp=%lu, localWsSizeDataPerRank=%lu, totalUbSize=%lu", info.totalWinSizeEp,
            info.localWsSizeDataPerRank, info.totalUbSize);
}

static ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context, const char *nodeName,
                                             MoeEpCombineInfo &info)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextStorageShape);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(nodeName, "contextShape dims must be 1."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *recvxShape = context->GetInputShape(RECVX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvxShape);
    OP_TILING_CHECK(recvxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "recvxShape dims must be 2."),
        return ge::GRAPH_FAILED);
    const int64_t recvxDim0 = recvxShape->GetStorageShape().GetDim(0);
    const int64_t recvxDim1 = recvxShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(recvxDim0 <= 0,
        OP_LOGE(nodeName, "recvxDim0(A) must be positive, got %ld.", recvxDim0),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((recvxDim1 < H_MIN) || (recvxDim1 > H_MAX),
        OP_LOGE(nodeName, "recvxDim1(hidden) is invalid, should be in [%ld, %ld], got %ld.",
            H_MIN, H_MAX, recvxDim1),
        return ge::GRAPH_FAILED);
    info.cfg.hidden = static_cast<uint32_t>(recvxDim1);

    const gert::StorageShape *topkIdxShape = context->GetInputShape(TOPK_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkIdxShape);
    OP_TILING_CHECK(topkIdxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "topkIdxShape dims must be 2."), return ge::GRAPH_FAILED);
    const int64_t topkDim0 = topkIdxShape->GetStorageShape().GetDim(0);
    const int64_t topkDim1 = topkIdxShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(topkDim0 <= 0,
        OP_LOGE(nodeName, "topkIdx dim0(numTokens) must be positive, got %ld.", topkDim0),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(topkDim1 <= 0 || topkDim1 > K_MAX,
        OP_LOGE(nodeName, "topkIdx dim1(topK) must be in (0, %ld], got %ld.", K_MAX, topkDim1),
        return ge::GRAPH_FAILED);
    info.cfg.numTokens = static_cast<uint32_t>(topkDim0);
    info.cfg.topK = static_cast<uint32_t>(topkDim1);

    const gert::StorageShape *recvSrcMetadataShape = context->GetInputShape(RECV_SRC_METADATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetadataShape);
    OP_TILING_CHECK(recvSrcMetadataShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "recvSrcMetadataShape dims must be 2."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(recvSrcMetadataShape->GetStorageShape().GetDim(1) != META_INNER_DIM,
        OP_LOGE(nodeName, "recvSrcMetadata dim1 must be %ld, got %ld.",
            META_INNER_DIM, recvSrcMetadataShape->GetStorageShape().GetDim(1)),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *numRecvPerExpertShape = context->GetInputShape(NUM_RECV_PER_EXPERT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpertShape);
    OP_TILING_CHECK(numRecvPerExpertShape->GetStorageShape().GetDimNum() != ONE_DIMS,
        OP_LOGE(nodeName, "numRecvPerExpertShape dims must be 1."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numRecvPerExpertShape->GetStorageShape().GetDim(0) !=
        static_cast<int64_t>(info.cfg.numLocalExperts),
        OP_LOGE(nodeName, "numRecvPerExpert dim0 must equal numLocalExperts=%u, got %ld.",
            info.cfg.numLocalExperts, numRecvPerExpertShape->GetStorageShape().GetDim(0)),
        return ge::GRAPH_FAILED);

    bool hasTopkWeights = (context->GetInputShape(TOPK_WEIGHTS_INDEX) != nullptr);
    if (hasTopkWeights) {
        const gert::StorageShape *topkWeightsShape = context->GetInputShape(TOPK_WEIGHTS_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, topkWeightsShape);
        OP_TILING_CHECK(topkWeightsShape->GetStorageShape().GetDimNum() != ONE_DIMS,
            OP_LOGE(nodeName, "topkWeightsShape dims must be 1."), return ge::GRAPH_FAILED);
    }
    info.hasTopkWeights = hasTopkWeights ? 1 : 0;

    bool hasBias0 = (context->GetInputShape(BIAS_0_INDEX) != nullptr);
    if (hasBias0) {
        const gert::StorageShape *bias0Shape = context->GetInputShape(BIAS_0_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, bias0Shape);
        OP_TILING_CHECK(bias0Shape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "bias0Shape dims must be 2."), return ge::GRAPH_FAILED);
    }

    bool hasBias1 = (context->GetInputShape(BIAS_1_INDEX) != nullptr);
    if (hasBias1) {
        const gert::StorageShape *bias1Shape = context->GetInputShape(BIAS_1_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context, bias1Shape);
        OP_TILING_CHECK(bias1Shape->GetStorageShape().GetDimNum() != TWO_DIMS,
            OP_LOGE(nodeName, "bias1Shape dims must be 2."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, const char *nodeName,
                                       MoeEpCombineInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);

    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nmtPtr == nullptr, OP_LOGE(nodeName, "nmtPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."),
        return ge::GRAPH_FAILED);

    int64_t epWorldSize = *epWorldSizePtr;
    OP_TILING_CHECK((epWorldSize < MIN_EP_WORLD_SIZE) || (epWorldSize > MAX_EP_WORLD_SIZE),
        OP_LOGE(nodeName, "epWorldSize is invalid, should be in [%ld, %ld], got %ld.",
            MIN_EP_WORLD_SIZE, MAX_EP_WORLD_SIZE, epWorldSize),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*epRankIdPtr < 0) || (*epRankIdPtr >= epWorldSize),
        OP_LOGE(nodeName, "epRankId is invalid, should be in [0, %ld), got %ld.",
            epWorldSize, *epRankIdPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK((*numExpertsPtr <= 0) || (*numExpertsPtr % epWorldSize != 0),
        OP_LOGE(nodeName, "numExperts must be positive and divisible by epWorldSize, got %ld.",
            *numExpertsPtr),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*nmtPtr <= 0, OP_LOGE(nodeName, "numMaxTokensPerRank must be positive, got %ld.", *nmtPtr),
        return ge::GRAPH_FAILED);

    info.cfg.epWorldSize = static_cast<uint32_t>(epWorldSize);
    info.cfg.epRankId = static_cast<uint32_t>(*epRankIdPtr);
    info.cfg.numExperts = static_cast<uint32_t>(*numExpertsPtr);
    info.cfg.numLocalExperts = static_cast<uint32_t>(*numExpertsPtr / epWorldSize);
    info.cfg.numMaxTokensPerRank = static_cast<uint32_t>(*nmtPtr);

    return ge::GRAPH_SUCCESS;
}

static uint64_t AlignUpWin(const uint64_t data)
{
    return (data + COMM_ALIGN - 1) / COMM_ALIGN * COMM_ALIGN;
}

static ge::graphStatus CheckWinSize(const gert::TilingContext *context, MoeEpCombineInfo &info,
                                    const char *nodeName)
{
    auto attrs = context->GetAttrs();
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);

    uint64_t maxWindowSize = static_cast<uint64_t>(*cclBufferSizePtr);
    uint64_t epWorldSize = static_cast<uint64_t>(info.cfg.epWorldSize);
    uint64_t moeExpertNumPerRank = static_cast<uint64_t>(info.cfg.numLocalExperts);
    uint64_t perWinDataNeed = static_cast<uint64_t>(info.cfg.topK) * static_cast<uint64_t>(*nmtPtr) *
        static_cast<uint64_t>(info.cfg.perSlotBytes);

    uint64_t dispatchWinStateSize = epWorldSize * AlignUpWin(moeExpertNumPerRank * sizeof(int32_t)) +
                                    2 * epWorldSize * COMM_ALIGN;
    uint64_t combineWinStateSize = static_cast<uint64_t>(info.cfg.topK) * static_cast<uint64_t>(*nmtPtr) *
                                   COMM_ALIGN + epWorldSize * COMM_ALIGN;
    uint32_t hiddenAlign = (info.cfg.hidden * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    uint32_t topKAlign = ((info.cfg.topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint64_t dispatchWinDataSize = epWorldSize * static_cast<uint64_t>(*nmtPtr) *
                                   AlignUpWin(static_cast<uint64_t>(hiddenAlign + topKAlign * 2 + UB_ALIGN));

    uint64_t winNeedTotal = dispatchWinStateSize + combineWinStateSize + dispatchWinDataSize + perWinDataNeed;
    OP_TILING_CHECK(winNeedTotal > maxWindowSize,
        OP_LOGE(nodeName, "HCCL_BUFFSIZE too small, need %luMB, available %luMB.",
            (winNeedTotal / MB_SIZE) + 1UL, maxWindowSize / MB_SIZE),
        return ge::GRAPH_FAILED);
    info.totalWinSizeEp = maxWindowSize;
    info.winStateOffset = dispatchWinStateSize;
    info.winDataOffset = dispatchWinStateSize + combineWinStateSize + dispatchWinDataSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpCombineTilingFunc(gert::TilingContext *context)
{
    context->SetScheduleMode(1U);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."), return ge::GRAPH_FAILED);

    MoeEpCombineTilingData *tilingData = context->GetTilingData<MoeEpCombineTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);
    OP_LOGI(nodeName, "Enter MoeEpCombine tiling func.");

    MoeEpCombineInfo &info = tilingData->moeEpCombineInfo;

    OP_TILING_CHECK(CheckAttrParams(context, nodeName, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    uint32_t hAlign32 = ((info.cfg.hidden * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    info.cfg.perSlotBytes = static_cast<uint32_t>(((hAlign32 + UB_ALIGN + COMM_ALIGN - 1UL)
                            / COMM_ALIGN) * COMM_ALIGN);

    OP_TILING_CHECK(CheckWinSize(context, info, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check window size failed."), return ge::GRAPH_FAILED);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    uint64_t perWinDataNeed = static_cast<uint64_t>(info.cfg.numMaxTokensPerRank) *
        static_cast<uint64_t>(info.cfg.topK) * static_cast<uint64_t>(info.cfg.perSlotBytes);
    uint64_t perWinStateNeed = static_cast<uint64_t>(info.cfg.numMaxTokensPerRank) *
        static_cast<uint64_t>(info.cfg.topK) * static_cast<uint64_t>(UB_ALIGN);
    info.localWsSizeDataPerRank = perWinDataNeed;
    info.localWsSizeStatusPerRank = perWinStateNeed;
    uint64_t stashBytes = static_cast<uint64_t>(info.cfg.epWorldSize) * (perWinDataNeed + perWinStateNeed);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + stashBytes;

    uint32_t tplHasTopkWeights = info.hasTopkWeights ? 1 : 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(tplHasTopkWeights, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    OP_LOGD(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u", tilingKey, blockDim, aivNum);
    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

struct MoeEpCombineCompileInfo {};
ge::graphStatus TilingParseForMoeEpCombine(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpCombine)
    .Tiling(MoeEpCombineTilingFunc)
    .TilingParse<MoeEpCombineCompileInfo>(TilingParseForMoeEpCombine);
} // namespace optiling