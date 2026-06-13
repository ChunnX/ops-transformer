/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file moe_ep_dispatch_epilogue_tiling.cpp
 * \brief Expert-sorted epilogue tiling — perSlotBytes = ALIGN_UP(hidden*2, 512) + 512,
 *        eAlloc from output shape, workspace for rank+expert prefix sums.
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
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling.h"
#include "../../op_kernel/moe_ep_dispatch_epilogue_tiling_key.h"
#include "../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"

#ifndef ALIGN_UP
#define ALIGN_UP(x, align) (((x) + (align)-1) & ~((align)-1))
#endif

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace optiling {

constexpr uint32_t CONTEXT_INDEX = 0U;
constexpr uint32_t DST_BUFFER_SLOT_IDX_INDEX = 1U;
constexpr uint32_t RECV_SRC_META_INTERNAL_INDEX = 2U;
constexpr uint32_t NUM_RECV_PER_RANK_HP_INDEX = 3U;
constexpr uint32_t NUM_RECV_PER_EXPERT_HP_INDEX = 4U;

constexpr uint32_t ATTR_EP_WORLD_SIZE_INDEX = 0;
constexpr uint32_t ATTR_EP_RANK_ID_INDEX = 1;
constexpr uint32_t ATTR_NUM_EXPERTS_INDEX = 2;
constexpr uint32_t ATTR_NUM_MAX_TPR_INDEX = 3;
constexpr uint32_t ATTR_CCL_BUFFER_SIZE_INDEX = 4;
constexpr uint32_t ATTR_EXPERT_ALIGNMENT_INDEX = 5;

constexpr uint32_t TWO_DIMS = 2U;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024U * 1024U;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr int64_t MAX_EP_WORLD_SIZE = 768;
constexpr int64_t MIN_EP_WORLD_SIZE = 2;
constexpr uint32_t HIDDEN_ALIGN = 32U;
constexpr uint64_t UB_ALIGN = 32UL;
constexpr uint64_t MAX_OUT_DTYPE_SIZE = 2UL;
constexpr uint64_t METADATA_DTYPE_SIZE = 4UL;   // sizeof(int32)=sizeof(float)=4

static void PrintTilingDataInfo(const char *nodeName, const MoeEpDispatchEpilogueInfo &info)
{
    OP_LOGD(nodeName, "epWorldSize=%u, epRankId=%u, numExperts=%u, numLocalExperts=%u",
              info.cfg.epWorldSize, info.cfg.epRankId, info.cfg.numExperts, info.cfg.numLocalExperts);
    OP_LOGD(nodeName, "hidden=%u, topK=%u, numTokens=%u, numMaxTokensPerRank=%u, perSlotBytes=%u",
              info.cfg.hidden, info.cfg.topK, info.cfg.numTokens, info.cfg.numMaxTokensPerRank, info.cfg.perSlotBytes);
    OP_LOGD(nodeName, "aivNum=%u", info.aivNum);
}

static ge::graphStatus CheckInputTensorShape(const gert::TilingContext *context, const char *nodeName)
{
    const gert::StorageShape *contextStorageShape = context->GetInputShape(CONTEXT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, contextStorageShape);
    OP_TILING_CHECK(contextStorageShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(nodeName, "contextShape dims must be 1."),
        return ge::GRAPH_FAILED);

    const gert::StorageShape *dstBufferSlotIdxShape = context->GetInputShape(DST_BUFFER_SLOT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, dstBufferSlotIdxShape);
    OP_TILING_CHECK(dstBufferSlotIdxShape->GetStorageShape().GetDimNum() != TWO_DIMS,
        OP_LOGE(nodeName, "dstBufferSlotIdxShape dims must be 2."), return ge::GRAPH_FAILED);

    const gert::StorageShape *recvSrcMetaInternalShape = context->GetInputShape(RECV_SRC_META_INTERNAL_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, recvSrcMetaInternalShape);

    const gert::StorageShape *numRecvPerRankHpShape = context->GetInputShape(NUM_RECV_PER_RANK_HP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerRankHpShape);
    OP_TILING_CHECK(numRecvPerRankHpShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(nodeName, "numRecvPerRankHpShape dims must be 1."), return ge::GRAPH_FAILED);

    const gert::StorageShape *numRecvPerExpertHpShape = context->GetInputShape(NUM_RECV_PER_EXPERT_HP_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, numRecvPerExpertHpShape);
    OP_TILING_CHECK(numRecvPerExpertHpShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(nodeName, "numRecvPerExpertHpShape dims must be 1."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAttrParams(const gert::TilingContext *context, const char *nodeName,
                                          MoeEpDispatchEpilogueInfo &info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(nodeName, "attrs is nullptr."), return ge::GRAPH_FAILED);

    auto epWorldSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_WORLD_SIZE_INDEX);
    auto epRankIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_EP_RANK_ID_INDEX);
    auto numExpertsPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_EXPERTS_INDEX);
    auto nmtPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_MAX_TPR_INDEX);
    auto cclBufferSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_CCL_BUFFER_SIZE_INDEX);
    auto expertAlignmentPtr = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_ALIGNMENT_INDEX);

    OP_TILING_CHECK(epWorldSizePtr == nullptr, OP_LOGE(nodeName, "epWorldSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(epRankIdPtr == nullptr, OP_LOGE(nodeName, "epRankIdPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(numExpertsPtr == nullptr, OP_LOGE(nodeName, "numExpertsPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(nmtPtr == nullptr, OP_LOGE(nodeName, "nmtPtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(cclBufferSizePtr == nullptr, OP_LOGE(nodeName, "cclBufferSizePtr is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertAlignmentPtr == nullptr, OP_LOGE(nodeName, "expertAlignmentPtr is null."), return ge::GRAPH_FAILED);

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
    info.cfg.expertAlignment = static_cast<uint32_t>(*expertAlignmentPtr);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MoeEpDispatchEpilogueTilingFunc(gert::TilingContext *context)
{
    context->SetScheduleMode(1U);
    const char *nodeName = context->GetNodeName();
    OP_TILING_CHECK(nodeName == nullptr, OP_LOGE("unKnownNodeName", "nodeName is nullptr."),
        return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueTilingData *tilingData = context->GetTilingData<MoeEpDispatchEpilogueTilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE(nodeName, "tilingData is nullptr."),
        return ge::GRAPH_FAILED);

    MoeEpDispatchEpilogueInfo &info = tilingData->moeEpDispatchEpilogueInfo;

    OP_TILING_CHECK(CheckAttrParams(context, nodeName, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check attr params failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputTensorShape(context, nodeName) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "Check input tensor shape failed."), return ge::GRAPH_FAILED);

    const gert::StorageShape *dstBufferSlotIdxShape = context->GetInputShape(DST_BUFFER_SLOT_IDX_INDEX);
    info.cfg.numTokens = static_cast<uint32_t>(dstBufferSlotIdxShape->GetStorageShape().GetDim(0));
    info.cfg.topK = static_cast<uint32_t>(dstBufferSlotIdxShape->GetStorageShape().GetDim(1));

    auto recvXShape = context->GetOutputShape(0);
    OP_TILING_CHECK(recvXShape == nullptr, OP_LOGE(nodeName, "recvX output shape is null."),
        return ge::GRAPH_FAILED);
    info.cfg.hidden = static_cast<uint32_t>(recvXShape->GetStorageShape().GetDim(1));

    OP_TILING_CHECK(info.cfg.hidden % HIDDEN_ALIGN != 0,
        OP_LOGE(nodeName, "hidden must be a multiple of %u for DataCopy alignment, got %u.",
            HIDDEN_ALIGN, info.cfg.hidden),
        return ge::GRAPH_FAILED);

    // info.cfg.perSlotBytes = static_cast<uint32_t>(
    //     ALIGN_UP(static_cast<uint64_t>(info.cfg.hidden) * 2UL, WIN_ADDR_ALIGN) + WIN_ADDR_ALIGN);
    uint32_t hAlign32 = ((info.cfg.hidden * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    uint32_t kAlign32 = ((info.cfg.topK * METADATA_DTYPE_SIZE + UB_ALIGN - 1UL) / UB_ALIGN) * UB_ALIGN;
    info.cfg.perSlotBytes = static_cast<uint32_t>((hAlign32 + kAlign32 * 2 + UB_ALIGN));

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(blockDim);
    info.aivNum = aivNum;
    info.totalUbSize = ubSize;

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(nodeName, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    uint64_t prefixSumBytes = ALIGN_UP(static_cast<uint64_t>(info.cfg.epWorldSize) * sizeof(int32_t), 8UL)
                            + static_cast<uint64_t>(info.cfg.numLocalExperts) * sizeof(int64_t);
    uint64_t hitCountBytes = static_cast<uint64_t>(aivNum) * info.cfg.numLocalExperts * sizeof(int32_t);
    uint64_t rowStartBytes = static_cast<uint64_t>(aivNum) * info.cfg.numLocalExperts * sizeof(int64_t);
    workSpaces[0] = SYSTEM_NEED_WORKSPACE + prefixSumBytes + hitCountBytes + rowStartBytes;

    uint64_t tilingKey = GET_TPL_TILING_KEY(TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(nodeName, "tilingKey=%lu, blockDim=%u, aivNum=%u", tilingKey, blockDim, aivNum);

    PrintTilingDataInfo(nodeName, info);
    return ge::GRAPH_SUCCESS;
}

struct MoeEpDispatchEpilogueCompileInfo {};
ge::graphStatus TilingParseForMoeEpDispatchEpilogue(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeEpDispatchEpilogue)
    .Tiling(MoeEpDispatchEpilogueTilingFunc)
    .TilingParse<MoeEpDispatchEpilogueCompileInfo>(TilingParseForMoeEpDispatchEpilogue);
}  // namespace optiling