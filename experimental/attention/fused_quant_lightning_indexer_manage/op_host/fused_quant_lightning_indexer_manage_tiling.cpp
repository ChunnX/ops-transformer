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
 * \file fused_quant_lightning_indexer_manage_tiling.cpp
 * \brief Host tiling implementation for FusedQuantLightningIndexerManage.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "../op_kernel/fused_quant_lightning_indexer_manage_tiling.h"
#include "../op_kernel/fused_quant_lightning_indexer_manage_fast_workspace.h"
#include "../op_kernel/lim_simt_config.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
enum InputIndex : size_t {
    INDEX_WEIGHTS = 0,
    QUERY_DEQUANT_SCALE,
    QUERY,
    INDEX_KEY_DEQUANT_SCALE,
    INDEX_KEY_CACHE,
    INDEX_BLOCK_TABLE,
    ACTUAL_SEQ_LENGTHS_QUERY,
    ACTUAL_SEQ_LENGTHS_KEY,
    OFFLOAD_SEQ_LENGTHS_KEY,
    NUM_CACHE_TOKENS,
    REQUEST_STATE,
    REQ_POOL_ENTRIES,
    CACHE_SLOTS_POOL,
};
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t MAX_ROUTES = 4;
constexpr int64_t MAX_SOURCE_CAPACITY = 1 << 18;
constexpr size_t ATTR_KEY_STRIDE = 0;
constexpr size_t ATTR_SCALE_STRIDE = 1;
constexpr size_t ATTR_WEIGHT_STRIDE = 2;

bool Rank(const gert::Shape &shape, size_t rank)
{
    return shape.GetDimNum() == rank;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFusedQuantLightningIndexerManage(gert::TilingContext *context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr)
        return ge::GRAPH_FAILED;
    for (size_t i = 0; i <= CACHE_SLOTS_POOL; ++i) {
        if (context->GetInputShape(i) == nullptr || context->GetInputDesc(i) == nullptr)
            return ge::GRAPH_FAILED;
    }
    const auto dtype = [context](size_t i) { return context->GetInputDesc(i)->GetDataType(); };
    if (dtype(INDEX_WEIGHTS) != ge::DT_BF16 || dtype(QUERY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(QUERY) != ge::DT_FLOAT8_E4M3FN || dtype(INDEX_KEY_DEQUANT_SCALE) != ge::DT_FLOAT ||
        dtype(INDEX_KEY_CACHE) != ge::DT_FLOAT8_E4M3FN)
        return ge::GRAPH_FAILED;
    for (size_t i = INDEX_BLOCK_TABLE; i <= CACHE_SLOTS_POOL; ++i)
        if (dtype(i) != ge::DT_INT32)
            return ge::GRAPH_FAILED;

    const auto weights = context->GetInputShape(INDEX_WEIGHTS)->GetStorageShape();
    const auto queryScale = context->GetInputShape(QUERY_DEQUANT_SCALE)->GetStorageShape();
    const auto query = context->GetInputShape(QUERY)->GetStorageShape();
    const auto keyScale = context->GetInputShape(INDEX_KEY_DEQUANT_SCALE)->GetStorageShape();
    const auto key = context->GetInputShape(INDEX_KEY_CACHE)->GetStorageShape();
    const auto blockTable = context->GetInputShape(INDEX_BLOCK_TABLE)->GetStorageShape();
    const auto queryEnds = context->GetInputShape(ACTUAL_SEQ_LENGTHS_QUERY)->GetStorageShape();
    const auto pool = context->GetInputShape(CACHE_SLOTS_POOL)->GetStorageShape();
    if (!Rank(weights, 2) || !Rank(queryScale, 2) || !Rank(query, 3) || !Rank(keyScale, 3) || !Rank(key, 4) ||
        !Rank(blockTable, 2) || !Rank(queryEnds, 1) || !Rank(pool, 2))
        return ge::GRAPH_FAILED;
    const int64_t totalQueries = query.GetDim(0);
    const int64_t heads = query.GetDim(1);
    const int64_t batch = queryEnds.GetDim(0);
    const int64_t sourceCapacity = blockTable.GetDim(1) * BLOCK_SIZE;
    if (batch <= 0 || totalQueries < batch || totalQueries > batch * MAX_ROUTES || (heads != 32 && heads != 64) ||
        query.GetDim(2) != HEAD_DIM || weights.GetDim(0) != totalQueries || weights.GetDim(1) != heads ||
        queryScale.GetDim(0) != totalQueries || queryScale.GetDim(1) != heads || key.GetDim(0) <= 0 ||
        key.GetDim(1) != BLOCK_SIZE || key.GetDim(2) != 1 || key.GetDim(3) != HEAD_DIM ||
        keyScale.GetDim(0) != key.GetDim(0) || keyScale.GetDim(1) != BLOCK_SIZE || keyScale.GetDim(2) != 1 ||
        blockTable.GetDim(0) != batch || sourceCapacity <= 0 || sourceCapacity > MAX_SOURCE_CAPACITY ||
        pool.GetDim(0) <= 0 || pool.GetDim(1) != sourceCapacity)
        return ge::GRAPH_FAILED;
    for (size_t i = ACTUAL_SEQ_LENGTHS_KEY; i <= REQ_POOL_ENTRIES; ++i) {
        const auto shape = context->GetInputShape(i)->GetStorageShape();
        if (!Rank(shape, 1) || shape.GetDim(0) != batch)
            return ge::GRAPH_FAILED;
    }

    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr)
        return ge::GRAPH_FAILED;
    const int64_t *keyStride = attrs->GetAttrPointer<int64_t>(ATTR_KEY_STRIDE);
    const int64_t *scaleStride = attrs->GetAttrPointer<int64_t>(ATTR_SCALE_STRIDE);
    const int64_t *weightStride = attrs->GetAttrPointer<int64_t>(ATTR_WEIGHT_STRIDE);
    if (keyStride == nullptr || scaleStride == nullptr || weightStride == nullptr ||
        *keyStride < BLOCK_SIZE * HEAD_DIM || *scaleStride < BLOCK_SIZE || *weightStride < heads ||
        static_cast<uint64_t>(*keyStride) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*scaleStride) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(*weightStride) > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t mixGroups = std::min<uint32_t>(platform.GetCoreNumAic(), platform.GetCoreNumAiv() / 2U);
    if (mixGroups == 0U)
        return ge::GRAPH_FAILED;
    // A5 switches 32-head launches to one four-route tile above 64 query
    // rows. Also switch when two reserved tiles per request would exceed
    // the available mix groups: Q=1/2 leaves every second tile empty, so
    // modulo scheduling would serialize requests on only half the cores.
    const uint32_t fastQueryTileSize =
        heads == 32 && (totalQueries > 64 || static_cast<uint32_t>(batch) * 2U > mixGroups) ? 4U : 2U;
    const uint32_t fastTaskCount = static_cast<uint32_t>(batch) * (MAX_ROUTES / fastQueryTileSize);
    const uint32_t usedCores = std::min<uint32_t>(fastTaskCount, mixGroups);
    const uint32_t s1Base = (256U + static_cast<uint32_t>(heads) - 1U) / static_cast<uint32_t>(heads);
    const uint64_t scoreStride =
        static_cast<uint64_t>(s1Base) * static_cast<uint64_t>(sourceCapacity) * sizeof(uint16_t);
    if (scoreStride > std::numeric_limits<uint32_t>::max() ||
        scoreStride * usedCores > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;
    const uint64_t fastScoreStride = 4U * static_cast<uint64_t>(sourceCapacity) * sizeof(uint16_t);
    const uint64_t fastWorkspace =
        fused_quant_li_manage_fast_workspace::TotalBytes(fastScoreStride, static_cast<uint64_t>(batch));
    if (fastScoreStride > std::numeric_limits<uint32_t>::max() || fastWorkspace > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;
    auto *tiling = context->GetTilingData<FusedQuantLightningIndexerManageTilingData>();
    if (tiling == nullptr || context->GetWorkspaceSizes(1) == nullptr)
        return ge::GRAPH_FAILED;
    tiling->usedCoreNum = usedCores;
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->totalQueryRows = static_cast<uint32_t>(totalQueries);
    tiling->poolSize = static_cast<uint32_t>(pool.GetDim(0));
    tiling->sourceCapacity = static_cast<uint32_t>(sourceCapacity);
    tiling->indexHeads = static_cast<uint32_t>(heads);
    tiling->maxBlockNumPerBatch = static_cast<uint32_t>(blockTable.GetDim(1));
    tiling->weightStride = static_cast<uint32_t>(*weightStride);
    tiling->keyStride = static_cast<uint32_t>(*keyStride);
    tiling->scaleStride = static_cast<uint32_t>(*scaleStride);
    tiling->scoreWorkspaceStride = static_cast<uint32_t>(scoreStride);
    // The fast manager now consumes route-pair workspace while publishing
    // statuses. Keep the two regions disjoint even for small core counts.
    tiling->managementWorkspaceOffset =
        static_cast<uint32_t>(std::max<uint64_t>(scoreStride * usedCores, fastWorkspace));
    tiling->fastScoreWorkspaceStride = static_cast<uint32_t>(fastScoreStride);
    tiling->fastScoreRowStride = static_cast<uint32_t>(sourceCapacity);
    tiling->fastQueryTileSize = fastQueryTileSize;
    tiling->fastPathEnabled = 1U;
    const uint64_t managementWorkspace = static_cast<uint64_t>(batch + totalQueries) * 32U;
    context->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() +
        std::max<uint64_t>(static_cast<uint64_t>(tiling->managementWorkspaceOffset) + managementWorkspace,
                           fastWorkspace);
    context->SetBlockDim(platform.CalcTschBlockDim(usedCores * 2U, usedCores, usedCores * 2U));
    context->SetScheduleMode(1);
#if LIM_SIMT_POSTPROCESS
    // Match the A5 reference: reserve 32 KiB for SIMT DCache and 8 KiB
    // for runtime/stack so TPipe UB buffers cannot overlap those regions.
    if (context->SetLocalMemorySize(LIM_SIMT_USER_UB_BYTES) != ge::GRAPH_SUCCESS)
        return ge::GRAPH_FAILED;
#endif
    return ge::GRAPH_SUCCESS;
}

struct FusedQuantLightningIndexerManageCompileInfo {};
static ge::graphStatus TilingParseFusedQuantLightningIndexerManage(gert::TilingParseContext *)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedQuantLightningIndexerManage)
    .Tiling(TilingFusedQuantLightningIndexerManage)
    .TilingParse<FusedQuantLightningIndexerManageCompileInfo>(TilingParseFusedQuantLightningIndexerManage);
} // namespace optiling
