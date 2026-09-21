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
 * \file ffn_to_attention_v2_tilling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <cstddef>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "../../op_kernel/ffn_to_attention_v2_tiling.h"
#include "../../../ffn_to_attention/op_kernel/ffn_to_attention_tiling.h"
#include "../../../ffn_to_attention/op_host/op_tiling/ffn_to_attention_tiling_base.h"
#include "../../op_kernel/ffn_to_attention_v2_tilling_key.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
// Base tiling writes the V1 prefix; V2 overwrites its two trailing slots after
// the base call and appends window address-table fields. Full V1/V2 sizes need not match.
#define CHECK_FFN_TILING_PREFIX(field) \
    static_assert(offsetof(FFNToAttentionV2Info, field) == offsetof(FFNToAttentionInfo, field), \
                  "FFNToAttention V1/V2 shared field offset mismatch: " #field)
CHECK_FFN_TILING_PREFIX(H);
CHECK_FFN_TILING_PREFIX(A);
CHECK_FFN_TILING_PREFIX(microBatchNum);
CHECK_FFN_TILING_PREFIX(BS);
CHECK_FFN_TILING_PREFIX(expertNumPerToken);
CHECK_FFN_TILING_PREFIX(HS);
CHECK_FFN_TILING_PREFIX(aivNum);
CHECK_FFN_TILING_PREFIX(worldSize);
CHECK_FFN_TILING_PREFIX(isInputRankTable);
CHECK_FFN_TILING_PREFIX(windowType);
#undef CHECK_FFN_TILING_PREFIX
static_assert(offsetof(FFNToAttentionV2Info, maxTokenNum) == offsetof(FFNToAttentionInfo, totalUbSize),
              "FFNToAttention base UB slot offset mismatch");
static_assert(offsetof(FFNToAttentionV2Info, cclBufferSize) == offsetof(FFNToAttentionInfo, totalWinSize),
              "FFNToAttention base window slot offset mismatch");
static_assert(offsetof(FFNToAttentionV2Info, addressTableWinOffset) >= sizeof(FFNToAttentionInfo),
              "FFNToAttention V2 extension overlaps base tiling");
static_assert(offsetof(FFNToAttentionV2TilingData, mc2InitTiling) ==
                      offsetof(FFNToAttentionTilingData, mc2InitTiling) &&
                  offsetof(FFNToAttentionV2TilingData, mc2CcTiling1) ==
                      offsetof(FFNToAttentionTilingData, mc2CcTiling1) &&
                  offsetof(FFNToAttentionV2TilingData, ffnToAttentionV2Info) ==
                      offsetof(FFNToAttentionTilingData, ffnToAttentionInfo),
              "FFNToAttention V1/V2 base tiling offsets mismatch");
static_assert(sizeof(FFNToAttentionV2TilingData) >= sizeof(FFNToAttentionTilingData),
              "FFNToAttention V2 tiling cannot contain base tiling");

constexpr size_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr size_t URMA_WORKSPACE_ALIGN = 32U;
constexpr size_t ADDRESS_ENTRY_SIZE = 2U * sizeof(uint64_t);
constexpr size_t ADDRESS_TABLE_ALIGN = 512U;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1U;
// Extra workspace headroom required by the URMA communication runtime.
constexpr size_t RESERVED_WORKSPACE_SIZE = 1024 * 1024 * 64LL;
// Mirrors of the kernel UB buffer constants in op_kernel/ffn_to_attention_urma.h; keep both sides in sync.
constexpr uint64_t TOKEN_DTYPE_BYTES = 2U;
constexpr uint64_t URMA_BUFFER_NUM = 2U;
constexpr uint64_t URMA_UB_ALIGN = 32U;
constexpr uint64_t URMA_HCOMM_INIT_SIZE = 512U;
constexpr uint64_t URMA_BATCH_BUFFER_SIZE = 256U * 64U;
constexpr uint64_t METADATA_CHUNK_BUFFER_SIZE = 256U * 4U * sizeof(int32_t);
constexpr uint64_t ADDRESS_TABLE_BUFFER_SIZE = 40U * 1024U;
constexpr uint64_t RANK_INFO_SEGMENT_NUM = 3U;

ge::graphStatus FFNToAttentionV2TilingFunc(gert::TilingContext *context)
{
    FFNToAttentionTilingConfig config;
    config.contextIndex = 0U;
    config.xIndex = 1U;
    config.sessionIdsIndex = 2U;
    config.microBatchIdsIndex = 3U;
    config.tokenIdsIndex = 4U;
    config.expertOffsetsIndex = 5U;
    config.actualTokenNumIndex = 6U;
    config.attnRankTableIndex = 7U;
    config.attrGroupIndex = 0U;
    config.attrWorldSizeIndex = 1U;
    config.attrTokenInfoTableShapeIndex = 2U;
    config.attrTokenDataShapeIndex = 3U;
    config.attrCclBufferSizeIndex = 4U;
    config.isMc2Context = true;

    auto ret = FFNToAttentionTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    OP_TILING_CHECK(context->SetScheduleMode(BATCH_MODE_SCHEDULE) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "failed to enable batch schedule mode"), return ge::GRAPH_FAILED);

    // Payloads are read directly from x. Flag sources and address tables occupy the local window tail.
    FFNToAttentionV2TilingData *tilingData = context->GetTilingData<FFNToAttentionV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData->ffnToAttentionV2Info.H > tilingData->ffnToAttentionV2Info.HS,
                    OP_LOGE(context->GetNodeName(), "token_data_shape HS must be greater than or equal to x H"),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *xShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "xShape"),
                    return ge::GRAPH_FAILED);
    const int64_t maxTokenNumDim = xShape->GetStorageShape().GetDim(0);
    OP_TILING_CHECK(maxTokenNumDim < 0 || maxTokenNumDim > std::numeric_limits<int32_t>::max(),
                    OP_LOGE(context->GetNodeName(), "x dim0 must fit non-negative int32 rank counts"),
                    return ge::GRAPH_FAILED);
    tilingData->ffnToAttentionV2Info.maxTokenNum = static_cast<uint64_t>(maxTokenNumDim);

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "workSpaces"),
                    return ge::GRAPH_FAILED);
    const size_t aivNum = static_cast<size_t>(tilingData->ffnToAttentionV2Info.aivNum);
    const size_t maxTokenNum = static_cast<size_t>(tilingData->ffnToAttentionV2Info.maxTokenNum);
    const size_t worldSize = static_cast<size_t>(tilingData->ffnToAttentionV2Info.worldSize);
    OP_TILING_CHECK(aivNum == 0U || worldSize == 0U,
                    OP_LOGE(context->GetNodeName(), "aivNum and worldSize must be greater than zero"),
                    return ge::GRAPH_FAILED);
    const size_t flagWindowSize = aivNum * URMA_FLAG_SLOT_SIZE;
    const auto &info = tilingData->ffnToAttentionV2Info;
    // Mirror the explicit InitBuffer allocations per AIV. The HCOMM buffers are allocated on
    // every AIV for the multi-channel relay, so the sum is each AIV's UB footprint.
    const uint64_t alignedPayloadBytes =
        (static_cast<uint64_t>(info.H) * TOKEN_DTYPE_BYTES + URMA_UB_ALIGN - 1U) / URMA_UB_ALIGN * URMA_UB_ALIGN;
    const uint64_t ubCountRowBytes =
        (static_cast<uint64_t>(worldSize) * sizeof(int32_t) + URMA_UB_ALIGN - 1U) / URMA_UB_ALIGN * URMA_UB_ALIGN;
    const uint64_t requiredUbBytes =
        URMA_BUFFER_NUM * alignedPayloadBytes + URMA_UB_ALIGN + METADATA_CHUNK_BUFFER_SIZE + ADDRESS_TABLE_BUFFER_SIZE +
        RANK_INFO_SEGMENT_NUM * ubCountRowBytes + URMA_HCOMM_INIT_SIZE + URMA_BATCH_BUFFER_SIZE;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t availableUbBytes = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, availableUbBytes);
    OP_TILING_CHECK(requiredUbBytes > availableUbBytes,
                    OP_LOGE(context->GetNodeName(), "URMA explicit buffers exceed available UB"),
                    return ge::GRAPH_FAILED);
    // Shape limits validated by the base tiling bound these products (MB=1, BS<=512, K<=17, HS<=8320).
    const uint64_t tokenSlotNum = static_cast<uint64_t>(info.microBatchNum) * info.BS * info.expertNumPerToken;
    const uint64_t tokenInfoBytes = (tokenSlotNum * sizeof(int32_t) + 511U) / 512U * 512U;
    const uint64_t tokenWindowEnd = tokenInfoBytes + tokenSlotNum * info.HS * TOKEN_DTYPE_BYTES;
    const uint64_t flagWindowOffset =
        (tokenWindowEnd + URMA_WORKSPACE_ALIGN - 1U) / URMA_WORKSPACE_ALIGN * URMA_WORKSPACE_ALIGN;
    const uint64_t rankTableStride =
        (static_cast<uint64_t>(maxTokenNum) * ADDRESS_ENTRY_SIZE + ADDRESS_TABLE_ALIGN - 1U) / ADDRESS_TABLE_ALIGN *
        ADDRESS_TABLE_ALIGN;
    const uint64_t countRowStride = (static_cast<uint64_t>(worldSize) * sizeof(int32_t) + ADDRESS_TABLE_ALIGN - 1U) /
                                    ADDRESS_TABLE_ALIGN * ADDRESS_TABLE_ALIGN;
    const uint64_t tableBytes = rankTableStride * worldSize + (static_cast<uint64_t>(aivNum) + 1U) * countRowStride;
    const uint64_t tableOffset =
        (flagWindowOffset + flagWindowSize + ADDRESS_TABLE_ALIGN - 1U) / ADDRESS_TABLE_ALIGN * ADDRESS_TABLE_ALIGN;
    const uint64_t requiredWindowSize = tableOffset + tableBytes;
    const auto cclBufferSize = context->GetAttrs()->GetAttrPointer<int64_t>(config.attrCclBufferSizeIndex);
    OP_TILING_CHECK(
        cclBufferSize == nullptr || *cclBufferSize <= 0 || static_cast<uint64_t>(*cclBufferSize) < requiredWindowSize,
        OP_LOGE(context->GetNodeName(), "ccl_buffer_size must include flag sources and address tables (%llu bytes)",
                static_cast<unsigned long long>(requiredWindowSize)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(workSpaces[0] > std::numeric_limits<size_t>::max() - RESERVED_WORKSPACE_SIZE,
                    OP_LOGE(context->GetNodeName(), "workspace size overflow"), return ge::GRAPH_FAILED);
    workSpaces[0] += RESERVED_WORKSPACE_SIZE;
    tilingData->ffnToAttentionV2Info.cclBufferSize = static_cast<uint64_t>(*cclBufferSize);
    tilingData->ffnToAttentionV2Info.addressTableWinOffset = tableOffset;
    tilingData->ffnToAttentionV2Info.addressTableBytes = tableBytes;

    // FFNToAttentionV2 on A5 uses the URMA communication implementation.
    bool rankTableMode = tilingData->ffnToAttentionV2Info.isInputRankTable;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A5, TILINGKEY_COMM_URMA);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "FFNToAttentionV2 cur case tilingKey is %lu", tilingKey);

    return ge::GRAPH_SUCCESS;
}

struct FFNToAttentionV2CompileInfo {};
ge::graphStatus TilingParseForFFNToAttentionV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FFNToAttentionV2)
    .Tiling(FFNToAttentionV2TilingFunc)
    .TilingParse<FFNToAttentionV2CompileInfo>(TilingParseForFFNToAttentionV2);
} // namespace MC2Tiling
