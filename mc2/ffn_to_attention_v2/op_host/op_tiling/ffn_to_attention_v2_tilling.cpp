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
#include "ffn_to_attension_v2_tiling.h"
#include "../../op_kernel/ffn_to_attention_v2_tiling.h"
#include "../../op_kernel/ffn_to_attention_v2_tilling_key.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
constexpr size_t URMA_FLAG_SLOT_SIZE = 32U;
constexpr size_t URMA_FLAG_WORKSPACE_ALIGN = 32U;

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

    // Reserve one GM flag slot for every AIV. WriteNbi uses the slot as the source of the remote token flag.
    FFNToAttentionV2TilingData *tilingData = context->GetTilingData<FFNToAttentionV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(tilingData->ffnToAttentionV2Info.H > tilingData->ffnToAttentionV2Info.HS,
                    OP_LOGE(context->GetNodeName(), "token_data_shape HS must be greater than or equal to x H"),
                    return ge::GRAPH_FAILED);
    const gert::StorageShape *xShape = context->GetInputShape(config.xIndex);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "xShape"),
                    return ge::GRAPH_FAILED);
    tilingData->ffnToAttentionV2Info.maxTokenNum =
        static_cast<uint64_t>(xShape->GetStorageShape().GetDim(0));

    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "workSpaces"),
                    return ge::GRAPH_FAILED);
    const size_t flagWorkspaceSize =
        static_cast<size_t>(tilingData->ffnToAttentionV2Info.aivNum) * URMA_FLAG_SLOT_SIZE;
    OP_TILING_CHECK(workSpaces[0] > std::numeric_limits<size_t>::max() - (URMA_FLAG_WORKSPACE_ALIGN - 1U),
                    OP_LOGE(context->GetNodeName(), "workspace alignment overflow"), return ge::GRAPH_FAILED);
    const size_t flagWorkspaceOffset =
        (workSpaces[0] + URMA_FLAG_WORKSPACE_ALIGN - 1U) / URMA_FLAG_WORKSPACE_ALIGN * URMA_FLAG_WORKSPACE_ALIGN;
    OP_TILING_CHECK(flagWorkspaceOffset > std::numeric_limits<size_t>::max() - flagWorkspaceSize,
                    OP_LOGE(context->GetNodeName(), "workspace size overflow"), return ge::GRAPH_FAILED);
    tilingData->ffnToAttentionV2Info.urmaFlagWorkspaceOffset = static_cast<uint64_t>(flagWorkspaceOffset);
    workSpaces[0] = flagWorkspaceOffset + flagWorkspaceSize;

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
