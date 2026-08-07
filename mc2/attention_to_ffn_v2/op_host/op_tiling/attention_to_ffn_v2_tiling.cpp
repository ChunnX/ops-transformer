/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_log.h"
#include "register/op_def_registry.h"

#include "attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling_key.h"

using namespace Mc2Tiling;

namespace MC2Tiling {
constexpr uint32_t SYNC_MODE = 1U;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2U;

ge::graphStatus AttentionToFFnV2TilingFunc(gert::TilingContext *context)
{
    AttentionToFFnTilingConfig config;
    config.contextIndex = 0U;
    config.xIndex = 1U;
    config.sessionIdIndex = 2U;
    config.microBatchIdIndex = 3U;
    config.layerIdIndex = 4U;
    config.expertIdsIndex = 5U;
    config.expertRankTableIndex = 6U;
    config.scalesIndex = 7U;
    config.activeMaskIndex = 8U;
    config.attrGroupIndex = 0U;
    config.attrWorldSizeIndex = 1U;
    config.attrFfnTokenInfoTableShapeIndex = 2U;
    config.attrFfnTokenDataShapeIndex = 3U;
    config.attrAttnTokenInfoTableShapeIndex = 4U;
    config.attrMoeExpertNumIndex = 5U;
    config.attrQuantModeIndex = 6U;
    config.attrSyncFlagIndex = 7U;
    config.attrFfnStartRankIdIndex = 8U;
    config.isMc2Context = true;

    auto ret = AttentionToFFnTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    AttentionToFFnV2TilingData *tilingData = context->GetTilingData<AttentionToFFnV2TilingData>();
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                    return ge::GRAPH_FAILED);
    const bool isQuant = tilingData->attentionToFFnV2Info.quantMode == DYNAMIC_QUANT_MODE;
    const bool isSync = tilingData->attentionToFFnV2Info.syncFlag == SYNC_MODE;
    const bool isActiveMask = tilingData->attentionToFFnV2Info.isActiveMask;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(isQuant, isSync, isActiveMask, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "AttentionToFFnV2 cur case tilingKey is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

struct AttentionToFFnV2CompileInfo {};

ge::graphStatus TilingParseForAttentionToFFnV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFFnV2)
    .Tiling(AttentionToFFnV2TilingFunc)
    .TilingParse<AttentionToFFnV2CompileInfo>(TilingParseForAttentionToFFnV2);
} // namespace MC2Tiling
