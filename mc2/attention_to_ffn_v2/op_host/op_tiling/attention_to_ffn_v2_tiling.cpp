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
 * \file attention_to_ffn_v2_tiling.cpp
 * \brief
 */

#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling.h"
#include "../../op_kernel/attention_to_ffn_v2_tiling_key.h"

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;

namespace MC2Tiling {
ge::graphStatus AttentionToFfnV2TilingFunc(gert::TilingContext *context)
{
    AttentionToFFNTilingConfig config;
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

    auto ret = AttentionToFFNTilingFuncBase(context, config);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Override tiling key from A3 to A5 after V1's tiling completes
    AttentionToFfnV2TilingData *tilingData = context->GetTilingData<AttentionToFfnV2TilingData>();
    uint32_t syncFlag = tilingData->attentionToFfnV2Info.syncFlag;
    uint32_t quantMode = tilingData->attentionToFfnV2Info.quantMode;
    bool activeMaskFlag = tilingData->attentionToFfnV2Info.isActiveMask;

    bool isSync = (syncFlag == 1U);
    bool isQuant = (quantMode == 2U); // DYNAMIC_QUANT_MODE
    const uint64_t tilingKey = GET_TPL_TILING_KEY(isQuant, isSync, activeMaskFlag, TILINGKEY_TPL_A5);
    context->SetTilingKey(tilingKey);
    OP_LOGD(context->GetNodeName(), "AttentionToFfnV2 cur case tilingKey is %lu", tilingKey);

    return ge::GRAPH_SUCCESS;
}

struct AttentionToFfnV2CompileInfo {};
ge::graphStatus TilingParseForAttentionToFfnV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFfnV2)
    .Tiling(AttentionToFfnV2TilingFunc)
    .TilingParse<AttentionToFfnV2CompileInfo>(TilingParseForAttentionToFfnV2);
} // namespace MC2Tiling
