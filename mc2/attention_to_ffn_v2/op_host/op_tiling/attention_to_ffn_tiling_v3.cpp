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
 * \file attention_to_ffn_tiling_v3.cpp
 * \brief Tiling entry for AttentionToFFNV3.
 */

#include "attention_to_ffn_tiling_base.h"
#include "mc2_log.h"

namespace MC2Tiling {

ge::graphStatus AttentionToFFNV3TilingFunc(gert::TilingContext *context)
{
    AttentionToFFNTilingConfig config{};

    // Input indexes. Keep them strictly aligned with AttentionToFFNV3's OpDef/proto.
    config.contextIndex = 0;
    config.xIndex = 1;
    config.sessionIdIndex = 2;
    config.microBatchIdIndex = 3;
    config.layerIdIndex = 4;
    config.expertIdsIndex = 5;
    config.expertRankTableIndex = 6;
    config.scalesIndex = 7;
    config.activeMaskIndex = 8;

    // Attribute indexes. The required attributes are followed by the optional attributes.
    config.attrWorldSizeIndex = 0;
    config.attrCclBufferSizeIndex = 1;
    config.attrFfnTokenInfoTableShapeIndex = 2;
    config.attrFfnTokenDataShapeIndex = 3;
    config.attrAttnTokenInfoTableShapeIndex = 4;
    config.attrMoeExpertNumIndex = 5;
    config.attrQuantModeIndex = 6;
    config.attrSyncFlagIndex = 7;
    config.attrFfnStartRankIdIndex = 8;

    config.isExplicitContext = true;

    const char *nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "Enter AttentionToFFNV3 tiling.");
    return AttentionToFFNTilingV3FuncBase(context, config);
}

struct AttentionToFFNV3CompileInfo {};

ge::graphStatus TilingParseForAttentionToFFNV3(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AttentionToFFNV3)
    .Tiling(AttentionToFFNV3TilingFunc)
    .TilingParse<AttentionToFFNV3CompileInfo>(TilingParseForAttentionToFFNV3);

} // namespace MC2Tiling
