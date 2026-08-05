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
ge::graphStatus FFNToAttentionV2TilingFunc(gert::TilingContext *context)
{
    auto ret = FFNToAttentionTilingFunc(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // Override tiling key from A3 to A5 after V1's tiling completes
    FFNToAttentionV2TilingData *tilingData = context->GetTilingData<FFNToAttentionV2TilingData>();
    bool rankTableMode = tilingData->ffnToAttentionV2Info.isInputRankTable;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(rankTableMode, TILINGKEY_TPL_A5);
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
