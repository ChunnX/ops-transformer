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
 * \file fused_causal_conv1d_tiling.cpp
 * \brief Main tiling entry for FusedCausalConv1d operator (unified, dispatches by runMode)
 */

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "fused_causal_conv1d_cut_bh_tiling_arch35.h"
#include "fused_causal_conv1d_cut_bsh_tiling_arch35.h"

namespace optiling {
// Run mode constants
constexpr int64_t RUN_MODE_BSH = 0;
constexpr int64_t RUN_MODE_BH = 1;

struct FusedCausalConv1dCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// Main tiling function that dispatches based on runMode attribute
static ge::graphStatus TilingFusedCausalConv1d(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "FusedCausalConv1dTiling tiling start");

    // Get runMode attribute to determine which tiling implementation to use
    int64_t runMode = 0;  // Default to BSH mode
    if (context->GetAttrs() != nullptr && context->GetAttrs()->GetInt(ATTR_RUN_MODE_INDEX) != nullptr) {
        runMode = *(context->GetAttrs()->GetInt(ATTR_RUN_MODE_INDEX));
    }

    OP_LOGD(context->GetNodeName(), "FusedCausalConv1d runMode=%ld (0=BSH, 1=BH)", runMode);

    // Dispatch to the appropriate tiling implementation based on runMode
    ge::graphStatus status = ge::GRAPH_FAILED;

    if (runMode == RUN_MODE_BH) {
        // Use BH tiling logic
        FusedCausalConv1dCutBHTiling tilingImpl(context);
        status = tilingImpl.DoTiling();
    } else if (runMode == RUN_MODE_BSH){
        // Use BSH tiling logic
        FusedCausalConv1dCutBSHTiling tilingImpl(context);
        status = tilingImpl.DoTiling();
    }

    OP_LOGD(context->GetNodeName(), "FusedCausalConv1dTiling tiling end, status=%d", status);
    return status;
}

static ge::graphStatus TilingPrepareFusedCausalConv1d(gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr,
                OP_LOGE("FusedCausalConv1d", "context is null"),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE(context->GetNodeName(), "platformInfo is null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Register the main tiling entry point for unified FusedCausalConv1d operator
IMPL_OP_OPTILING(FusedCausalConv1d)
    .Tiling(TilingFusedCausalConv1d)
    .TilingParse<FusedCausalConv1dCompileInfo>(TilingPrepareFusedCausalConv1d);
} // namespace optiling
