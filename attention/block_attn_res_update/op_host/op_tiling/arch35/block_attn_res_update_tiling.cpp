/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attn_res_update_tiling.h"

#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr const char *OP_NAME = "BlockAttnResUpdate";
} // namespace

ge::graphStatus Tiling4BlockAttnResUpdate(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "context is nullptr."), return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistryArch::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4BlockAttnResUpdate(gert::TilingParseContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(OP_NAME, "context is nullptr."), return ge::GRAPH_FAILED);
    auto *compileInfo = context->GetCompiledInfo<BlockAttnResUpdateCompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(OP_NAME, "compileInfo is nullptr."), return ge::GRAPH_FAILED);
    auto *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(OP_NAME, "platformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->coreNum == 0 || compileInfo->ubSize == 0,
                OP_LOGE(OP_NAME, "Invalid platform data: coreNum=%u, ubSize=%lu", compileInfo->coreNum,
                        compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockAttnResUpdate)
    .Tiling(Tiling4BlockAttnResUpdate)
    .TilingParse<BlockAttnResUpdateCompileInfo>(TilingPrepare4BlockAttnResUpdate);

} // namespace optiling
