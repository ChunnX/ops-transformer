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
 * \file moe_fused_topk_tiling_base.cpp
 * \brief Tiling registry entry for MoeFusedTopk.
 */

#include "moe_fused_topk_tiling.h"
#include "register/op_def_registry.h"
#include "err/ops_err.h"
#include "log/log.h"
#include "platform/platform_info.h"

using Ops::Transformer::OpTiling::TilingRegistry;

namespace optiling {
static ge::graphStatus TilingForMoeFusedTopk(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepare4MoeFusedTopk(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<MoeFusedTopkCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->ubSize = ubSize;

    OP_CHECK_IF(compileInfo->coreNum <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "MoeFusedTopk GetHardwareInfo Failed, vectorCoreNum: %u",
                                            compileInfo->coreNum),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compileInfo->ubSize <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(),
                                            "MoeFusedTopk GetHardwareInfo Failed, ubSize: %lu", compileInfo->ubSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeFusedTopk)
    .Tiling(TilingForMoeFusedTopk)
    .TilingParse<MoeFusedTopkCompileInfo>(TilingPrepare4MoeFusedTopk);
} // namespace optiling
