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
 * \file moe_token_permute_tiling_base.cpp
 * \brief tiling 胶水层：按 TilingRegistry 分流到 base(910b) 或 arch35(ascend950)
 */
#include "moe_token_permute_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

using Ops::Transformer::OpTiling::TilingRegistry;

namespace optiling {
static ge::graphStatus TilingForMoeTokenPermute(gert::TilingContext *context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeTokenPermute(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepareForMoeTokenPermute enter.");

    auto compileInfo = context->GetCompiledInfo<MoeTokenPermuteCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->aivNum <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "compileInfo->aivNum",
                                                      std::to_string(compileInfo->aivNum), "failed to get core num"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = ubSize;
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
                                                      std::to_string(compileInfo->ubSize), "failed to get ub size"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTokenPermute)
    .Tiling(TilingForMoeTokenPermute)
    .TilingParse<MoeTokenPermuteCompileInfo>(TilingPrepareForMoeTokenPermute);
} // namespace optiling
