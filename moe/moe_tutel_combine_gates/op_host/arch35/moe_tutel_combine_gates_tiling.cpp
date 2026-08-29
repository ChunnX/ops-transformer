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
 * \file moe_tutel_combine_gates_tiling.cpp
 * \brief MoeTutelCombineGates Ascend 950 tiling registration.
 */

#include "moe_tutel_combine_gates_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {

static ge::graphStatus Tiling4MoeTutelCombineGates(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_CHECK_IF(!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context),
                OP_LOGE(context->GetNodeName(), "MoeTutelCombineGates is only supported on Ascend 950"),
                return ge::GRAPH_FAILED);
    MoeTutelCombineGatesTiling tilingObject(context);
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepare4MoeTutelCombineGates(gert::TilingParseContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto compileInfo = context->GetCompiledInfo<MoeTutelCombineGatesCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum == 0, OP_LOGE(context, "get AIV core num failed"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->ubSize == 0, OP_LOGE(context, "get UB size failed"), return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = static_cast<uint32_t>(Ops::Base::GetUbBlockSize(context));
    OP_CHECK_IF(compileInfo->ubBlockSize == 0, OP_LOGE(context, "get UB block size failed"), return ge::GRAPH_FAILED);
    compileInfo->vectorLength = static_cast<uint32_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF(compileInfo->vectorLength == 0, OP_LOGE(context, "get vector length failed"), return ge::GRAPH_FAILED);
    compileInfo->workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTutelCombineGates)
    .Tiling(Tiling4MoeTutelCombineGates)
    .TilingParse<MoeTutelCombineGatesCompileInfo>(TilingPrepare4MoeTutelCombineGates);

} // namespace optiling
