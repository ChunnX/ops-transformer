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
 * \brief
 */

#include "moe_tutel_combine_gates_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {

static ge::graphStatus Tiling4MoeTutelCombineGates(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    // 本仓只交付 Ascend950 / arch35；其它芯片由 canndev 侧已上库实现承载（★A5 不碰 A2★）。
    OP_CHECK_IF(!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context),
                OP_LOGE(context->GetNodeName(), "MoeTutelCombineGates is only supported on Ascend950 in this repo"),
                return ge::GRAPH_FAILED);
    MoeTutelCombineGatesTiling tilingObject(context);
    return tilingObject.DoTiling();
}

static ge::graphStatus TilingPrepare4MoeTutelCombineGates(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4MoeTutelCombineGates enter.");
    auto compileInfo = context->GetCompiledInfo<MoeTutelCombineGatesCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum == 0, OP_LOGE(context, "get core num failed"), return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSize = ubSizePlatForm;
    OP_CHECK_IF(compileInfo->ubSize == 0, OP_LOGE(context, "get ub size failed"), return ge::GRAPH_FAILED);

    compileInfo->ubBlockSize = static_cast<uint32_t>(Ops::Base::GetUbBlockSize(context));
    OP_CHECK_IF(compileInfo->ubBlockSize == 0, OP_LOGE(context, "get ub block size failed"), return ge::GRAPH_FAILED);
    compileInfo->vectorLength = static_cast<uint32_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF(compileInfo->vectorLength == 0, OP_LOGE(context, "get vector length failed"),
                return ge::GRAPH_FAILED);

    OP_LOGD(context, "TilingPrepare4MoeTutelCombineGates exit.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTutelCombineGates)
    .Tiling(Tiling4MoeTutelCombineGates)
    .TilingParse<MoeTutelCombineGatesCompileInfo>(TilingPrepare4MoeTutelCombineGates);

} // namespace optiling
