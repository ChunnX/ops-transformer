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
 * \file moe_tutel_combine_x_tiling_base.cpp
 * \brief Entry of the MoeTutelCombineX host tiling: dispatch through the
 *        TilingRegistry and register the IMPL_OP_OPTILING interface.
 */
#include "moe_tutel_combine_x_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

namespace optiling {

ge::graphStatus Tiling4MoeTutelCombineX(gert::TilingContext *context)
{
    OP_LOGD(context, "Tiling4MoeTutelCombineX running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4MoeTutelCombineX(gert::TilingParseContext *context)
{
    OP_LOGD(context, "TilingPrepare4MoeTutelCombineX running.");
    auto compileInfo = context->GetCompiledInfo<MoeTutelCombineXCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->numBlocks = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeTutelCombineX)
    .Tiling(Tiling4MoeTutelCombineX)
    .TilingInputsDataDependency({2, 3})
    .TilingParse<MoeTutelCombineXCompileInfo>(TilingPrepare4MoeTutelCombineX);
} // namespace optiling
