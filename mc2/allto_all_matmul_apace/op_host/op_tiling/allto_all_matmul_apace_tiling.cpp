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
 * \file allto_all_matmul_apace_tiling.cpp
 * \brief Tiling implementation for AlltoAllMatmulApace
 */

#include "arch35/allto_all_matmul_apace_tiling_base.h"

#include <cstdio>
#include <string>
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>
#include "graph/types.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;

namespace MC2Tiling {

using Ops::Transformer::OpTiling::TilingRegistryArch;
using Ops::Transformer::OpTiling::TilingRegistryNew;

static ge::graphStatus AlltoAllMatmulApaceTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmulApace(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulApaceCompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmulApace)
    .Tiling(AlltoAllMatmulApaceTilingFunc)
    .TilingParse<AlltoAllMatmulApaceCompileInfo>(TilingParseForAlltoAllMatmulApace);

} // namespace MC2Tiling
