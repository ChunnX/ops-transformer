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
 * \file matmul_all_reduce_v2_tiling.cc
 * \brief
 */

#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>
#include "matmul_all_reduce_v2_tiling_base.h"

using namespace ge;
using Ops::Transformer::OpTiling::TilingRegistryNew;
using Ops::Transformer::OpTiling::TilingRegistryArch;

namespace optiling {

ge::graphStatus MatmulAllReduceV2TilingFunc(gert::TilingContext* context);
ge::graphStatus TilingParseForMatmulAllReduceV2(gert::TilingParseContext* context);

ge::graphStatus MatmulAllReduceV2TilingFunc(gert::TilingContext* context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

struct MatmulAllReduceV2CompileInfo {
};

ge::graphStatus TilingParseForMatmulAllReduceV2(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulAllReduceV2)
    .Tiling(MatmulAllReduceV2TilingFunc)
    .TilingParse<MatmulAllReduceV2CompileInfo>(TilingParseForMatmulAllReduceV2);

} // namespace optiling
