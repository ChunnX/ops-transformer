/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "arch35/mega_mhc_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"

namespace optiling {
struct MegaMhcCompileInfo {};

/** No numerical transform: resource capacities are read from TilingContext at launch time. */
static ge::graphStatus ParseMegaMhc(gert::TilingParseContext *context)
{
    return context == nullptr ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
}

/** Dispatch the graph op to the 950 planner; no 910 fallback exists. */
static ge::graphStatus TilingMegaMhc(gert::TilingContext *context)
{
    if (context == nullptr || !Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        return ge::GRAPH_FAILED;
    }
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}
IMPL_OP_OPTILING(MegaMhc).Tiling(TilingMegaMhc).TilingParse<MegaMhcCompileInfo>(ParseMegaMhc);
} // namespace optiling
