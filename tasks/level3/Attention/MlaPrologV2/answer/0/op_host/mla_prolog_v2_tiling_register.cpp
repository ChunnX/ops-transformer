/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 * \file mla_prolog_v2_tiling_register.cpp
 */
#include "mla_prolog_v2_tiling.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingPrepareForMlaPrologV2(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MlaPrologV2)
    .Tiling(TilingMlaProlog)
    .TilingParse<MlaPrologCompileInfo>(TilingPrepareForMlaPrologV2);
} // namespace optiling
