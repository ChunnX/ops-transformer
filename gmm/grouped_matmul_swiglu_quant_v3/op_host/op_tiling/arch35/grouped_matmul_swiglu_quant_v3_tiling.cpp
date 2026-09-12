/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Ascend 950 MXFP8 weight-NZ tiling entry reserved for V3.
 * The concrete tiling calculation is intentionally deferred until a 950
 * build environment is available.
 */

#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
namespace {
constexpr size_t RESERVED_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

ge::graphStatus GroupedMatmulSwigluQuantV3TilingFunc(gert::TilingContext *context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = RESERVED_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_OPTILING(GroupedMatmulSwigluQuantV3)
    .Tiling(GroupedMatmulSwigluQuantV3TilingFunc);
} // namespace optiling
