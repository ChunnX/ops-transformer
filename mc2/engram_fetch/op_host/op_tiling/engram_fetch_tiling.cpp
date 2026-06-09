/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetch tiling implementation
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "../../op_kernel/engram_fetch_tiling.h"

using namespace ge;
using namespace op;

namespace optiling {

static ge::graphStatus EngramFetchTilingFunc(gert::TilingContext *context)
{
    // Get tiling data
    EngramFetchTilingData *tilingData = context->GetTilingData<EngramFetchTilingData>();

    // Set basic parameters (skeleton version)
    tilingData->numTokens = 1;
    tilingData->hidden = 1;
    tilingData->placeholder = 0;

    // Set workspace size
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = 0; // Skeleton: no workspace needed

    // Set block dimension
    context->SetBlockDim(1);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetch).Tiling(EngramFetchTilingFunc);

} // namespace optiling