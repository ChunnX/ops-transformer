/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetchWait tiling implementation
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "../../op_kernel/engram_fetch_tiling.h"

using namespace ge;
using namespace op;

namespace optiling {

static ge::graphStatus EngramFetchWaitTilingFunc(gert::TilingContext *context)
{
    // Get tiling data
    EngramFetchWaitTilingData *tilingData = context->GetTilingData<EngramFetchWaitTilingData>();

    // Set basic parameters (skeleton version)
    tilingData->placeholder = 0;

    // Set workspace size
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = 0;

    // Set block dimension
    context->SetBlockDim(1);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(EngramFetchWait).Tiling(EngramFetchWaitTilingFunc);

} // namespace optiling