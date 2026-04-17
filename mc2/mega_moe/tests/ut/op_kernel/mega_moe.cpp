/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Stub version of mega_moe kernel for CPU UT testing.
 * All complex dependencies are stubbed out.
 */

#include "kernel_operator.h"
#include "mega_moe_tiling_def.h"

extern "C" __global__ __aicore__ void mega_moe(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR weight1, GM_ADDR weightScales1, GM_ADDR weight2, GM_ADDR weightScales2,
    GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingData, tilingData, tilingGM);
    // Stub implementation - no actual computation
}