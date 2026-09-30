#include "../host_runtime/dynamic_tiling_planner.h"

IMPL_OP_OPTILING(SageAttention2).Tiling(optiling::ComputeSageAttention2Tiling)
    .TilingParse<optiling::PlatformHardwareSpec>(optiling::ParseHardwareSpec);
