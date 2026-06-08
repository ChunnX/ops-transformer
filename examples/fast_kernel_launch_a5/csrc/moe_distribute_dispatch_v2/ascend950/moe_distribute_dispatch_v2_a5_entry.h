#ifndef MOE_DISPATCH_ENTRY_A5_H
#define MOE_DISPATCH_ENTRY_A5_H
#include "op_kernel/moe_distribute_dispatch_v2_info.h"
#include <cstdint>

void moe_distribute_dispatch_v2_a5_entry(
    int32_t tilingKey, uint32_t blockDim, void* stream,
    void* x, void* expertIds,
    void* scales, void* xActiveMask, void* expertScales, void* performanceInfo,
    void* expandXOut, void* dynamicScalesOut, void* assistInfoOut,
    void* expertTokenNumsOut, void* epSendCountsOut,
    void* expandScalesOut, void* workspaceGM, void* mc2Context,
    MoeDistributeDispatchV2Info tilingData);

#endif
