#ifndef MOE_COMBINE_ENTRY_A5_H
#define MOE_COMBINE_ENTRY_A5_H
#include "op_kernel/moe_distribute_combine_v2_info.h"
#include <cstdint>

void moe_distribute_combine_v2_a5_entry(
    int32_t tilingKey, uint32_t blockDim, void* stream,
    void* expandX, void* expertIds, void* assistInfo,
    void* epSendCounts, void* expertScales, void* tpSendCounts, void* xActiveMask,
    void* sharedExpertX, void* elasticInfo, void* oriX,
    void* constExpertAlpha1, void* constExpertAlpha2, void* constExpertV,
    void* performanceInfo, void* XOut, void* workspaceGM,
    void* mc2Context,
    MoeDistributeCombineV2Info tilingData);

#endif
