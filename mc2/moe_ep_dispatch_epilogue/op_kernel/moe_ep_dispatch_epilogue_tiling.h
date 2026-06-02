#ifndef MOE_EP_DISPATCH_EPILOGUE_TILING_H
#define MOE_EP_DISPATCH_EPILOGUE_TILING_H

#include "mc2_common_def.h"
#include "mc2_moe_context.h"
#include "../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"

struct MoeEpDispatchEpilogueInfo {
    MoeEpCommonTilingData cfg;
    uint32_t aAlloc             = 0;
    uint32_t numLocalExperts    = 0;
    uint64_t windowBaseAddr     = 0;
    uint64_t numRecvPerRankAddr = 0;
    uint64_t numRecvPerExpertAddr = 0;
    uint32_t aivNum             = 0;
    uint64_t totalUbSize        = 0;
};

struct MoeEpDispatchEpilogueTilingData {
    MoeEpDispatchEpilogueInfo moeEpDispatchEpilogueInfo;
};



#endif  // MOE_EP_DISPATCH_EPILOGUE_TILING_H