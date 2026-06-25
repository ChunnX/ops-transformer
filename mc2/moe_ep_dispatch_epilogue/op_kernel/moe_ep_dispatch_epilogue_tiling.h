#ifndef MOE_EP_DISPATCH_EPILOGUE_TILING_H
#define MOE_EP_DISPATCH_EPILOGUE_TILING_H

#if __has_include("../moe_ep_dispatch/moe_ep_dispatch_tiling.h")
#include "../moe_ep_dispatch/moe_ep_dispatch_tiling.h"
#else
#include "../../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"
#endif

struct MoeEpDispatchEpilogueInfo {
    MoeEpCommonTilingData cfg;
    uint32_t aivNum             = 0;
    uint64_t totalUbSize        = 0;
    uint32_t cached             = 0;   // 0 = non-cached path, 1 = cached path
    uint32_t hasScales          = 0;   // 0 = no recvScales output, 1 = copy scales for fp8 path
    uint32_t scaleBytes         = 0;   // bytes per expanded row in recvScales
};

struct MoeEpDispatchEpilogueTilingData {
    MoeEpDispatchEpilogueInfo moeEpDispatchEpilogueInfo;
};

#endif  // MOE_EP_DISPATCH_EPILOGUE_TILING_H