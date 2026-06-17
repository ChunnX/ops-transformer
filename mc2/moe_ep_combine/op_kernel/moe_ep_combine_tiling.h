#ifndef MOE_EP_COMBINE_TILING_H
#define MOE_EP_COMBINE_TILING_H

#include "../../common/op_kernel/mc2_common_def.h"
#include "../../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"

struct MoeEpCombineInfo {
    MoeEpCommonTilingData cfg;
    uint32_t hasTopkWeights     = 0;
    uint32_t aivNum             = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
    
};

struct MoeEpCombineTilingData {
    MoeEpCombineInfo moeEpCombineInfo;
};



#endif  // MOE_EP_COMBINE_TILING_H