#ifndef MOE_EP_COMBINE_TILING_DATA_H
#define MOE_EP_COMBINE_TILING_DATA_H

#include "mc2_common_def.h"
#include "mc2_moe_context.h"
#include "mc2_init_tiling.h"
#include "mc2_cc_tiling.h"
#include "../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling_data.h"

struct MoeEpCombineInfo {
    MoeEpCommonTilingData cfg;
    uint32_t hasTopkWeights     = 0;
    uint32_t aivNum             = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
};

struct MoeEpCombineTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeEpCombineInfo moeEpCombineInfo;
};

struct MoeEpCombineA2Info {
    MoeEpCommonTilingData cfg;
    uint32_t hasTopkWeights     = 0;
    uint32_t aivNum             = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
    bool reserved1              = false;
};

struct MoeEpCombineA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeEpCombineA2Info moeEpCombineInfo;
};

#endif  // MOE_EP_COMBINE_TILING_DATA_H