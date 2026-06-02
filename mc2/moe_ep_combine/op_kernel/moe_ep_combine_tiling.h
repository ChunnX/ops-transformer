#ifndef MOE_EP_COMBINE_TILING_H
#define MOE_EP_COMBINE_TILING_H

#if __has_include("mc2_common_def.h")
#include "mc2_common_def.h"
#else
#include "../../common/op_kernel/mc2_common_def.h"
#endif
#if __has_include("mc2_moe_context.h")
#include "mc2_moe_context.h"
#else
#include "../../common/op_kernel/mc2_moe_context.h"
#endif
#if __has_include("../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h")
#include "../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"
#elif __has_include("../../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h")
#include "../../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"
#else
#include "moe_ep_dispatch_tiling.h"
#endif

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