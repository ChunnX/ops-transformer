#ifndef MOE_EP_DISPATCH_EPILOGUE_TILING_H
#define MOE_EP_DISPATCH_EPILOGUE_TILING_H

#include "../../common/op_kernel/mc2_common_def.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../moe_ep_dispatch/op_kernel/moe_ep_dispatch_tiling.h"

struct MoeEpDispatchEpilogueInfo {
    MoeEpCommonTilingData cfg;
    uint32_t aivNum             = 0;
    uint64_t totalUbSize        = 0;
};

struct MoeEpDispatchEpilogueTilingData {
    MoeEpDispatchEpilogueInfo moeEpDispatchEpilogueInfo;
};

#endif  // MOE_EP_DISPATCH_EPILOGUE_TILING_H