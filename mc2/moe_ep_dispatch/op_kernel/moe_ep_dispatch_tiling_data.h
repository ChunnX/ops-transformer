#ifndef MOE_EP_DISPATCH_TILING_DATA_H
#define MOE_EP_DISPATCH_TILING_DATA_H

#include "mc2_common_def.h"
#include "mc2_moe_context.h"
#include "mc2_init_tiling.h"
#include "mc2_cc_tiling.h"

struct MoeEpCommonTilingData {
    uint32_t epWorldSize        = 0;
    uint32_t epRankId           = 0;
    uint32_t numExperts         = 0;
    uint32_t numLocalExperts    = 0;
    uint32_t numTokens          = 0;
    uint32_t hidden             = 0;
    uint32_t topK               = 0;
    uint32_t numMaxTokensPerRank = 0;
    uint32_t perSlotBytes       = 0;
    uint32_t expertAlignment    = 1;
};

struct MoeEpDispatchInfo {
    MoeEpCommonTilingData cfg;
    uint32_t doCpuSync          = 1;
    uint32_t isCached           = 0;
    uint32_t networkMode        = 0;
    uint32_t numScaleupRanks    = 0;
    uint32_t numScaleoutRanks   = 0;
    uint32_t numAivStage1       = 0;
    uint32_t numAivStage2       = 0;
    uint32_t aivNum             = 0;
    uint64_t hostPinnedCounterAddr = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
};

struct MoeEpDispatchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeEpDispatchInfo moeEpDispatchInfo;
};

struct MoeEpDispatchA2Info {
    MoeEpCommonTilingData cfg;
    uint32_t doCpuSync          = 1;
    uint32_t isCached           = 0;
    uint32_t networkMode        = 0;
    uint32_t numScaleupRanks    = 0;
    uint32_t numScaleoutRanks   = 0;
    uint32_t numAivStage1       = 0;
    uint32_t numAivStage2       = 0;
    uint32_t aivNum             = 0;
    uint64_t hostPinnedCounterAddr = 0;
    uint64_t totalWinSizeEp     = 0;
    uint64_t totalUbSize        = 0;
    bool reserved1              = false;
    bool reserved2              = false;
};

struct MoeEpDispatchA2TilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MoeEpDispatchA2Info moeEpDispatchInfo;
};

#endif  // MOE_EP_DISPATCH_TILING_DATA_H