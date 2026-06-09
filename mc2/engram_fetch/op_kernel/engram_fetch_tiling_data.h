#ifndef ASCENDC_ENGRAM_FETCH_TILING_H
#define ASCENDC_ENGRAM_FETCH_TILING_H

#include "kernel_tiling/kernel_tiling.h"

struct EngramFetchInfo {
    uint32_t numTokens;
    uint32_t numEntriesPerRank;
    uint32_t hiddenDim;
    uint32_t hiddenBytes;
    uint32_t aivNum;
};

struct EngramFetchTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    EngramFetchInfo engramFetchInfo;
};
#endif