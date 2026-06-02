#ifndef DISPATCH_NORMAL_TILING_H
#define DISPATCH_NORMAL_TILING_H

struct DispatchNormalInfo {
    uint32_t epWorldSize;
    uint32_t epRankId;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t realMaxBs;
    uint32_t globalBs;
    uint32_t round;
    uint32_t perRoundTokens;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t aivNum;
    bool isQuant;
    bool isEnableDiagnose;
    bool isMc2Context;
    bool reserved3;
    uint64_t totalUbSize;
    uint64_t totalWinSize;
};

struct DispatchNormalTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling1;
    DispatchNormalInfo dispatchNormalInfo;
};

#endif
