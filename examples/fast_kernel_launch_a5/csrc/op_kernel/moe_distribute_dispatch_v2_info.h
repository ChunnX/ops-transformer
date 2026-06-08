#ifndef MOE_DISPATCH_INFO_H
#define MOE_DISPATCH_INFO_H
#include <cstdint>

struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;
    bool isTokenMask;
    bool isExpertMask;
    bool hasElasticInfo;
    bool isPerformance;
    bool isQuant;
    bool isMc2Context;
    bool reserved1;
    bool reserved2;
    uint64_t totalUbSize;
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
    uint32_t expertTokenNumsType;
    int32_t zeroComputeExpertNum;
    uint32_t maxSizeForUbBuffer;
    uint64_t scalesRow;
    uint64_t scalesCol;
    uint32_t scalesTypeSize;
    uint64_t scalesCount;
};
#endif
