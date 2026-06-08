#ifndef MOE_COMBINE_INFO_H
#define MOE_COMBINE_INFO_H
#include <cstdint>

struct MoeDistributeCombineV2Info {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t zeroExpertNum;
    uint32_t copyExpertNum;
    uint32_t constExpertNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;
    bool isTokenMask;
    bool isExpertMask;
    bool hasSharedExpertX;
    bool hasElasticInfo;
    bool isPerformance;
    bool isMc2Context;
    bool reserved1;
    bool reserved2;
    uint64_t totalUbSize;
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
    float armAvgFactor;
    float epsilon;
    uint32_t bufferNum;
};
#endif
