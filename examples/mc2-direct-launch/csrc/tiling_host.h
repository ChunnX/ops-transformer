/**
 * @file tiling_host.h
 * @brief Host-compatible tiling POD structs
 * Layout MUST match the kernel-side definitions in moe_distribute_dispatch_v2_tiling.h
 * and moe_distribute_combine_v2_tiling.h exactly.
 */

#ifndef MC2_TILING_HOST_H
#define MC2_TILING_HOST_H

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
    uint32_t expertTokenNumsType;
    int32_t  zeroComputeExpertNum;
    uint32_t maxSizeForUbBuffer;
    uint64_t scalesRow;
    uint64_t scalesCol;
    uint32_t scalesTypeSize;
    uint64_t scalesCount;
};

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
    float  armAvgFactor;
    float  epsilon;
    uint32_t bufferNum;
};

#endif // MC2_TILING_HOST_H
