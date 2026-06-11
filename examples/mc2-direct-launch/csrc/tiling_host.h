#ifndef MC2_TILING_HOST_H
#define MC2_TILING_HOST_H

#include <cstdint>

// 与 kernel 侧 struct 逐字段一致，不依赖 SDK 类型

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

void mc2_dispatch_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2Context, void *x, void *expertIds,
    void *scales, void *xActiveMask, void *expertScales,
    void *elasticInfo, void *performanceInfo,
    void *expandXOut, void *dynamicScalesOut,
    void *assistInfoOut, void *expertTokenNumsOut,
    void *epSendCountsOut, void *tpSendCountsOut,
    void *expandScalesOut, void *workspaceGM,
    MoeDistributeDispatchV2Info tilingInfo);

void mc2_combine_launch(int32_t tilingKey, uint32_t blockDim, void *stream,
    void *mc2Context, void *expandX, void *expertIds,
    void *assistInfo, void *epSendCount, void *tpSendCount,
    void *residualX, void *gamma, void *expertScales,
    void *xActiveMask, void *sharedExpertX,
    void *elasticInfo, void *oriX,
    void *constExpertAlpha1, void *constExpertAlpha2, void *constExpertV,
    void *performanceInfo, void *yOut, void *rstdOut, void *XOut,
    void *workspaceGM, MoeDistributeCombineV2Info tilingInfo);

#endif
