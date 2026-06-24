#pragma once

#include <cstdint>

struct CausalConv1dDecodeCutBHTilingData {
    int64_t usedCoreNum;
    int64_t dimCoreCnt;
    int64_t batchCoreCnt;
    int64_t dimMainCoreCnt;
    int64_t dimTailCoreCnt;
    int64_t mainCoreDimLen;
    int64_t tailCoreDimLen;
    int64_t batchMainCoreCnt;
    int64_t batchTailCoreCnt;
    int64_t mainCoreBatchNum;
    int64_t tailCoreBatchNum;
    int64_t loopNumBS;
    int64_t loopNumDim;
    int64_t ubMainFactorBS;
    int64_t ubTailFactorBS;
    int64_t ubMainFactorDim;
    int64_t ubTailFactorDim;
    int64_t tailBlockLoopNumBS;
    int64_t tailBlockLoopNumDim;
    int64_t tailBlockUbFactorBS;
    int64_t tailBlockUbTailFactorBS;
    int64_t tailBlockUbFactorDim;
    int64_t tailBlockUbTailFactorDim;
    int64_t batchSize;
    int64_t seqLen;
    int64_t dim;
    int64_t kernelWidth;
    int64_t stateLen;
    int64_t xStride;
    int64_t cacheStride0;
    int64_t cacheStride1;
    int64_t padSlotId;
    int64_t xInputMode;
    int64_t hasBias;
    int64_t hasQueryStartLoc;
    int64_t hasCacheIndices;
    int64_t hasNumAcceptedTokens;
    int64_t dimRemainderElems;
    int64_t lastCoreLoopNumDim;
    int64_t lastCoreUbMainFactorDim;
    int64_t lastCoreUbTailFactorDim;
};
