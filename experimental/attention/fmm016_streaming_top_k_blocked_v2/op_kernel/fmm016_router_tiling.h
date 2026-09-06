/**
 * Copyright (c) 2026 fk2323
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the repository.
 */

#ifndef FMM016_ROUTER_TILING_H
#define FMM016_ROUTER_TILING_H

#include <cstdint>

struct Fmm016StreamingTopKTilingData {
    uint32_t bhCount;
    uint32_t leaves;
    uint32_t indexHeads;
    uint32_t indexDim;
    uint32_t topK;
    uint32_t totalRows;
    float invSqrtIndexDim;
};

struct Fmm016StreamingTopKBlockedTilingData {
    uint32_t bhCount;
    uint32_t leaves;
    uint32_t paddedLeaves;
    uint32_t indexHeads;
    uint32_t indexDim;
    uint32_t topK;
    uint32_t totalRows;
    float invSqrtIndexDim;
};

struct Fmm016CausalSpanTopKTilingData {
    uint32_t batch;
    uint32_t heads;
    uint32_t leaves;
    uint32_t paddedLeaves;
    uint32_t indexHeads;
    uint32_t indexDim;
    uint32_t kvGroupSize;
    uint32_t kvGroups;
    uint32_t totalRows;
    float invSqrtIndexDim;
};

#endif
