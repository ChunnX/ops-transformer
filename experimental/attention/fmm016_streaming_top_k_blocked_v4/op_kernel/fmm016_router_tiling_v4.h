/**
 * Copyright (c) 2026 fk2323
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fmm016_router_tiling_v4.h
 * \brief
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
