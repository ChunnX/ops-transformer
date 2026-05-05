/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* kernel-side mirror of host MegaMoeV2TilingData; v1 fields + wave/ring/arrival params (DESIGN §5.1) */

#ifndef ASCENDC_MEGA_MOE_V2_TILING
#define ASCENDC_MEGA_MOE_V2_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "moe_init_routing_v3/arch35/moe_init_routing_v3_arch35_tiling_def.h"

using namespace AscendC;

struct MegaMoeV2TilingData {
    MoeInitRoutingV3Arch35TilingData moeInitRoutingTilingData;

    // ---- v1 carryover ----
    uint32_t expertPerRank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t epWorldSize;
    uint32_t maxOutputSize;        // dispatch ring upper bound (no longer ptrA size)
    uint32_t rankId;
    uint32_t topK;
    int32_t  dispatchRows;         // kept for CopyGMToGMPerTokenV2 inner pingpong tile
    uint8_t  groupListType;
    bool     transX;
    bool     transW;
    bool     transW2;

    // ---- V2 wave / ring / arrival ----
    uint32_t maxTokensPerExpert;   // upper bound recv per local expert across all ranks (for ring sizing)
    uint32_t kExpertsPerWave;      // experts handled per wave
    uint32_t kNumWaves;            // = ceilDiv(expertPerRank, kExpertsPerWave)
    uint32_t chunkDepth;           // ring buffer depth (slots concurrently alive)
    uint32_t chunkM;               // rows per ring slot (epilogue singleM_ aligned, =128)
    uint32_t numChunksPerWave;     // ceilDiv(maxTokensPerExpert*kExpertsPerWave, chunkM)
    uint32_t dispatchArrivalLen;   // = chunkDepth*kNumWaves*numChunksPerWave (uint32 GM counters)
    uint32_t l2ArrivalMaskLen;     // same length, GM uint64 bitmask array
    uint32_t gmm2ToCombineLen;     // same length, GM uint32 counters
    uint32_t commVecCount;         // = 32 (subBlockIdx_=1)
    uint32_t epiVecCount;          // = 32 (subBlockIdx_=0)

    // ---- V2 dispatch ----
    uint32_t dispatchVecPerRank;   // = commVecCount / epWorldSize (replaces v1 blockNumPerEP)
    uint32_t dispatchRingTokens;   // = chunkDepth * chunkM * kExpertsPerWave
};
#endif
