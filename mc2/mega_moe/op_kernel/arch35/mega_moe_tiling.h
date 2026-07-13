/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_tiling.h
 * \brief
 */

#ifndef ASCENDC_MEGA_MOE_TILING
#define ASCENDC_MEGA_MOE_TILING

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

// Keep phase-1 batches bounded so dispatch cannot run too far ahead of the
// GMM2 AIV1 consumer. The mode-4 UB release itself is acknowledged every batch.
constexpr uint32_t MEGA_MOE_MAX_EXPERTS_PER_BATCH = 14U;
// Flag regions scale with maxOutputSize. Clear them in a bounded UB window
// so extreme legal shapes do not turn the reset scratch into a full copy.
constexpr uint32_t MEGA_MOE_RESET_CHUNK_INT32 = 2048U;

struct MegaMoeTilingData {
    uint32_t expertPerRank;
    uint32_t bs;
    uint32_t h;
    uint32_t hiddenDim;
    uint32_t epWorldSize;
    uint32_t blockNumPerEP;
    uint32_t maxOutputSize;
    uint32_t topK;
    uint32_t aicNum;
    uint32_t blockAivNum;
    int64_t combineQuantMode;
    float clampLimit;
    uint8_t groupedMatmulMode;
    int64_t topoType;
    // Keep scheduling extensions at the tail to preserve all existing field offsets.
    uint32_t expertsPerBatch;
};
#endif
