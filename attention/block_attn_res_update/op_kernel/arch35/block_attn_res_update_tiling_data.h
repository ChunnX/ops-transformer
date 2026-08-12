/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_TILING_DATA_H
#define BLOCK_ATTN_RES_UPDATE_TILING_DATA_H

#include <cstdint>

enum class BlockAttnResUpdatePhase2Mode : uint8_t {
    TOKEN = 0,
    BATCH = 1,
};

#pragma pack(push, 8)
struct alignas(8) BlockAttnResUpdateTilingData {
    // All used cores except the last process tPerCore T indices.
    uint32_t tPerCore;
    // The last used core processes the remaining T indices.
    uint32_t lastTPerCore;
    uint32_t usedCoreNum;
    uint32_t dSize;
    // Aligned D strides in elements for the FP32 and BF16 [T, D] UB views.
    uint32_t dAlignFp32;
    uint32_t dAlignBf16;
    // Maximum T size processed by one UB tile.
    uint32_t tileT;
    // Element stride between FP32 stats planes, aligned from tileT according to phase2Mode.
    uint32_t statsTStride;
    // Number of FP32 VREG chunks required to cover D.
    uint32_t dVregNum;
    float eps;
    // Host-selected primary Phase 2 path. A short tail tile may still use TOKEN.
    uint8_t phase2Mode;
};
#pragma pack(pop)

#endif // BLOCK_ATTN_RES_UPDATE_TILING_DATA_H
