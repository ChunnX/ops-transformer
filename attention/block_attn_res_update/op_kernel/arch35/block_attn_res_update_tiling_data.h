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
    ROW_LOCAL = 0,
    ROW_BATCH = 1,
};

#pragma pack(push, 8)
struct BlockAttnResUpdateTilingData {
    uint32_t rowsPerCoreBase;
    uint32_t rowsRemainder;
    uint32_t usedCoreNum;
    uint32_t cols;
    // Element strides after per-row 32-byte alignment. FP32 and BF16 need different element granularities.
    uint32_t colsAlignFp32;
    uint32_t colsAlignBf16;
    uint32_t tileRows;
    // Stats row stride: VREG-aligned for the small-D token-batch path, otherwise 32-byte aligned.
    uint32_t scalarAlign;
    uint32_t colsLoops;
    float epsilon;
    float invCols;
    // Host-selected primary Phase 2 path. A short tail tile may still use the row-local path.
    uint8_t phase2Mode;
};
#pragma pack(pop)

#endif // BLOCK_ATTN_RES_UPDATE_TILING_DATA_H
