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
 * \file moe_tutel_combine_x_tiling_struct.h
 * \brief Tiling data struct and tiling key constants shared between the host
 *        tiling and the device kernel of MoeTutelCombineX (arch35).
 */
#ifndef MOE_TUTEL_COMBINE_X_TILING_STRUCT_H
#define MOE_TUTEL_COMBINE_X_TILING_STRUCT_H

#include <cstdint>

// ---------------------------------------------------------------------------
// Tiling keys (set by the host tiling, dispatched by the kernel entry):
//   0 - normal, no batch (y_grad is 2D)
//   1 - empty tensor shortcut (output x_grad is zeroed)
//   2 - normal, with batch (y_grad is 3D)
// ---------------------------------------------------------------------------
enum MoetutelcombinexTilingKey {
    TILING_KEY_NORMAL_NO_BATCH = 0,
    TILING_KEY_EMPTY = 1,
    TILING_KEY_NORMAL_BATCH = 2
};

constexpr int64_t kMaxInputSlots = 4;  // y_grad, gates, indices, locations
constexpr int64_t kMaxOutputSlots = 1; // x_grad

// ---------------------------------------------------------------------------
// MoetutelcombinexTilingData — complete tiling data shared host<->kernel.
// All element-count fields are in units of "elements" (not bytes).
// ---------------------------------------------------------------------------
struct MoetutelcombinexTilingData {
    // shape / metadata
    int64_t batchSize; // B (no batch = 1)
    int64_t samples;   // S tokens per batch
    int64_t hidden;    // H hidden dim
    int64_t capacity;  // C = eC / E
    int64_t eC;        // rows of y_grad (E*C)

    // multi-core token split (tokenTotal = B*S)
    int64_t tokenTotal;
    int64_t coreTokenBase; // tokens per base core
    int32_t bigCoreCnt;    // cores taking one extra token
    int32_t usedCoreNum;   // active cores (0 on empty)

    // UB split over the H dim
    int64_t hChunk;     // chunk size in elements
    int64_t hChunkCnt;  // ceil(H / hChunk)
    int64_t hChunkTail; // tail chunk size
};

#endif // MOE_TUTEL_COMBINE_X_TILING_STRUCT_H
