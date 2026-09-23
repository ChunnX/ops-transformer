/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SMLA_L1_LAYOUT_H
#define SMLA_L1_LAYOUT_H

#include <stdint.h>

// Experimental build selector: 0 original, 1 split reference, 2 control,
// 3 balanced, 4 balanced with slot1's logical halves exchanged.
#ifndef SMLA_L1_LAYOUT_MODE
#define SMLA_L1_LAYOUT_MODE 0
#endif

// 0 original readiness, 1 unified MTE2 readiness, 2 unified readiness + half FREE.
#ifndef SMLA_L1_SYNC_MODE
#define SMLA_L1_SYNC_MODE 0
#endif
static_assert(SMLA_L1_SYNC_MODE <= 2, "Unsupported sparse KV synchronization");

// 0 original AB ping-pong; 1 single A scratch control; 2 single A + two resident Q quarters.
#ifndef SMLA_L0A_MODE
#define SMLA_L0A_MODE 0
#endif
static_assert(SMLA_L0A_MODE <= 2, "Unsupported sparse L0A policy");

namespace SMLAKernel {
template <uint32_t Mode>
struct SparseKvL1Layout {
    static_assert(Mode <= 4, "Unsupported sparse KV L1 layout");
    static constexpr uint32_t KIB = 1024;
    static constexpr uint32_t HALF_BYTES = 64 * KIB;
    static constexpr uint32_t QUARTER_ELEMENTS = 128 * 128;
    static constexpr bool SPLIT = Mode != 0;
    static_assert((2 * 16 + 3 * 32 + 6 * 64) * KIB == 512 * KIB, "L1 capacity mismatch");

    __aicore__ static constexpr uint32_t QueryBase(uint32_t slot)
    {
        if constexpr (Mode <= 1) {
            return (32 + slot * 32) * KIB;
        } else {
            return (slot == 0 ? 32 : (slot == 1 ? 256 : 288)) * KIB;
        }
    }

    __aicore__ static constexpr uint32_t HalfBase(uint32_t slot, uint32_t half)
    {
        if constexpr (Mode <= 1) {
            return (128 + slot * 128 + half * 64) * KIB;
        } else if constexpr (Mode == 2) {
            if (slot == 0) {
                return (half == 0 ? 64 : 128) * KIB;
            }
            if (slot == 1) {
                return (half == 0 ? 192 : 320) * KIB;
            }
            return (half == 0 ? 384 : 448) * KIB;
        } else {
            if constexpr (Mode == 4) {
                half ^= slot & 1U;
            }
            return (64 + slot * 64 + half * 256) * KIB;
        }
    }

    // Row pitch is the aligned actual S, including GM-staged tail tiles.
    __aicore__ static constexpr uint32_t QuarterBase(uint32_t slot, uint32_t quarter, uint32_t rows)
    {
        if constexpr (Mode == 0) {
            return HalfBase(slot, 0) + quarter * rows * 128 * 2;
        } else {
            return HalfBase(slot, quarter / 2) + (quarter % 2) * rows * 128 * 2;
        }
    }
};

using ActiveSparseKvL1Layout = SparseKvL1Layout<SMLA_L1_LAYOUT_MODE>;
} // namespace SMLAKernel

#endif // SMLA_L1_LAYOUT_H
