/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "common.h"
namespace Sage {
constexpr uint32_t M = 128, N = 256, HALF = M / 2, PIPE_SLOTS = 3, LOOKAHEAD = PIPE_SLOTS - 1;
// Shift adjacent NZ planes by one 32-byte row; MTE3 skips the padding.
constexpr uint32_t P_NZ_PAD_ROWS = 1, P_SLOT_BYTES = (HALF + P_NZ_PAD_ROWS) * N;
constexpr uint32_t ALPHA_SLOTS = 4;
constexpr uint32_t P_L1 = 65536, B_L1 = P_L1 + PIPE_SLOTS * M * N;
static_assert(M * 128 <= P_L1 && P_L1 + PIPE_SLOTS * M * N <= B_L1, "L1 A/P/B buffers overlap");
static_assert(B_L1 + 2 * N * 128 <= Arch::L1_SIZE, "L1 B exceeds capacity");
// 32 KiB is one operand slot, not the full L0A/L0B capacity.
static_assert(M * N <= 32 * 1024 && N * 128 <= 32 * 1024 && M * 128 <= 32 * 1024,
              "L0 operands exceed their 32 KiB slots");
// UB layout derives from tile sizes. The final output reuses P after PV has
// consumed it; the next tile cannot produce P until this AIV finishes writing.
// Three score/P slots provide two-block lookahead. Four alpha slots retain
// alpha(i) until the pair O(i),O(i+1) is consumed after Softmax(i+3).
// FP8 PV accumulates in FP32 L0C and stays FP32 in both UB slots.
constexpr uint32_t PV_SLOTS = 2, PV_SLOT_BYTES = HALF * 128 * 4;
static_assert(LOOKAHEAD == 2 && PV_SLOTS == 2 && ALPHA_SLOTS == 4, "paired O ownership requires these ring depths");
constexpr uint32_t SCORE_UB = 0, PV_UB = PIPE_SLOTS * HALF * N * 2, ACCUM_UB = PV_UB + PV_SLOTS * PV_SLOT_BYTES;
constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4, MU_UB = P_UB + PIPE_SLOTS * P_SLOT_BYTES;
constexpr uint32_t KS_UB = MU_UB + 512, QS_UB = KS_UB + 2 * N * 2;
constexpr uint32_t MAX_UB = QS_UB + 512, DEN_UB = MAX_UB + 512, ALPHA_UB = DEN_UB + 512;
constexpr uint32_t SUM_UB = ALPHA_UB + ALPHA_SLOTS * 512, UB_END = SUM_UB + HALF * 4;
// Do not reuse score: AIC may already be writing the next query tile's QK.
constexpr uint32_t LSE_UB = P_UB;
static_assert(HALF * sizeof(float) <= P_SLOT_BYTES, "LSE exceeds retired P0");
static_assert(UB_END <= Arch::UB_SIZE, "UB exceeds capacity");
static_assert(HALF * N >= HALF * 128 * 2, "P buffer cannot hold final FP16/BF16 output");
// 128 KiB is one accumulator slot, not the full L0C capacity.
static_assert(M * N * 4 <= 128 * 1024 && M * 128 * 4 <= 128 * 1024, "L0C exceeds its 128 KiB slot");
// MODE4 counted notifications, shared by all Cube/Vector groups.
constexpr uint32_t QK_READY = 0, P_READY = 1, PV_READY = 2, PV_RELEASE = 3, AIV_FLAG_OFFSET = 16;
// Local hard-event ownership. Scale slots use consecutive IDs 4 and 5.
constexpr uint32_t QK_EVENT = 2, PV_EVENT = 3, SCALE_EVENT_BASE = 4;
} // namespace Sage
