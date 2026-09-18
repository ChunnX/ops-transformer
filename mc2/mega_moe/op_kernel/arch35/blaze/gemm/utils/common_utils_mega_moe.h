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
 * \file common_utils_mega_moe.h
 * \brief MXFP8FP4 grouped-matmul pipeline constants extracted from blaze for mega_moe.
 */
#pragma once

namespace Blaze {
namespace Gemm {

constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_256 = 256UL;
constexpr uint64_t MX_FP8FP4_L1_K_CONFIG_512 = 512UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128UL;
constexpr uint64_t MX_FP8FP4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD = 256UL;
constexpr uint64_t MX_FP8FP4_SCALE_K_L1_SIZE = 4096UL;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint64_t SYNC_MODE4 = 4;

// Shared/Layered retain the original A2/B4 layout and demand-driven copies.
struct MegaMoeMxL1StaticLayout {
    static constexpr bool dynamic = false;
    static constexpr uint64_t aStages = 2;
    static constexpr uint64_t bStages = 4;
    static constexpr uint64_t aSlotBytes = 64 * 1024;
    static constexpr uint64_t aOffsets[2] = {128 * 1024, 384 * 1024};
    static constexpr uint64_t bOffsets[4] = {0, 256 * 1024, 64 * 1024, 320 * 1024};
    static constexpr uint64_t scaleAOffsets[2] = {192 * 1024, 448 * 1024};
    static constexpr uint64_t scaleBOffsets[2] = {224 * 1024, 480 * 1024};
};

// Wave A8W4 computes one allocation for its persistent AIC/AIV0 resources.
struct MegaMoeMxL1Allocation {
    static constexpr bool dynamic = true;
    static constexpr uint64_t L1_BYTES = 512 * 1024;
    static constexpr uint64_t MAX_A_STAGES = 6;
    static constexpr uint64_t MAX_B_STAGES = 8;
    uint64_t aStages = 0;
    uint64_t bStages = 0;
    uint64_t tileK = 0;
    uint64_t scaleK = 0;
    uint64_t aSlotBytes = 0;
    uint64_t bSlotBytes = 0;
    uint64_t scaleASlotBytes = 0;
    uint64_t scaleBSlotBytes = 0;
    uint64_t aOffsets[MAX_A_STAGES] = {};
    uint64_t bOffsets[MAX_B_STAGES] = {};
    uint64_t scaleAOffsets[2] = {};
    uint64_t scaleBOffsets[2] = {};

    __aicore__ static inline uint64_t Align(uint64_t value, uint64_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    // The caller supplies positive tile dimensions: KTile is a multiple of 64,
    // and each FP8 A/B tile fits in 64 KiB (Prologue has two 32 KiB W8 halves).
    template <uint64_t M, uint64_t N, uint64_t KTile>
    __aicore__ static inline MegaMoeMxL1Allocation Build(uint64_t maxK)
    {
        MegaMoeMxL1Allocation result;
        const uint64_t m = Align(M, 16);
        const uint64_t n = Align(N, 32);
        const uint64_t kt = KTile;
        const uint64_t aBytes = Align(m * kt, 512);
        const uint64_t bBytes = Align(n * kt, 512);
        const uint64_t maxScalePerRow = Align(maxK, 64) / 32;
        const uint64_t maxScaleBytes = Align(m * maxScalePerRow, 512) + Align(n * maxScalePerRow, 512);
        const uint64_t minScaleA = Align(m * (kt / 32), 512);
        const uint64_t minScaleB = Align(n * (kt / 32), 512);
        uint64_t aCount = maxScaleBytes < L1_BYTES ? (L1_BYTES - maxScaleBytes) / (aBytes + bBytes) : 0;
        // A full-K scale reservation is only a sizing estimate; large K still
        // runs with at least one A stage and a shorter double-buffered scale window.
        aCount = aCount == 0 ? 1 : aCount;
        aCount = aCount < MAX_A_STAGES ? aCount : MAX_A_STAGES;
        const uint64_t minScaleBytes = 2 * (minScaleA + minScaleB);
        uint64_t bCount = (L1_BYTES - aCount * aBytes - minScaleBytes) / bBytes;
        bCount = bCount < MAX_B_STAGES ? bCount : MAX_B_STAGES;
        const uint64_t remaining = L1_BYTES - aCount * aBytes - bCount * bBytes;
        uint64_t windows = remaining / (2 * (minScaleA + minScaleB));
        const uint64_t maxWindows = maxK == 0 ? 1 : (maxK + kt - 1) / kt;
        windows = windows < maxWindows ? windows : maxWindows;
        result.aStages = aCount;
        result.bStages = bCount;
        result.tileK = kt;
        result.scaleK = windows * kt;
        result.aSlotBytes = aBytes;
        result.bSlotBytes = bBytes;
        result.scaleASlotBytes = Align(m * (result.scaleK / 32), 512);
        result.scaleBSlotBytes = Align(n * (result.scaleK / 32), 512);
        uint64_t offset = 0;
        for (uint64_t i = 0; i < aCount; ++i) {
            result.aOffsets[i] = offset;
            offset += aBytes;
        }
        // Interleave the logical B ring across its two physical groups.
        for (uint64_t i = 0; i < bCount; ++i) {
            const uint64_t physical = (i & 1) * ((bCount + 1) / 2) + i / 2;
            result.bOffsets[i] = offset + physical * bBytes;
        }
        offset += bCount * bBytes;
        for (uint64_t i = 0; i < 2; ++i) {
            result.scaleAOffsets[i] = offset;
            offset += result.scaleASlotBytes;
            result.scaleBOffsets[i] = offset;
            offset += result.scaleBSlotBytes;
        }
        return result;
    }
};

} // namespace Gemm
} // namespace Blaze
