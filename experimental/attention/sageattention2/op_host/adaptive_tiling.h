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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace optiling {

// Evaluated metrics for a specific tile candidate (M, N)
struct TileCandidate {
    uint32_t m;
    uint32_t n;
    uint32_t totalTasks;
    uint32_t waveCount;
    uint32_t tailCores;
    float waveUtilization;
    float tailBubbleRatio;
    uint64_t ubBytes;
    uint64_t l1Bytes;
    bool fitsHardware;
};

// Result of adaptive tile selection
struct TileSelection {
    uint32_t tileM;
    uint32_t tileN;
    uint32_t totalTasks;
    uint32_t waveCount;
    uint32_t tailCores;
    float waveUtilization;
    float tailBubbleRatio;
    bool isFallbackM64;
    uint64_t ubBytes;
    uint64_t l1Bytes;
};

// Compute Unified Buffer (UB) footprint in bytes for a given (M, N, DP)
// Derivation based on attention_common.h / attention_block_vec_wide.h layout:
//   - Standard mode (N=256): AttentionBlockVec layout (END <= 248 KiB)
//   - Wide stream mode (N=512): WideVectorLayout with dual 256-subtiles (END = 246 KiB <= 248 KiB)
inline uint64_t EstimateUbFootprint(uint32_t m, uint32_t n, uint32_t dp)
{
    const uint32_t half = m / 2;
    const uint32_t nTile = (n > 256) ? 256 : n;
    constexpr uint32_t pvSlots = 2;
    constexpr uint32_t alphaSlots = 4;
    constexpr uint32_t pNzPadRows = 1;

    const uint32_t pSlotBytes = (half + pNzPadRows) * nTile;
    const uint32_t pvSlotBytes = half * dp * sizeof(float);

    const uint32_t scoreBytes = (n > 256 ? 4 : 3) * half * nTile * sizeof(uint16_t);
    const uint32_t pvBytes = pvSlots * pvSlotBytes;
    const uint32_t accumBytes = half * dp * sizeof(float);
    const uint32_t pBytes = (n > 256 ? 1 : 3) * pSlotBytes;
    const uint32_t miscBytes = 512 + 2 * (n > 256 ? 512 : nTile) * sizeof(uint16_t) + 512 + 512 + 512 + alphaSlots * 512 + half * 4;

    return static_cast<uint64_t>(scoreBytes + pvBytes + accumBytes + pBytes + miscBytes);
}

// Compute L1 buffer footprint in bytes for a given (M, N, DP)
//   - P_L1: 64 KiB
//   - B_L1: P_L1 + PIPE_SLOTS * M * N_tile
//   - Operands: B_L1 + 2 * N_tile * DP
inline uint64_t EstimateL1Footprint(uint32_t m, uint32_t n, uint32_t dp)
{
    constexpr uint32_t pL1 = 65536; // 64 KiB
    const uint32_t nTile = (n > 256) ? 256 : n;
    const uint32_t bL1 = pL1 + 3 * m * nTile;
    return static_cast<uint64_t>(bL1 + 2 * nTile * dp);
}

// Evaluate hardware wave scheduling efficiency and resource boundaries for a candidate (M, N)
inline TileCandidate EvaluateCandidate(uint32_t b, uint32_t hq, uint32_t sq, uint32_t sk,
                                       uint32_t dp, uint32_t m, uint32_t n, uint32_t cores)
{
    TileCandidate c{};
    c.m = m;
    c.n = n;
    const uint32_t tilesM = (sq + m - 1) / m;
    c.totalTasks = b * hq * tilesM;
    c.waveCount = (c.totalTasks + cores - 1) / cores;
    c.tailCores = c.totalTasks % cores;
    if (c.tailCores == 0 && c.totalTasks > 0) {
        c.tailCores = cores;
    }
    const uint32_t totalCoreSlots = c.waveCount * cores;
    c.waveUtilization = (totalCoreSlots > 0) ? (static_cast<float>(c.totalTasks) / static_cast<float>(totalCoreSlots)) : 0.0f;
    c.tailBubbleRatio = 1.0f - (static_cast<float>(c.tailCores) / static_cast<float>(cores));
    c.ubBytes = EstimateUbFootprint(m, n, dp);
    c.l1Bytes = EstimateL1Footprint(m, n, dp);

    // Ascend 950 hardware bounds:
    // UB: 256 KiB per core, 384 KiB combined safe threshold for dual AIV
    // L1: 512 KiB per core
    constexpr uint64_t UB_HARDWARE_LIMIT = 384ULL * 1024ULL;
    constexpr uint64_t L1_HARDWARE_LIMIT = 512ULL * 1024ULL;
    c.fitsHardware = (c.ubBytes <= UB_HARDWARE_LIMIT) && (c.l1Bytes <= L1_HARDWARE_LIMIT);
    return c;
}

// Adaptive Tile Selection Engine & Wave Balancer (自适应动态分块引擎与波次均衡调度器)
// Targets:
//   1. Maximize WaveUtilization = tasks / (ceil(tasks / cores) * cores)
//   2. Eliminate tail-wave compute bubbles when M=128 causes low core occupancy (< 85% or severe bubble < 75%)
//   3. Strictly guarantee UB (<= 384 KiB) and L1 (<= 512 KiB) resource constraints
//   4. Enable Wide-stream N=512 for long sequences (SK >= 4096, DP=128)
inline TileSelection AdaptiveTileSelect(uint32_t b, uint32_t hq, uint32_t hk,
                                        uint32_t sq, uint32_t sk, uint32_t d,
                                        bool causal, uint32_t cores = 24)
{
    const uint32_t dp = d <= 64 ? 64 : 128;

    // Candidate N selection: Wide stream mode (N=512) for long sequences when DP=128
    uint32_t n = 256;
    if (dp == 128 && sk >= 4096 && (sk % 512 == 0)) {
        n = 512;
    }

    // Evaluate M=128 (default large tile, optimal Cube arithmetic intensity)
    TileCandidate c128 = EvaluateCandidate(b, hq, sq, sk, dp, 128, n, cores);

    // Evaluate M=64 (fine-grained tile for wave balancing and tail bubble elimination)
    TileCandidate c64 = EvaluateCandidate(b, hq, sq, sk, dp, 64, n, cores);

    TileSelection sel{};

    // Optimization threshold:
    // If M=128 yields high wave utilization (>= 85%), keep M=128 for maximum Cube GEMM throughput.
    // If M=128 suffers from tail wave bubbles (utilization < 85%) and M=64 achieves strictly higher
    // wave utilization, dynamically downgrade to M=64.
    constexpr float TARGET_UTIL_GATE = 0.85f;
    if (c128.waveUtilization < TARGET_UTIL_GATE && c64.waveUtilization > c128.waveUtilization && c64.fitsHardware) {
        sel.tileM = c64.m;
        sel.tileN = c64.n;
        sel.totalTasks = c64.totalTasks;
        sel.waveCount = c64.waveCount;
        sel.tailCores = c64.tailCores;
        sel.waveUtilization = c64.waveUtilization;
        sel.tailBubbleRatio = c64.tailBubbleRatio;
        sel.isFallbackM64 = true;
        sel.ubBytes = c64.ubBytes;
        sel.l1Bytes = c64.l1Bytes;
    } else {
        sel.tileM = c128.m;
        sel.tileN = c128.n;
        sel.totalTasks = c128.totalTasks;
        sel.waveCount = c128.waveCount;
        sel.tailCores = c128.tailCores;
        sel.waveUtilization = c128.waveUtilization;
        sel.tailBubbleRatio = c128.tailBubbleRatio;
        sel.isFallbackM64 = false;
        sel.ubBytes = c128.ubBytes;
        sel.l1Bytes = c128.l1Bytes;
    }

    return sel;
}

} // namespace optiling
