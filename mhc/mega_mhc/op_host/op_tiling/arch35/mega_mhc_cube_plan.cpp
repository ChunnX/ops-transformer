/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mega_mhc_cube_plan.h"

#include <algorithm>
#include <numeric>

namespace optiling::mega_mhc {
namespace {

constexpr uint64_t kMAlign = 16;
constexpr uint64_t kCubeMTile = 32;
constexpr uint64_t kLargeCubeMTile = 96;
constexpr uint64_t kFiveCubeMTile = 192;
constexpr uint64_t kLargeMThreshold = 3072;
constexpr uint64_t kKAlign = 256;
constexpr uint64_t kL1KAlign = 128;
constexpr uint64_t kL0KAlign = 32;
constexpr uint64_t kNzNAlign = 16;
constexpr uint64_t kFp32Bytes = sizeof(float);
constexpr uint64_t kSequentialPartialK = 1024;
constexpr uint64_t kSequentialPartialThreshold = 2048;
constexpr uint64_t kMaxWaveRowsPerAiv = 16;
constexpr uint64_t kWaveProjectionElements = 8192;

/** Return ceil(value/divisor), with zero for an invalid divisor. */
uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

/** Round a host-selected element count up to the required hardware alignment. */
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

/** Round a host-selected element count down without crossing its capacity. */
uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

/** Check both ping-pong operands against their dedicated per-core L1 budgets. */
bool FitsL1(uint64_t m, uint64_t q, uint64_t kL1, const CubePlanRequest &request)
{
    constexpr uint64_t kPingPong = 2;
    // l1ABytes/l1BBytes are each operand's complete two-slot budget.
    return kPingPong * m * kL1 * kFp32Bytes <= request.l1ABytes &&
           kPingPong * AlignUp(q, kNzNAlign) * kL1 * kFp32Bytes <= request.l1BBytes;
}

/** Check whether aligned K ownership retains every lane: sequential partials use quotient/remainder;
 * otherwise ceil(K/align256(ceil(K/g))) must equal g to preserve the complete-row wave layout.
 */
bool KeepsWaveKGrid(uint64_t k, uint64_t grid)
{
    const uint64_t shard = AlignUp(CeilDiv(k, grid), kKAlign);
    return shard >= kSequentialPartialThreshold || CeilDiv(k, shard) == grid;
}

} // namespace

/*
 * Formula/plan: waves minimize busiest-core M*K work over two transfer-budget choices, preferring larger M on ties.
 * Wave R adapts to T for at least two batches: M=2*R*kGrid and L1K<=8192/(R*kGrid).
 * Non-wave plans use M96 or M192 for M>=3072, according to preferWideTile; otherwise M32.
 * mGrid=min(activeAic,ceil(M/mTile)), then
 * kGrid=min(floor(activeAic/mGrid),ceil(K/256)); splitK is rounded to 256
 * or to an ordered 1024-element partial group. kL1 is the largest 128-aligned
 * block (a divisor of 1024 for sequential partials), up to 1024, that fits both double-buffered L1
 * operands. baseK is the minimum L0A/L0B/L1 capacity rounded down to 32. All loop counts and tails
 * are returned for the kernel; no device-side tile search is needed.
 */
bool BuildCubePlan(const CubePlanRequest &request, CubePlan &plan)
{
    if (request.rows == 0 || request.k == 0 || request.q == 0 || request.aicCount == 0 || request.aivCount < 2 ||
        request.l1ABytes == 0 || request.l1BBytes == 0 || request.l0ABytes == 0 || request.l0BBytes == 0 ||
        request.l0CBytes < kFp32Bytes * 32 * 32) {
        return false;
    }

    const uint64_t active = std::min(request.aicCount, request.aivCount / 2);
    if (active == 0 || (request.wavePipeline && (active < 4U || request.schedule != ScheduleMode::MK))) {
        return false;
    }

    // Wider M tiles amortize FP32 MMAD setup for large M. L1 K shrinks to fit
    // both ping-pong operands; reduction batching is planned independently so
    // the epilogue can safely reuse the projection UB allocation.
    const uint64_t largeMTile = request.preferWideTile ? kFiveCubeMTile : kLargeCubeMTile;
    // Select K ownership using at least two waves: 2*C*R rows per wave, R<=16 and a power of two.
    // Increase K parallelism as R shrinks so the Cube M tile remains 64 or 128 rows.
    const uint64_t rowBudget = std::min(kMaxWaveRowsPerAiv, std::max<uint64_t>(1U, request.rows / (4U * active)));
    uint64_t waveRowsPerAiv = 1U;
    while (waveRowsPerAiv * 2U <= rowBudget) {
        waveRowsPerAiv *= 2U;
    }
    // Prefer complete waves when the minimum full-core batch divides T; otherwise retain capacity for a tail.
    if (request.preferCompleteWaves && request.rows % (2U * active) == 0U) {
        while (waveRowsPerAiv > 1U && request.rows % (2U * active * waveRowsPerAiv) != 0U) {
            waveRowsPerAiv /= 2U;
        }
    }
    const uint64_t waveKProduct = kWaveProjectionElements / waveRowsPerAiv;
    // Since M=2*R*g, busiest-core M*K work is proportional to g*ceil(partials/g).
    // Ties retain the larger M tile; the two candidates follow the same transfer budget.
    const uint64_t maxWaveKGrid = waveKProduct / kL1KAlign;
    const uint64_t minWaveKGrid = maxWaveKGrid / 2U;
    const uint64_t partialCount = CeilDiv(request.k, kSequentialPartialK);
    const uint64_t preferredWaveKGrid =
        AlignUp(partialCount, minWaveKGrid) < AlignUp(partialCount, maxWaveKGrid) ? minWaveKGrid : maxWaveKGrid;
    uint64_t waveKGrid = std::gcd(preferredWaveKGrid, active);
    // Try the other existing transfer-budget candidate before falling back to staged execution.
    // This is a legality check, not a shape-specific performance exception.
    if (request.wavePipeline && !KeepsWaveKGrid(request.k, waveKGrid)) {
        const uint64_t alternate = std::gcd(preferredWaveKGrid == minWaveKGrid ? maxWaveKGrid : minWaveKGrid, active);
        if (alternate >= minWaveKGrid && KeepsWaveKGrid(request.k, alternate)) {
            waveKGrid = alternate;
        }
    }
    if (request.wavePipeline && waveKGrid < minWaveKGrid) {
        return false;
    }
    // Widen M without changing K ownership: R*g*L1K<=8192 and L1K>=128.
    // Keep at least two waves and preserve the selected complete-wave constraint.
    const uint64_t waveRowCapacity =
        std::min(request.rows / (4U * active), kWaveProjectionElements / (waveKGrid * kL1KAlign));
    while (waveRowsPerAiv * 2U <= waveRowCapacity &&
           (!request.preferCompleteWaves || request.rows % (2U * active) != 0U ||
            request.rows % (4U * active * waveRowsPerAiv) == 0U)) {
        waveRowsPerAiv *= 2U;
    }
    uint64_t selectedMTile = request.wavePipeline ? 2U * waveRowsPerAiv * waveKGrid :
                                                    (request.rows >= kLargeMThreshold ? largeMTile : kCubeMTile);
    // ND keeps one M lane and partitions K=N*D. BS owns full K and partitions token tiles.
    // Cap M by both L0C and the minimum legal double-buffered L1 transfer; all counts stay host planned.
    if (request.schedule != ScheduleMode::MK) {
        const uint64_t mCapacity = AlignDown(std::min<uint64_t>({256U, request.l0CBytes / (2U * 32U * kFp32Bytes),
                                                                 request.l1ABytes / (2U * kL1KAlign * kFp32Bytes)}),
                                             kMAlign);
        if (mCapacity == 0U) {
            return false;
        }
        const uint64_t targetRows = request.schedule == ScheduleMode::ND ? request.rows : CeilDiv(request.rows, active);
        selectedMTile = std::min(mCapacity, AlignUp(targetRows, kMAlign));
    }
    // Preserve the original L1 K grain so M balancing does not change RMS partial sums.
    const uint64_t transferMTile = selectedMTile;
    // FP32 keeps a fixed 32-element Mmad K step while balancing M ownership.
    // HF32 retains its existing plan because baseK changes its accumulation order.
    if (!request.wavePipeline && request.schedule == ScheduleMode::MK && request.preferWideTile &&
        request.mode == ProjectionMode::FP32) {
        const uint64_t occupiedM = std::min(active, CeilDiv(request.rows, selectedMTile));
        const uint64_t occupiedK = std::min(active / occupiedM, CeilDiv(request.k, kKAlign));
        // Rebalance only unsplit K with idle cores; keep existing K ownership and full grids.
        if (occupiedK == 1U && occupiedM < active) {
            selectedMTile = std::min(selectedMTile, AlignUp(CeilDiv(request.rows, active), kMAlign));
        }
    }
    const uint64_t selectedMGrid =
        request.schedule == ScheduleMode::ND ?
            1U :
            std::min(request.wavePipeline ? active / waveKGrid : active, CeilDiv(request.rows, selectedMTile));
    const uint64_t kBlocks = CeilDiv(request.k, kKAlign);
    uint64_t selectedKGrid = request.schedule == ScheduleMode::BS ? 1U : std::min(active / selectedMGrid, kBlocks);
    if (selectedKGrid == 0) {
        return false;
    }

    const uint64_t rawSplitK = AlignUp(CeilDiv(request.k, selectedKGrid), kKAlign);
    // Match mhc_pre: decide from the initial 256-aligned singleCoreK, before adjusting K ownership.
    // Small shards accumulate once per core; large shards publish ordered 1024-element partials.
    const bool useSequentialPartials = rawSplitK >= kSequentialPartialThreshold;
    const uint64_t partialGroupK = useSequentialPartials ? kSequentialPartialK : rawSplitK;
    const uint64_t splitK = useSequentialPartials ?
                                CeilDiv(CeilDiv(request.k, kSequentialPartialK), selectedKGrid) * kSequentialPartialK :
                                rawSplitK;
    // Ordered partials can fill every wave lane with quotient/remainder ownership;
    // compacting ceil-sized shards here would discard usable lanes before balancing them.
    if (!request.wavePipeline || !useSequentialPartials) {
        selectedKGrid = CeilDiv(request.k, splitK);
    }
    // Distribute ordered partials by quotient/remainder; keep the launch grid and partial size unchanged.
    const uint64_t remainder = useSequentialPartials ? partialCount % selectedKGrid : 0U;
    const uint64_t largeKCoreCount = remainder == 0U ? selectedKGrid : remainder;
    const uint64_t shortSplitK = useSequentialPartials ? partialCount / selectedKGrid * partialGroupK : splitK;
    const uint64_t selectedActive = selectedMGrid * selectedKGrid;
    const uint64_t rowsPerWave = selectedMGrid * selectedMTile;
    const uint64_t mWaveCount = CeilDiv(request.rows, rowsPerWave);
    const uint64_t l1KMax =
        std::min<uint64_t>({request.wavePipeline ? kWaveProjectionElements / (waveRowsPerAiv * waveKGrid) : 1024U,
                            request.l1ABytes / (2 * transferMTile * kFp32Bytes),
                            request.l1BBytes / (2 * AlignUp(request.q, kNzNAlign) * kFp32Bytes)});
    // L1 transfers must not cross an ordered partial boundary. A small unsplit shard may end with an L1 tail.
    uint64_t l1K = AlignDown(std::min(l1KMax, partialGroupK), kL1KAlign);
    while (useSequentialPartials && l1K != 0 && partialGroupK % l1K != 0) {
        l1K -= kL1KAlign;
    }
    if (l1K == 0 || !FitsL1(selectedMTile, request.q, l1K, request)) {
        return false;
    }

    const uint64_t baseM =
        AlignDown(std::min<uint64_t>(selectedMTile, request.l0CBytes / (2 * 32 * kFp32Bytes)), kMAlign);
    constexpr uint64_t baseN = 32;
    // Both L0C slots hold a complete M tile. Reject a smaller
    // hardware footprint instead of silently introducing an unplanned M loop.
    if (baseM < selectedMTile || 2 * baseM * baseN * kFp32Bytes > request.l0CBytes) {
        return false;
    }
    const uint64_t l0AK = request.l0ABytes / (2 * baseM * kFp32Bytes);
    const uint64_t l0BK = request.l0BBytes / (2 * baseN * kFp32Bytes);
    const uint64_t baseK = AlignDown(std::min({l0AK, l0BK, l1K}), kL0KAlign);
    if (baseK == 0) {
        return false;
    }
    const uint64_t nzM = AlignUp(selectedMTile, kMAlign);
    const uint64_t nzN = AlignUp(request.q, kNzNAlign);
    const uint64_t aL1SlotElements = nzM * l1K;
    const uint64_t bL1SlotElements = nzN * l1K;
    const uint64_t l1BOffsetElements = 2 * aL1SlotElements;
    const uint64_t l0ASlotElements = baseM * baseK;
    const uint64_t l0BSlotElements = baseN * baseK;
    const uint64_t l0CSlotElements = baseM * baseN;

    const uint64_t lastWaveRows = request.rows - (mWaveCount - 1) * rowsPerWave;
    const uint64_t lastWaveMGrid = CeilDiv(lastWaveRows, selectedMTile);
    const uint64_t lastMRows = lastWaveRows - (lastWaveMGrid - 1) * selectedMTile;
    const uint64_t largeBeforeLast = std::min(selectedKGrid - 1U, largeKCoreCount);
    const uint64_t lastKElements =
        request.k - largeBeforeLast * splitK - (selectedKGrid - 1U - largeBeforeLast) * shortSplitK;
    const uint64_t l1FullLoops = splitK / l1K;
    const uint64_t l1Tail = splitK % l1K;
    const uint64_t lastCoreL1FullLoops = lastKElements / l1K;
    const uint64_t lastCoreL1Tail = lastKElements % l1K;
    const uint64_t l0FullLoops = l1K / baseK;
    const uint64_t l0Tail = l1K % baseK;
    const uint64_t tailL1K = l1Tail == 0 ? l1K : l1Tail;
    const uint64_t tailL1L0FullLoops = tailL1K / baseK;
    const uint64_t tailL1L0Tail = tailL1K % baseK;
    const uint64_t lastCoreTailL1K = lastCoreL1Tail == 0 ? l1K : lastCoreL1Tail;
    const uint64_t lastCoreTailL1L0FullLoops = lastCoreTailL1K / baseK;
    const uint64_t lastCoreTailL1L0Tail = lastCoreTailL1K % baseK;
    const uint64_t mL0FullLoops = selectedMTile / baseM;
    const uint64_t mL0TailRows = selectedMTile % baseM;
    const uint64_t lastMCoreL0FullLoops = lastMRows / baseM;
    const uint64_t lastMCoreL0TailRows = lastMRows % baseM;
    const uint64_t fullMmadKStep = request.mode == ProjectionMode::FP32 ? kL0KAlign : baseK;
    const uint64_t l0TailMmadKStep = request.mode == ProjectionMode::FP32 ? kL0KAlign : l0Tail;
    const uint64_t tailL1TailMmadKStep = request.mode == ProjectionMode::FP32 ? kL0KAlign : tailL1L0Tail;
    const uint64_t lastCoreTailMmadKStep = request.mode == ProjectionMode::FP32 ? kL0KAlign : lastCoreTailL1L0Tail;
    const uint64_t l0TailMmadFinalKStep =
        request.mode == ProjectionMode::FP32 ? (l0Tail % kL0KAlign == 0 ? kL0KAlign : l0Tail % kL0KAlign) : l0Tail;
    const uint64_t tailL1TailMmadFinalKStep =
        request.mode == ProjectionMode::FP32 ? (tailL1L0Tail % kL0KAlign == 0 ? kL0KAlign : tailL1L0Tail % kL0KAlign) :
                                               tailL1L0Tail;
    const uint64_t lastCoreTailMmadFinalKStep =
        request.mode == ProjectionMode::FP32 ?
            (lastCoreTailL1L0Tail % kL0KAlign == 0 ? kL0KAlign : lastCoreTailL1L0Tail % kL0KAlign) :
            lastCoreTailL1L0Tail;

    plan.activeAic = static_cast<uint32_t>(selectedActive);
    plan.activeAiv = static_cast<uint32_t>(2 * selectedActive);
    plan.mGrid = static_cast<uint32_t>(selectedMGrid);
    plan.kGrid = static_cast<uint32_t>(selectedKGrid);
    plan.mWaveCount = static_cast<uint32_t>(mWaveCount);
    plan.rowsPerWave = static_cast<uint32_t>(rowsPerWave);
    plan.lastWaveMGrid = static_cast<uint32_t>(lastWaveMGrid);
    plan.lastWaveRows = static_cast<uint32_t>(lastWaveRows);
    plan.mTile = static_cast<uint32_t>(selectedMTile);
    plan.splitK = static_cast<uint32_t>(splitK);
    plan.largeKCoreCount = static_cast<uint32_t>(largeKCoreCount);
    plan.shortSplitK = static_cast<uint32_t>(shortSplitK);
    plan.shortKCoreL1FullLoops = static_cast<uint32_t>(shortSplitK / l1K);
    plan.l1K = static_cast<uint32_t>(l1K);
    plan.l1KFullLoops = static_cast<uint32_t>(l1FullLoops);
    plan.l1KTail = static_cast<uint32_t>(l1Tail);
    plan.lastKCoreL1FullLoops = static_cast<uint32_t>(lastCoreL1FullLoops);
    plan.lastKCoreL1Tail = static_cast<uint32_t>(lastCoreL1Tail);
    plan.lastMCoreRows = static_cast<uint32_t>(lastMRows);
    plan.lastKCoreElements = static_cast<uint32_t>(lastKElements);
    plan.mL0FullLoops = static_cast<uint32_t>(mL0FullLoops);
    plan.mL0TailRows = static_cast<uint32_t>(mL0TailRows);
    plan.lastMCoreL0FullLoops = static_cast<uint32_t>(lastMCoreL0FullLoops);
    plan.lastMCoreL0TailRows = static_cast<uint32_t>(lastMCoreL0TailRows);
    plan.loadMStride = static_cast<uint32_t>(nzM / 16U);
    plan.loadNStride = static_cast<uint32_t>(nzN / 16U);
    plan.fullL0LoadKStep = static_cast<uint32_t>(baseK / 8U);
    plan.l0TailLoadKStep = static_cast<uint32_t>(l0Tail / 8U);
    plan.tailL1TailLoadKStep = static_cast<uint32_t>(tailL1L0Tail / 8U);
    plan.lastCoreTailLoadKStep = static_cast<uint32_t>(lastCoreTailL1L0Tail / 8U);
    plan.baseM = static_cast<uint32_t>(baseM);
    plan.baseN = static_cast<uint32_t>(baseN);
    plan.baseK = static_cast<uint32_t>(baseK);
    plan.nzM = static_cast<uint32_t>(nzM);
    plan.nzN = static_cast<uint32_t>(nzN);
    plan.l1ASlotElements = static_cast<uint32_t>(aL1SlotElements);
    plan.l1BOffsetElements = static_cast<uint32_t>(l1BOffsetElements);
    plan.l1BSlotElements = static_cast<uint32_t>(bL1SlotElements);
    plan.l0ASlotElements = static_cast<uint32_t>(l0ASlotElements);
    plan.l0BSlotElements = static_cast<uint32_t>(l0BSlotElements);
    plan.l0CSlotElements = static_cast<uint32_t>(l0CSlotElements);
    // Fixpipe stores the padded Cube N tile. Keep Q=24's valid coefficients in
    // the first columns and reserve the full 32-column destination stride.
    plan.fixpipeRowStride = static_cast<uint32_t>(nzN);
    plan.fullMmadKStep = static_cast<uint32_t>(fullMmadKStep);
    plan.fullMmadKLoopCount = static_cast<uint32_t>(request.mode == ProjectionMode::FP32 ? baseK / kL0KAlign : 1);
    plan.l0TailMmadKStep = static_cast<uint32_t>(l0TailMmadKStep);
    plan.l0TailMmadKLoopCount = static_cast<uint32_t>(
        l0Tail == 0 ? 0 : (request.mode == ProjectionMode::FP32 ? CeilDiv(l0Tail, kL0KAlign) : 1));
    plan.l0TailMmadFinalKStep = static_cast<uint32_t>(l0Tail == 0 ? 0 : l0TailMmadFinalKStep);
    plan.tailL1TailMmadKStep = static_cast<uint32_t>(tailL1TailMmadKStep);
    plan.tailL1TailMmadKLoopCount = static_cast<uint32_t>(
        tailL1L0Tail == 0 ? 0 : (request.mode == ProjectionMode::FP32 ? CeilDiv(tailL1L0Tail, kL0KAlign) : 1));
    plan.tailL1TailMmadFinalKStep = static_cast<uint32_t>(tailL1L0Tail == 0 ? 0 : tailL1TailMmadFinalKStep);
    plan.lastCoreTailMmadKStep = static_cast<uint32_t>(lastCoreTailMmadKStep);
    plan.lastCoreTailMmadKLoopCount = static_cast<uint32_t>(
        lastCoreTailL1L0Tail == 0 ?
            0 :
            (request.mode == ProjectionMode::FP32 ? CeilDiv(lastCoreTailL1L0Tail, kL0KAlign) : 1));
    plan.lastCoreTailMmadFinalKStep = static_cast<uint32_t>(lastCoreTailL1L0Tail == 0 ? 0 : lastCoreTailMmadFinalKStep);
    plan.l0KLoops = static_cast<uint32_t>(l0FullLoops);
    plan.l0KTail = static_cast<uint32_t>(l0Tail);
    plan.tailL1L0FullLoops = static_cast<uint32_t>(tailL1L0FullLoops);
    plan.tailL1L0Tail = static_cast<uint32_t>(tailL1L0Tail);
    plan.lastCoreTailL1L0FullLoops = static_cast<uint32_t>(lastCoreTailL1L0FullLoops);
    plan.lastCoreTailL1L0Tail = static_cast<uint32_t>(lastCoreTailL1L0Tail);
    plan.partialGroupK = static_cast<uint32_t>(partialGroupK);
    plan.partialGroupCount = static_cast<uint32_t>(CeilDiv(request.k, plan.partialGroupK));
    return true;
}

} // namespace optiling::mega_mhc
