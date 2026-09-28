/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../op_host/op_tiling/arch35/mega_mhc_cube_plan.h"
#include "../../op_host/op_tiling/arch35/mega_mhc_memory_plan.h"

#include <algorithm>
#include <initializer_list>

using namespace optiling::mega_mhc;
#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            return __LINE__; \
        } \
    } while (false)

/** Exercise the actual host planner: exact M/K coverage, partial ownership, tails and physical capacities. */
extern "C"
#if defined(_WIN32)
    __declspec(dllexport)
#endif
    int
    run_plan_tests()
{
    for (auto schedule : {ScheduleMode::ND, ScheduleMode::BS}) {
        for (uint64_t m : {1U, 17U, 33U, 255U, 256U, 257U, 512U, 1025U, 3072U, 8192U, 8193U}) {
            for (uint64_t d = 2560U; d <= 7168U; d += 256U) {
                for (auto mode : {ProjectionMode::FP32, ProjectionMode::HF32}) {
                    CubePlanRequest r;
                    r.schedule = schedule;
                    r.rows = m;
                    r.k = 4U * d;
                    r.q = 24U;
                    r.aicCount = 32U;
                    r.aivCount = 64U;
                    r.l1ABytes = r.l1BBytes = 256U * 1024U;
                    r.l0ABytes = r.l0BBytes = 64U * 1024U;
                    r.l0CBytes = 128U * 1024U;
                    r.mode = mode;
                    CubePlan p;
                    REQUIRE(BuildCubePlan(r, p));
                    REQUIRE(schedule == ScheduleMode::ND ? p.mGrid == 1U : p.kGrid == 1U);
                    REQUIRE(p.activeAic == p.mGrid * p.kGrid && p.activeAic <= r.aicCount);
                    REQUIRE((p.mWaveCount - 1ULL) * p.rowsPerWave + p.lastWaveRows == m);
                    REQUIRE((p.lastWaveMGrid - 1ULL) * p.mTile + p.lastMCoreRows == p.lastWaveRows);
                    const uint64_t largeBeforeLast = std::min(p.kGrid - 1U, p.largeKCoreCount);
                    REQUIRE(largeBeforeLast * p.splitK + (p.kGrid - 1U - largeBeforeLast) * p.shortSplitK +
                                p.lastKCoreElements ==
                            r.k);
                    REQUIRE(p.lastKCoreElements > 0U && p.lastKCoreElements <= p.splitK);
                    REQUIRE(p.partialGroupK < 2048U);
                    REQUIRE(2ULL * p.l0ASlotElements * 4U <= r.l0ABytes);
                    REQUIRE(2ULL * p.l0BSlotElements * 4U <= r.l0BBytes);
                    REQUIRE(2ULL * p.l0CSlotElements * 4U <= r.l0CBytes);
                    REQUIRE(ProjectionUbBytes(r.k, p.l1K, (p.mTile + 1U) / 2U) <= 248U * 1024U);
                }
            }
        }
    }
    const bool midModes[] = {false, true};
    const uint64_t rows[] = {1,   2,   15,   16,   17,   31,   32,   33,   63,   64,   160,  192,          257,
                             512, 768, 1024, 1536, 1537, 2048, 3071, 3072, 3073, 4096, 6145, 4294967295ULL};
    const uint64_t widths[] = {2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864,
                               5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168};
    const uint64_t coreCounts[] = {1, 7, 24, 32, 48};
    for (auto m : rows) {
        for (auto d : widths) {
            for (auto cores : coreCounts) {
                for (uint32_t variant = 0; variant < 4; ++variant) {
                    const uint32_t mode = variant & 1U;
                    CubePlanRequest request;
                    request.rows = m;
                    const bool hasMid = (variant & 2U) != 0U;
                    request.preferWideTile = true;
                    request.k = 4 * d;
                    request.q = 24;
                    request.aicCount = cores;
                    request.aivCount = 2 * cores;
                    request.l1ABytes = 256 * 1024;
                    request.l1BBytes = 256 * 1024;
                    request.l0ABytes = 64 * 1024;
                    request.l0BBytes = 64 * 1024;
                    request.l0CBytes = 128 * 1024;
                    request.mode = static_cast<ProjectionMode>(mode);
                    request.wavePipeline =
                        !hasMid &&
                        ((m == 3072 && cores >= 4 && cores % 2 == 0) ||
                         (cores == 32 && (m == 512 || m == 768 || m == 1024 || m == 1536 || m == 2048 || m == 4096)));
                    CubePlan p;
                    REQUIRE(BuildCubePlan(request, p));
                    // K-tail compaction can invalidate complete-row ownership; the host then uses its non-wave plan.
                    if (request.wavePipeline &&
                        (p.mTile % (2U * p.kGrid) != 0U || p.mWaveCount <= 1U || p.lastKCoreL1FullLoops < 2U)) {
                        request.wavePipeline = false;
                        REQUIRE(BuildCubePlan(request, p));
                    }
                    // M balancing must preserve K ownership and RMS reduction grain for every tested shape.
                    if (!request.wavePipeline && request.mode == ProjectionMode::FP32) {
                        CubePlanRequest referenceRequest = request;
                        referenceRequest.mode = ProjectionMode::HF32;
                        CubePlan referencePlan;
                        REQUIRE(BuildCubePlan(referenceRequest, referencePlan));
                        REQUIRE(p.kGrid == referencePlan.kGrid && p.partialGroupK == referencePlan.partialGroupK);
                        REQUIRE(p.l1K == referencePlan.l1K);
                        REQUIRE(p.activeAic >= referencePlan.activeAic);
                    }
                    REQUIRE(p.activeAic <= cores && p.activeAic == p.mGrid * p.kGrid);
                    REQUIRE(p.activeAiv == 2 * p.activeAic);
                    if (request.wavePipeline) {
                        // Each producer batch must finish before the next wave's projection reads it.
                        const uint32_t members = p.kGrid * 2U;
                        const uint32_t producerRows = p.mTile / members;
                        REQUIRE(producerRows > 0 && producerRows * members == p.mTile);
                        const uint32_t fullProducers = p.lastWaveRows / producerRows;
                        const uint32_t tailProducerRows = p.lastWaveRows % producerRows;
                        REQUIRE(fullProducers <= p.activeAiv);
                        REQUIRE(fullProducers * producerRows + tailProducerRows == p.lastWaveRows);
                        const uint32_t ticksAvailable = p.lastKCoreL1FullLoops - 1U;
                        const uint32_t rowsPerTick = (producerRows + ticksAvailable - 1U) / ticksAvailable;
                        const uint32_t ticks = (producerRows + rowsPerTick - 1U) / rowsPerTick;
                        const uint32_t tickStride = ticksAvailable / ticks;
                        REQUIRE(tickStride > 0U);
                        REQUIRE(1U + (ticks - 1U) * tickStride < p.lastKCoreL1FullLoops);
                        const uint64_t producerUb = UbAllocation(2ULL * d * 2U) + UbAllocation(2ULL * request.k * 2U) +
                                                    UbAllocation(d * 2U) + UbAllocation(d * 4U) +
                                                    UbAllocation(256U * 4U) + UbAllocation(8U) + UbAllocation(32U);
                        REQUIRE(producerUb + ProjectionUbBytes(request.k, p.l1K, p.mTile / 2U) <= 248U * 1024U);
                    }
                    REQUIRE((uint64_t(p.mWaveCount) - 1) * p.rowsPerWave + p.lastWaveRows == m);
                    REQUIRE((p.lastWaveMGrid - 1) * p.mTile + p.lastMCoreRows == p.lastWaveRows);
                    REQUIRE(p.lastMCoreRows > 0 && p.lastMCoreRows <= p.mTile);
                    uint64_t coveredK = 0;
                    for (uint32_t shard = 0; shard < p.kGrid; ++shard) {
                        const uint32_t count = shard + 1U == p.kGrid ?
                                                   p.lastKCoreElements :
                                                   (shard < p.largeKCoreCount ? p.splitK : p.shortSplitK);
                        REQUIRE(coveredK % p.partialGroupK == 0U);
                        REQUIRE(count > 0U);
                        REQUIRE(p.splitK - count < p.partialGroupK * 2U);
                        coveredK += count;
                    }
                    REQUIRE(coveredK == request.k);
                    REQUIRE(p.shortKCoreL1FullLoops * p.l1K == p.shortSplitK || p.largeKCoreCount == p.kGrid);
                    REQUIRE(p.lastKCoreElements > 0 && p.lastKCoreElements <= p.splitK);
                    REQUIRE(p.splitK % p.partialGroupK == 0);
                    if (p.splitK >= 2048) {
                        REQUIRE(p.partialGroupK == 1024);
                        REQUIRE(p.partialGroupK % p.l1K == 0);
                    } else {
                        REQUIRE(p.partialGroupK == p.splitK);
                        REQUIRE(p.partialGroupCount == p.kGrid);
                    }
                    REQUIRE(uint64_t(p.partialGroupCount) * p.partialGroupK >= request.k);
                    REQUIRE(uint64_t(p.partialGroupCount - 1) * p.partialGroupK < request.k);
                    REQUIRE(p.l1KFullLoops * p.l1K + p.l1KTail == p.splitK);
                    REQUIRE(p.lastKCoreL1FullLoops * p.l1K + p.lastKCoreL1Tail == p.lastKCoreElements);
                    REQUIRE(p.l0KLoops * p.baseK + p.l0KTail == p.l1K);
                    REQUIRE(p.baseK % 32 == 0 && p.l1K % 128 == 0);
                    REQUIRE(p.fullMmadKLoopCount * p.fullMmadKStep == p.baseK);
                    REQUIRE(mode != 0 || p.fullMmadKStep == 32);
                    REQUIRE(2ULL * p.l1ASlotElements * 4 <= request.l1ABytes);
                    REQUIRE(2ULL * p.l1BSlotElements * 4 <= request.l1BBytes);
                    REQUIRE(2ULL * p.l0ASlotElements * 4 <= request.l0ABytes);
                    REQUIRE(2ULL * p.l0BSlotElements * 4 <= request.l0BBytes);
                    REQUIRE(uint64_t(p.l0CSlotElements) * 4 * 2 <= request.l0CBytes);
                    REQUIRE(p.baseM == p.nzM && p.fixpipeRowStride == p.nzN && p.nzN >= request.q);
                    for (bool mid : midModes) {
                        const uint32_t gammaTile = MidGammaTile(d, 248U * 1024U);
                        REQUIRE(gammaTile > 0U && gammaTile % 64U == 0U);
                        REQUIRE(ProducerUbBytes(d, mid, false, gammaTile) <= 248U * 1024U);
                        const uint32_t reductionRowLimit =
                            request.wavePipeline ?
                                static_cast<uint32_t>((m + p.activeAiv - 1U) / p.activeAiv) :
                                std::min(p.mTile / 2U, static_cast<uint32_t>((m + p.activeAiv - 1U) / p.activeAiv));
                        const uint32_t capacityRows = FitReductionRows(reductionRowLimit, d, p.partialGroupCount,
                                                                       p.fixpipeRowStride, p.kGrid, 248U * 1024U);
                        const uint32_t reduceRows = BalanceReductionRows(reductionRowLimit, capacityRows);
                        REQUIRE(reduceRows > 0 && reduceRows <= reductionRowLimit);
                        REQUIRE(EpilogueUbBytes(d, p.partialGroupCount, p.fixpipeRowStride, reduceRows, p.kGrid) <=
                                248U * 1024U);
                        REQUIRE(ProjectionUbBytes(4 * d, p.l1K, p.mTile / 2) <= 248U * 1024U);
                    }
                }
            }
        }
    }
    // Tail plans must cover each producer row exactly once, including empty producer members.
    for (uint64_t m : {513ULL, 576ULL, 640ULL, 1023ULL, 1025ULL, 2176ULL, 2560ULL, 2561ULL, 8192ULL, 8193ULL}) {
        for (auto d : widths) {
            CubePlanRequest request;
            request.rows = m;
            request.k = 4U * d;
            request.q = 24U;
            request.l1ABytes = request.l1BBytes = 256U * 1024U;
            request.l0ABytes = request.l0BBytes = 64U * 1024U;
            request.l0CBytes = 128U * 1024U;
            request.wavePipeline = true;
            CubePlan p;
            request.aicCount = 32U;
            request.aivCount = 64U;
            bool found = false;
            for (bool complete : {true, false}) {
                request.preferCompleteWaves = complete;
                if (BuildCubePlan(request, p) && p.mTile % (2U * p.kGrid) == 0U && p.mWaveCount > 1U &&
                    p.lastKCoreL1FullLoops >= 2U &&
                    ProducerUbBytes(d, false, true) + ProjectionUbBytes(request.k, p.l1K, p.mTile / 2U) <=
                        248U * 1024U) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
            const uint32_t rowsPerProducer = p.mTile / (2U * p.kGrid);
            REQUIRE(p.activeAiv * rowsPerProducer == p.rowsPerWave);
            const uint32_t full = p.lastWaveRows / rowsPerProducer;
            const uint32_t tail = p.lastWaveRows % rowsPerProducer;
            uint64_t covered = (p.mWaveCount - 1ULL) * p.rowsPerWave;
            for (uint32_t member = 0U; member < p.activeAiv; ++member) {
                covered += member < full ? rowsPerProducer : (member == full ? tail : 0U);
            }
            REQUIRE(covered == m);
        }
    }
    // Equal 64-row ownership must use two 32-row batches instead of stranding cores with 96 versus 48 rows.
    REQUIRE(BalanceReductionRows(64, FitReductionRows(64, 2560, 10, 32, 2, 248U * 1024U)) == 32U);
    REQUIRE(BalanceReductionRows(48, 48) == 48U);
    REQUIRE(BalanceReductionRows(0, 48) == 0U);
    REQUIRE(BalanceReductionRows(64, 0) == 0U);
    CubePlanRequest invalid;
    CubePlan unused;
    REQUIRE(!BuildCubePlan(invalid, unused));
    return 0;
}

#ifndef MEGA_MHC_TEST_DLL
/** Return zero only when every independent plan invariant passed. */
int main()
{
    return run_plan_tests();
}
#endif
