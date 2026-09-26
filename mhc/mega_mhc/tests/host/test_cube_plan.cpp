/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include "../../op_host/op_tiling/arch35/mega_mhc_cube_plan.h"
#include "../../op_host/op_tiling/arch35/mega_mhc_memory_plan.h"

using namespace optiling::mega_mhc;
#define REQUIRE(condition) do { if (!(condition)) { return __LINE__; } } while (false)

/** Exercise the actual host planner: exact M/K coverage, partial ownership, tails and physical capacities. */
extern "C"
#if defined(_WIN32)
__declspec(dllexport)
#endif
int run_plan_tests()
{
    const bool midModes[] = {false, true};
    const uint64_t rows[] = {1, 2, 15, 16, 17, 31, 32, 33, 63, 257, 768, 1537, 3071, 3072, 3073, 4096, 6145, 4294967295ULL};
    const uint64_t widths[] = {2560, 5120};
    const uint64_t coreCounts[] = {1, 7, 24, 32, 48};
    for (auto m : rows) {
        for (auto d : widths) {
            for (auto cores : coreCounts) {
                for (uint32_t variant = 0; variant < 4; ++variant) {
                    const uint32_t mode = variant & 1U;
                    CubePlanRequest request;
                    request.rows = m;
                    request.hasMid = (variant & 2U) != 0U;
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
                    CubePlan p;
                    REQUIRE(BuildCubePlan(request, p));
                    REQUIRE(p.activeAic <= cores && p.activeAic == p.mGrid * p.kGrid);
                    REQUIRE(p.activeAiv == 2 * p.activeAic);
                    REQUIRE((uint64_t(p.mWaveCount) - 1) * p.rowsPerWave + p.lastWaveRows == m);
                    REQUIRE((p.lastWaveMGrid - 1) * p.mTile + p.lastMCoreRows == p.lastWaveRows);
                    REQUIRE(p.lastMCoreRows > 0 && p.lastMCoreRows <= p.mTile);
                    REQUIRE(uint64_t(p.kGrid - 1) * p.splitK + p.lastKCoreElements == request.k);
                    REQUIRE(p.lastKCoreElements > 0 && p.lastKCoreElements <= p.splitK);
                    REQUIRE(p.splitK % p.partialGroupK == 0);
                    REQUIRE(p.partialGroupK <= 1024);
                    REQUIRE(p.partialGroupK % p.l1K == 0);
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
                        uint32_t r = p.mTile / 2;
                        while (r && ProducerUbBytes(r, d, p.l1K, mid, p.mTile / 2) > 248 * 1024) { --r; }
                        REQUIRE(r > 0);
                        REQUIRE(ProducerUbBytes(r, d, p.l1K, mid, p.mTile / 2) <= 248 * 1024);
                        REQUIRE(r == p.mTile / 2 || ProducerUbBytes(r + 1, d, p.l1K, mid, p.mTile / 2) > 248 * 1024);
                        const uint32_t reduceRows = ReductionRows(p.mTile / 2, p.l1K, p.partialGroupCount, p.fixpipeRowStride, mid);
                        REQUIRE(reduceRows > 0 && reduceRows <= p.mTile / 2);
                        REQUIRE(EpilogueUbBytes(d, p.partialGroupCount, p.fixpipeRowStride, reduceRows, p.kGrid) <= 248 * 1024);
                        // Overlapped four-op epilogue aliases the FP32 stage tile only after its transfer completes.
                        REQUIRE(mid || uint64_t(reduceRows) * p.partialGroupCount * p.fixpipeRowStride <= (p.mTile / 2) * p.l1K);
                        REQUIRE(ProjectionUbBytes(4 * d, p.l1K, p.mTile / 2) <= 248 * 1024);
                        // Four-op reuses its Post output allocation for a packed projection half-tile.
                        REQUIRE(mid || p.kGrid != 1 || uint64_t(r) * 4 * d >= (p.mTile / 2) * p.l1K);
                    }
                }
            }
        }
    }
    CubePlanRequest invalid;
    // A complete second input row fits the target M96 plan, but must fall back for a wider K-L1 tile.
    REQUIRE(ProducerUbBytes(1, 5120, 256, false, 48, true) <= 248U * 1024U);
    REQUIRE(ProducerUbBytes(1, 5120, 1024, false, 16, true) > 248U * 1024U);
    REQUIRE(ProducerUbBytes(1, 5120, 1024, false, 16, false) <= 248U * 1024U);
    CubePlan unused;
    REQUIRE(!BuildCubePlan(invalid, unused));
    return 0;
}

#ifndef MEGA_MHC_TEST_DLL
/** Return zero only when every independent plan invariant passed. */
int main() { return run_plan_tests(); }
#endif
