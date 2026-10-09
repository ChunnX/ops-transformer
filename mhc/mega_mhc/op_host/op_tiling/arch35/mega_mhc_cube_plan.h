/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_CUBE_PLAN_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_CUBE_PLAN_H

#include <cstdint>

namespace optiling::mega_mhc {

enum class ProjectionMode : uint32_t {
    FP32 = 0,
    HF32 = 1
};

enum class ScheduleMode : uint32_t {
    MK = 0,
    ND = 1,
    BS = 2
};

struct CubePlanRequest {
    ScheduleMode schedule = ScheduleMode::MK;
    uint64_t rows = 0;
    uint64_t k = 0;
    uint64_t q = 0;
    uint64_t aicCount = 0;
    uint64_t aivCount = 0;
    uint64_t l1ABytes = 0;
    uint64_t l1BBytes = 0;
    uint64_t l0ABytes = 0;
    uint64_t l0BBytes = 0;
    uint64_t l0CBytes = 0;
    bool preferWideTile = false;
    bool wavePipeline = false;
    bool preferCompleteWaves = true;
    ProjectionMode mode = ProjectionMode::FP32;
};

struct CubePlan {
    uint32_t activeAic = 0;
    uint32_t activeAiv = 0;
    uint32_t mGrid = 0;
    uint32_t kGrid = 0;
    uint32_t mWaveCount = 0;
    uint32_t rowsPerWave = 0;
    uint32_t lastWaveMGrid = 0;
    uint32_t lastWaveRows = 0;
    uint32_t mTile = 0;
    uint32_t splitK = 0;
    uint32_t largeKCoreCount = 0;
    uint32_t shortSplitK = 0;
    uint32_t shortKCoreL1FullLoops = 0;
    uint32_t l1K = 0;
    uint32_t l1KFullLoops = 0;
    uint32_t l1KTail = 0;
    uint32_t lastKCoreL1FullLoops = 0;
    uint32_t lastKCoreL1Tail = 0;
    uint32_t lastMCoreRows = 0;
    uint32_t lastKCoreElements = 0;
    uint32_t mL0FullLoops = 0;
    uint32_t mL0TailRows = 0;
    uint32_t lastMCoreL0FullLoops = 0;
    uint32_t lastMCoreL0TailRows = 0;
    uint32_t loadMStride = 0;
    uint32_t loadNStride = 0;
    uint32_t fullL0LoadKStep = 0;
    uint32_t l0TailLoadKStep = 0;
    uint32_t tailL1TailLoadKStep = 0;
    uint32_t lastCoreTailLoadKStep = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t nzM = 0;
    uint32_t nzN = 0;
    uint32_t l1ASlotElements = 0;
    uint32_t l1BOffsetElements = 0;
    uint32_t l1BSlotElements = 0;
    uint32_t l0ASlotElements = 0;
    uint32_t l0BSlotElements = 0;
    uint32_t l0CSlotElements = 0;
    uint32_t fixpipeRowStride = 0;
    uint32_t fullMmadKLoopCount = 0;
    uint32_t l0KLoops = 0;
    uint32_t l0KTail = 0;
    uint32_t tailL1L0FullLoops = 0;
    uint32_t tailL1L0Tail = 0;
    uint32_t lastCoreTailL1L0FullLoops = 0;
    uint32_t lastCoreTailL1L0Tail = 0;
    uint32_t fullMmadKStep = 0;
    uint32_t l0TailMmadKStep = 0;
    uint32_t l0TailMmadKLoopCount = 0;
    uint32_t l0TailMmadFinalKStep = 0;
    uint32_t tailL1TailMmadKStep = 0;
    uint32_t tailL1TailMmadKLoopCount = 0;
    uint32_t tailL1TailMmadFinalKStep = 0;
    uint32_t lastCoreTailMmadKStep = 0;
    uint32_t lastCoreTailMmadKLoopCount = 0;
    uint32_t lastCoreTailMmadFinalKStep = 0;
    uint32_t partialGroupK = 0;
    uint32_t partialGroupCount = 0;
};

/*
 * Plans the M/K launch grid and all L1/L0/Mmad tiles on the host.
 * Returns false when no candidate satisfies the supplied per-core capacities.
 * Every dimension is in elements; capacity and workspace fields are in bytes.
 */
bool BuildCubePlan(const CubePlanRequest &request, CubePlan &plan);

} // namespace optiling::mega_mhc

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_CUBE_PLAN_H
