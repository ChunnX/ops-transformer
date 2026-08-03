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
 * \file mega_moe_trace.h
 * \brief Compile-time-contained diagnostic trace contract for the MTE A8W8 MegaMoeWave path.
 */

#ifndef MEGA_MOE_TRACE_H
#define MEGA_MOE_TRACE_H

#include "kernel_operator.h"

#ifndef MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE
#define MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE 1
#endif

#if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0 && MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 1
#error "MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE must be 0 or 1"
#endif

namespace MegaMoeImpl {

#if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0

static constexpr bool CRITICAL_PATH_TRACE_ENABLED =
    static_cast<bool>(MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE);
static constexpr uint32_t CRITICAL_PATH_TRACE_MAGIC = 0x43505452U;
static constexpr uint32_t CRITICAL_PATH_TRACE_PROTOCOL_VERSION = 5U;
static constexpr uint32_t CRITICAL_PATH_TRACE_SCHEMA_ID = 0xD32D8A49U;
static constexpr uint32_t CRITICAL_PATH_TRACE_HEADER_WORDS = 30U;
static constexpr uint32_t CRITICAL_PATH_TRACE_RECORD_WORDS = 18U;
static constexpr uint32_t CRITICAL_PATH_TRACE_ROLE_COUNT = 3U;
static constexpr uint32_t CRITICAL_PATH_TRACE_MAX_WAVES = 4U;
static constexpr uint32_t CRITICAL_PATH_TRACE_PATH_MTE_A8W8_WAVE = 1U;
static constexpr uint32_t CRITICAL_PATH_TRACE_FLAGS = 0xFFU;

// Word-wise FNV-1a. The repository runner recomputes this from checked header
// fields, so a stale installed kernel or a different path/config fails closed.
__aicore__ inline uint32_t CriticalPathTraceHashWord(uint32_t hash, uint32_t value)
{
    return (hash ^ value) * 16777619U;
}

__aicore__ inline uint32_t CriticalPathTraceConfigHash(
    uint32_t pathId, uint32_t bs, uint32_t h, uint32_t hiddenDim, uint32_t topK,
    uint32_t moeExpertsPerRank, uint32_t sharedExpertCount, uint32_t expertsPerWave,
    uint32_t worldSize, uint32_t aicCount, uint32_t quantMode, uint32_t combineQuantMode,
    uint32_t topkWeightsPrefetch, uint32_t weight1Interleaved)
{
    uint32_t hash = 2166136261U;
    hash = CriticalPathTraceHashWord(hash, pathId);
    hash = CriticalPathTraceHashWord(hash, bs);
    hash = CriticalPathTraceHashWord(hash, h);
    hash = CriticalPathTraceHashWord(hash, hiddenDim);
    hash = CriticalPathTraceHashWord(hash, topK);
    hash = CriticalPathTraceHashWord(hash, moeExpertsPerRank);
    hash = CriticalPathTraceHashWord(hash, sharedExpertCount);
    hash = CriticalPathTraceHashWord(hash, expertsPerWave);
    hash = CriticalPathTraceHashWord(hash, worldSize);
    hash = CriticalPathTraceHashWord(hash, aicCount);
    hash = CriticalPathTraceHashWord(hash, quantMode);
    hash = CriticalPathTraceHashWord(hash, combineQuantMode);
    hash = CriticalPathTraceHashWord(hash, topkWeightsPrefetch);
    hash = CriticalPathTraceHashWord(hash, weight1Interleaved);
    return hash;
}

// Role-dependent compact slot:
//   AIC  stage0=GMM1,     stage1=GMM2
//   AIV0 stage0=SwiGLU,   stage1=unused
//   AIV1 stage0=Dispatch, stage1=Combine
struct CriticalPathWaveTrace {
    uint32_t stage0Begin = 0U;
    uint32_t stage0End = 0U;
    uint32_t stage1Begin = 0U;
    uint32_t stage1End = 0U;
};

struct CriticalPathTrace {
    uint32_t processBegin = 0U;
    uint32_t workEnd = 0U;
    uint32_t activeWave = 0U;
    CriticalPathWaveTrace wave0{};
    CriticalPathWaveTrace wave1{};
    CriticalPathWaveTrace wave2{};
    CriticalPathWaveTrace wave3{};

    static __aicore__ inline uint32_t Now()
    {
        const uint32_t cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
        return cycle == 0U ? 1U : cycle;
    }

    __aicore__ inline CriticalPathWaveTrace *Wave(uint32_t wave)
    {
        switch (wave) {
            case 0U:
                return &wave0;
            case 1U:
                return &wave1;
            case 2U:
                return &wave2;
            case 3U:
                return &wave3;
            default:
                return nullptr;
        }
    }

    __aicore__ inline void SetActiveWave(uint32_t wave)
    {
        activeWave = wave;
    }

    __aicore__ inline bool NeedsStage0Begin()
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage0Begin == 0U;
    }

    __aicore__ inline bool NeedsStage1Begin()
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage1Begin == 0U;
    }

    __aicore__ inline bool HasStage0Work()
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage0Begin != 0U;
    }

    __aicore__ inline bool HasStage1Work()
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage1Begin != 0U;
    }

    __aicore__ inline void BeginStage0(uint32_t value)
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage0Begin == 0U) {
            slot->stage0Begin = value;
        }
    }

    __aicore__ inline void EndStage0(uint32_t value)
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage0Begin != 0U) {
            slot->stage0End = value;
        }
    }

    __aicore__ inline void BeginStage1(uint32_t value)
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage1Begin == 0U) {
            slot->stage1Begin = value;
        }
    }

    __aicore__ inline void EndStage1(uint32_t value)
    {
        CriticalPathWaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage1Begin != 0U) {
            slot->stage1End = value;
        }
    }

    __aicore__ inline void BeginDispatch(uint32_t wave, uint32_t value)
    {
        CriticalPathWaveTrace *slot = Wave(wave);
        if (slot != nullptr && slot->stage0Begin == 0U) {
            slot->stage0Begin = value;
        }
    }

    __aicore__ inline void EndDispatch(uint32_t wave, uint32_t value)
    {
        CriticalPathWaveTrace *slot = Wave(wave);
        if (slot != nullptr && slot->stage0Begin != 0U) {
            slot->stage0End = value;
        }
    }

    static __aicore__ inline void WriteWave(
        __gm__ uint32_t *record, uint32_t word, const CriticalPathWaveTrace &wave)
    {
        AscendC::WriteGmByPassDCache(record + word + 0U, wave.stage0Begin);
        AscendC::WriteGmByPassDCache(record + word + 1U, wave.stage0End);
        AscendC::WriteGmByPassDCache(record + word + 2U, wave.stage1Begin);
        AscendC::WriteGmByPassDCache(record + word + 3U, wave.stage1End);
    }

    __aicore__ inline void WriteRecord(__gm__ uint32_t *record) const
    {
        AscendC::WriteGmByPassDCache(record + 0U, processBegin);
        AscendC::WriteGmByPassDCache(record + 1U, workEnd);
        WriteWave(record, 2U, wave0);
        WriteWave(record, 6U, wave1);
        WriteWave(record, 10U, wave2);
        WriteWave(record, 14U, wave3);
    }
};

#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TRACE_H
