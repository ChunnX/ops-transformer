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

#ifndef MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE
#define MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE 1
#endif

#if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0 && \
    MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 1
#error "MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE must be 0 or 1"
#endif

namespace MegaMoeImpl {

#if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0

static constexpr bool CRITICAL_PATH_TRACE_ENABLED =
    static_cast<bool>(MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE);
static constexpr bool CRITICAL_PATH_FINE_TRACE_ENABLED =
    static_cast<bool>(MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE);
static constexpr uint32_t CRITICAL_PATH_TRACE_MAGIC = 0x43505452U;
#if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
static constexpr uint32_t CRITICAL_PATH_TRACE_PROTOCOL_VERSION = 7U;
static constexpr uint32_t CRITICAL_PATH_TRACE_SCHEMA_ID = 0x3EB1C6B3U;
static constexpr uint32_t CRITICAL_PATH_TRACE_FLAGS = 0x3FFU;
#else
static constexpr uint32_t CRITICAL_PATH_TRACE_PROTOCOL_VERSION = 5U;
static constexpr uint32_t CRITICAL_PATH_TRACE_SCHEMA_ID = 0xD32D8A49U;
static constexpr uint32_t CRITICAL_PATH_TRACE_FLAGS = 0xFFU;
#endif
static constexpr uint32_t CRITICAL_PATH_TRACE_HEADER_WORDS = 30U;
static constexpr uint32_t CRITICAL_PATH_TRACE_ROLE_COUNT = 3U;
static constexpr uint32_t CRITICAL_PATH_TRACE_COARSE_RECORD_WORDS = 18U;
static constexpr uint32_t CRITICAL_PATH_TRACE_COARSE_MAX_WAVES = 4U;
static constexpr uint32_t CRITICAL_PATH_TRACE_DYNAMIC_RECORD_BASE_WORDS = 2U;
static constexpr uint32_t CRITICAL_PATH_TRACE_DYNAMIC_WORDS_PER_EXPERT = 4U;
static constexpr uint32_t CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT = 3U;
static constexpr uint32_t CRITICAL_PATH_TRACE_SCRATCH_STAGE_COUNT = 2U;
// Arch35 tiling always appends this diagnostic-safe reserve after WorkspaceInfo::workspaceSize.
// Keep synchronized with RESERVED_WORKSPACE_SIZE in op_host/op_tiling/arch35/mega_moe_tiling.cpp.
static constexpr uint64_t CRITICAL_PATH_TRACE_SCRATCH_RESERVED_BYTES = 50ULL * 1024ULL * 1024ULL;
static constexpr uint32_t CRITICAL_PATH_TRACE_PATH_MTE_A8W8_WAVE = 1U;

__aicore__ inline uint32_t CriticalPathTraceRecordWords(uint32_t expertCount)
{
#if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
    return CRITICAL_PATH_TRACE_DYNAMIC_RECORD_BASE_WORDS +
           CRITICAL_PATH_TRACE_DYNAMIC_WORDS_PER_EXPERT * expertCount;
#else
    (void)expertCount;
    return CRITICAL_PATH_TRACE_COARSE_RECORD_WORDS;
#endif
}

__aicore__ inline uint32_t CriticalPathTraceScratchRecordWords(uint32_t expertCount)
{
#if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
    return CRITICAL_PATH_TRACE_SCRATCH_STAGE_COUNT *
           CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT * expertCount;
#else
    (void)expertCount;
    return 0U;
#endif
}

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

struct CriticalPathTrace {
    uint32_t processBegin = 0U;
    uint32_t workEnd = 0U;
#if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
    static constexpr uint32_t INVALID_OCCURRENCE = 0xFFFFFFFFU;
    __gm__ uint32_t *scratchRecord = nullptr;
    uint32_t occurrenceCapacity = 0U;
    uint32_t stage0ActiveOccurrence = INVALID_OCCURRENCE;
    uint32_t stage1ActiveOccurrence = INVALID_OCCURRENCE;
    uint32_t stage0ActiveBegin = 0U;
    uint32_t stage1ActiveBegin = 0U;
    uint32_t stage0CapturedCount = 0U;
    uint32_t stage1CapturedCount = 0U;
#else
    // Coarse v5 keeps the original four named wave slots and exact wire layout.
    struct WaveTrace {
        uint32_t stage0Begin = 0U;
        uint32_t stage0End = 0U;
        uint32_t stage1Begin = 0U;
        uint32_t stage1End = 0U;
    };
    uint32_t activeWave = 0U;
    WaveTrace wave0{};
    WaveTrace wave1{};
    WaveTrace wave2{};
    WaveTrace wave3{};
#endif

    static __aicore__ inline uint32_t Now()
    {
        const uint32_t cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
        return cycle == 0U ? 1U : cycle;
    }

#if MEGA_MOE_ENABLE_FINE_GRAIN_CRITICAL_PATH_TRACE != 0
    __aicore__ inline void Init(__gm__ uint32_t *scratch, uint32_t expertCount)
    {
        scratchRecord = scratch;
        occurrenceCapacity = expertCount;
    }

    __aicore__ inline void PrepareExpertStage0(uint32_t occurrence)
    {
        stage0ActiveOccurrence = occurrence;
        stage0ActiveBegin = 0U;
    }

    __aicore__ inline void PrepareExpertStage1(uint32_t occurrence)
    {
        stage1ActiveOccurrence = occurrence;
        stage1ActiveBegin = 0U;
    }

    __aicore__ inline bool NeedsExpertStage0Begin()
    {
        return stage0ActiveOccurrence != INVALID_OCCURRENCE && stage0ActiveBegin == 0U;
    }

    __aicore__ inline bool NeedsExpertStage1Begin()
    {
        return stage1ActiveOccurrence != INVALID_OCCURRENCE && stage1ActiveBegin == 0U;
    }

    __aicore__ inline void BeginExpertStage0(uint32_t value)
    {
        if (stage0ActiveOccurrence != INVALID_OCCURRENCE && stage0ActiveBegin == 0U) {
            stage0ActiveBegin = value;
        }
    }

    __aicore__ inline void EndExpertStage0(uint32_t value)
    {
        if (scratchRecord != nullptr && stage0ActiveOccurrence < occurrenceCapacity &&
            stage0ActiveBegin != 0U && stage0CapturedCount < occurrenceCapacity) {
            __gm__ uint32_t *entry =
                scratchRecord + stage0CapturedCount * CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT;
            AscendC::WriteGmByPassDCache(entry + 0U, stage0ActiveOccurrence);
            AscendC::WriteGmByPassDCache(entry + 1U, stage0ActiveBegin);
            AscendC::WriteGmByPassDCache(entry + 2U, value);
            ++stage0CapturedCount;
        }
    }

    __aicore__ inline bool HasExpertStage0Work() const
    {
        return stage0ActiveOccurrence != INVALID_OCCURRENCE && stage0ActiveBegin != 0U;
    }

    __aicore__ inline void FinishExpertStage0()
    {
        stage0ActiveOccurrence = INVALID_OCCURRENCE;
        stage0ActiveBegin = 0U;
    }

    __aicore__ inline void BeginExpertStage1(uint32_t value)
    {
        if (stage1ActiveOccurrence != INVALID_OCCURRENCE && stage1ActiveBegin == 0U) {
            stage1ActiveBegin = value;
        }
    }

    __aicore__ inline void EndExpertStage1(uint32_t value)
    {
        if (scratchRecord != nullptr && stage1ActiveOccurrence < occurrenceCapacity &&
            stage1ActiveBegin != 0U && stage1CapturedCount < occurrenceCapacity) {
            __gm__ uint32_t *entry =
                scratchRecord + occurrenceCapacity * CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT +
                stage1CapturedCount * CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT;
            AscendC::WriteGmByPassDCache(entry + 0U, stage1ActiveOccurrence);
            AscendC::WriteGmByPassDCache(entry + 1U, stage1ActiveBegin);
            AscendC::WriteGmByPassDCache(entry + 2U, value);
            ++stage1CapturedCount;
        }
    }

    __aicore__ inline bool HasExpertStage1Work() const
    {
        return stage1ActiveOccurrence != INVALID_OCCURRENCE && stage1ActiveBegin != 0U;
    }

    __aicore__ inline void FinishExpertStage1()
    {
        stage1ActiveOccurrence = INVALID_OCCURRENCE;
        stage1ActiveBegin = 0U;
    }

    __aicore__ inline void WriteScratchStage(
        __gm__ uint32_t *record, __gm__ uint32_t *stageScratch, uint32_t count, uint32_t beginWord) const
    {
        for (uint32_t index = 0U; index < count; ++index) {
            __gm__ uint32_t *entry =
                stageScratch + index * CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT;
            const uint32_t occurrence = AscendC::ReadGmByPassDCache(entry + 0U);
            const uint32_t begin = AscendC::ReadGmByPassDCache(entry + 1U);
            const uint32_t end = AscendC::ReadGmByPassDCache(entry + 2U);
            if (occurrence >= occurrenceCapacity) {
                continue;
            }
            const uint32_t word = CRITICAL_PATH_TRACE_DYNAMIC_RECORD_BASE_WORDS +
                                  occurrence * CRITICAL_PATH_TRACE_DYNAMIC_WORDS_PER_EXPERT + beginWord;
            AscendC::WriteGmByPassDCache(record + word + 0U, begin);
            AscendC::WriteGmByPassDCache(record + word + 1U, end);
        }
    }

    __aicore__ inline void WriteRecord(__gm__ uint32_t *record) const
    {
        const uint32_t recordWords = CriticalPathTraceRecordWords(occurrenceCapacity);
        for (uint32_t word = 0U; word < recordWords; ++word) {
            AscendC::WriteGmByPassDCache(record + word, 0U);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WriteGmByPassDCache(record + 0U, processBegin);
        AscendC::WriteGmByPassDCache(record + 1U, workEnd);
        WriteScratchStage(record, scratchRecord, stage0CapturedCount, 0U);
        WriteScratchStage(
            record, scratchRecord + occurrenceCapacity * CRITICAL_PATH_TRACE_SCRATCH_WORDS_PER_STAGE_EXPERT,
            stage1CapturedCount, 2U);
    }
#else
    __aicore__ inline WaveTrace *Wave(uint32_t wave)
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
        WaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage0Begin == 0U;
    }

    __aicore__ inline bool NeedsStage1Begin()
    {
        WaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage1Begin == 0U;
    }

    __aicore__ inline bool HasStage0Work()
    {
        WaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage0Begin != 0U;
    }

    __aicore__ inline bool HasStage1Work()
    {
        WaveTrace *slot = Wave(activeWave);
        return slot != nullptr && slot->stage1Begin != 0U;
    }

    __aicore__ inline void BeginStage0(uint32_t value)
    {
        WaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage0Begin == 0U) {
            slot->stage0Begin = value;
        }
    }

    __aicore__ inline void EndStage0(uint32_t value)
    {
        WaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage0Begin != 0U) {
            slot->stage0End = value;
        }
    }

    __aicore__ inline void BeginStage1(uint32_t value)
    {
        WaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage1Begin == 0U) {
            slot->stage1Begin = value;
        }
    }

    __aicore__ inline void EndStage1(uint32_t value)
    {
        WaveTrace *slot = Wave(activeWave);
        if (slot != nullptr && slot->stage1Begin != 0U) {
            slot->stage1End = value;
        }
    }

    __aicore__ inline void BeginDispatch(uint32_t wave, uint32_t value)
    {
        WaveTrace *slot = Wave(wave);
        if (slot != nullptr && slot->stage0Begin == 0U) {
            slot->stage0Begin = value;
        }
    }

    __aicore__ inline void EndDispatch(uint32_t wave, uint32_t value)
    {
        WaveTrace *slot = Wave(wave);
        if (slot != nullptr && slot->stage0Begin != 0U) {
            slot->stage0End = value;
        }
    }

    static __aicore__ inline void WriteWave(
        __gm__ uint32_t *record, uint32_t word, const WaveTrace &wave)
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
#endif
};

#endif

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TRACE_H
