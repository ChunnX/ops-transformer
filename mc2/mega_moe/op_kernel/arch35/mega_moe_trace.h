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
 * \brief 仅供诊断构建使用的 A4W4 Wave 轻量关键路径打点。
 */

#ifndef MEGA_MOE_TRACE_H
#define MEGA_MOE_TRACE_H

#include "kernel_operator.h"
#include "common/mega_moe_utils.h"

#ifndef MEGA_MOE_ENABLE_A4W4_WAVE_TRACE
#define MEGA_MOE_ENABLE_A4W4_WAVE_TRACE 1
#endif

#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0 && MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 1
#error "MEGA_MOE_ENABLE_A4W4_WAVE_TRACE must be 0 or 1"
#endif

namespace MegaMoeImpl {

#if MEGA_MOE_ENABLE_A4W4_WAVE_TRACE != 0

constexpr uint32_t A4W4_WAVE_TRACE_MAGIC = 0x41345452U; // "A4TR"
constexpr uint32_t A4W4_WAVE_TRACE_VERSION = 3U;
constexpr uint32_t A4W4_WAVE_TRACE_ROLE_COUNT = 3U;
constexpr uint32_t A4W4_WAVE_TRACE_HEADER_WORDS = 24U;
constexpr uint32_t A4W4_WAVE_TRACE_RECORD_HEADER_WORDS = 4U;
constexpr uint32_t A4W4_WAVE_TRACE_EVENT_WORDS = 9U;
constexpr uint32_t A4W4_WAVE_TRACE_MAX_EVENTS_PER_RECORD = 256U;
constexpr uint64_t A4W4_WAVE_TRACE_OUTPUT_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t A4W4_WAVE_TRACE_SCRATCH_BYTES = 50ULL * 1024ULL * 1024ULL;

enum class A4W4WaveTraceRole : uint32_t {
    AIC = 0U,
    AIV0 = 1U,
    AIV1 = 2U,
};

enum class A4W4WaveTraceEvent : uint32_t {
    INVALID = 0U,
    DISPATCH = 1U,
    GMM1 = 2U,
    GMM1_PROLOGUE = 3U,
    ACTIVATION = 4U,
    GMM2 = 5U,
    GMM2_PROLOGUE = 6U,
    COMBINE = 7U,
    WAIT_FOR_DISPATCH = 8U,
    WAIT_FOR_GMM1 = 9U,
    WAIT_FOR_ACTIVATION = 10U,
    WAIT_FOR_GMM2 = 11U,
};

/*
 * 每个 AIC/AIV0/AIV1 角色拥有一条独立记录，事件先写入预留 workspace，算子结束后统一覆盖到 y。
 * 这种诊断构建会破坏 y 的精度输出；header 最后发布，解析器只有看到 magic 才接受数据。
 */
struct A4W4WaveTrace {
    __gm__ uint32_t *scratchRecord = nullptr;
    __gm__ uint32_t *traceOutput = nullptr;
    uint32_t coreCount = 0U;
    uint32_t coreIndex = 0U;
    uint32_t role = 0U;
    uint32_t eventCapacity = 0U;
    uint32_t eventCount = 0U;
    uint32_t processBegin = 0U;
    uint32_t bs = 0U;
    uint32_t h = 0U;
    uint32_t hiddenDim = 0U;
    uint32_t topK = 0U;
    uint32_t moeExpertCount = 0U;
    uint32_t worldSize = 0U;
    uint32_t waveMGroupTarget = 0U;
    CriticalPathTraceEventWriter eventWriter{};
    bool enabled = false;
    uint32_t truncated = 0U;

    static __aicore__ inline uint32_t Now()
    {
        uint32_t cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
        return cycle == 0U ? 1U : cycle;
    }

    __aicore__ inline uint32_t RecordWords() const
    {
        return A4W4_WAVE_TRACE_RECORD_HEADER_WORDS + eventCapacity * A4W4_WAVE_TRACE_EVENT_WORDS;
    }

    __aicore__ inline void Init(GM_ADDR workspaceBase, int64_t workspaceBytes, GM_ADDR outputBase, uint64_t outputBytes,
                                uint32_t logicalCoreCount, A4W4WaveTraceRole traceRole, uint32_t logicalCoreIndex,
                                uint32_t tokenCount, uint32_t hiddenSize, uint32_t intermediateSize, uint32_t topKValue,
                                uint32_t expertCount, uint32_t epWorldSize)
    {
        coreCount = logicalCoreCount;
        coreIndex = logicalCoreIndex;
        role = static_cast<uint32_t>(traceRole);
        bs = tokenCount;
        h = hiddenSize;
        hiddenDim = intermediateSize;
        topK = topKValue;
        moeExpertCount = expertCount;
        worldSize = epWorldSize;

        const uint32_t recordCount = A4W4_WAVE_TRACE_ROLE_COUNT * coreCount;
        const uint64_t outputWords = outputBytes / sizeof(uint32_t);
        const uint64_t fixedOutputWords =
            A4W4_WAVE_TRACE_HEADER_WORDS + static_cast<uint64_t>(recordCount) * A4W4_WAVE_TRACE_RECORD_HEADER_WORDS;
        const uint64_t outputEventCapacity =
            outputWords > fixedOutputWords ?
                (outputWords - fixedOutputWords) / (static_cast<uint64_t>(recordCount) * A4W4_WAVE_TRACE_EVENT_WORDS) :
                0U;
        const uint64_t scratchFixedWords = static_cast<uint64_t>(recordCount) * A4W4_WAVE_TRACE_RECORD_HEADER_WORDS;
        const uint64_t scratchWords = A4W4_WAVE_TRACE_SCRATCH_BYTES / sizeof(uint32_t);
        const uint64_t scratchEventCapacity =
            scratchWords > scratchFixedWords ? (scratchWords - scratchFixedWords) /
                                                   (static_cast<uint64_t>(recordCount) * A4W4_WAVE_TRACE_EVENT_WORDS) :
                                               0U;
        uint64_t capacity = outputEventCapacity < scratchEventCapacity ? outputEventCapacity : scratchEventCapacity;
        capacity = capacity < A4W4_WAVE_TRACE_MAX_EVENTS_PER_RECORD ? capacity : A4W4_WAVE_TRACE_MAX_EVENTS_PER_RECORD;
        eventCapacity = static_cast<uint32_t>(capacity);
        enabled = workspaceBase != nullptr && outputBase != nullptr && coreCount != 0U && coreIndex < coreCount &&
                  eventCapacity != 0U && workspaceBytes >= 0;
        if (!enabled) {
            return;
        }

        const uint32_t recordIndex = role * coreCount + coreIndex;
        const uint32_t recordWords = RecordWords();
        __gm__ uint32_t *scratchBase = reinterpret_cast<__gm__ uint32_t *>(workspaceBase + workspaceBytes);
        scratchRecord = scratchBase + static_cast<uint64_t>(recordIndex) * recordWords;
        traceOutput = reinterpret_cast<__gm__ uint32_t *>(outputBase);
        eventWriter.record = scratchRecord;
        eventWriter.eventCount = &eventCount;
        eventWriter.truncated = &truncated;
        eventWriter.eventCapacity = eventCapacity;
        eventWriter.recordHeaderWords = A4W4_WAVE_TRACE_RECORD_HEADER_WORDS;
        eventWriter.eventWords = A4W4_WAVE_TRACE_EVENT_WORDS;
        processBegin = Now();
    }

    __aicore__ inline void SetWaveMGroupTarget(uint32_t target)
    {
        waveMGroupTarget = target;
    }

    __aicore__ inline void Record(A4W4WaveTraceEvent event, uint32_t waveIndex, uint32_t expertIndex,
                                  uint32_t tokenStartIndexInExpert, uint32_t tokenCount, uint32_t mGroupCount,
                                  uint32_t beginCycle, uint32_t endCycle)
    {
        eventWriter.Record(static_cast<uint32_t>(event), waveIndex, expertIndex, tokenStartIndexInExpert, tokenCount,
                           mGroupCount, static_cast<uint32_t>(A4W4WaveTraceEvent::INVALID), beginCycle, endCycle);
    }

    __aicore__ inline CriticalPathWaitTraceContext CreateWaitTraceContext(A4W4WaveTraceEvent waitEvent,
                                                                          A4W4WaveTraceEvent parentEvent,
                                                                          uint32_t waveIndex, uint32_t expertIndex,
                                                                          uint32_t tokenStartIndexInExpert,
                                                                          uint32_t tokenCount, uint32_t mGroupCount)
    {
        CriticalPathWaitTraceContext context{};
        context.writer = eventWriter;
        context.waitEventType = static_cast<uint32_t>(waitEvent);
        context.parentEventType = static_cast<uint32_t>(parentEvent);
        context.iterationIndex = waveIndex;
        context.logicalUnitIndex = expertIndex;
        context.unitStart = tokenStartIndexInExpert;
        context.unitCount = tokenCount;
        context.workCount = mGroupCount;
        return context;
    }

    __aicore__ inline void Finish()
    {
        if (!enabled) {
            return;
        }
        const uint32_t processEnd = Now();

        // 先等正常 y 写回完成，再用 trace 覆盖 y；随后等待所有角色完成记录复制，最后发布 magic。
        AscendC::SyncAll<false>();
        AscendC::WriteGmByPassDCache(scratchRecord + 0U, processBegin);
        AscendC::WriteGmByPassDCache(scratchRecord + 1U, processEnd);
        AscendC::WriteGmByPassDCache(scratchRecord + 2U, eventCount);
        AscendC::WriteGmByPassDCache(scratchRecord + 3U, truncated);

        const uint32_t recordIndex = role * coreCount + coreIndex;
        const uint32_t recordWords = RecordWords();
        __gm__ uint32_t *outputRecord =
            traceOutput + A4W4_WAVE_TRACE_HEADER_WORDS + static_cast<uint64_t>(recordIndex) * recordWords;
        for (uint32_t word = 0U; word < A4W4_WAVE_TRACE_RECORD_HEADER_WORDS + eventCount * A4W4_WAVE_TRACE_EVENT_WORDS;
             ++word) {
            uint32_t value = AscendC::ReadGmByPassDCache(scratchRecord + word);
            AscendC::WriteGmByPassDCache(outputRecord + word, value);
        }

        AscendC::SyncAll<false>();
        if (role != static_cast<uint32_t>(A4W4WaveTraceRole::AIC) || coreIndex != 0U) {
            return;
        }
        AscendC::WriteGmByPassDCache(traceOutput + 1U, A4W4_WAVE_TRACE_VERSION);
        AscendC::WriteGmByPassDCache(traceOutput + 2U, A4W4_WAVE_TRACE_HEADER_WORDS);
        AscendC::WriteGmByPassDCache(traceOutput + 3U, recordWords);
        AscendC::WriteGmByPassDCache(traceOutput + 4U, A4W4_WAVE_TRACE_EVENT_WORDS);
        AscendC::WriteGmByPassDCache(traceOutput + 5U, A4W4_WAVE_TRACE_ROLE_COUNT * coreCount);
        AscendC::WriteGmByPassDCache(traceOutput + 6U, coreCount);
        AscendC::WriteGmByPassDCache(traceOutput + 7U, eventCapacity);
        AscendC::WriteGmByPassDCache(traceOutput + 8U, bs);
        AscendC::WriteGmByPassDCache(traceOutput + 9U, h);
        AscendC::WriteGmByPassDCache(traceOutput + 10U, hiddenDim);
        AscendC::WriteGmByPassDCache(traceOutput + 11U, topK);
        AscendC::WriteGmByPassDCache(traceOutput + 12U, moeExpertCount);
        AscendC::WriteGmByPassDCache(traceOutput + 13U, worldSize);
        AscendC::WriteGmByPassDCache(traceOutput + 14U, waveMGroupTarget);
        AscendC::WriteGmByPassDCache(traceOutput + 15U, 32U);
        for (uint32_t word = 16U; word < A4W4_WAVE_TRACE_HEADER_WORDS; ++word) {
            AscendC::WriteGmByPassDCache(traceOutput + word, 0U);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WriteGmByPassDCache(traceOutput + 0U, A4W4_WAVE_TRACE_MAGIC);
    }
};

#endif

} // namespace MegaMoeImpl

#endif
