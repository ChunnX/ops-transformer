/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef MEGA_MOE_WAVE_TRACE_H
#define MEGA_MOE_WAVE_TRACE_H

#include "kernel_operator.h"

#ifndef MEGA_MOE_ENABLE_WAVE_TRACE
#define MEGA_MOE_ENABLE_WAVE_TRACE 1
#endif

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0 && MEGA_MOE_ENABLE_WAVE_TRACE != 1
#error "MEGA_MOE_ENABLE_WAVE_TRACE must be 0 or 1"
#endif

namespace MegaMoeImpl {

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0

constexpr uint32_t MEGA_MOE_WAVE_TRACE_MAGIC = 0x41575452U; // "AWTR"
constexpr uint32_t MEGA_MOE_WAVE_TRACE_VERSION = 3U;
constexpr uint32_t MEGA_MOE_WAVE_TRACE_ROLE_COUNT = 3U;
constexpr uint32_t MEGA_MOE_WAVE_TRACE_HEADER_WORDS = 24U;
constexpr uint32_t MEGA_MOE_WAVE_TRACE_RECORD_HEADER_WORDS = 4U;
constexpr uint32_t MEGA_MOE_WAVE_TRACE_EVENT_WORDS = 9U;
// BS=4096 时单核可能跨越较多 expert slice；512 仍可被 BS=512 的 BF16 y 轻松承载。
constexpr uint32_t MEGA_MOE_WAVE_TRACE_MAX_EVENTS_PER_RECORD = 512U;
constexpr uint64_t MEGA_MOE_WAVE_TRACE_SCRATCH_BYTES = 50ULL * 1024ULL * 1024ULL;

enum class MegaMoeWaveTraceRole : uint32_t {
    AIC = 0U,
    AIV0 = 1U,
    AIV1 = 2U,
};

// Keep the validated MegaMoe v3 event ids so the skill-provided report tool can parse the capture directly.
enum class MegaMoeWaveTraceEvent : uint32_t {
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
    // Aggregated per-wave Dispatch transfer metrics. These are emitted as
    // synthetic child intervals so one event captures many short waits without
    // overflowing the trace record.
    DISPATCH_TRANSFER_PIPELINE = 12U,
    WAIT_DISPATCH_MTE3_REUSE = 13U,
    WAIT_DISPATCH_MTE3_DRAIN = 14U,
    WAIT_DISPATCH_READY_SYNC = 15U,
    // Aggregated per-wave Combine metrics. Their durations are exact sums;
    // their synthetic child intervals are not real timeline placement.
    COMBINE_META_INFO_COPY = 16U,
    COMBINE_ROW_PREPARE = 17U,
    COMBINE_MTE3_ISSUE = 18U,
    WAIT_COMBINE_MTE3_REUSE = 19U,
    WAIT_COMBINE_MTE3_DRAIN = 20U,
    // Aggregated per-wave Dispatch stage metrics. Their durations are exact
    // sums; their synthetic child intervals are not real timeline placement.
    DISPATCH_COUNT_PREP = 21U,
    DISPATCH_RANK_ROUND_PLAN = 22U,
    DISPATCH_MASK_LOAD_AND_GATHER = 23U,
    DISPATCH_READY_PUBLISH = 24U,
};

namespace MegaMoeWaveTraceDetail {

struct EventScope {
    uint32_t wave = 0U;
    uint32_t expert = 0U;
    uint32_t rowBegin = 0U;
    uint32_t rowCount = 0U;
    uint32_t mGroupCount = 0U;
};

struct EventWriter {
    __gm__ uint32_t *record = nullptr;
    uint32_t *eventCount = nullptr;
    uint32_t *truncated = nullptr;
    uint32_t eventCapacity = 0U;

    __aicore__ inline bool IsEnabled() const
    {
        return record != nullptr && eventCount != nullptr && truncated != nullptr;
    }

    __aicore__ inline void Record(uint32_t eventType, const EventScope &scope, uint32_t parentEventType,
                                  uint32_t beginCycle, uint32_t endCycle)
    {
        if (!IsEnabled()) {
            return;
        }
        if (*eventCount >= eventCapacity) {
            *truncated = 1U;
            return;
        }
        __gm__ uint32_t *dst = record + MEGA_MOE_WAVE_TRACE_RECORD_HEADER_WORDS +
            static_cast<uint64_t>(*eventCount) * MEGA_MOE_WAVE_TRACE_EVENT_WORDS;
        AscendC::WriteGmByPassDCache(dst + 0U, eventType);
        AscendC::WriteGmByPassDCache(dst + 1U, scope.wave);
        AscendC::WriteGmByPassDCache(dst + 2U, scope.expert);
        AscendC::WriteGmByPassDCache(dst + 3U, scope.rowBegin);
        AscendC::WriteGmByPassDCache(dst + 4U, scope.rowCount);
        AscendC::WriteGmByPassDCache(dst + 5U, scope.mGroupCount);
        AscendC::WriteGmByPassDCache(dst + 6U, parentEventType);
        AscendC::WriteGmByPassDCache(dst + 7U, beginCycle);
        AscendC::WriteGmByPassDCache(dst + 8U, endCycle);
        ++(*eventCount);
    }
};

struct WaitContext {
    EventWriter writer{};
    EventScope scope{};
    uint32_t waitEventType = 0U;
    uint32_t parentEventType = 0U;
};

__aicore__ inline uint32_t Now()
{
    uint32_t cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
    return cycle == 0U ? 1U : cycle;
}

__aicore__ inline uint32_t BeginWaitInterval(WaitContext *context)
{
    return context == nullptr || !context->writer.IsEnabled() ? 0U : Now();
}

__aicore__ inline void EndWaitInterval(WaitContext *context, uint32_t beginCycle)
{
    if (context == nullptr || !context->writer.IsEnabled()) {
        return;
    }
    context->writer.Record(context->waitEventType, context->scope, context->parentEventType,
                           beginCycle, Now());
}

} // namespace MegaMoeWaveTraceDetail

struct MegaMoeWaveTrace {
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
    uint32_t truncated = 0U;
    bool enabled = false;
    MegaMoeWaveTraceDetail::EventWriter writer{};

    __aicore__ inline uint32_t RecordWords() const
    {
        return MEGA_MOE_WAVE_TRACE_RECORD_HEADER_WORDS +
               eventCapacity * MEGA_MOE_WAVE_TRACE_EVENT_WORDS;
    }

    __aicore__ inline void Init(GM_ADDR workspaceBase, int64_t businessWorkspaceBytes,
                                GM_ADDR outputBase, uint64_t outputBytes,
                                uint32_t logicalCoreCount, MegaMoeWaveTraceRole traceRole,
                                uint32_t logicalCoreIndex, uint32_t tokenCount,
                                uint32_t hiddenSize, uint32_t intermediateSize,
                                uint32_t topKValue, uint32_t expertCount, uint32_t epWorldSize)
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

        if (workspaceBase == nullptr || outputBase == nullptr || businessWorkspaceBytes < 0 ||
            coreCount == 0U || coreIndex >= coreCount) {
            return;
        }

        const uint32_t recordCount = MEGA_MOE_WAVE_TRACE_ROLE_COUNT * coreCount;
        const uint64_t outputWords = outputBytes / sizeof(uint32_t);
        const uint64_t fixedOutputWords = MEGA_MOE_WAVE_TRACE_HEADER_WORDS +
            static_cast<uint64_t>(recordCount) * MEGA_MOE_WAVE_TRACE_RECORD_HEADER_WORDS;
        const uint64_t outputCapacity = outputWords > fixedOutputWords ?
            (outputWords - fixedOutputWords) /
                (static_cast<uint64_t>(recordCount) * MEGA_MOE_WAVE_TRACE_EVENT_WORDS) : 0U;
        const uint64_t scratchWords = MEGA_MOE_WAVE_TRACE_SCRATCH_BYTES / sizeof(uint32_t);
        const uint64_t scratchFixedWords =
            static_cast<uint64_t>(recordCount) * MEGA_MOE_WAVE_TRACE_RECORD_HEADER_WORDS;
        const uint64_t scratchCapacity = scratchWords > scratchFixedWords ?
            (scratchWords - scratchFixedWords) /
                (static_cast<uint64_t>(recordCount) * MEGA_MOE_WAVE_TRACE_EVENT_WORDS) : 0U;
        uint64_t capacity = outputCapacity < scratchCapacity ? outputCapacity : scratchCapacity;
        capacity = capacity < MEGA_MOE_WAVE_TRACE_MAX_EVENTS_PER_RECORD ?
            capacity : MEGA_MOE_WAVE_TRACE_MAX_EVENTS_PER_RECORD;
        eventCapacity = static_cast<uint32_t>(capacity);
        enabled = eventCapacity != 0U;
        if (!enabled) {
            return;
        }

        const uint32_t recordIndex = role * coreCount + coreIndex;
        const uint32_t recordWords = RecordWords();
        __gm__ uint32_t *scratchBase = reinterpret_cast<__gm__ uint32_t *>(
            workspaceBase + businessWorkspaceBytes);
        scratchRecord = scratchBase + static_cast<uint64_t>(recordIndex) * recordWords;
        traceOutput = reinterpret_cast<__gm__ uint32_t *>(outputBase);
        writer.record = scratchRecord;
        writer.eventCount = &eventCount;
        writer.truncated = &truncated;
        writer.eventCapacity = eventCapacity;
        processBegin = MegaMoeWaveTraceDetail::Now();
    }

    __aicore__ inline void SetWaveMGroupTarget(uint32_t target)
    {
        waveMGroupTarget = target;
    }

    __aicore__ inline void Record(MegaMoeWaveTraceEvent event,
                                  const MegaMoeWaveTraceDetail::EventScope &scope,
                                  uint32_t beginCycle, uint32_t endCycle)
    {
        writer.Record(static_cast<uint32_t>(event), scope,
                      static_cast<uint32_t>(MegaMoeWaveTraceEvent::INVALID), beginCycle, endCycle);
    }

    __aicore__ inline uint32_t RecordAggregatedChild(
        MegaMoeWaveTraceEvent event, MegaMoeWaveTraceEvent parentEvent,
        const MegaMoeWaveTraceDetail::EventScope &scope,
        uint32_t parentBeginCycle, uint32_t parentEndCycle,
        uint32_t accumulatedCycles, uint32_t beginOffset = 0U)
    {
        uint32_t parentDuration = parentEndCycle - parentBeginCycle;
        if (accumulatedCycles == 0U || beginOffset >= parentDuration) {
            return beginOffset;
        }
        uint32_t remaining = parentDuration - beginOffset;
        uint32_t recordedCycles = accumulatedCycles < remaining ? accumulatedCycles : remaining;
        uint32_t beginCycle = parentBeginCycle + beginOffset;
        writer.Record(static_cast<uint32_t>(event), scope, static_cast<uint32_t>(parentEvent),
                      beginCycle, beginCycle + recordedCycles);
        return beginOffset + recordedCycles;
    }

    __aicore__ inline MegaMoeWaveTraceDetail::WaitContext CreateWaitContext(
        MegaMoeWaveTraceEvent waitEvent, MegaMoeWaveTraceEvent parentEvent,
        const MegaMoeWaveTraceDetail::EventScope &scope) const
    {
        MegaMoeWaveTraceDetail::WaitContext context{};
        context.writer = writer;
        context.scope = scope;
        context.waitEventType = static_cast<uint32_t>(waitEvent);
        context.parentEventType = static_cast<uint32_t>(parentEvent);
        return context;
    }

    __aicore__ inline void Finish()
    {
        if (!enabled) {
            return;
        }
        const uint32_t processEnd = MegaMoeWaveTraceDetail::Now();
        AscendC::SyncAll<false>();
        AscendC::WriteGmByPassDCache(scratchRecord + 0U, processBegin);
        AscendC::WriteGmByPassDCache(scratchRecord + 1U, processEnd);
        AscendC::WriteGmByPassDCache(scratchRecord + 2U, eventCount);
        AscendC::WriteGmByPassDCache(scratchRecord + 3U, truncated);

        const uint32_t recordIndex = role * coreCount + coreIndex;
        const uint32_t recordWords = RecordWords();
        __gm__ uint32_t *outputRecord = traceOutput + MEGA_MOE_WAVE_TRACE_HEADER_WORDS +
            static_cast<uint64_t>(recordIndex) * recordWords;
        const uint32_t usedWords = MEGA_MOE_WAVE_TRACE_RECORD_HEADER_WORDS +
            eventCount * MEGA_MOE_WAVE_TRACE_EVENT_WORDS;
        for (uint32_t word = 0U; word < usedWords; ++word) {
            uint32_t value = AscendC::ReadGmByPassDCache(scratchRecord + word);
            AscendC::WriteGmByPassDCache(outputRecord + word, value);
        }

        AscendC::SyncAll<false>();
        if (role != static_cast<uint32_t>(MegaMoeWaveTraceRole::AIC) || coreIndex != 0U) {
            return;
        }
        AscendC::WriteGmByPassDCache(traceOutput + 1U, MEGA_MOE_WAVE_TRACE_VERSION);
        AscendC::WriteGmByPassDCache(traceOutput + 2U, MEGA_MOE_WAVE_TRACE_HEADER_WORDS);
        AscendC::WriteGmByPassDCache(traceOutput + 3U, recordWords);
        AscendC::WriteGmByPassDCache(traceOutput + 4U, MEGA_MOE_WAVE_TRACE_EVENT_WORDS);
        AscendC::WriteGmByPassDCache(traceOutput + 5U, MEGA_MOE_WAVE_TRACE_ROLE_COUNT * coreCount);
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
        for (uint32_t word = 16U; word < MEGA_MOE_WAVE_TRACE_HEADER_WORDS; ++word) {
            AscendC::WriteGmByPassDCache(traceOutput + word, 0U);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WriteGmByPassDCache(traceOutput + 0U, MEGA_MOE_WAVE_TRACE_MAGIC);
    }
};

#endif // MEGA_MOE_ENABLE_WAVE_TRACE

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WAVE_TRACE_H
