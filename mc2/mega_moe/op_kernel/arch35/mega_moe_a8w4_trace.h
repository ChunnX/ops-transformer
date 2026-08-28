/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This software is licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef MEGA_MOE_A8W4_TRACE_H
#define MEGA_MOE_A8W4_TRACE_H

#include "kernel_operator.h"
#include "common/mega_moe_a8w4_trace_config.h"

namespace MegaMoeImpl {

#if MEGA_MOE_ENABLE_A8W4_CRITICAL_PATH_TRACE

constexpr uint32_t A8W4_TRACE_MAGIC = 0x41385452U; // "A8TR"
constexpr uint32_t A8W4_TRACE_VERSION = 1U;
constexpr uint32_t A8W4_TRACE_ROLE_COUNT = 3U;
constexpr uint32_t A8W4_TRACE_HEADER_WORDS = 24U;
constexpr uint32_t A8W4_TRACE_RECORD_HEADER_WORDS = 8U;
constexpr uint32_t A8W4_TRACE_EVENT_WORDS = 13U;
constexpr uint32_t A8W4_TRACE_MAX_EVENTS_PER_RECORD = 1024U;
constexpr uint32_t A8W4_TRACE_NO_SCOPE = 0xffffffffU;

enum class A8W4TraceRole : uint32_t {
    AIC = 0U,
    AIV0 = 1U,
    AIV1 = 2U
};

enum class A8W4TraceEvent : uint32_t {
    INVALID = 0U,
    OP_PRE_MOE = 1U,
    DISPATCH_WORK = 2U,
    WAIT_FOR_DISPATCH = 3U,
    GMM1_PROLOGUE = 4U,
    GMM1_COMPUTE = 5U,
    WAIT_FOR_GMM1 = 6U,
    ACTIVATION_WORK = 7U,
    WAIT_FOR_ACTIVATION = 8U,
    GMM2_PROLOGUE = 9U,
    GMM2_COMPUTE = 10U,
    WAIT_FOR_GMM2 = 11U,
    COMBINE_WORK = 12U,
    FINAL_COMBINE_OR_DRAIN = 13U,
    COUNT_EXPORT = 14U,
    UNPERMUTE = 15U,
    OP_POST_MOE = 16U,
    CROSS_RANK_INPUT_SYNC = 17U,
    CROSS_RANK_OUTPUT_SYNC = 18U,
    SHARED_GMM1_ACTIVATION = 19U,
    SHARED_GMM2 = 20U,
};

enum A8W4TraceEventFlags : uint32_t {
    A8W4_TRACE_END_COMPLETE = 1U,
    A8W4_TRACE_END_SUBMIT = 2U,
    A8W4_TRACE_EXPLICIT_WAIT = 4U,
    A8W4_TRACE_ONLY_COMPLETION = 8U,
};

struct A8W4TraceScope {
    uint32_t wave = A8W4_TRACE_NO_SCOPE;
    uint32_t expert = A8W4_TRACE_NO_SCOPE;
    uint32_t tile = A8W4_TRACE_NO_SCOPE;
    uint32_t tokenStart = 0U;
    uint32_t tokenCount = 0U;
    uint32_t workCount = 0U;
};

struct A8W4CriticalPathTrace {
    __gm__ uint32_t *scratchRecord = nullptr;
    __gm__ uint32_t *traceOutput = nullptr;
    uint32_t rank = 0U;
    uint32_t coreCount = 0U;
    uint32_t coreIndex = 0U;
    uint32_t role = 0U;
    uint32_t eventCapacity = 0U;
    uint32_t eventCount = 0U;
    uint32_t droppedCount = 0U;
    uint64_t processBegin = 0U;
    uint32_t bs = 0U;
    uint32_t h = 0U;
    uint32_t hiddenDim = 0U;
    uint32_t topK = 0U;
    uint32_t expertCount = 0U;
    uint32_t worldSize = 0U;
    uint32_t waveMGroupTarget = 0U;
    bool enabled = false;

    static __aicore__ inline uint64_t Now()
    {
        return AscendC::GetSystemCycle();
    }

    __aicore__ inline uint32_t RecordWords() const
    {
        return A8W4_TRACE_RECORD_HEADER_WORDS + eventCapacity * A8W4_TRACE_EVENT_WORDS;
    }

    __aicore__ inline void Init(GM_ADDR workspaceBase, int64_t formalWorkspaceBytes, GM_ADDR outputBase,
                                uint32_t logicalCoreCount, A8W4TraceRole traceRole, uint32_t logicalCoreIndex,
                                uint32_t rankId, uint32_t tokenCount, uint32_t hiddenSize, uint32_t intermediateSize,
                                uint32_t topKValue, uint32_t moeExpertCount, uint32_t epWorldSize)
    {
        rank = rankId;
        coreCount = logicalCoreCount;
        coreIndex = logicalCoreIndex;
        role = static_cast<uint32_t>(traceRole);
        bs = tokenCount;
        h = hiddenSize;
        hiddenDim = intermediateSize;
        topK = topKValue;
        expertCount = moeExpertCount;
        worldSize = epWorldSize;

        uint32_t recordCount = A8W4_TRACE_ROLE_COUNT * coreCount;
        uint64_t outputWords = A8W4_TRACE_OUTPUT_BYTES / sizeof(uint32_t);
        uint64_t fixedWords =
            A8W4_TRACE_HEADER_WORDS + static_cast<uint64_t>(recordCount) * A8W4_TRACE_RECORD_HEADER_WORDS;
        uint64_t capacity =
            outputWords > fixedWords ?
                (outputWords - fixedWords) / (static_cast<uint64_t>(recordCount) * A8W4_TRACE_EVENT_WORDS) :
                0U;
        if (capacity > A8W4_TRACE_MAX_EVENTS_PER_RECORD) {
            capacity = A8W4_TRACE_MAX_EVENTS_PER_RECORD;
        }
        eventCapacity = static_cast<uint32_t>(capacity);
        uint64_t requiredScratchBytes = static_cast<uint64_t>(recordCount) *
                                        (A8W4_TRACE_RECORD_HEADER_WORDS + eventCapacity * A8W4_TRACE_EVENT_WORDS) *
                                        sizeof(uint32_t);
        enabled = workspaceBase != nullptr && outputBase != nullptr && formalWorkspaceBytes >= 0 && coreCount != 0U &&
                  coreIndex < coreCount && role < A8W4_TRACE_ROLE_COUNT && eventCapacity != 0U &&
                  requiredScratchBytes <= A8W4_TRACE_SCRATCH_BYTES;
        if (!enabled) {
            return;
        }
        uint32_t recordIndex = role * coreCount + coreIndex;
        __gm__ uint32_t *scratchBase = reinterpret_cast<__gm__ uint32_t *>(workspaceBase + formalWorkspaceBytes);
        scratchRecord = scratchBase + static_cast<uint64_t>(recordIndex) * RecordWords();
        traceOutput = reinterpret_cast<__gm__ uint32_t *>(outputBase);
        if (role == static_cast<uint32_t>(A8W4TraceRole::AIC) && coreIndex == 0U) {
            AscendC::WriteGmByPassDCache(traceOutput, 0U);
        }
        processBegin = Now();
    }

    __aicore__ inline void SetWaveMGroupTarget(uint32_t target)
    {
        waveMGroupTarget = target;
    }

    __aicore__ inline void Record(A8W4TraceEvent event, A8W4TraceEvent parent, uint32_t flags,
                                  const A8W4TraceScope &scope, uint64_t begin, uint64_t end)
    {
        if (!enabled) {
            return;
        }
        if (eventCount >= eventCapacity) {
            ++droppedCount;
            return;
        }
        __gm__ uint32_t *dst =
            scratchRecord + A8W4_TRACE_RECORD_HEADER_WORDS + static_cast<uint64_t>(eventCount) * A8W4_TRACE_EVENT_WORDS;
        AscendC::WriteGmByPassDCache(dst + 0U, static_cast<uint32_t>(event));
        AscendC::WriteGmByPassDCache(dst + 1U, static_cast<uint32_t>(parent));
        AscendC::WriteGmByPassDCache(dst + 2U, flags);
        AscendC::WriteGmByPassDCache(dst + 3U, scope.wave);
        AscendC::WriteGmByPassDCache(dst + 4U, scope.expert);
        AscendC::WriteGmByPassDCache(dst + 5U, scope.tile);
        AscendC::WriteGmByPassDCache(dst + 6U, scope.tokenStart);
        AscendC::WriteGmByPassDCache(dst + 7U, scope.tokenCount);
        AscendC::WriteGmByPassDCache(dst + 8U, scope.workCount);
        AscendC::WriteGmByPassDCache(dst + 9U, static_cast<uint32_t>(begin));
        AscendC::WriteGmByPassDCache(dst + 10U, static_cast<uint32_t>(begin >> 32U));
        AscendC::WriteGmByPassDCache(dst + 11U, static_cast<uint32_t>(end));
        AscendC::WriteGmByPassDCache(dst + 12U, static_cast<uint32_t>(end >> 32U));
        ++eventCount;
    }

    __aicore__ inline void RecordWait(A8W4TraceEvent event, A8W4TraceEvent parent, const A8W4TraceScope &scope,
                                      uint64_t begin, uint64_t end)
    {
        Record(event, parent, A8W4_TRACE_END_COMPLETE | A8W4_TRACE_EXPLICIT_WAIT, scope, begin, end);
    }

    __aicore__ inline void Finish()
    {
        if (!enabled) {
            return;
        }
        uint64_t processEnd = Now();

        // This barrier is only for post-compute raw publication; it is not an event completion boundary.
        AscendC::SyncAll<false>();
        AscendC::WriteGmByPassDCache(scratchRecord + 0U, static_cast<uint32_t>(processBegin));
        AscendC::WriteGmByPassDCache(scratchRecord + 1U, static_cast<uint32_t>(processBegin >> 32U));
        AscendC::WriteGmByPassDCache(scratchRecord + 2U, static_cast<uint32_t>(processEnd));
        AscendC::WriteGmByPassDCache(scratchRecord + 3U, static_cast<uint32_t>(processEnd >> 32U));
        AscendC::WriteGmByPassDCache(scratchRecord + 4U, eventCount);
        AscendC::WriteGmByPassDCache(scratchRecord + 5U, droppedCount);
        AscendC::WriteGmByPassDCache(scratchRecord + 6U, rank);
        AscendC::WriteGmByPassDCache(scratchRecord + 7U, (role << 16U) | coreIndex);

        uint32_t recordIndex = role * coreCount + coreIndex;
        __gm__ uint32_t *outputRecord =
            traceOutput + A8W4_TRACE_HEADER_WORDS + static_cast<uint64_t>(recordIndex) * RecordWords();
        uint32_t writtenWords = A8W4_TRACE_RECORD_HEADER_WORDS + eventCount * A8W4_TRACE_EVENT_WORDS;
        for (uint32_t word = 0U; word < writtenWords; ++word) {
            uint32_t value = AscendC::ReadGmByPassDCache(scratchRecord + word);
            AscendC::WriteGmByPassDCache(outputRecord + word, value);
        }

        AscendC::SyncAll<false>();
        if (role != static_cast<uint32_t>(A8W4TraceRole::AIC) || coreIndex != 0U) {
            return;
        }
        uint32_t recordWords = RecordWords();
        AscendC::WriteGmByPassDCache(traceOutput + 1U, A8W4_TRACE_VERSION);
        AscendC::WriteGmByPassDCache(traceOutput + 2U, A8W4_TRACE_HEADER_WORDS);
        AscendC::WriteGmByPassDCache(traceOutput + 3U, recordWords);
        AscendC::WriteGmByPassDCache(traceOutput + 4U, A8W4_TRACE_EVENT_WORDS);
        AscendC::WriteGmByPassDCache(traceOutput + 5U, A8W4_TRACE_ROLE_COUNT * coreCount);
        AscendC::WriteGmByPassDCache(traceOutput + 6U, coreCount);
        AscendC::WriteGmByPassDCache(traceOutput + 7U, eventCapacity);
        AscendC::WriteGmByPassDCache(traceOutput + 8U, rank);
        AscendC::WriteGmByPassDCache(traceOutput + 9U, bs);
        AscendC::WriteGmByPassDCache(traceOutput + 10U, h);
        AscendC::WriteGmByPassDCache(traceOutput + 11U, hiddenDim);
        AscendC::WriteGmByPassDCache(traceOutput + 12U, topK);
        AscendC::WriteGmByPassDCache(traceOutput + 13U, expertCount);
        AscendC::WriteGmByPassDCache(traceOutput + 14U, worldSize);
        AscendC::WriteGmByPassDCache(traceOutput + 15U, waveMGroupTarget);
        AscendC::WriteGmByPassDCache(traceOutput + 16U, 64U);
        for (uint32_t word = 17U; word < A8W4_TRACE_HEADER_WORDS; ++word) {
            AscendC::WriteGmByPassDCache(traceOutput + word, 0U);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WriteGmByPassDCache(traceOutput + 0U, A8W4_TRACE_MAGIC);
    }
};

template <typename AddressInfo>
__aicore__ inline A8W4CriticalPathTrace *GetA8W4Trace(const AddressInfo &info)
{
    return reinterpret_cast<A8W4CriticalPathTrace *>(info.criticalPathTrace);
}

template <typename AddressInfo>
__aicore__ inline A8W4TraceScope GetA8W4TraceScope(const AddressInfo &info, uint32_t tile = A8W4_TRACE_NO_SCOPE)
{
    return {info.traceWave,       info.traceExpert,     tile == A8W4_TRACE_NO_SCOPE ? info.traceTile : tile,
            info.traceTokenStart, info.traceTokenCount, info.traceWorkCount};
}

#endif

} // namespace MegaMoeImpl

#endif
