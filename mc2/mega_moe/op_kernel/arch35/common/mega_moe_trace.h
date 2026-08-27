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
 * \brief Diagnostic critical-path trace for the MTE A8W8 token-group Wave path.
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

constexpr uint32_t CRITICAL_PATH_TRACE_MAGIC = 0x43505452U;
constexpr uint32_t CRITICAL_PATH_TRACE_PROTOCOL_VERSION = 14U;
constexpr uint32_t CRITICAL_PATH_TRACE_SCHEMA_ID = 0x0AABA4B5U;
constexpr uint32_t CRITICAL_PATH_TRACE_HEADER_WORDS = 30U;
constexpr uint32_t CRITICAL_PATH_TRACE_ROLE_COUNT = 3U;
constexpr uint32_t CRITICAL_PATH_TRACE_RECORD_WORDS = 40U;
constexpr uint32_t CRITICAL_PATH_TRACE_FIXED_TIMESTAMP_WORDS = 30U;
constexpr uint32_t CRITICAL_PATH_TRACE_SCRATCH_HEADER_WORDS = 36U;
constexpr uint32_t CRITICAL_PATH_TRACE_SEGMENT_WORDS = 4U;
constexpr uint32_t CRITICAL_PATH_TRACE_STAGE_COUNT = 5U;
constexpr uint32_t CRITICAL_PATH_TRACE_M_GROUP_ROWS = 256U;
constexpr uint32_t CRITICAL_PATH_TRACE_REQUIRED_FLAGS = 0x1FFFU;
constexpr uint32_t CRITICAL_PATH_TRACE_DISPATCH_GROUP_READY_FLAG = 1U << 13U;
constexpr uint32_t CRITICAL_PATH_TRACE_PATH_MTE_A8W8_TOKEN_GROUP_WAVE = 2U;
// arch35 tiling appends this diagnostic reserve after WorkspaceInfo::workspaceSize.
constexpr uint64_t CRITICAL_PATH_TRACE_SCRATCH_RESERVED_BYTES = 50ULL * 1024ULL * 1024ULL;

enum class CriticalPathTraceRole : uint32_t {
    AIC = 0U,
    AIV0 = 1U,
    AIV1 = 2U,
};

__aicore__ inline uint64_t CriticalPathTraceMaxMGroups(uint32_t bs, uint32_t epWorldSize, uint32_t topK,
                                                       uint32_t expertCount)
{
    if (expertCount == 0U) {
        return 0ULL;
    }
    const uint64_t localExpertFanout = topK < expertCount ? topK : expertCount;
    const uint64_t maxRoutedRows = static_cast<uint64_t>(bs) * epWorldSize * localExpertFanout;
    const uint64_t firstMGroups = maxRoutedRows < expertCount ? maxRoutedRows : expertCount;
    return firstMGroups + (maxRoutedRows - firstMGroups) / CRITICAL_PATH_TRACE_M_GROUP_ROWS;
}

// Conservative upper bound for the number of token-group Waves. Distributing
// one row to an expert creates its first M group; each further group needs 256
// additional rows. The actual runtime count can only be smaller.
__aicore__ inline uint64_t CriticalPathTraceWaveCapacity(uint32_t bs, uint32_t epWorldSize, uint32_t topK,
                                                         uint32_t expertCount, uint32_t mGroupsPerWave)
{
    if (expertCount == 0U || mGroupsPerWave == 0U) {
        return 0ULL;
    }
    const uint64_t maxMGroups = CriticalPathTraceMaxMGroups(bs, epWorldSize, topK, expertCount);
    const uint64_t maxWaves = (maxMGroups + mGroupsPerWave - 1ULL) / mGroupsPerWave;
    return maxWaves == 0ULL ? 1ULL : maxWaves;
}

// Dispatch/Combine can contribute one segment per expert, while a hot expert
// can contribute one GMM problem per M group in the most fragmented case.
__aicore__ inline uint64_t CriticalPathTraceRequiredSegmentCapacity(uint32_t bs, uint32_t epWorldSize, uint32_t topK,
                                                                    uint32_t expertCount)
{
    const uint64_t maxMGroups = CriticalPathTraceMaxMGroups(bs, epWorldSize, topK, expertCount);
    return maxMGroups < expertCount ? expertCount : maxMGroups;
}

__aicore__ inline uint32_t CriticalPathTraceScratchRecordWords(uint32_t segmentCapacity)
{
    return CRITICAL_PATH_TRACE_SCRATCH_HEADER_WORDS +
           CRITICAL_PATH_TRACE_STAGE_COUNT * CRITICAL_PATH_TRACE_SEGMENT_WORDS * segmentCapacity;
}

__aicore__ inline uint64_t CriticalPathTraceScratchCapacityLimit(uint32_t aicCount)
{
    const uint64_t recordCount = static_cast<uint64_t>(CRITICAL_PATH_TRACE_ROLE_COUNT) * aicCount;
    if (recordCount == 0ULL) {
        return 0ULL;
    }
    const uint64_t reserveWords = CRITICAL_PATH_TRACE_SCRATCH_RESERVED_BYTES / sizeof(int32_t);
    const uint64_t wordsPerRecord = reserveWords / recordCount;
    if (wordsPerRecord <= CRITICAL_PATH_TRACE_SCRATCH_HEADER_WORDS) {
        return 0ULL;
    }
    return (wordsPerRecord - CRITICAL_PATH_TRACE_SCRATCH_HEADER_WORDS) /
           (CRITICAL_PATH_TRACE_STAGE_COUNT * CRITICAL_PATH_TRACE_SEGMENT_WORDS);
}

// Word-wise FNV-1a. The host adapter recomputes this from checked runtime fields.
__aicore__ inline uint32_t CriticalPathTraceHashWord(uint32_t hash, uint32_t value)
{
    return (hash ^ value) * 16777619U;
}

__aicore__ inline uint32_t CriticalPathTraceConfigHash(uint32_t pathId, uint32_t bs, uint32_t h, uint32_t hiddenDim,
                                                       uint32_t topK, uint32_t moeExpertsPerRank,
                                                       uint32_t sharedExpertCount, uint32_t mGroupsPerWave,
                                                       uint32_t worldSize, uint32_t aicCount, uint32_t quantMode,
                                                       uint32_t combineQuantMode, uint32_t topkWeightsPrefetch,
                                                       uint32_t weight1Interleaved)
{
    uint32_t hash = 2166136261U;
    hash = CriticalPathTraceHashWord(hash, pathId);
    hash = CriticalPathTraceHashWord(hash, bs);
    hash = CriticalPathTraceHashWord(hash, h);
    hash = CriticalPathTraceHashWord(hash, hiddenDim);
    hash = CriticalPathTraceHashWord(hash, topK);
    hash = CriticalPathTraceHashWord(hash, moeExpertsPerRank);
    hash = CriticalPathTraceHashWord(hash, sharedExpertCount);
    hash = CriticalPathTraceHashWord(hash, mGroupsPerWave);
    hash = CriticalPathTraceHashWord(hash, worldSize);
    hash = CriticalPathTraceHashWord(hash, aicCount);
    hash = CriticalPathTraceHashWord(hash, quantMode);
    hash = CriticalPathTraceHashWord(hash, combineQuantMode);
    hash = CriticalPathTraceHashWord(hash, topkWeightsPrefetch);
    hash = CriticalPathTraceHashWord(hash, weight1Interleaved);
    return hash;
}

class CriticalPathTrace {
public:
    static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFFU;

    __aicore__ inline void Init(__gm__ int32_t *scratch, uint32_t segmentCapacity)
    {
        scratchRecord_ = scratch;
        segmentCapacity_ = segmentCapacity;
    }

    static __aicore__ inline uint32_t Now()
    {
        const uint32_t cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
        return cycle == 0U ? 1U : cycle;
    }

    static __aicore__ inline void WriteWord(__gm__ int32_t *address, uint32_t value)
    {
        AscendC::WriteGmByPassDCache(address, static_cast<int32_t>(value));
    }

    static __aicore__ inline uint32_t ReadWord(__gm__ int32_t *address)
    {
        return static_cast<uint32_t>(AscendC::ReadGmByPassDCache(address));
    }

    __aicore__ inline void BeginProcess()
    {
        processBegin_ = Now();
    }

    __aicore__ inline void EndWork()
    {
        workEnd_ = Now();
    }

    __aicore__ inline void BeginInputBufferInit()
    {
        inputBufferInitBegin_ = Now();
    }

    __aicore__ inline void EndInputBufferInit()
    {
        inputBufferInitEnd_ = Now();
    }

    __aicore__ inline void BeginInputQuantize()
    {
        inputQuantizeBegin_ = Now();
    }

    __aicore__ inline void EndInputQuantize()
    {
        inputQuantizeEnd_ = Now();
    }

    __aicore__ inline void BeginRouteIndex()
    {
        routeIndexBegin_ = Now();
    }

    __aicore__ inline void EndRouteIndex()
    {
        routeIndexEnd_ = Now();
    }

    __aicore__ inline void BeginWorkspaceReset()
    {
        workspaceResetBegin_ = Now();
    }

    __aicore__ inline void EndWorkspaceReset()
    {
        workspaceResetEnd_ = Now();
    }

    __aicore__ inline void BeginLocalPrepareSync()
    {
        localPrepareSyncBegin_ = Now();
    }

    __aicore__ inline void EndLocalPrepareSync()
    {
        localPrepareSyncEnd_ = Now();
    }

    __aicore__ inline void BeginCrossRankSync()
    {
        crossRankSyncBegin_ = Now();
    }

    __aicore__ inline void EndCrossRankSync()
    {
        crossRankSyncEnd_ = Now();
    }

    __aicore__ inline void BeginExpertBufferInit()
    {
        expertBufferInitBegin_ = Now();
    }

    __aicore__ inline void EndExpertBufferInit()
    {
        expertBufferInitEnd_ = Now();
    }

    __aicore__ inline void BeginFinalCombineDrain()
    {
        finalCombineDrainBegin_ = Now();
    }

    __aicore__ inline void EndFinalCombineDrain()
    {
        finalCombineDrainEnd_ = Now();
    }

    __aicore__ inline void BeginMoeEpilogue()
    {
        moeEpilogueBegin_ = Now();
    }

    __aicore__ inline void EndMoeEpilogue()
    {
        moeEpilogueEnd_ = Now();
    }

    __aicore__ inline void BeginMoeTailSync()
    {
        moeTailSyncBegin_ = Now();
    }

    __aicore__ inline void EndMoeTailSync()
    {
        moeTailSyncEnd_ = Now();
    }

    __aicore__ inline void BeginSharedExpertGmm2()
    {
        sharedExpertGmm2Begin_ = Now();
    }

    __aicore__ inline void EndSharedExpertGmm2()
    {
        sharedExpertGmm2End_ = Now();
    }

    __aicore__ inline void BeginOutputCrossRankSync()
    {
        outputCrossRankSyncBegin_ = Now();
    }

    __aicore__ inline void EndOutputCrossRankSync()
    {
        outputCrossRankSyncEnd_ = Now();
    }

    __aicore__ inline void BeginUnpermuteInit()
    {
        unpermuteInitBegin_ = Now();
    }

    __aicore__ inline void EndUnpermuteInit()
    {
        unpermuteInitEnd_ = Now();
    }

    __aicore__ inline void BeginUnpermute()
    {
        unpermuteBegin_ = Now();
    }

    __aicore__ inline void EndUnpermute()
    {
        unpermuteEnd_ = Now();
    }

    __aicore__ inline void PrepareStage0(uint32_t waveIdx, uint32_t expertIdx)
    {
        stage0ActiveWave_ = waveIdx;
        stage0ActiveExpert_ = expertIdx;
        stage0ActiveBegin_ = 0U;
    }

    __aicore__ inline void PrepareStage1(uint32_t waveIdx, uint32_t expertIdx)
    {
        stage1ActiveWave_ = waveIdx;
        stage1ActiveExpert_ = expertIdx;
        stage1ActiveBegin_ = 0U;
    }

    __aicore__ inline bool NeedsStage0Begin() const
    {
        return stage0ActiveWave_ != INVALID_INDEX && stage0ActiveBegin_ == 0U;
    }

    __aicore__ inline bool NeedsStage1Begin() const
    {
        return stage1ActiveWave_ != INVALID_INDEX && stage1ActiveBegin_ == 0U;
    }

    __aicore__ inline bool HasStage0Begin() const
    {
        return stage0ActiveBegin_ != 0U;
    }

    __aicore__ inline bool HasStage1Begin() const
    {
        return stage1ActiveBegin_ != 0U;
    }

    // GMM1 的有效 MMAD 段只从 ready 后开始。下面三类诊断段填充相邻
    // MMAD problem 之间的白色区间，但不会把它们并入实际 GMM1 耗时。
    __aicore__ inline void PrepareGmm1Gap(uint32_t waveIdx, uint32_t expertIdx, uint32_t begin)
    {
        // TRACE_POINT: gmm1_scalar_setup_begin_S
        if (gmm1SetupBegin_ == 0U) {
            gmm1SetupBegin_ = begin;
        }
        gmm1GapWave_ = waveIdx;
        gmm1GapExpert_ = expertIdx;
    }

    __aicore__ inline bool HasGmm1Gap() const
    {
        return gmm1GapWave_ != INVALID_INDEX;
    }

    // AIV1 只保留相邻两个有效 Dispatch task 之间的 Scalar 区间。调用方在连续
    // 预取调用结束后丢弃未闭合尾段，避免把后续 GMM2/Combine 误算进来。
    __aicore__ inline bool HasDispatchGap() const
    {
        return dispatchGapBegin_ != 0U;
    }

    __aicore__ inline void BeginDispatchGap(uint32_t value)
    {
        dispatchGapBegin_ = value;
    }

    __aicore__ inline void EndDispatchGap(uint32_t waveIdx, uint32_t expertIdx, uint32_t value)
    {
        if (dispatchGapBegin_ == 0U) {
            return;
        }
        AppendSegment(waveIdx, expertIdx, dispatchGapBegin_, value, 2U, stage2CapturedCount_);
        dispatchGapBegin_ = 0U;
    }

    __aicore__ inline void CancelDispatchGap()
    {
        dispatchGapBegin_ = 0U;
    }

    __aicore__ inline void BeginGmm1DispatchWait(uint32_t value)
    {
        if (gmm1SetupBegin_ == 0U) {
            return;
        }
        EndGmm1Setup(value);
        // 排除 EndGmm1Setup 写 trace scratch 的开销，避免把打点本身误算为 Dispatch-ready 等待。
        gmm1DispatchWaitBegin_ = Now();
    }

    __aicore__ inline void EndGmm1DispatchWait(uint32_t value)
    {
        if (gmm1DispatchWaitBegin_ == 0U) {
            return;
        }
        AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1DispatchWaitBegin_, value, 3U, stage3CapturedCount_);
        gmm1DispatchWaitBegin_ = 0U;
    }

    __aicore__ inline void BeginGmm1VectorWait(uint32_t value)
    {
        if (gmm1GapWave_ != INVALID_INDEX) {
            gmm1VectorWaitBegin_ = value;
        }
    }

    __aicore__ inline void EndGmm1VectorWait(uint32_t value)
    {
        if (gmm1VectorWaitBegin_ == 0U) {
            return;
        }
        AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1VectorWaitBegin_, value, 4U, stage4CapturedCount_);
        gmm1VectorWaitBegin_ = 0U;
    }

    // 无 owner tile 或空专家不会进入 ready wait；退出 problem 时把剩余时间
    // 归入 Scalar 准备/ownership 判断，然后清理活动诊断状态。
    __aicore__ inline void EndGmm1Gap(uint32_t value)
    {
        EndGmm1Setup(value);
        if (gmm1DispatchWaitBegin_ != 0U) {
            AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1DispatchWaitBegin_, value, 3U, stage3CapturedCount_);
        }
        if (gmm1VectorWaitBegin_ != 0U) {
            AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1VectorWaitBegin_, value, 4U, stage4CapturedCount_);
        }
        ClearGmm1Gap();
    }

    __aicore__ inline void CompleteGmm1Gap()
    {
        ClearGmm1Gap();
    }

    __aicore__ inline void BeginStage0(uint32_t value)
    {
        if (NeedsStage0Begin()) {
            stage0ActiveBegin_ = value;
        }
    }

    __aicore__ inline void BeginStage1(uint32_t value)
    {
        if (NeedsStage1Begin()) {
            stage1ActiveBegin_ = value;
        }
    }

    __aicore__ inline void EndStage0(uint32_t value)
    {
        AppendSegment(stage0ActiveWave_, stage0ActiveExpert_, stage0ActiveBegin_, value, 0U, stage0CapturedCount_);
        ClearStage0();
    }

    __aicore__ inline void CancelStage0()
    {
        ClearStage0();
    }

    __aicore__ inline void EndStage1(uint32_t value)
    {
        AppendSegment(stage1ActiveWave_, stage1ActiveExpert_, stage1ActiveBegin_, value, 1U, stage1CapturedCount_);
        ClearStage1();
    }

    __aicore__ inline void CancelStage1()
    {
        ClearStage1();
    }

    __aicore__ inline void PublishSummary() const
    {
        if (scratchRecord_ == nullptr) {
            return;
        }
        WriteWord(scratchRecord_ + 0U, processBegin_);
        WriteWord(scratchRecord_ + 1U, workEnd_);
        WriteWord(scratchRecord_ + 2U, inputBufferInitBegin_);
        WriteWord(scratchRecord_ + 3U, inputBufferInitEnd_);
        WriteWord(scratchRecord_ + 4U, inputQuantizeBegin_);
        WriteWord(scratchRecord_ + 5U, inputQuantizeEnd_);
        WriteWord(scratchRecord_ + 6U, routeIndexBegin_);
        WriteWord(scratchRecord_ + 7U, routeIndexEnd_);
        WriteWord(scratchRecord_ + 8U, workspaceResetBegin_);
        WriteWord(scratchRecord_ + 9U, workspaceResetEnd_);
        WriteWord(scratchRecord_ + 10U, localPrepareSyncBegin_);
        WriteWord(scratchRecord_ + 11U, localPrepareSyncEnd_);
        WriteWord(scratchRecord_ + 12U, crossRankSyncBegin_);
        WriteWord(scratchRecord_ + 13U, crossRankSyncEnd_);
        WriteWord(scratchRecord_ + 14U, expertBufferInitBegin_);
        WriteWord(scratchRecord_ + 15U, expertBufferInitEnd_);
        WriteWord(scratchRecord_ + 16U, finalCombineDrainBegin_);
        WriteWord(scratchRecord_ + 17U, finalCombineDrainEnd_);
        WriteWord(scratchRecord_ + 18U, moeEpilogueBegin_);
        WriteWord(scratchRecord_ + 19U, moeEpilogueEnd_);
        WriteWord(scratchRecord_ + 20U, moeTailSyncBegin_);
        WriteWord(scratchRecord_ + 21U, moeTailSyncEnd_);
        WriteWord(scratchRecord_ + 22U, sharedExpertGmm2Begin_);
        WriteWord(scratchRecord_ + 23U, sharedExpertGmm2End_);
        WriteWord(scratchRecord_ + 24U, outputCrossRankSyncBegin_);
        WriteWord(scratchRecord_ + 25U, outputCrossRankSyncEnd_);
        WriteWord(scratchRecord_ + 26U, unpermuteInitBegin_);
        WriteWord(scratchRecord_ + 27U, unpermuteInitEnd_);
        WriteWord(scratchRecord_ + 28U, unpermuteBegin_);
        WriteWord(scratchRecord_ + 29U, unpermuteEnd_);
        WriteWord(scratchRecord_ + 30U, stage0CapturedCount_);
        WriteWord(scratchRecord_ + 31U, stage1CapturedCount_);
        WriteWord(scratchRecord_ + 32U, stage2CapturedCount_);
        WriteWord(scratchRecord_ + 33U, stage3CapturedCount_);
        WriteWord(scratchRecord_ + 34U, stage4CapturedCount_);
        WriteWord(scratchRecord_ + 35U, static_cast<uint32_t>(truncated_));
    }

    __aicore__ inline void WriteStagePayloadTo(__gm__ int32_t *traceOut, uint32_t stage, uint32_t offset,
                                               uint32_t count) const
    {
        if (scratchRecord_ == nullptr || stage >= CRITICAL_PATH_TRACE_STAGE_COUNT || count > segmentCapacity_) {
            return;
        }
        WriteStagePayload(traceOut + offset, StageScratch(stage), count);
    }

private:
    __aicore__ inline void EndGmm1Setup(uint32_t value)
    {
        if (gmm1SetupBegin_ == 0U) {
            return;
        }
        // TRACE_POINT: gmm1_scalar_setup_end_S
        AppendSegment(gmm1GapWave_, gmm1GapExpert_, gmm1SetupBegin_, value, 2U, stage2CapturedCount_);
        gmm1SetupBegin_ = 0U;
    }

    __aicore__ inline void ClearStage0()
    {
        stage0ActiveWave_ = INVALID_INDEX;
        stage0ActiveExpert_ = INVALID_INDEX;
        stage0ActiveBegin_ = 0U;
    }

    __aicore__ inline void ClearStage1()
    {
        stage1ActiveWave_ = INVALID_INDEX;
        stage1ActiveExpert_ = INVALID_INDEX;
        stage1ActiveBegin_ = 0U;
    }

    __aicore__ inline void ClearGmm1Gap()
    {
        gmm1GapWave_ = INVALID_INDEX;
        gmm1GapExpert_ = INVALID_INDEX;
        gmm1SetupBegin_ = 0U;
        gmm1DispatchWaitBegin_ = 0U;
        gmm1VectorWaitBegin_ = 0U;
    }

    __aicore__ inline __gm__ int32_t *StageScratch(uint32_t stage) const
    {
        return scratchRecord_ + CRITICAL_PATH_TRACE_SCRATCH_HEADER_WORDS +
               static_cast<uint64_t>(stage) * segmentCapacity_ * CRITICAL_PATH_TRACE_SEGMENT_WORDS;
    }

    __aicore__ inline void AppendSegment(uint32_t waveIdx, uint32_t expertIdx, uint32_t begin, uint32_t end,
                                         uint32_t stage, uint32_t &capturedCount)
    {
        if (scratchRecord_ == nullptr || waveIdx == INVALID_INDEX || expertIdx == INVALID_INDEX || begin == 0U) {
            return;
        }
        if (capturedCount >= segmentCapacity_) {
            truncated_ = true;
            return;
        }
        __gm__ int32_t *entry =
            StageScratch(stage) + static_cast<uint64_t>(capturedCount) * CRITICAL_PATH_TRACE_SEGMENT_WORDS;
        WriteWord(entry + 0U, waveIdx);
        WriteWord(entry + 1U, expertIdx);
        WriteWord(entry + 2U, begin);
        WriteWord(entry + 3U, end);
        ++capturedCount;
    }

    static __aicore__ inline void WriteStagePayload(__gm__ int32_t *output, __gm__ int32_t *scratch, uint32_t count)
    {
        for (uint32_t index = 0U; index < count; ++index) {
            __gm__ int32_t *source = scratch + static_cast<uint64_t>(index) * CRITICAL_PATH_TRACE_SEGMENT_WORDS;
            __gm__ int32_t *target = output + static_cast<uint64_t>(index) * CRITICAL_PATH_TRACE_SEGMENT_WORDS;
            WriteWord(target + 0U, ReadWord(source + 0U));
            WriteWord(target + 1U, ReadWord(source + 1U));
            WriteWord(target + 2U, ReadWord(source + 2U));
            WriteWord(target + 3U, ReadWord(source + 3U));
        }
    }

    __gm__ int32_t *scratchRecord_ = nullptr;
    uint32_t segmentCapacity_ = 0U;
    uint32_t processBegin_ = 0U;
    uint32_t workEnd_ = 0U;
    uint32_t inputBufferInitBegin_ = 0U;
    uint32_t inputBufferInitEnd_ = 0U;
    uint32_t inputQuantizeBegin_ = 0U;
    uint32_t inputQuantizeEnd_ = 0U;
    uint32_t routeIndexBegin_ = 0U;
    uint32_t routeIndexEnd_ = 0U;
    uint32_t workspaceResetBegin_ = 0U;
    uint32_t workspaceResetEnd_ = 0U;
    uint32_t localPrepareSyncBegin_ = 0U;
    uint32_t localPrepareSyncEnd_ = 0U;
    uint32_t crossRankSyncBegin_ = 0U;
    uint32_t crossRankSyncEnd_ = 0U;
    uint32_t expertBufferInitBegin_ = 0U;
    uint32_t expertBufferInitEnd_ = 0U;
    uint32_t finalCombineDrainBegin_ = 0U;
    uint32_t finalCombineDrainEnd_ = 0U;
    uint32_t moeEpilogueBegin_ = 0U;
    uint32_t moeEpilogueEnd_ = 0U;
    uint32_t moeTailSyncBegin_ = 0U;
    uint32_t moeTailSyncEnd_ = 0U;
    uint32_t sharedExpertGmm2Begin_ = 0U;
    uint32_t sharedExpertGmm2End_ = 0U;
    uint32_t outputCrossRankSyncBegin_ = 0U;
    uint32_t outputCrossRankSyncEnd_ = 0U;
    uint32_t unpermuteInitBegin_ = 0U;
    uint32_t unpermuteInitEnd_ = 0U;
    uint32_t unpermuteBegin_ = 0U;
    uint32_t unpermuteEnd_ = 0U;
    uint32_t stage0ActiveWave_ = INVALID_INDEX;
    uint32_t stage1ActiveWave_ = INVALID_INDEX;
    uint32_t stage0ActiveExpert_ = INVALID_INDEX;
    uint32_t stage1ActiveExpert_ = INVALID_INDEX;
    uint32_t stage0ActiveBegin_ = 0U;
    uint32_t stage1ActiveBegin_ = 0U;
    uint32_t gmm1GapWave_ = INVALID_INDEX;
    uint32_t gmm1GapExpert_ = INVALID_INDEX;
    uint32_t gmm1SetupBegin_ = 0U;
    uint32_t gmm1DispatchWaitBegin_ = 0U;
    uint32_t gmm1VectorWaitBegin_ = 0U;
    uint32_t dispatchGapBegin_ = 0U;
    uint32_t stage0CapturedCount_ = 0U;
    uint32_t stage1CapturedCount_ = 0U;
    uint32_t stage2CapturedCount_ = 0U;
    uint32_t stage3CapturedCount_ = 0U;
    uint32_t stage4CapturedCount_ = 0U;
    bool truncated_ = false;
};

#endif // MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TRACE_H
