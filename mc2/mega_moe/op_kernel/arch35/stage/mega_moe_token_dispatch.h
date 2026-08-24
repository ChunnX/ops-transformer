/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_DISPATCH_H
#define MEGA_MOE_TOKEN_DISPATCH_H

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

// Token Dispatch uses a logical block job for work partitioning and a physical
// workspace slot for count/flag publication. Neither index is derived here.
struct TokenDispatchConfig {
    MoeStageCommonConfig common;
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    uint64_t maxOutputSize;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint32_t maskAlignSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    int32_t dispatchFlagSlotsPerExpert;
};

template <typename ActivationType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<int32_t> topkIndexTensor;
    LocalTensor<int32_t> sendCntTensor;
    LocalTensor<uint8_t> maskBatchTensor;
    LocalTensor<uint32_t> maskBatchU32Tensor;
    LocalTensor<int32_t> expertTokenCntTensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    // Contiguous runtime-sized ring. A slot view is built from this UB base on demand.
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
    uint64_t cumsumRevCntInRank;
};

/** 表示当前 Dispatch core 在当前专家 row 空间中负责的相对 row range。 */
struct DispatchExpertRange {
    // 当前 Dispatch core 负责的起始专家 row offset（包含）。
    int32_t coreExpertRowBegin;
    // 当前 Dispatch core 负责的结束专家 row offset（不包含）。
    int32_t coreExpertRowEnd;
};

/** 描述覆盖当前 core 的专家 row range 的 source-rank 范围。 */
struct DispatchExpertRankRange {
    // 第一个与当前 core 的专家 row range 相交的 source rank index。
    int32_t firstRankIdx;
    // 最后一个与当前 core 的专家 row range 相交的 source rank index。
    // 当没有相交的 source rank 时为 -1。
    int32_t lastRankIdx;
    // 第一个相交 source-rank segment 在当前专家 row 空间中的起始 offset。
    int32_t firstRankOffset;
};

// Trace-only accounting is passed explicitly through the Dispatch helpers so
// the production path remains unchanged when MEGA_MOE_ENABLE_WAVE_TRACE=0.
// All counters are accumulated within one Dispatch wave on one AIV1 core.
struct DispatchMte3TraceStats {
    uint32_t transferPipelineCycles = 0U;
    uint32_t reuseWaitCycles = 0U;
    uint32_t drainWaitCycles = 0U;
    uint32_t readySyncWaitCycles = 0U;
};

// Minimal state required to keep the Dispatch MTE ring alive across route
// batches and source-rank changes inside one expert tile. Only the last
// fetched token is pending; there is no materialized descriptor queue.
struct DispatchTransferStreamState {
    int32_t issuedTokenCount = 0;
    int32_t pendingDestinationRow = 0;
    int32_t pendingTopkIndex = 0;
};

__aicore__ inline uint32_t BeginDispatchTransferTrace(DispatchMte3TraceStats *stats)
{
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    return stats == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#else
    (void)stats;
    return 0U;
#endif
}

__aicore__ inline void EndDispatchTransferTrace(
    DispatchMte3TraceStats *stats, uint32_t beginCycle, uint32_t &accumulator)
{
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (stats != nullptr && beginCycle != 0U) {
        accumulator += MegaMoeWaveTraceDetail::Now() - beginCycle;
    }
#else
    (void)stats;
    (void)beginCycle;
    (void)accumulator;
#endif
}

struct ExpertTokenCountExportScratch {
    LocalTensor<int32_t> stridedTensor;
    LocalTensor<int32_t> compactTensor;
};

template <typename ActivationType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig &context,
                                                     TokenDispatchScratch<ActivationType> &scratch,
                                                     int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType> &remoteRankGlobalTensor,
                                                     DispatchMte3TraceStats *traceStats = nullptr)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(context.common.topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
    if constexpr (IsBufferReuse) {
        uint32_t waitBegin = BeginDispatchTransferTrace(traceStats);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        if (traceStats != nullptr) {
            EndDispatchTransferTrace(traceStats, waitBegin, traceStats->reuseWaitCycles);
        }
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        uint32_t waitBegin = BeginDispatchTransferTrace(traceStats);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
        if (traceStats != nullptr) {
            EndDispatchTransferTrace(traceStats, waitBegin, traceStats->reuseWaitCycles);
        }
    }
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX,
                                                                topkIndex % static_cast<int32_t>(context.common.topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
    int32_t dstIdx, GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor, GlobalTensor<int32_t> &metaInfoGlobalTensor,
    int32_t topkIndex)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(context.common.topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    DataCopyPad(tokenRevGlobalTensor[dstIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[dstIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[dstIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Prototype: MegaMoe::CopyGMToGMPerToken. Copies one dispatch shard and emits its metadata.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t rowDstOffset, int32_t remoteRankIdx,
    int32_t copyStartIdx, int32_t copyNum)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t bufferCount = bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr +
                                                  rowDstOffset * scratch.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr +
                                                  rowDstOffset * scratch.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr +
                                           rowDstOffset * INT32_PER_256B * sizeof(int32_t)));

    // The only caller enters this function for a non-empty match interval, so copyNum is at least one.
    // GatherMask's V-to-S synchronization covers the first scalar read, and the drain below covers
    // cross-call reuse of the copy and metadata rings.
    int32_t firstTopkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, scratch, 0, firstTopkIndex, remoteRankIdx,
                                                              remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, scratch, issueIdx, topkIndex, remoteRankIdx,
                                                                  remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx));
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyBufferIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx));
    }

    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, (copyNum - 1) % bufferCount, copyNum - 1, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyNum - 1));

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// Append one already selected wave task directly to the existing MTE ring.
// sourceRank is a property of this task, so changing rank does not drain the
// ring. The last fetched token remains pending until the next task or tile end.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void AppendDispatchTokenToStream(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t remoteRankIdx,
    int32_t topkIndex, int32_t destinationRow, DispatchTransferStreamState &stream,
    DispatchMte3TraceStats *traceStats = nullptr)
{
    int32_t bufferCount = context.bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));

    int32_t issueBufferIdx = stream.issuedTokenCount % bufferCount;
    if (stream.issuedTokenCount >= bufferCount) {
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(
            context, scratch, issueBufferIdx, topkIndex, remoteRankIdx,
            remoteRankGlobalTensor, traceStats);
    } else {
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(
            context, scratch, issueBufferIdx, topkIndex, remoteRankIdx,
            remoteRankGlobalTensor, traceStats);
    }
    if (stream.issuedTokenCount > 0) {
        int32_t pendingBufferIdx = (stream.issuedTokenCount - 1) % bufferCount;
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, pendingBufferIdx, stream.pendingDestinationRow,
            tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            stream.pendingTopkIndex);
    }
    stream.pendingDestinationRow = destinationRow;
    stream.pendingTopkIndex = topkIndex;
    ++stream.issuedTokenCount;
}

// Store the final pending token and wait only for the slots used by the current
// stream. The interleaved path invokes this at a GMM1 tile-ready boundary.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void FinishInterleavedDispatchStream(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, DispatchTransferStreamState &stream,
    DispatchMte3TraceStats *traceStats = nullptr)
{
    if (stream.issuedTokenCount <= 0) {
        return;
    }
    int32_t bufferCount = context.bufferConfig.bufferCount;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));

    int32_t lastBufferIdx = (stream.issuedTokenCount - 1) % bufferCount;
    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, lastBufferIdx, stream.pendingDestinationRow,
        tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
        stream.pendingTopkIndex);

    uint32_t drainBegin = BeginDispatchTransferTrace(traceStats);
    int32_t firstUseEnd = stream.issuedTokenCount < bufferCount ? stream.issuedTokenCount : bufferCount;
    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
    if (traceStats != nullptr) {
        EndDispatchTransferTrace(traceStats, drainBegin, traceStats->drainWaitCycles);
    }
}

/** 根据当前 core 的专家 row range 定位相交的 source-rank segment。 */
template <typename ActivationType>
__aicore__ inline DispatchExpertRankRange FindDispatchExpertRankRange(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch,
    uint32_t localExpertId, int32_t expertGlobalRowBegin,
    int32_t coreExpertRowBegin, int32_t coreExpertRowEnd)
{
    DispatchExpertRankRange range{static_cast<int32_t>(context.common.worldSize), -1, expertGlobalRowBegin};
    int32_t previousRankPrefix = expertGlobalRowBegin;
    for (uint32_t remoteRankIdx = 0U; remoteRankIdx < context.common.worldSize; ++remoteRankIdx) {
        int32_t rankPrefix = scratch.cumsumInfoTensor.GetValue(
            localExpertId * context.common.worldSize + remoteRankIdx);
        if (range.firstRankIdx == static_cast<int32_t>(context.common.worldSize) &&
            rankPrefix - expertGlobalRowBegin > coreExpertRowBegin) {
            range.firstRankIdx = static_cast<int32_t>(remoteRankIdx);
            range.firstRankOffset = previousRankPrefix - expertGlobalRowBegin;
        }
        if (rankPrefix - expertGlobalRowBegin >= coreExpertRowEnd) {
            range.lastRankIdx = static_cast<int32_t>(remoteRankIdx);
            break;
        }
        previousRankPrefix = rankPrefix;
    }
    return range;
}

template <typename ActivationType>
__aicore__ inline int32_t GetDispatchExpertRankTokenCount(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch,
    uint32_t localExpertId, int32_t expertGlobalRowBegin, uint32_t remoteRankIdx)
{
    int32_t rankRowEnd = scratch.cumsumInfoTensor.GetValue(
        localExpertId * context.common.worldSize + remoteRankIdx) - expertGlobalRowBegin;
    int32_t rankRowBegin = remoteRankIdx == 0U ? 0 :
        scratch.cumsumInfoTensor.GetValue(
            localExpertId * context.common.worldSize + remoteRankIdx - 1U) - expertGlobalRowBegin;
    return rankRowEnd - rankRowBegin;
}

constexpr int32_t DISPATCH_RANK_SCRATCH_STRIDE = ALIGN_32 / sizeof(int32_t);
constexpr int32_t DISPATCH_RANK_COUNT_LANE = 0;
constexpr int32_t DISPATCH_RANK_LOCAL_TASK_COUNT_LANE = 1;
constexpr int32_t DISPATCH_ACTIVE_RANK_LIST_LANE = 2;
constexpr int32_t DISPATCH_TASK_ALIGN_ITEMS = ALIGN_32 / sizeof(int32_t);

// Per-core task descriptors live in topkIndexTensor. The rank and topk arrays
// are independently 32-byte aligned; the remaining suffix is kept 256-item
// aligned for CreateVecIndex. Only rows owned by this core are materialized.
struct DispatchCoreTaskTable {
    int32_t taskCount = 0;
    int32_t taskCapacity = 0;
    int32_t rankListOffset = 0;
    int32_t topkListOffset = 0;
    int32_t routeIndexOffset = 0;
    int32_t routeCapacity = 0;
};

template <typename ActivationType>
__aicore__ inline void LoadDispatchExpertRankCounts(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch,
    uint32_t localExpertId, int32_t expertGlobalRowBegin)
{
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankTokenCount = GetDispatchExpertRankTokenCount(
            context, scratch, localExpertId, expertGlobalRowBegin, rankIdx);
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        scratch.sendCntTensor.SetValue(rankScratchBase + DISPATCH_RANK_COUNT_LANE, rankTokenCount);
        scratch.sendCntTensor.SetValue(rankScratchBase + DISPATCH_RANK_LOCAL_TASK_COUNT_LANE, 0);
    }
}

__aicore__ inline int32_t GetFirstDispatchRowOwnedByCore(
    const TokenDispatchConfig &context, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd)
{
    if (segmentRowBegin >= segmentRowEnd) {
        return segmentRowEnd;
    }
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t coreIdx = static_cast<int32_t>(context.blockJob.jobIndex);
    int32_t firstDestinationRow = expertGlobalRowBegin + segmentRowBegin;
    int32_t owner = firstDestinationRow % coreCount;
    int32_t ownerDelta = coreIdx - owner;
    if (ownerDelta < 0) {
        ownerDelta += coreCount;
    }
    return segmentRowBegin + ownerDelta;
}

// Build the final rank-round-robin descriptors for this core only. Each
// interval covers rounds with the same active-rank set; therefore the number
// of planning intervals is bounded by the number of source ranks, not rows.
template <typename ActivationType>
__aicore__ inline DispatchCoreTaskTable BuildDispatchCoreTaskTable(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch,
    int32_t expertGlobalRowBegin, int32_t segmentRowBegin, int32_t segmentRowEnd,
    int32_t firstOwnedRow)
{
    DispatchCoreTaskTable table{};
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    if (firstOwnedRow >= segmentRowEnd) {
        return table;
    }

    int32_t expectedTaskCount =
        (segmentRowEnd - 1 - firstOwnedRow) / coreCount + 1;
    table.taskCapacity = Ops::Base::CeilAlign(
        expectedTaskCount, DISPATCH_TASK_ALIGN_ITEMS);
    table.rankListOffset = 0;
    table.topkListOffset = table.taskCapacity;
    table.routeIndexOffset = table.taskCapacity * 2;
    table.routeCapacity = context.bufferConfig.routeItemsPerBatch - table.routeIndexOffset;
    table.routeCapacity = table.routeCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    if (table.routeCapacity <= 0) {
        table.taskCapacity = 0;
        return table;
    }

    int32_t roundBegin = 0;
    int32_t intervalRowBegin = 0;
    while (true) {
        int32_t activeRankCount = 0;
        int32_t roundEnd = 0x7fffffff;
        for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
            int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
            int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
                rankScratchBase + DISPATCH_RANK_COUNT_LANE);
            if (rankTokenCount <= roundBegin) {
                continue;
            }
            int32_t activeListBase = activeRankCount * DISPATCH_RANK_SCRATCH_STRIDE;
            scratch.sendCntTensor.SetValue(
                activeListBase + DISPATCH_ACTIVE_RANK_LIST_LANE,
                static_cast<int32_t>(rankIdx));
            ++activeRankCount;
            if (rankTokenCount < roundEnd) {
                roundEnd = rankTokenCount;
            }
        }
        if (activeRankCount == 0) {
            break;
        }

        int32_t intervalRowEnd = intervalRowBegin +
            (roundEnd - roundBegin) * activeRankCount;
        int32_t overlapBegin = intervalRowBegin > segmentRowBegin ?
            intervalRowBegin : segmentRowBegin;
        int32_t overlapEnd = intervalRowEnd < segmentRowEnd ?
            intervalRowEnd : segmentRowEnd;
        if (overlapBegin < overlapEnd) {
            int32_t firstIntervalOwnedRow = GetFirstDispatchRowOwnedByCore(
                context, expertGlobalRowBegin, overlapBegin, overlapEnd);
            for (int32_t expertRow = firstIntervalOwnedRow;
                 expertRow < overlapEnd; expertRow += coreCount) {
                int32_t withinInterval = expertRow - intervalRowBegin;
                int32_t activeRankPosition = withinInterval % activeRankCount;
                int32_t rankOrdinal = roundBegin + withinInterval / activeRankCount;
                int32_t activeListBase =
                    activeRankPosition * DISPATCH_RANK_SCRATCH_STRIDE;
                int32_t rankIdx = scratch.sendCntTensor.GetValue(
                    activeListBase + DISPATCH_ACTIVE_RANK_LIST_LANE);
                int32_t taskIdx = table.taskCount++;
                scratch.topkIndexTensor.SetValue(
                    table.rankListOffset + taskIdx, rankIdx);
                // Negative values are unresolved compact ordinals. Gather
                // replaces them in place with the non-negative topkIndex.
                scratch.topkIndexTensor.SetValue(
                    table.topkListOffset + taskIdx, -rankOrdinal - 1);
                int32_t rankScratchBase = rankIdx * DISPATCH_RANK_SCRATCH_STRIDE;
                int32_t localRankTaskCount = scratch.sendCntTensor.GetValue(
                    rankScratchBase + DISPATCH_RANK_LOCAL_TASK_COUNT_LANE);
                scratch.sendCntTensor.SetValue(
                    rankScratchBase + DISPATCH_RANK_LOCAL_TASK_COUNT_LANE,
                    localRankTaskCount + 1);
            }
        }

        intervalRowBegin = intervalRowEnd;
        roundBegin = roundEnd;
        if (intervalRowBegin >= segmentRowEnd) {
            break;
        }
    }
    if (table.taskCount != expectedTaskCount) {
        table.taskCount = 0;
    }
    return table;
}

// Resolve only this core's compact ordinals for one rank. GatherMask still
// compacts the rank bitmap in vector code, but there is no full compact-list
// copy: matching local descriptors are patched directly in the task table.
template <typename ActivationType>
__aicore__ inline int32_t ResolveDispatchCoreRankTasks(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    uint32_t remoteRankIdx, const DispatchCoreTaskTable &table)
{
    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    int32_t unresolvedTaskCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_LOCAL_TASK_COUNT_LANE);
    if (unresolvedTaskCount <= 0) {
        return 0;
    }

    uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    LocalTensor<int32_t> routeIndexTensor = scratch.topkIndexTensor[table.routeIndexOffset];
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
    int32_t matchedRouteCount = 0;
    int32_t resolvedTaskCount = 0;

    for (int32_t routeBegin = 0;
         static_cast<uint64_t>(routeBegin) < sendTotalNum && unresolvedTaskCount > 0;
         routeBegin += table.routeCapacity) {
        int32_t validRouteCount = static_cast<int32_t>(
            sendTotalNum - static_cast<uint64_t>(routeBegin));
        if (validRouteCount > table.routeCapacity) {
            validRouteCount = table.routeCapacity;
        }
        // The bitmap GM copy must remain 32-byte aligned. GatherMask still
        // receives validRouteCount, so padded tail bits are never consumed.
        int32_t maskSliceBytes = Ops::Base::CeilAlign(
            validRouteCount, static_cast<int32_t>(ALIGN_256)) / 8;
        int32_t maskByteBegin = routeBegin / 8;
        if (maskByteBegin + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
            maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - maskByteBegin;
        }

        remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
            params.peermemInfo.maskRecvPtr +
            (static_cast<uint64_t>(localExpertId) * context.common.worldSize + remoteRankIdx) * maskSlotSize +
            static_cast<uint64_t>(maskByteBegin)));
        DataCopy(scratch.maskBatchTensor, remoteRankMaskGlobal, static_cast<uint32_t>(maskSliceBytes));
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
        CreateVecIndex(routeIndexTensor, routeBegin, table.routeCapacity);
        uint64_t batchMatchedRouteCount = 0U;
        GatherMask(scratch.validTopkIndexTensor, routeIndexTensor, scratch.maskBatchU32Tensor, true,
                   static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

        int32_t batchMatchCount = static_cast<int32_t>(batchMatchedRouteCount);
        int32_t batchOrdinalBegin = matchedRouteCount;
        int32_t batchOrdinalEnd = matchedRouteCount + batchMatchCount;
        for (int32_t taskIdx = 0; taskIdx < table.taskCount && unresolvedTaskCount > 0; ++taskIdx) {
            int32_t taskRank = scratch.topkIndexTensor.GetValue(
                table.rankListOffset + taskIdx);
            if (taskRank != static_cast<int32_t>(remoteRankIdx)) {
                continue;
            }
            int32_t encodedOrdinal = scratch.topkIndexTensor.GetValue(
                table.topkListOffset + taskIdx);
            if (encodedOrdinal >= 0) {
                continue;
            }
            int32_t rankOrdinal = -encodedOrdinal - 1;
            if (rankOrdinal >= batchOrdinalBegin && rankOrdinal < batchOrdinalEnd) {
                int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
                    rankOrdinal - batchOrdinalBegin);
                scratch.topkIndexTensor.SetValue(
                    table.topkListOffset + taskIdx, topkIndex);
                --unresolvedTaskCount;
                ++resolvedTaskCount;
            }
        }
        matchedRouteCount = batchOrdinalEnd;
    }
    return resolvedTaskCount;
}

template <typename ActivationType>
__aicore__ inline bool ResolveDispatchCoreTaskTable(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    const DispatchCoreTaskTable &table)
{
    int32_t resolvedTaskCount = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        resolvedTaskCount += ResolveDispatchCoreRankTasks(
            context, params, scratch, localExpertId, rankIdx, table);
    }
    return resolvedTaskCount == table.taskCount;
}

__aicore__ inline void PublishInterleavedDispatchTileReady(
    const TokenDispatchConfig &context, const Params &params,
    uint32_t localExpertId, int32_t tileIdx, int32_t readyRowCount,
    DispatchMte3TraceStats *traceStats = nullptr)
{
    if (readyRowCount <= 0) {
        return;
    }
    uint32_t waitBegin = BeginDispatchTransferTrace(traceStats);
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    if (traceStats != nullptr) {
        EndDispatchTransferTrace(traceStats, waitBegin, traceStats->readySyncWaitCycles);
    }
    __gm__ int32_t *gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * context.dispatchFlagSlotsPerExpert;
    AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE,
              readyRowCount);
}

template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void FinishInterleavedDispatchTile(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t tileIdx, int32_t readyRowCount, DispatchTransferStreamState &stream,
    DispatchMte3TraceStats *traceStats = nullptr, uint32_t transferBegin = 0U)
{
    if (stream.issuedTokenCount <= 0 || readyRowCount <= 0) {
        return;
    }
    if (transferBegin == 0U) {
        transferBegin = BeginDispatchTransferTrace(traceStats);
    }
    FinishInterleavedDispatchStream<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, scratch, stream, traceStats);
    if (traceStats != nullptr) {
        EndDispatchTransferTrace(traceStats, transferBegin, traceStats->transferPipelineCycles);
    }
    PublishInterleavedDispatchTileReady(
        context, params, localExpertId, tileIdx, readyRowCount, traceStats);
    stream.issuedTokenCount = 0;
}

// 发布一个 source-rank 段覆盖到的 GMM1 tile ready 计数。
__aicore__ inline void PublishGmm1TileReady(
    const TokenDispatchConfig &context, const Params &params, uint32_t localExpertId,
    int32_t gmm1TileRowCount, int32_t segmentExpertRowBegin, int32_t segmentExpertRowEnd)
{
    if (segmentExpertRowBegin >= segmentExpertRowEnd) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    __gm__ int32_t *gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * context.dispatchFlagSlotsPerExpert;
    int32_t firstTileIdx = segmentExpertRowBegin / gmm1TileRowCount;
    int32_t lastTileIdx = (segmentExpertRowEnd - 1) / gmm1TileRowCount;
    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileExpertRowBegin = tileIdx * gmm1TileRowCount;
        int32_t tileExpertRowEnd = tileExpertRowBegin + gmm1TileRowCount;
        int32_t overlapRowBegin =
            segmentExpertRowBegin > tileExpertRowBegin ? segmentExpertRowBegin : tileExpertRowBegin;
        int32_t overlapRowEnd = segmentExpertRowEnd < tileExpertRowEnd ? segmentExpertRowEnd : tileExpertRowEnd;
        AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE,
                  overlapRowEnd - overlapRowBegin);
    }
}

/**
 * Dispatch 第 3 层：扫描单个 source rank 的 route mask，搬运选中的 token
 * 和 metadata，并发布 GMM1 tile ready count。
 */
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchRankTokens(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t remoteRankIdx,
    int32_t rankSegmentRowBegin,
    int32_t segmentMatchOrdinalBegin, int32_t segmentMatchOrdinalEnd,
    int32_t gmm1TileRowCount)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
    int32_t matchedRouteCount = 0;
    int32_t dispatchedRowCount = 0;
    for (int32_t batchIdx = 0;
         batchIdx < bufferConfig.routeBatchCount && matchedRouteCount < segmentMatchOrdinalEnd;
         ++batchIdx) {
        int32_t batchRouteBegin = batchIdx * bufferConfig.routeItemsPerBatch;
        bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
        int32_t validRouteCount = bufferConfig.routeItemsPerBatch;
        int32_t maskSliceBytes = bufferConfig.routeItemsPerBatch / 8;
        if (isLastBatch) {
            uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
            validRouteCount = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchRouteBegin));
            if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
                maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - batchRouteBegin / 8;
            }
        }

        remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
            params.peermemInfo.maskRecvPtr +
            (static_cast<uint64_t>(localExpertId) * context.common.worldSize + remoteRankIdx) * maskSlotSize +
            static_cast<uint64_t>(batchRouteBegin / 8)));
        DataCopy(scratch.maskBatchTensor, remoteRankMaskGlobal, static_cast<uint32_t>(maskSliceBytes));
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
        CreateVecIndex(scratch.topkIndexTensor, batchRouteBegin, bufferConfig.routeItemsPerBatch);
        uint64_t batchMatchedRouteCount = 0;
        GatherMask(scratch.validTopkIndexTensor, scratch.topkIndexTensor, scratch.maskBatchU32Tensor, true,
                   static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

        int32_t batchMatchOrdinalBegin = matchedRouteCount;
        int32_t batchMatchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);
        int32_t dispatchMatchOrdinalBegin =
            batchMatchOrdinalBegin > segmentMatchOrdinalBegin ? batchMatchOrdinalBegin : segmentMatchOrdinalBegin;
        int32_t dispatchMatchOrdinalEnd =
            batchMatchOrdinalEnd < segmentMatchOrdinalEnd ? batchMatchOrdinalEnd : segmentMatchOrdinalEnd;
        if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
            int32_t batchLocalMatchBegin = dispatchMatchOrdinalBegin - batchMatchOrdinalBegin;
            int32_t batchDispatchRowCount = dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin;
            int32_t dispatchDstRowBegin =
                rankSegmentRowBegin + expertGlobalRowBegin + dispatchMatchOrdinalBegin;
            CopyTokensAndMetaForDispatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, params, winRankAddr, scratch, dispatchDstRowBegin, remoteRankIdx,
                batchLocalMatchBegin, batchDispatchRowCount);
            dispatchedRowCount += batchDispatchRowCount;
        }
        matchedRouteCount = batchMatchOrdinalEnd;
    }

    if (dispatchedRowCount > 0) {
        int32_t segmentExpertRowBegin = rankSegmentRowBegin + segmentMatchOrdinalBegin;
        int32_t segmentExpertRowEnd = segmentExpertRowBegin + dispatchedRowCount;
        PublishGmm1TileReady(context, params, localExpertId, gmm1TileRowCount,
                             segmentExpertRowBegin, segmentExpertRowEnd);
    }
}

// Small-route fallback is defined below the rank-round path. Declaring it here
// lets every core make one uniform capacity decision before building a local
// task table; no core can silently skip ready publication on a small shape.
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensByRows(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount,
    uint32_t expertRowOffset = 0U);

// Dispatch one expert segment from the current core's compact task table.
// Planning, topk resolution and execution are all proportional to this core's
// owned rows; no core walks or stages the complete expert segment.
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensRankRoundRobin(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount, uint32_t expertRowOffset,
    DispatchMte3TraceStats *traceStats = nullptr)
{
    if (dispatchRowCount == 0U) {
        return;
    }

    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t maxCoreTaskCount = Ops::Base::CeilDiv(
        static_cast<int32_t>(dispatchRowCount), coreCount);
    int32_t maxTaskCapacity = Ops::Base::CeilAlign(
        maxCoreTaskCount, DISPATCH_TASK_ALIGN_ITEMS);
    int32_t routeCapacity = context.bufferConfig.routeItemsPerBatch - maxTaskCapacity * 2;
    routeCapacity = routeCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    if (routeCapacity <= 0) {
        DispatchExpertTokensByRows<ActivationType, QuantScaleType, PipelineTileM,
                                   TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, localExpertId,
            expertGlobalRowBegin, dispatchRowCount, expertRowOffset);
        return;
    }

    int32_t segmentRowBegin = static_cast<int32_t>(expertRowOffset);
    int32_t segmentRowEnd = segmentRowBegin + static_cast<int32_t>(dispatchRowCount);
    int32_t firstOwnedRow = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    if (firstOwnedRow >= segmentRowEnd) {
        return;
    }

    LoadDispatchExpertRankCounts(context, scratch, localExpertId, expertGlobalRowBegin);
    DispatchCoreTaskTable taskTable = BuildDispatchCoreTaskTable(
        context, scratch, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd,
        firstOwnedRow);
    if (taskTable.taskCount <= 0 ||
        !ResolveDispatchCoreTaskTable(context, params, scratch, localExpertId, taskTable)) {
        return;
    }

    DispatchTransferStreamState stream{};
    int32_t currentTileIdx = -1;
    int32_t currentTileRowCount = 0;
    uint32_t currentTileTransferBegin = 0U;
    for (int32_t taskIdx = 0; taskIdx < taskTable.taskCount; ++taskIdx) {
        int32_t expertRow = firstOwnedRow + taskIdx * coreCount;
        int32_t tileIdx = expertRow / gmm1TileRowCount;
        if (currentTileIdx >= 0 && tileIdx != currentTileIdx) {
            FinishInterleavedDispatchTile<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, params, scratch, localExpertId, currentTileIdx,
                currentTileRowCount, stream, traceStats, currentTileTransferBegin);
            currentTileIdx = -1;
            currentTileRowCount = 0;
            currentTileTransferBegin = 0U;
        }
        if (currentTileIdx < 0) {
            currentTileIdx = tileIdx;
            currentTileTransferBegin = BeginDispatchTransferTrace(traceStats);
        }
        int32_t rankIdx = scratch.topkIndexTensor.GetValue(
            taskTable.rankListOffset + taskIdx);
        int32_t topkIndex = scratch.topkIndexTensor.GetValue(
            taskTable.topkListOffset + taskIdx);
        int32_t destinationRow = expertGlobalRowBegin + expertRow;
        AppendDispatchTokenToStream<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, rankIdx,
            topkIndex, destinationRow, stream, traceStats);
        ++currentTileRowCount;
    }
    if (currentTileIdx >= 0) {
        FinishInterleavedDispatchTile<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, localExpertId, currentTileIdx,
            currentTileRowCount, stream, traceStats, currentTileTransferBegin);
    }
}

/**
 * Dispatch 第 2 层：将当前专家中本 core 负责的 row range 映射到重叠的
 * source-rank segment，并逐段调用 DispatchRankTokens。
 */
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensByRank(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, const DispatchExpertRange &expertRange,
    const DispatchExpertRankRange &rankRange,
    int32_t gmm1TileRowCount)
{
    int32_t rankSegmentRowBegin = rankRange.firstRankOffset;
    int32_t rankSegmentRowEnd = 0;
    for (int32_t remoteRankIdx = static_cast<int32_t>(rankRange.firstRankIdx);
         remoteRankIdx <= rankRange.lastRankIdx; ++remoteRankIdx) {
        if (remoteRankIdx > static_cast<int32_t>(rankRange.firstRankIdx)) {
            rankSegmentRowBegin = rankSegmentRowEnd;
        }
        rankSegmentRowEnd = scratch.cumsumInfoTensor.GetValue(
                                localExpertId * context.common.worldSize + static_cast<uint32_t>(remoteRankIdx)) -
                            expertGlobalRowBegin;

        int32_t segmentMatchOrdinalBegin =
            expertRange.coreExpertRowBegin > rankSegmentRowBegin ?
                expertRange.coreExpertRowBegin - rankSegmentRowBegin :
                0;
        int32_t segmentMatchOrdinalEnd = expertRange.coreExpertRowEnd - rankSegmentRowBegin;
        int32_t rankSegmentRowCount = rankSegmentRowEnd - rankSegmentRowBegin;
        if (segmentMatchOrdinalEnd > rankSegmentRowCount) {
            segmentMatchOrdinalEnd = rankSegmentRowCount;
        }
        if (segmentMatchOrdinalEnd <= segmentMatchOrdinalBegin) {
            continue;
        }

        DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, localExpertId,
            expertGlobalRowBegin, static_cast<uint32_t>(remoteRankIdx),
            rankSegmentRowBegin, segmentMatchOrdinalBegin, segmentMatchOrdinalEnd,
            gmm1TileRowCount);
    }
}

/** Dispatch one expert with the rank-shard partition required by A8W4 dynamic waves. */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensByRankShard(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId)
{
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    int32_t expertGlobalRowBegin = localExpertId == 0U ? 0 :
                                                         scratch.cumsumInfoTensor.GetValue(localExpertId * context.common.worldSize - 1U);
    uint32_t rankShardCount = context.countWorkspace.blockNum / context.common.worldSize;
    rankShardCount = rankShardCount == 0U ? 1U : rankShardCount;

    for (uint32_t dispatchShardIdx = context.blockJob.jobIndex;
         dispatchShardIdx < context.common.worldSize * rankShardCount;
         dispatchShardIdx += context.blockJob.totalJobs) {
        uint32_t remoteRankIdx = dispatchShardIdx / rankShardCount;
        uint32_t rankShardIdx = dispatchShardIdx % rankShardCount;
        uint32_t rankSegmentDstRowBegin =
            (remoteRankIdx == 0U && localExpertId == 0U) ?
                0U :
                static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(
                    localExpertId * context.common.worldSize + remoteRankIdx - 1U));
        if (rankSegmentDstRowBegin >= context.maxOutputSize) {
            continue;
        }

        int32_t rankTokenCount =
            scratch.cumsumInfoTensor.GetValue(localExpertId * context.common.worldSize + remoteRankIdx) -
            static_cast<int32_t>(rankSegmentDstRowBegin);
        int32_t rankDispatchRowCount =
            rankSegmentDstRowBegin + rankTokenCount > context.maxOutputSize ?
                static_cast<int32_t>(context.maxOutputSize - rankSegmentDstRowBegin) :
                rankTokenCount;
        int32_t rowsPerRankShard =
            Ops::Base::CeilDiv(rankDispatchRowCount, static_cast<int32_t>(rankShardCount));
        int32_t rankShardRowBegin = static_cast<int32_t>(rankShardIdx) * rowsPerRankShard;
        int32_t dispatchRowCount =
            rankShardRowBegin + rowsPerRankShard > rankDispatchRowCount ?
                rankDispatchRowCount - rankShardRowBegin :
                rowsPerRankShard;
        if (dispatchRowCount <= 0) {
            continue;
        }

        DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, localExpertId, expertGlobalRowBegin,
            remoteRankIdx, static_cast<int32_t>(rankSegmentDstRowBegin) - expertGlobalRowBegin,
            rankShardRowBegin, rankShardRowBegin + dispatchRowCount, gmm1TileRowCount);
    }
}

/**
 * Dispatch 第 1 层：将当前专家接收的 row 均分给 Dispatch core，再将本 core
 * 的 row range 交给 DispatchExpertTokensByRank 处理。
 */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensByRows(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount,
    uint32_t expertRowOffset)
{
    if (dispatchRowCount == 0U) {
        return;
    }

    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t dispatchCoreCount = context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs;
    int32_t rowsPerCore = Ops::Base::CeilDiv(
        static_cast<int32_t>(dispatchRowCount), static_cast<int32_t>(dispatchCoreCount));
    int32_t coreDispatchRowBegin = static_cast<int32_t>(context.blockJob.jobIndex) * rowsPerCore;
    int32_t coreDispatchRowEnd = coreDispatchRowBegin + rowsPerCore;
    if (coreDispatchRowEnd > static_cast<int32_t>(dispatchRowCount)) {
        coreDispatchRowEnd = static_cast<int32_t>(dispatchRowCount);
    }
    if (coreDispatchRowBegin >= coreDispatchRowEnd) {
        return;
    }

    DispatchExpertRange expertRange{
        static_cast<int32_t>(expertRowOffset) + coreDispatchRowBegin,
        static_cast<int32_t>(expertRowOffset) + coreDispatchRowEnd};
    DispatchExpertRankRange rankRange = FindDispatchExpertRankRange(
        context, scratch, localExpertId, expertGlobalRowBegin,
        expertRange.coreExpertRowBegin, expertRange.coreExpertRowEnd);
    if (rankRange.firstRankIdx == static_cast<int32_t>(context.common.worldSize) || rankRange.lastRankIdx < 0) {
        return;
    }

    DispatchExpertTokensByRank<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, winRankAddr, scratch, localExpertId, expertGlobalRowBegin,
        expertRange, rankRange, gmm1TileRowCount);
}

// 计算一个专家的接收 token 数，并发布就绪 flag。
template <typename ActivationType, bool EnableA8W4, bool TopkWeightsPrefetch>
__aicore__ inline void ComputeExpertTokenCountAndNotify(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId, uint64_t &sendCnt)
{
    sendCnt = 0;
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    GlobalTensor<int32_t> countSrcGlobal;
    countSrcGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * context.common.worldSize * maskSlotSize +
        context.maskAlignSize));
    DataCopyExtParams countCopyParams{static_cast<uint16_t>(context.common.worldSize),
                                      static_cast<uint32_t>(sizeof(int32_t)),
                                      static_cast<uint32_t>(maskSlotSize - sizeof(int32_t)), 0U, 0U};
    DataCopyPadExtParams<int32_t> countPad{true, 0U, 0U, 0U};
    DataCopyPad(scratch.sendCntTensor, countSrcGlobal, countCopyParams, countPad);

    if constexpr (EnableA8W4) {
        if (localExpertId != 0) {
            DataCopyPad(
                scratch.cumsumInfoTensor, scratch.cumsumInfoGlobalTensor,
                {1U, static_cast<uint32_t>(context.common.worldSize * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        }
    }
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    if constexpr (TopkWeightsPrefetch) {
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }
    constexpr int32_t countStrideI32 = ALIGN_32 / sizeof(int32_t);
    for (uint32_t rankIdx = 0; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankCount = scratch.sendCntTensor.GetValue(rankIdx * countStrideI32);
        sendCnt += static_cast<uint64_t>(rankCount);
        scratch.cumsumRevCntInRank += static_cast<uint64_t>(rankCount);
        scratch.cumsumInfoTensor.SetValue(localExpertId * context.common.worldSize + rankIdx,
                                          static_cast<int32_t>(scratch.cumsumRevCntInRank));
    }

    scratch.expertTokenCntTensor.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    uint64_t countOffset = localExpertId * INT32_PER_256B * context.countWorkspace.blockNum +
                           INT32_PER_256B * context.countWorkspace.blockIdx;
    DataCopy<int32_t>(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenCntTensor, INT32_PER_256B);
    if constexpr (EnableA8W4) {
        DataCopyPad(
            scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
            {1U, static_cast<uint32_t>(context.common.worldSize * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t *sendCntFlag =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(localExpertId) * context.countWorkspace.blockNum * INT_CACHELINE +
        static_cast<uint64_t>(context.countWorkspace.blockIdx) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// 导出 Token Dispatch 生成的逐专家接收 token 数。
template <typename ActivationType, bool ReloadCumsumFromGm>
__aicore__ inline void ExportExpertTokenCounts(const TokenDispatchConfig &context,
                                               TokenDispatchScratch<ActivationType> &scratch,
                                               const Params &params)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if constexpr (ReloadCumsumFromGm) {
        DataCopyPad(scratch.cumsumInfoTensor, scratch.cumsumInfoGlobalTensor,
                    {1U,
                     static_cast<uint32_t>(
                         context.common.worldSize * context.common.moeExpertPerRank * sizeof(int32_t)),
                     0U, 0U, 0U},
                    {true, 0U, 0U, 0U});
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(context.common.worldSize - 1U);
    scratch.expertTokenNumsOutTensor.SetValue(0, scratch.cumsumInfoTensor.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < static_cast<int32_t>(context.common.moeExpertPerRank); ++expertIdx) {
        int32_t currentCount = scratch.cumsumInfoTensor.GetValue(
            expertIdx * static_cast<int32_t>(context.common.worldSize) + lastRankIdx);
        int32_t previousCount = scratch.cumsumInfoTensor.GetValue(
            (expertIdx - 1) * static_cast<int32_t>(context.common.worldSize) + lastRankIdx);
        scratch.expertTokenNumsOutTensor.SetValue(expertIdx, currentCount - previousCount);
    }
    SyncFuncStatic<HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, static_cast<uint32_t>(context.common.moeExpertPerRank * sizeof(int32_t)), 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// 从稳定的物理 workspace 槽导出专家 token 数。
__aicore__ inline void ExportExpertTokenCountsFromWorkspace(
    const TokenDispatchConfig &context, const Params &params, ExpertTokenCountExportScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || context.blockJob.jobIndex != 0U) {
        return;
    }
    uint32_t expertPerRank = context.common.moeExpertPerRank;
    GlobalTensor<int32_t> expertRevTokenNums;
    expertRevTokenNums.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.expertRevTokenNumsPtr));
    uint32_t expertCountStrideBytes =
        context.countWorkspace.blockNum * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    DataCopyPad(scratch.stridedTensor,
                expertRevTokenNums[context.countWorkspace.blockIdx * static_cast<uint32_t>(INT32_PER_256B)],
                {static_cast<uint16_t>(expertPerRank), static_cast<uint32_t>(sizeof(int32_t)),
                 expertCountStrideBytes - static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U},
                {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    for (uint32_t expertIdx = 0U; expertIdx < expertPerRank; ++expertIdx) {
        int32_t tokenCount =
            scratch.stridedTensor.GetValue(expertIdx * static_cast<uint32_t>(INT32_PER_256B));
        scratch.compactTensor.SetValue(expertIdx, tokenCount);
    }
    SyncFuncStatic<HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.compactTensor,
                {1U, static_cast<uint32_t>(expertPerRank * sizeof(int32_t)), 0U, 0U, 0U});
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

// 执行一个 MoE 专家的 SendCount 和 Token Dispatch。
template <typename ActivationType, typename QuantScaleType, bool EnableA8W4, uint32_t Gmm1TileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void RunMoeExpertDispatchStage(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t expertIdx, uint64_t &sendCnt)
{
    sendCnt = 0U;
    if constexpr (g_coreType == AIV) {
        if (GetSubBlockIdx() != 1U) {
            return;
        }
        ComputeExpertTokenCountAndNotify<ActivationType, EnableA8W4, TopkWeightsPrefetch>(
            context, params, scratch, expertIdx, sendCnt);
        if (sendCnt != 0U) {
            uint64_t expertGlobalRowBegin = expertIdx == 0U ? 0U :
                                                              static_cast<uint64_t>(
                                                                  scratch.cumsumInfoTensor.GetValue(expertIdx * context.common.worldSize - 1U));
            uint64_t dispatchRowCount = expertGlobalRowBegin < context.maxOutputSize ?
                                            context.maxOutputSize - expertGlobalRowBegin :
                                            0U;
            if (dispatchRowCount > sendCnt) {
                dispatchRowCount = sendCnt;
            }
            if constexpr (EnableA8W4) {
                DispatchExpertTokensByRankShard<ActivationType, QuantScaleType, Gmm1TileM, TopkWeightsPrefetch>(
                    context, params, winRankAddr, scratch, expertIdx);
            } else {
                DispatchExpertTokensByRows<ActivationType, QuantScaleType, Gmm1TileM, TopkWeightsPrefetch>(
                    context, params, winRankAddr, scratch, expertIdx,
                    static_cast<int32_t>(expertGlobalRowBegin), static_cast<uint32_t>(dispatchRowCount));
            }
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_H
