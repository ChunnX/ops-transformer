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
    // Per-rank physical mask scan state for one expert. This is deliberately
    // isolated from sendCntTensor: counts and scan cursors have independent
    // ownership and lifetimes even though both are indexed by source rank.
    LocalTensor<int32_t> rankScanCursorTensor;
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
                                                     GlobalTensor<ActivationType> &remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(context.common.topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
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
constexpr int32_t DISPATCH_RANK_SCAN_CURSOR_STRIDE = 2;
constexpr int32_t DISPATCH_RANK_SCAN_ROUTE_BEGIN_LANE = 0;
constexpr int32_t DISPATCH_RANK_SCAN_MATCHED_COUNT_LANE = 1;
constexpr int32_t DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS = ALIGN_32 / sizeof(int32_t);
constexpr uint32_t DISPATCH_ROUTE_RANK_SHIFT = 24U;
constexpr uint32_t DISPATCH_ROUTE_TOPK_MASK = (1U << DISPATCH_ROUTE_RANK_SHIFT) - 1U;
constexpr uint32_t DISPATCH_ROUTE_RANK_CAPACITY = 1U << (32U - DISPATCH_ROUTE_RANK_SHIFT);

// One compact route word is staged per local task. Queue position determines
// destination row; the word keeps only source rank and source topkIndex.
struct DispatchCoreRouteQueue {
    int32_t taskCount = 0;
    int32_t taskCapacity = 0;
    int32_t routeIndexOffset = 0;
    int32_t routeCapacity = 0;
};

__aicore__ inline int32_t PackDispatchRouteEntry(uint32_t rankIdx, int32_t topkIndex)
{
    return static_cast<int32_t>(
        (rankIdx << DISPATCH_ROUTE_RANK_SHIFT) |
        (static_cast<uint32_t>(topkIndex) & DISPATCH_ROUTE_TOPK_MASK));
}

__aicore__ inline uint32_t GetDispatchRouteRank(int32_t routeEntry)
{
    return static_cast<uint32_t>(routeEntry) >> DISPATCH_ROUTE_RANK_SHIFT;
}

__aicore__ inline int32_t GetDispatchRouteTopkIndex(int32_t routeEntry)
{
    return static_cast<int32_t>(
        static_cast<uint32_t>(routeEntry) & DISPATCH_ROUTE_TOPK_MASK);
}

struct DispatchRankRoundInterval {
    int32_t roundBegin = 0;
    int32_t roundEnd = 0;
    int32_t intervalRowBegin = 0;
    int32_t activeRankCount = 0;
    int32_t rankPosition = -1;
};

struct DispatchOwnedOrdinalRange {
    int32_t firstDelta = 0;
    int32_t endDelta = 0;
    int32_t stride = 1;
};

// Physical mask position corresponding to a logical per-rank token ordinal.
// routeBegin is always ALIGN_256 route-item aligned; matchedCount is the
// number of set mask bits strictly before routeBegin. Keeping both values lets
// a later segment resume GatherMask without rescanning the expert prefix.
struct DispatchRankScanCursor {
    int32_t routeBegin = 0;
    int32_t matchedCount = 0;
};

struct DispatchCoreRankPlan {
    int32_t taskCount = 0;
    int32_t segmentOrdinalBegin = 0;
    int32_t segmentOrdinalEnd = 0;
};

template <typename ActivationType>
__aicore__ inline void ResetDispatchRankScanCursors(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch)
{
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t cursorBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCAN_CURSOR_STRIDE;
        scratch.rankScanCursorTensor.SetValue(
            cursorBase + DISPATCH_RANK_SCAN_ROUTE_BEGIN_LANE, 0);
        scratch.rankScanCursorTensor.SetValue(
            cursorBase + DISPATCH_RANK_SCAN_MATCHED_COUNT_LANE, 0);
    }
}

template <typename ActivationType>
__aicore__ inline DispatchRankScanCursor LoadDispatchRankScanCursor(
    const TokenDispatchScratch<ActivationType> &scratch, uint32_t remoteRankIdx)
{
    int32_t cursorBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCAN_CURSOR_STRIDE;
    return {
        scratch.rankScanCursorTensor.GetValue(
            cursorBase + DISPATCH_RANK_SCAN_ROUTE_BEGIN_LANE),
        scratch.rankScanCursorTensor.GetValue(
            cursorBase + DISPATCH_RANK_SCAN_MATCHED_COUNT_LANE)};
}

template <typename ActivationType>
__aicore__ inline void StoreDispatchRankScanCursor(
    TokenDispatchScratch<ActivationType> &scratch, uint32_t remoteRankIdx,
    const DispatchRankScanCursor &cursor)
{
    int32_t cursorBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCAN_CURSOR_STRIDE;
    scratch.rankScanCursorTensor.SetValue(
        cursorBase + DISPATCH_RANK_SCAN_ROUTE_BEGIN_LANE, cursor.routeBegin);
    scratch.rankScanCursorTensor.SetValue(
        cursorBase + DISPATCH_RANK_SCAN_MATCHED_COUNT_LANE, cursor.matchedCount);
}

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

__aicore__ inline int32_t DispatchPositiveGcd(int32_t lhs, int32_t rhs)
{
    while (rhs != 0) {
        int32_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

template <typename ActivationType>
__aicore__ inline bool GetDispatchRankRoundInterval(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch,
    uint32_t remoteRankIdx, int32_t roundBegin, int32_t intervalRowBegin,
    DispatchRankRoundInterval &interval)
{
    int32_t activeRankCount = 0;
    int32_t rankPosition = -1;
    int32_t roundEnd = 0x7fffffff;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_COUNT_LANE);
        if (rankTokenCount <= roundBegin) {
            continue;
        }
        if (rankIdx == remoteRankIdx) {
            rankPosition = activeRankCount;
        }
        ++activeRankCount;
        if (rankTokenCount < roundEnd) {
            roundEnd = rankTokenCount;
        }
    }
    if (activeRankCount == 0 || rankPosition < 0) {
        return false;
    }
    interval = {roundBegin, roundEnd, intervalRowBegin, activeRankCount, rankPosition};
    return true;
}

__aicore__ inline bool GetDispatchOwnedOrdinalRange(
    const TokenDispatchConfig &context, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd,
    int32_t ordinalWindowBegin, int32_t ordinalWindowEnd,
    const DispatchRankRoundInterval &interval, DispatchOwnedOrdinalRange &ownedRange)
{
    int32_t intervalRoundCount = interval.roundEnd - interval.roundBegin;
    int32_t rankRowBase = interval.intervalRowBegin + interval.rankPosition;
    if (intervalRoundCount <= 0 || rankRowBase >= segmentRowEnd ||
        ordinalWindowEnd <= interval.roundBegin || ordinalWindowBegin >= interval.roundEnd) {
        return false;
    }

    int32_t firstDelta = 0;
    if (rankRowBase < segmentRowBegin) {
        firstDelta = Ops::Base::CeilDiv(
            segmentRowBegin - rankRowBase, interval.activeRankCount);
    }
    int32_t endDelta = intervalRoundCount;
    if (rankRowBase >= segmentRowEnd) {
        endDelta = 0;
    } else {
        int32_t segmentEndDelta = Ops::Base::CeilDiv(
            segmentRowEnd - rankRowBase, interval.activeRankCount);
        if (segmentEndDelta < endDelta) {
            endDelta = segmentEndDelta;
        }
    }

    int32_t windowFirstDelta = ordinalWindowBegin > interval.roundBegin ?
        ordinalWindowBegin - interval.roundBegin : 0;
    int32_t windowEndDelta = ordinalWindowEnd - interval.roundBegin;
    if (windowEndDelta > intervalRoundCount) {
        windowEndDelta = intervalRoundCount;
    }
    if (windowFirstDelta > firstDelta) {
        firstDelta = windowFirstDelta;
    }
    if (windowEndDelta < endDelta) {
        endDelta = windowEndDelta;
    }
    if (firstDelta >= endDelta) {
        return false;
    }

    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t coreIdx = static_cast<int32_t>(context.blockJob.jobIndex);
    int32_t stride = coreCount / DispatchPositiveGcd(coreCount, interval.activeRankCount);
    int32_t baseDestinationRow = expertGlobalRowBegin + rankRowBase;
    int32_t searchEnd = firstDelta + stride;
    if (searchEnd > endDelta) {
        searchEnd = endDelta;
    }
    while (firstDelta < searchEnd &&
           (baseDestinationRow + firstDelta * interval.activeRankCount) % coreCount != coreIdx) {
        ++firstDelta;
    }
    if (firstDelta >= searchEnd) {
        return false;
    }
    ownedRange = {firstDelta, endDelta, stride};
    return true;
}

__aicore__ inline int32_t GetDispatchRankOrdinalBeforeRow(
    const DispatchRankRoundInterval &interval, int32_t expertRow)
{
    if (expertRow <= interval.intervalRowBegin) {
        return interval.roundBegin;
    }
    int32_t intervalRoundCount = interval.roundEnd - interval.roundBegin;
    int32_t intervalRowCount = intervalRoundCount * interval.activeRankCount;
    int32_t coveredRowCount = expertRow - interval.intervalRowBegin;
    if (coveredRowCount >= intervalRowCount) {
        return interval.roundEnd;
    }
    int32_t coveredRoundCount = coveredRowCount / interval.activeRankCount;
    int32_t partialRoundRows = coveredRowCount % interval.activeRankCount;
    int32_t ordinal = interval.roundBegin + coveredRoundCount;
    if (partialRoundRows > interval.rankPosition) {
        ++ordinal;
    }
    return ordinal;
}

template <typename ActivationType>
__aicore__ inline DispatchCoreRankPlan GetDispatchCoreRankPlan(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType> &scratch,
    uint32_t remoteRankIdx, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd)
{
    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_COUNT_LANE);
    DispatchCoreRankPlan plan{};
    bool ordinalBeginResolved = false;
    bool ordinalEndResolved = false;
    int32_t roundBegin = 0;
    int32_t intervalRowBegin = 0;
    while (roundBegin < rankTokenCount && intervalRowBegin < segmentRowEnd) {
        DispatchRankRoundInterval interval{};
        if (!GetDispatchRankRoundInterval(
                context, scratch, remoteRankIdx, roundBegin, intervalRowBegin, interval)) {
            break;
        }
        DispatchOwnedOrdinalRange ownedRange{};
        if (GetDispatchOwnedOrdinalRange(
                context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd,
                0, rankTokenCount, interval, ownedRange)) {
            plan.taskCount += (ownedRange.endDelta - 1 - ownedRange.firstDelta) /
                ownedRange.stride + 1;
        }
        int32_t intervalRowEnd = interval.intervalRowBegin +
            (interval.roundEnd - interval.roundBegin) * interval.activeRankCount;
        if (!ordinalBeginResolved && segmentRowBegin <= intervalRowEnd) {
            plan.segmentOrdinalBegin = GetDispatchRankOrdinalBeforeRow(
                interval, segmentRowBegin);
            ordinalBeginResolved = true;
        }
        if (!ordinalEndResolved && segmentRowEnd <= intervalRowEnd) {
            plan.segmentOrdinalEnd = GetDispatchRankOrdinalBeforeRow(
                interval, segmentRowEnd);
            ordinalEndResolved = true;
        }
        intervalRowBegin = intervalRowEnd;
        roundBegin = interval.roundEnd;
    }
    // A rank that exhausted its tokens before either segment boundary
    // contributes its complete ordinal range to all later logical rows.
    if (!ordinalBeginResolved) {
        plan.segmentOrdinalBegin = rankTokenCount;
    }
    if (!ordinalEndResolved) {
        plan.segmentOrdinalEnd = rankTokenCount;
    }
    return plan;
}

template <typename ActivationType>
__aicore__ inline int32_t LowerBoundDispatchRouteIndex(
    const TokenDispatchScratch<ActivationType> &scratch,
    int32_t valueCount, int32_t targetRouteIndex)
{
    int32_t lower = 0;
    int32_t upper = valueCount;
    while (lower < upper) {
        int32_t middle = lower + (upper - lower) / 2;
        if (scratch.validTopkIndexTensor.GetValue(middle) < targetRouteIndex) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower;
}

template <typename ActivationType>
__aicore__ inline int32_t PlanDispatchCoreRankRoutes(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    uint32_t remoteRankIdx, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd, int32_t firstOwnedRow,
    const DispatchCoreRouteQueue &queue, const DispatchCoreRankPlan &rankPlan)
{
    if (rankPlan.taskCount <= 0) {
        return 0;
    }

    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_COUNT_LANE);
    uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    LocalTensor<int32_t> routeIndexTensor = scratch.topkIndexTensor[queue.routeIndexOffset];
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t alignedSendTotalNum = Ops::Base::CeilAlign(
        static_cast<int32_t>(sendTotalNum), static_cast<int32_t>(ALIGN_256));
    DispatchRankScanCursor scanCursor{};
    if (segmentRowBegin > 0) {
        scanCursor = LoadDispatchRankScanCursor(scratch, remoteRankIdx);
    }
    bool validScanCursor =
        scanCursor.routeBegin >= 0 && scanCursor.routeBegin <= alignedSendTotalNum &&
        scanCursor.routeBegin % static_cast<int32_t>(ALIGN_256) == 0 &&
        scanCursor.matchedCount >= 0 && scanCursor.matchedCount <= rankTokenCount &&
        scanCursor.matchedCount <= rankPlan.segmentOrdinalBegin;
    if (!validScanCursor) {
        scanCursor = {};
    }
    int32_t matchedRouteCount = scanCursor.matchedCount;
    int32_t plannedTaskCount = 0;
    bool nextCursorStored = false;

    // GatherMask returns this rank's routes in monotonically increasing ordinal
    // order. Keep one interval cursor for the whole rank scan instead of
    // rebuilding the rank-round decomposition from round zero for every mask
    // batch.
    int32_t planRoundBegin = 0;
    int32_t planIntervalRowBegin = 0;
    DispatchRankRoundInterval planInterval{};
    bool hasPlanInterval = GetDispatchRankRoundInterval(
        context, scratch, remoteRankIdx, planRoundBegin,
        planIntervalRowBegin, planInterval);

    for (int32_t routeBegin = scanCursor.routeBegin;
         static_cast<uint64_t>(routeBegin) < sendTotalNum &&
         matchedRouteCount < rankTokenCount &&
         (plannedTaskCount < rankPlan.taskCount || !nextCursorStored);
         routeBegin += queue.routeCapacity) {
        int32_t validRouteCount = static_cast<int32_t>(
            sendTotalNum - static_cast<uint64_t>(routeBegin));
        if (validRouteCount > queue.routeCapacity) {
            validRouteCount = queue.routeCapacity;
        }
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
        // routeIndexTensor does not depend on the mask payload. Generate it
        // while MTE2 is loading the mask, then fence only before GatherMask.
        CreateVecIndex(routeIndexTensor, routeBegin, queue.routeCapacity);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
        uint64_t batchMatchedRouteCount = 0U;
        GatherMask(scratch.validTopkIndexTensor, routeIndexTensor, scratch.maskBatchU32Tensor, true,
                   static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

        int32_t batchOrdinalBegin = matchedRouteCount;
        int32_t batchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);

        // Skip intervals that end before this mask batch. This cursor only
        // moves forward across batches, so each rank-round interval is derived
        // at most once for the complete rank scan.
        while (hasPlanInterval && planInterval.roundEnd <= batchOrdinalBegin) {
            planIntervalRowBegin = planInterval.intervalRowBegin +
                (planInterval.roundEnd - planInterval.roundBegin) *
                    planInterval.activeRankCount;
            planRoundBegin = planInterval.roundEnd;
            hasPlanInterval = GetDispatchRankRoundInterval(
                context, scratch, remoteRankIdx, planRoundBegin,
                planIntervalRowBegin, planInterval);
        }

        while (hasPlanInterval && planInterval.roundBegin < batchOrdinalEnd &&
               planInterval.intervalRowBegin < segmentRowEnd) {
            DispatchOwnedOrdinalRange ownedRange{};
            if (GetDispatchOwnedOrdinalRange(
                    context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd,
                    batchOrdinalBegin, batchOrdinalEnd, planInterval, ownedRange)) {
                for (int32_t delta = ownedRange.firstDelta;
                     delta < ownedRange.endDelta; delta += ownedRange.stride) {
                    int32_t rankOrdinal = planInterval.roundBegin + delta;
                    int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
                        rankOrdinal - batchOrdinalBegin);
                    int32_t expertRow = planInterval.intervalRowBegin +
                        delta * planInterval.activeRankCount + planInterval.rankPosition;
                    int32_t taskIdx = (expertRow - firstOwnedRow) / coreCount;
                    scratch.topkIndexTensor.SetValue(
                        taskIdx, PackDispatchRouteEntry(remoteRankIdx, topkIndex));
                    ++plannedTaskCount;
                }
            }

            // The next mask batch can still lie in the current round interval.
            // Preserve it in that case; otherwise advance exactly once.
            if (batchOrdinalEnd < planInterval.roundEnd) {
                break;
            }
            planIntervalRowBegin = planInterval.intervalRowBegin +
                (planInterval.roundEnd - planInterval.roundBegin) *
                    planInterval.activeRankCount;
            planRoundBegin = planInterval.roundEnd;
            hasPlanInterval = GetDispatchRankRoundInterval(
                context, scratch, remoteRankIdx, planRoundBegin,
                planIntervalRowBegin, planInterval);
        }
        if (!nextCursorStored && batchOrdinalEnd >= rankPlan.segmentOrdinalEnd) {
            DispatchRankScanCursor nextCursor{};
            if (rankPlan.segmentOrdinalEnd >= rankTokenCount) {
                nextCursor = {alignedSendTotalNum, rankTokenCount};
            } else if (rankPlan.segmentOrdinalEnd == batchOrdinalEnd) {
                nextCursor = {
                    routeBegin + validRouteCount,
                    batchOrdinalEnd};
            } else {
                int32_t nextTopkIndex = scratch.validTopkIndexTensor.GetValue(
                    rankPlan.segmentOrdinalEnd - batchOrdinalBegin);
                int32_t nextRouteBegin = nextTopkIndex /
                    static_cast<int32_t>(ALIGN_256) * static_cast<int32_t>(ALIGN_256);
                int32_t matchedInBatchBeforeNextRoute = LowerBoundDispatchRouteIndex(
                    scratch, static_cast<int32_t>(batchMatchedRouteCount), nextRouteBegin);
                nextCursor = {
                    nextRouteBegin,
                    batchOrdinalBegin + matchedInBatchBeforeNextRoute};
            }
            StoreDispatchRankScanCursor(scratch, remoteRankIdx, nextCursor);
            nextCursorStored = true;
        }
        matchedRouteCount = batchOrdinalEnd;
    }
    return plannedTaskCount;
}

template <typename ActivationType>
__aicore__ inline DispatchCoreRouteQueue PlanDispatchCoreRouteQueue(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, int32_t segmentRowBegin, int32_t segmentRowEnd,
    int32_t firstOwnedRow, int32_t expectedTaskCount, int32_t taskCapacity)
{
    DispatchCoreRouteQueue queue{};
    queue.taskCapacity = taskCapacity;
    queue.routeIndexOffset = queue.taskCapacity;
    queue.routeCapacity = context.bufferConfig.routeItemsPerBatch - queue.routeIndexOffset;
    queue.routeCapacity = queue.routeCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    if (queue.routeCapacity <= 0) {
        queue.taskCapacity = 0;
        return queue;
    }

    int32_t plannedTotalTaskCount = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        DispatchCoreRankPlan rankPlan = GetDispatchCoreRankPlan(
            context, scratch, rankIdx, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
        int32_t plannedTaskCount = PlanDispatchCoreRankRoutes(
            context, params, scratch, localExpertId, rankIdx, expertGlobalRowBegin,
            segmentRowBegin, segmentRowEnd, firstOwnedRow, queue, rankPlan);
        if (plannedTaskCount != rankPlan.taskCount) {
            queue.taskCount = 0;
            return queue;
        }
        plannedTotalTaskCount += plannedTaskCount;
    }
    queue.taskCount = plannedTotalTaskCount == expectedTaskCount ? plannedTotalTaskCount : 0;
    return queue;
}

// Issue one compact route task into the Dispatch MTE2 ring. Changing source
// rank only redirects the GM view; it does not drain or restart the ring.
template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline int32_t FetchInterleavedDispatchRouteTask(
    const TokenDispatchConfig &context, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t taskIdx,
    int32_t bufferIdx, GlobalTensor<ActivationType> &remoteRankGlobalTensor,
    int32_t &currentRankIdx)
{
    int32_t routeEntry = scratch.topkIndexTensor.GetValue(taskIdx);
    int32_t remoteRankIdx = static_cast<int32_t>(GetDispatchRouteRank(routeEntry));
    int32_t topkIndex = GetDispatchRouteTopkIndex(routeEntry);
    if (remoteRankIdx != currentRankIdx) {
        remoteRankGlobalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ ActivationType *>(
                winRankAddr[remoteRankIdx] + context.quantWinOffset));
        currentRankIdx = remoteRankIdx;
    }
    FetchDispatchTokenAndMetaInfo<IsBufferReuse, TopkWeightsPrefetch>(
        context, scratch, bufferIdx, topkIndex, remoteRankIdx,
        remoteRankGlobalTensor);
    return topkIndex;
}

// Consume the current core's compact route queue with the same four-phase MTE
// ring used by the compact baseline: primer, first-fill, steady reuse and
// epilogue. Route planning and rank changes are outside the synchronization
// decisions, so the hot loops contain no per-token stream-state branches.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyInterleavedDispatchRouteQueue(
    const TokenDispatchConfig &context, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t destinationRowBase,
    int32_t destinationRowStride, int32_t taskCount,
    GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor,
    GlobalTensor<int32_t> &metaInfoGlobalTensor)
{
    if (taskCount <= 0) {
        return;
    }
    int32_t bufferCount = context.bufferConfig.bufferCount;
    if (bufferCount <= 0) {
        return;
    }

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    int32_t currentRankIdx = -1;
    int32_t pendingTopkIndex =
        FetchInterleavedDispatchRouteTask<false, TopkWeightsPrefetch>(
            context, winRankAddr, scratch, 0, 0, remoteRankGlobalTensor,
            currentRankIdx);

    // A one-slot ring cannot overlap the next fetch with the current store.
    // Keep it correct without adding a buffer-count branch to the normal path.
    if (bufferCount == 1) {
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, 0, destinationRowBase, tokenRevGlobalTensor,
            scaleRevGlobalTensor, metaInfoGlobalTensor, pendingTopkIndex);
        for (int32_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
            pendingTopkIndex =
                FetchInterleavedDispatchRouteTask<true, TopkWeightsPrefetch>(
                    context, winRankAddr, scratch, taskIdx, 0,
                    remoteRankGlobalTensor, currentRankIdx);
            StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, scratch, 0,
                destinationRowBase + taskIdx * destinationRowStride,
                tokenRevGlobalTensor, scaleRevGlobalTensor,
                metaInfoGlobalTensor, pendingTopkIndex);
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(0));
        return;
    }

    int32_t firstUseEnd = taskCount < bufferCount ? taskCount : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t currentTopkIndex =
            FetchInterleavedDispatchRouteTask<false, TopkWeightsPrefetch>(
                context, winRankAddr, scratch, issueIdx, issueIdx,
                remoteRankGlobalTensor, currentRankIdx);
        int32_t copyIdx = issueIdx - 1;
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyIdx,
            destinationRowBase + copyIdx * destinationRowStride,
            tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            pendingTopkIndex);
        pendingTopkIndex = currentTopkIndex;
    }

    for (int32_t issueIdx = bufferCount; issueIdx < taskCount; ++issueIdx) {
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t currentTopkIndex =
            FetchInterleavedDispatchRouteTask<true, TopkWeightsPrefetch>(
                context, winRankAddr, scratch, issueIdx, issueBufferIdx,
                remoteRankGlobalTensor, currentRankIdx);
        int32_t copyIdx = issueIdx - 1;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyBufferIdx,
            destinationRowBase + copyIdx * destinationRowStride,
            tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            pendingTopkIndex);
        pendingTopkIndex = currentTopkIndex;
    }

    int32_t lastTaskIdx = taskCount - 1;
    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, lastTaskIdx % bufferCount,
        destinationRowBase + lastTaskIdx * destinationRowStride,
        tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
        pendingTopkIndex);

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// The rank-round path keeps one MTE stream alive for the complete expert
// segment. After that stream is drained once, publish this core's exact row
// contribution to every GMM1 tile covered by the segment.
__aicore__ inline void PublishInterleavedDispatchTilesReady(
    const TokenDispatchConfig &context, const Params &params,
    uint32_t localExpertId, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd, int32_t gmm1TileRowCount)
{
    if (segmentRowBegin >= segmentRowEnd) {
        return;
    }

    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();

    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t firstTileIdx = segmentRowBegin / gmm1TileRowCount;
    int32_t lastTileIdx = (segmentRowEnd - 1) / gmm1TileRowCount;
    __gm__ int32_t *gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * context.dispatchFlagSlotsPerExpert;

    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileRowBegin = tileIdx * gmm1TileRowCount;
        int32_t tileRowEnd = tileRowBegin + gmm1TileRowCount;
        int32_t overlapRowBegin = segmentRowBegin > tileRowBegin ? segmentRowBegin : tileRowBegin;
        int32_t overlapRowEnd = segmentRowEnd < tileRowEnd ? segmentRowEnd : tileRowEnd;
        int32_t firstOwnedRow = GetFirstDispatchRowOwnedByCore(
            context, expertGlobalRowBegin, overlapRowBegin, overlapRowEnd);
        if (firstOwnedRow >= overlapRowEnd) {
            continue;
        }
        int32_t readyRowCount = (overlapRowEnd - 1 - firstOwnedRow) / coreCount + 1;
        AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE,
                  readyRowCount);
    }
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
// lets every core make one uniform capacity decision before reserving the
// compact route queue; no core can silently skip ready publication.
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensByRows(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount,
    uint32_t expertRowOffset = 0U);

// Dispatch one expert segment with the rank-round destination layout. Planning
// writes one compact {rank, topkIndex} word directly to its final local task
// position. Execution then walks monotonically increasing destination rows;
// no ordinal/destination descriptor or inverse layer mapping is needed.
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensRankRoundRobin(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount, uint32_t expertRowOffset)
{
    if (dispatchRowCount == 0U) {
        return;
    }

    int32_t segmentRowBegin = static_cast<int32_t>(expertRowOffset);
    int32_t segmentRowEnd = segmentRowBegin + static_cast<int32_t>(dispatchRowCount);
    int32_t expertGlobalRowEnd = scratch.cumsumInfoTensor.GetValue(
        localExpertId * context.common.worldSize + context.common.worldSize - 1U);
    int32_t expertRowCount = expertGlobalRowEnd - expertGlobalRowBegin;
    uint64_t remainingOutputRows = static_cast<uint64_t>(expertGlobalRowBegin) < context.maxOutputSize ?
        context.maxOutputSize - static_cast<uint64_t>(expertGlobalRowBegin) : 0U;
    if (static_cast<uint64_t>(expertRowCount) > remainingOutputRows) {
        expertRowCount = static_cast<int32_t>(remainingOutputRows);
    }
    // The first segment never consumes a stale cursor. Only a genuinely split
    // expert needs every core (including a core with no first-segment task) to
    // clear its private state for a later continuation.
    if (segmentRowBegin == 0 && segmentRowEnd < expertRowCount) {
        ResetDispatchRankScanCursors(context, scratch);
    }
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t maxCoreTaskCount = Ops::Base::CeilDiv(
        static_cast<int32_t>(dispatchRowCount), coreCount);
    int32_t maxTaskCapacity = Ops::Base::CeilAlign(
        maxCoreTaskCount, DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS);
    int32_t routeCapacity = context.bufferConfig.routeItemsPerBatch - maxTaskCapacity;
    routeCapacity = routeCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    uint64_t sendTotalNum =
        static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    bool canPackRouteEntry =
        context.common.worldSize <= DISPATCH_ROUTE_RANK_CAPACITY &&
        sendTotalNum <= static_cast<uint64_t>(DISPATCH_ROUTE_TOPK_MASK) + 1U;
    if (routeCapacity <= 0 || !canPackRouteEntry) {
        DispatchExpertTokensByRows<ActivationType, QuantScaleType, PipelineTileM,
                                   TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, localExpertId,
            expertGlobalRowBegin, dispatchRowCount, expertRowOffset);
        return;
    }

    int32_t firstOwnedRow = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    if (firstOwnedRow >= segmentRowEnd) {
        return;
    }
    int32_t expectedTaskCount =
        (segmentRowEnd - 1 - firstOwnedRow) / coreCount + 1;

    LoadDispatchExpertRankCounts(context, scratch, localExpertId, expertGlobalRowBegin);
    DispatchCoreRouteQueue queue = PlanDispatchCoreRouteQueue(
        context, params, scratch, localExpertId, expertGlobalRowBegin,
        segmentRowBegin, segmentRowEnd, firstOwnedRow,
        expectedTaskCount, maxTaskCapacity);
    if (queue.taskCount != expectedTaskCount) {
        return;
    }

    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));

    CopyInterleavedDispatchRouteQueue<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, winRankAddr, scratch, expertGlobalRowBegin + firstOwnedRow,
        coreCount, expectedTaskCount, tokenRevGlobalTensor,
        scaleRevGlobalTensor, metaInfoGlobalTensor);
    PublishInterleavedDispatchTilesReady(
        context, params, localExpertId, expertGlobalRowBegin,
        segmentRowBegin, segmentRowEnd, gmm1TileRowCount);
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
