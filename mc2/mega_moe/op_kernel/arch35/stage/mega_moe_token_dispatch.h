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

// Accumulated wall-clock intervals for one Dispatch wave. The trace writer
// emits these counters once per wave, avoiding per-rank and per-batch records.
struct WaveDispatchTraceMetrics {
    uint32_t prepareDispatchCountsCycles = 0U;
    uint32_t rankRoundPlanCycles = 0U;
    uint32_t maskLoadAndGatherCycles = 0U;
    uint32_t transferPipelineCycles = 0U;
    uint32_t readyPublishCycles = 0U;
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

// 每个本核任务保存一个紧凑 route word。queue 下标决定目的 row，word 中只保存
// source rank 和 source topkIndex，避免额外保存目的 row。
struct DispatchCoreRouteQueue {
    // 本核在当前专家分段中最终需要搬运的 token 数。
    int32_t taskCount = 0;
    // topkIndexTensor 前部为最终 route queue 预留的容量。
    int32_t taskCapacity = 0;
    // 临时 CreateVecIndex 区紧跟 route queue，二者复用同一块 UB。
    int32_t routeIndexOffset = 0;
    // 每批最多扫描的 sendMask bit 数，按 ALIGN_256 向下对齐。
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
    // 当前仍然存活的 source rank 的 token ordinal 区间 [roundBegin, roundEnd)。
    int32_t roundBegin = 0;
    int32_t roundEnd = 0;
    // roundBegin 对应的第一个逻辑专家 row。
    int32_t intervalRowBegin = 0;
    // 该 ordinal 区间仍有 token 的 rank 数。
    int32_t activeRankCount = 0;
    // 当前 remoteRank 在轮次偏移后的活跃 rank 顺序中的位置。
    int32_t rankPosition = -1;
};

struct DispatchOwnedOrdinalRange {
    int32_t firstDelta = 0;
    int32_t endDelta = 0;
    int32_t stride = 1;
};

// 保存“rank 内逻辑 token ordinal”对应的物理 mask 扫描位置。routeBegin 始终按
// ALIGN_256 个 route item 对齐；matchedCount 表示 routeBegin 之前有效 bit 的数量。
// 两者共同保存后，跨 wave 的专家分段可以继续 GatherMask，而不必重扫专家前缀。
struct DispatchRankScanCursor {
    int32_t routeBegin = 0;
    int32_t matchedCount = 0;
};

struct DispatchCoreRankPlan {
    // 当前 core 从该 source rank 实际取得的 token 数。
    int32_t taskCount = 0;
    // 当前专家分段边界映射到该 rank compact 序列后的 ordinal 范围。
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
    // 目的 workspace row 仍按 globalRow % coreCount 分核。轮次偏移只改变
    // 每个目的 row 对应哪个 source rank，不改变目的 row 的所有权。
    int32_t firstDestinationRow = expertGlobalRowBegin + segmentRowBegin;
    int32_t owner = firstDestinationRow % coreCount;
    // ownerDelta 把 segment 首行推进到本核拥有的第一行，后续固定跨 coreCount 行。
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
    // 对同一 ordinal round，只有 token 数大于 roundBegin 的 rank 仍参与交织。
    int32_t activeRankCount = 0;
    int32_t rankPosition = -1;
    // 最短 rank 的 token 数决定这一组活跃 rank 能共同维持到哪个 ordinal。
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

    // 当 coreCount 与 activeRankCount 存在公因子时，固定目的 row 步长会与 rank
    // 交织周期发生共振，典型情况是 28 核/4 rank。每当原 core/rank 映射即将重复，
    // 就切分一次 rank-round interval，并将活跃 rank 顺序左移一位。
    // 目的 row 所有权保持不变，只改变该 row 对应的 source rank，因此原有本核
    // route queue、MTE 环形流水和 Dispatch -> GMM1 ready 计数均可保持不变。
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    // 原映射每经过 coreCount/gcd(coreCount, activeRankCount) 个 rank round，
    // 同一 core 又会命中同一 rank；在这个重复点把 rank 顺序左移一位。
    int32_t roundsPerRotation =
        coreCount / DispatchPositiveGcd(coreCount, activeRankCount);
    int32_t rotationEpoch = roundBegin / roundsPerRotation;
    int32_t nextRotationRound = (rotationEpoch + 1) * roundsPerRotation;
    if (nextRotationRound < roundEnd) {
        roundEnd = nextRotationRound;
    }
    int32_t rankRotation = rotationEpoch % activeRankCount;
    rankPosition -= rankRotation;
    if (rankPosition < 0) {
        rankPosition += activeRankCount;
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
    // 当前 rank 在该 interval 中的第 delta 个 token 位于：
    // intervalRowBegin + rankPosition + delta * activeRankCount。
    int32_t intervalRoundCount = interval.roundEnd - interval.roundBegin;
    int32_t rankRowBase = interval.intervalRowBegin + interval.rankPosition;
    if (intervalRoundCount <= 0 || rankRowBase >= segmentRowEnd ||
        ordinalWindowEnd <= interval.roundBegin || ordinalWindowBegin >= interval.roundEnd) {
        return false;
    }

    // 先把完整 interval 裁剪到当前 wave 中该专家的 [segmentRowBegin, segmentRowEnd)。
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

    // 再与当前 GatherMask batch 覆盖的 compact ordinal window 求交。
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
    // 在 rankPosition 固定的 interval 内，目的 row 每次增加 activeRankCount；
    // stride 是再次命中同一 core 所需跨过的 rank ordinal 数。
    int32_t stride = coreCount / DispatchPositiveGcd(coreCount, interval.activeRankCount);
    int32_t baseDestinationRow = expertGlobalRowBegin + rankRowBase;
    int32_t searchEnd = firstDelta + stride;
    if (searchEnd > endDelta) {
        searchEnd = endDelta;
    }
    // 最多检查一个周期，找到该 core 在此 interval 中拥有的第一个 ordinal。
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
    // rankTokenCount 是该专家从 remoteRank 接收的 compact token 总数。
    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_COUNT_LANE);
    DispatchCoreRankPlan plan{};
    bool ordinalBeginResolved = false;
    bool ordinalEndResolved = false;
    int32_t roundBegin = 0;
    int32_t intervalRowBegin = 0;
    // 同时受“某些 rank 提前耗尽”和“轮次偏移边界”影响，一个专家会被拆成
    // 若干 rank-round interval；逐段累计本核任务并解析分段 ordinal 边界。
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
    // 如果该 rank 在专家分段边界前已经耗尽，则后续逻辑 row 对应的 ordinal
    // 边界固定为该 rank 的完整 token 数。
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
    const DispatchCoreRouteQueue &queue, const DispatchCoreRankPlan &rankPlan,
    WaveDispatchTraceMetrics *traceMetrics)
{
    if (rankPlan.taskCount <= 0) {
        return 0;
    }

    // 该函数只处理“当前专家、当前 source rank、当前 core”这一条规划支路：
    // 读取该 rank 的 sendMask，压紧得到 topkIndex，再把属于本核的条目写到
    // 最终本地 route queue 位置。这里不搬 token，也不发布 GMM1 ready。
    // rankTokenCount 决定 compact ordinal 上界；sendTotalNum 决定物理 mask 上界。
    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_COUNT_LANE);
    uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    // 每个 [expert][rank] mask 槽尾部还包含一个 32B 的 count 区域。
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    // topkIndexTensor 前部保存最终 task，后部临时生成当前 mask batch 的物理索引。
    LocalTensor<int32_t> routeIndexTensor = scratch.topkIndexTensor[queue.routeIndexOffset];
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t alignedSendTotalNum = Ops::Base::CeilAlign(
        static_cast<int32_t>(sendTotalNum), static_cast<int32_t>(ALIGN_256));
    // 专家续段优先从上一 wave 保存的位置继续；专家首段必须从 mask 0 开始。
    DispatchRankScanCursor scanCursor{};
    if (segmentRowBegin > 0) {
        scanCursor = LoadDispatchRankScanCursor(scratch, remoteRankIdx);
    }
    // cursor 必须满足物理对齐、compact 计数范围及“不越过本段起点”三个条件；
    // 任一条件不满足都安全回退为从头扫描，只影响性能，不影响结果。
    bool validScanCursor =
        scanCursor.routeBegin >= 0 && scanCursor.routeBegin <= alignedSendTotalNum &&
        scanCursor.routeBegin % static_cast<int32_t>(ALIGN_256) == 0 &&
        scanCursor.matchedCount >= 0 && scanCursor.matchedCount <= rankTokenCount &&
        scanCursor.matchedCount <= rankPlan.segmentOrdinalBegin;
    if (!validScanCursor) {
        scanCursor = {};
    }
    // matchedRouteCount 是当前物理 mask 位置之前已有多少个有效 bit，即下一个
    // GatherMask batch 在 rank compact 序列中的起始 ordinal。
    int32_t matchedRouteCount = scanCursor.matchedCount;
    int32_t plannedTaskCount = 0;
    bool nextCursorStored = false;

    // GatherMask 按 ordinal 单调递增地返回该 rank 的有效 route。整个 rank 扫描
    // 共用一个 interval 游标，避免每个 mask batch 都从 round 0 重建交织区间。
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
        // 当前 batch 的物理 route 范围为 [routeBegin, routeBegin + validRouteCount)。
        int32_t validRouteCount = static_cast<int32_t>(
            sendTotalNum - static_cast<uint64_t>(routeBegin));
        if (validRouteCount > queue.routeCapacity) {
            validRouteCount = queue.routeCapacity;
        }
        // sendMask 每个 route 占 1 bit，因此 byte 数为对齐后的 route 数除以 8。
        int32_t maskSliceBytes = Ops::Base::CeilAlign(
            validRouteCount, static_cast<int32_t>(ALIGN_256)) / 8;
        int32_t maskByteBegin = routeBegin / 8;
        if (maskByteBegin + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
            maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - maskByteBegin;
        }

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        uint32_t maskTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
        remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
            params.peermemInfo.maskRecvPtr +
            (static_cast<uint64_t>(localExpertId) * context.common.worldSize + remoteRankIdx) * maskSlotSize +
            static_cast<uint64_t>(maskByteBegin)));
        DataCopy(scratch.maskBatchTensor, remoteRankMaskGlobal, static_cast<uint32_t>(maskSliceBytes));
        // routeIndexTensor 与 mask 内容无关，因此可在 MTE2 搬 mask 时并行生成；
        // 只在 GatherMask 真正消费 mask 前等待 MTE2 完成。
        CreateVecIndex(routeIndexTensor, routeBegin, queue.routeCapacity);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
        uint64_t batchMatchedRouteCount = 0U;
        GatherMask(scratch.validTopkIndexTensor, routeIndexTensor, scratch.maskBatchU32Tensor, true,
                   static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (traceMetrics != nullptr) {
            traceMetrics->maskLoadAndGatherCycles += MegaMoeWaveTraceDetail::Now() - maskTraceBegin;
        }
#endif

        // GatherMask 返回本 batch 的有效项数量，由此得到该批在 rank compact
        // 序列中的 ordinal 范围 [batchOrdinalBegin, batchOrdinalEnd)。
        int32_t batchOrdinalBegin = matchedRouteCount;
        int32_t batchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);

        // 跳过完全位于当前 mask batch 之前的 interval。游标只向前移动，因此一次
        // 完整 rank 扫描中，每个 rank-round interval 最多推导一次。
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
                    // delta 是当前 interval 内的 rank ordinal 偏移；加上 interval
                    // 起点后得到 GatherMask compact 结果中的绝对 ordinal。
                    int32_t rankOrdinal = planInterval.roundBegin + delta;
                    // validTopkIndexTensor 保存真正的物理 topkIndex，可直接反推出
                    // source tokenId 与 topk slot。
                    int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
                        rankOrdinal - batchOrdinalBegin);
                    // 轮次偏移已经体现在 planInterval.rankPosition 中；该公式把
                    // rank ordinal 映射回当前专家的逻辑 row。
                    int32_t expertRow = planInterval.intervalRowBegin +
                        delta * planInterval.activeRankCount + planInterval.rankPosition;
                    // 本核目的 row 仍按 coreCount 等步长递增，因此可直接算出最终
                    // queue 下标，来自不同 rank 的任务会在此按目的 row 自动归并。
                    int32_t taskIdx = (expertRow - firstOwnedRow) / coreCount;
                    scratch.topkIndexTensor.SetValue(
                        taskIdx, PackDispatchRouteEntry(remoteRankIdx, topkIndex));
                    ++plannedTaskCount;
                }
            }

            // 下一 mask batch 可能仍落在当前 interval 内；这种情况下保留当前
            // interval，否则只向下一个 interval 推进一步。
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
        // 一旦扫描覆盖本专家分段终点，就保存下一 wave 可恢复的物理 mask 位置与
        // compact ordinal。终点落在 batch 中部时回退到其所在 ALIGN_256 块起点。
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
    int32_t firstOwnedRow, int32_t expectedTaskCount, int32_t taskCapacity,
    WaveDispatchTraceMetrics *traceMetrics)
{
    DispatchCoreRouteQueue queue{};
    // 同一 topkIndexTensor 分成两段：前 taskCapacity 个 int32 保存最终 queue，
    // 剩余空间作为 CreateVecIndex 的 mask 扫描临时区。
    queue.taskCapacity = taskCapacity;
    queue.routeIndexOffset = queue.taskCapacity;
    queue.routeCapacity = context.bufferConfig.routeItemsPerBatch - queue.routeIndexOffset;
    queue.routeCapacity = queue.routeCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    if (queue.routeCapacity <= 0) {
        queue.taskCapacity = 0;
        return queue;
    }

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t planTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
    uint32_t maskCyclesBefore = traceMetrics == nullptr ? 0U : traceMetrics->maskLoadAndGatherCycles;
#endif
    // 分别扫描各 source rank 的 mask，但都写入同一个按目的 row 排序的本核 queue。
    int32_t plannedTotalTaskCount = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        DispatchCoreRankPlan rankPlan = GetDispatchCoreRankPlan(
            context, scratch, rankIdx, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
        int32_t plannedTaskCount = PlanDispatchCoreRankRoutes(
            context, params, scratch, localExpertId, rankIdx, expertGlobalRowBegin,
            segmentRowBegin, segmentRowEnd, firstOwnedRow, queue, rankPlan, traceMetrics);
        // 规划数量不一致时不能继续搬运，否则 GMM1 ready 计数会宣告未写入的 row
        // 已就绪；统一返回空 queue 进入保护路径。
        if (plannedTaskCount != rankPlan.taskCount) {
            queue.taskCount = 0;
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
            if (traceMetrics != nullptr) {
                uint32_t planCycles = MegaMoeWaveTraceDetail::Now() - planTraceBegin;
                uint32_t maskCycles = traceMetrics->maskLoadAndGatherCycles - maskCyclesBefore;
                traceMetrics->rankRoundPlanCycles += planCycles > maskCycles ? planCycles - maskCycles : 0U;
            }
#endif
            return queue;
        }
        plannedTotalTaskCount += plannedTaskCount;
    }
    // 所有 rank 的任务总和必须等于由目的 row 分核公式计算出的本核任务数。
    queue.taskCount = plannedTotalTaskCount == expectedTaskCount ? plannedTotalTaskCount : 0;
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        uint32_t planCycles = MegaMoeWaveTraceDetail::Now() - planTraceBegin;
        uint32_t maskCycles = traceMetrics->maskLoadAndGatherCycles - maskCyclesBefore;
        traceMetrics->rankRoundPlanCycles += planCycles > maskCycles ? planCycles - maskCycles : 0U;
    }
#endif
    return queue;
}

// 将一个紧凑 route 任务下发到 Dispatch MTE2 环形流水。source rank 改变时只
// 重定向 GM 视图，不清空也不重启环形流水。
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

// 使用与紧凑基线一致的四阶段 MTE 环形流水消费本核 route queue：首任务预取、
// 首轮填充、稳态复用和尾部回收。route 规划与 rank 切换不参与事件同步决策，
// 因此热循环中没有逐 token 的流水状态分支。
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

    // 单槽环形缓冲区无法重叠下一次读取与当前写出，单独走正确性路径，避免在
    // 正常多槽热路径中增加 bufferCount 判断。
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

// rank-round 路径在完整专家分段内维持同一条 MTE 流水。尾部统一回收后，再向
// 分段覆盖的每个 GMM1 tile 发布本核实际完成的 row 数。
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

// 小容量回退路径定义在 rank-round 路径之后，先在此声明，使所有核在预留紧凑
// route queue 前基于相同条件选择路径，避免个别核跳过 ready 发布。
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensByRows(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount,
    uint32_t expertRowOffset = 0U);

/**
 * 按 rank-round 逻辑布局处理一个专家在当前 wave 中的分段。
 *
 * 数据索引的转换关系：
 *   sendMask 物理 bit/topkIndex
 *       -> GatherMask 后的 rank 内 compact ordinal
 *       -> rank-round 交织后的专家内 expertRow
 *       -> expertGlobalRowBegin + expertRow 对应的 workspace 全局 row
 *       -> globalRow % coreCount 对应的目的 core。
 *
 * 本函数只为当前 core 完成以下六个阶段：
 *   1. 确定当前专家分段及截断后的专家总行数；
 *   2. 为跨 wave 的专家维护每个 rank 的 mask 扫描 cursor；
 *   3. 按目的 row 均分计算本核 firstOwnedRow 和 expectedTaskCount；
 *   4. 扫描各 rank 的 sendMask，把 {rank, topkIndex} 直接写到最终本核 queue；
 *   5. 复用同一 MTE 环形流水，按 queue 顺序读取远端 token/metaInfo 并写目的 row；
 *   6. MTE3 全部完成后，按 GMM1 tile 发布本核实际完成的 row 数。
 *
 * 轮次偏移发生在第4阶段的 rank-round 映射中。它只改变 source rank 到
 * expertRow 的排列，不改变目的 row 分核、queue 大小、MTE 写地址或 ready 语义。
 */
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokensRankRoundRobin(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount, uint32_t expertRowOffset,
    WaveDispatchTraceMetrics *traceMetrics)
{
    if (dispatchRowCount == 0U) {
        return;
    }

    // 阶段1：确定当前调用覆盖的专家分段和真实专家边界。
    // 当前调用只处理该专家落入本 wave 的 [segmentRowBegin, segmentRowEnd)。
    int32_t segmentRowBegin = static_cast<int32_t>(expertRowOffset);
    int32_t segmentRowEnd = segmentRowBegin + static_cast<int32_t>(dispatchRowCount);
    // cumsum 的专家末值减去专家全局起点，得到未考虑 maxOutputSize 前的专家总行数。
    int32_t expertGlobalRowEnd = scratch.cumsumInfoTensor.GetValue(
        localExpertId * context.common.worldSize + context.common.worldSize - 1U);
    int32_t expertRowCount = expertGlobalRowEnd - expertGlobalRowBegin;
    // 尾部专家可能被 workspace 容量截断，cursor 的合法范围必须使用截断值。
    uint64_t remainingOutputRows = static_cast<uint64_t>(expertGlobalRowBegin) < context.maxOutputSize ?
        context.maxOutputSize - static_cast<uint64_t>(expertGlobalRowBegin) : 0U;
    if (static_cast<uint64_t>(expertRowCount) > remainingOutputRows) {
        expertRowCount = static_cast<int32_t>(remainingOutputRows);
    }
    // 阶段2：维护跨 wave 的 mask 扫描起点。
    // 专家首段不能读取旧 cursor。只有确实跨 wave 的专家才需要所有核清空私有
    // cursor；即使某核在首段没有任务，也必须为后续专家续段建立一致初始状态。
    if (segmentRowBegin == 0 && segmentRowEnd < expertRowCount) {
        ResetDispatchRankScanCursors(context, scratch);
    }
    // 阶段3：划分本核目的 row，并为本地 route queue 预留 UB。
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    // 目的 row 均分后，单核任务上界为 ceil(dispatchRowCount/coreCount)，按32B
    // 对齐后作为最终 route queue 的保留容量。
    int32_t maxCoreTaskCount = Ops::Base::CeilDiv(
        static_cast<int32_t>(dispatchRowCount), coreCount);
    int32_t maxTaskCapacity = Ops::Base::CeilAlign(
        maxCoreTaskCount, DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS);
    // 剩余 UB 可用于每批 mask 扫描；GatherMask 要求按 ALIGN_256 route 对齐。
    int32_t routeCapacity = context.bufferConfig.routeItemsPerBatch - maxTaskCapacity;
    routeCapacity = routeCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    uint64_t sendTotalNum =
        static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    // 紧凑 route word 高8位保存 rank、低24位保存 topkIndex；超过编码范围时
    // 回退到直接按 row/rank 处理的兼容路径。
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

    // 计算本核在该专家分段拥有的第一条目的 row；之后每隔 coreCount 行一条。
    int32_t firstOwnedRow = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    if (firstOwnedRow >= segmentRowEnd) {
        return;
    }
    // 首行和末行确定后，可精确算出本核应有任务数，用于校验 route 规划无遗漏。
    int32_t expectedTaskCount =
        (segmentRowEnd - 1 - firstOwnedRow) / coreCount + 1;

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t countTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    // 阶段4：把各 source rank 的 mask 解析为按本核目的 row 排序的 route queue。
    // 从专家 cumsum 计算每个 source rank 的独立 token 数，供活跃 rank 轮次
    // 切分、轮次偏移和 ordinal 边界计算共同使用。
    LoadDispatchExpertRankCounts(context, scratch, localExpertId, expertGlobalRowBegin);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->prepareDispatchCountsCycles += MegaMoeWaveTraceDetail::Now() - countTraceBegin;
    }
#endif
    // 规划结果按本核目的 row 顺序存放，每项仅包含 {sourceRank, topkIndex}。
    DispatchCoreRouteQueue queue = PlanDispatchCoreRouteQueue(
        context, params, scratch, localExpertId, expertGlobalRowBegin,
        segmentRowBegin, segmentRowEnd, firstOwnedRow,
        expectedTaskCount, maxTaskCapacity, traceMetrics);
    if (queue.taskCount != expectedTaskCount) {
        return;
    }

    // 阶段5：绑定目的 workspace，并让 MTE 环形流水消费本核 route queue。
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t transferTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    // queue 已按目的 row 排序。MTE 环形流水逐项从对应 rank 取 token/metaInfo，
    // 并写入 expertGlobalRowBegin + firstOwnedRow + taskIdx * coreCount。
    CopyInterleavedDispatchRouteQueue<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, winRankAddr, scratch, expertGlobalRowBegin + firstOwnedRow,
        coreCount, expectedTaskCount, tokenRevGlobalTensor,
        scaleRevGlobalTensor, metaInfoGlobalTensor);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferPipelineCycles += MegaMoeWaveTraceDetail::Now() - transferTraceBegin;
    }
    uint32_t readyTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    // 阶段6：发布 Dispatch -> GMM1 的 tile 级 ready 行数。
    // 只有在本核全部 MTE3 写出回收后，才按 tile 原子累加本核完成的精确 row 数；
    // GMM1 等待该计数达到 tile 实际行数后启动。
    PublishInterleavedDispatchTilesReady(
        context, params, localExpertId, expertGlobalRowBegin,
        segmentRowBegin, segmentRowEnd, gmm1TileRowCount);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->readyPublishCycles += MegaMoeWaveTraceDetail::Now() - readyTraceBegin;
    }
#endif
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
