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

// 保留旧的 per-core sendMask 扫描路径作为编译期隔离回退。正常 MTE wave
// 路径使用共享 Route Table，避免每个 wave、每个 core 重复 GatherMask。
#ifndef MEGA_MOE_USE_SHARED_DISPATCH_ROUTE_TABLE
#define MEGA_MOE_USE_SHARED_DISPATCH_ROUTE_TABLE 1
#endif

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
    uint32_t maskCopyIssueCycles = 0U;
    uint32_t routeIndexIssueCycles = 0U;
    uint32_t maskIndexReadyWaitCycles = 0U;
    uint32_t gatherMaskIssueCycles = 0U;
    uint32_t gatherMaskReadyWaitCycles = 0U;
    uint32_t gatherMaskCallCount = 0U;
    uint32_t routeTableLoadCycles = 0U;
    uint32_t transferPipelineCycles = 0U;
    uint32_t transferRoutePrepareCycles = 0U;
    uint32_t transferMte3ReuseWaitCycles = 0U;
    uint32_t transferMte2IssueCycles = 0U;
    uint32_t transferMte2ReadyWaitCycles = 0U;
    uint32_t transferMetaInfoPrepareCycles = 0U;
    uint32_t transferMetaInfoReadyWaitCycles = 0U;
    uint32_t transferMte3IssueCycles = 0U;
    uint32_t transferMte3DrainWaitCycles = 0U;
    uint32_t readyPublishCycles = 0U;
    uint32_t readySyncWaitCycles = 0U;
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
                                                     WaveDispatchTraceMetrics *traceMetrics = nullptr)
{
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t routePrepareBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(context.common.topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferRoutePrepareCycles += MegaMoeWaveTraceDetail::Now() - routePrepareBegin;
    }
    uint32_t reuseWaitBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr && IsBufferReuse) {
        traceMetrics->transferMte3ReuseWaitCycles += MegaMoeWaveTraceDetail::Now() - reuseWaitBegin;
    }
    uint32_t mte2IssueBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMte2IssueCycles += MegaMoeWaveTraceDetail::Now() - mte2IssueBegin;
        reuseWaitBegin = MegaMoeWaveTraceDetail::Now();
    }
#endif
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr && IsBufferReuse) {
        traceMetrics->transferMte3ReuseWaitCycles += MegaMoeWaveTraceDetail::Now() - reuseWaitBegin;
    }
    uint32_t metaInfoPrepareBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX,
                                                                topkIndex % static_cast<int32_t>(context.common.topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMetaInfoPrepareCycles +=
            MegaMoeWaveTraceDetail::Now() - metaInfoPrepareBegin;
    }
#endif
}

template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
    int32_t dstIdx, GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor, GlobalTensor<int32_t> &metaInfoGlobalTensor,
    int32_t topkIndex, WaveDispatchTraceMetrics *traceMetrics = nullptr)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t mte2ReadyWaitBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMte2ReadyWaitCycles +=
            MegaMoeWaveTraceDetail::Now() - mte2ReadyWaitBegin;
    }
#endif
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if constexpr (TopkWeightsPrefetch) {
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        mte2ReadyWaitBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (traceMetrics != nullptr) {
            traceMetrics->transferMte2ReadyWaitCycles +=
                MegaMoeWaveTraceDetail::Now() - mte2ReadyWaitBegin;
        }
        uint32_t metaInfoPrepareBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(context.common.topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (traceMetrics != nullptr) {
            traceMetrics->transferMetaInfoPrepareCycles +=
                MegaMoeWaveTraceDetail::Now() - metaInfoPrepareBegin;
        }
#endif
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t mte3IssueBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    // token/scale 不依赖 Scalar 生成的 metadata，先下发这两笔 MTE3 以掩盖 S_MTE3 等待；
    // 只有 metadata 写出必须在 Scalar 对 MTE3 可见后才能下发，保持 upstream 的原始流水顺序。
    DataCopyPad(tokenRevGlobalTensor[dstIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[dstIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMte3IssueCycles += MegaMoeWaveTraceDetail::Now() - mte3IssueBegin;
    }
    uint32_t metaInfoReadyWaitBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMetaInfoReadyWaitCycles +=
            MegaMoeWaveTraceDetail::Now() - metaInfoReadyWaitBegin;
        mte3IssueBegin = MegaMoeWaveTraceDetail::Now();
    }
#endif
    DataCopy(metaInfoGlobalTensor[dstIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMte3IssueCycles += MegaMoeWaveTraceDetail::Now() - mte3IssueBegin;
    }
#endif
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
constexpr int32_t DISPATCH_RANK_TASK_COUNT_LANE = 1;
// 共享 Route Table 路径不使用 per-rank task count，复用 lane 1 按 active
// position 保存当前 rank-round interval 的 rankId。
constexpr int32_t DISPATCH_ACTIVE_RANK_CACHE_LANE = 1;
constexpr int32_t DISPATCH_RANK_DESCRIPTOR_BEGIN_LANE = 2;
// 共享 Route Table 路径不再构造 grouped descriptor，因此复用 lane 2
// 保存该 rank slice 在 validTopkIndexTensor 中的起点。
constexpr int32_t DISPATCH_RANK_ROUTE_UB_BASE_LANE = 2;
constexpr int32_t DISPATCH_RANK_DESCRIPTOR_CURSOR_LANE = 3;
// 共享 Route Table 路径复用 lane 3 标记本核是否实际访问该 source rank。
constexpr int32_t DISPATCH_RANK_ROUTE_USED_LANE = 3;
constexpr int32_t DISPATCH_RANK_SEGMENT_ORDINAL_BEGIN_LANE = 4;
constexpr int32_t DISPATCH_RANK_SEGMENT_ORDINAL_END_LANE = 5;
constexpr int32_t DISPATCH_RANK_BEGIN_RESOLVED_LANE = 6;
constexpr int32_t DISPATCH_RANK_END_RESOLVED_LANE = 7;
constexpr int32_t DISPATCH_RANK_SCAN_CURSOR_STRIDE = 2;
constexpr int32_t DISPATCH_RANK_SCAN_ROUTE_BEGIN_LANE = 0;
constexpr int32_t DISPATCH_RANK_SCAN_MATCHED_COUNT_LANE = 1;
constexpr int32_t DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS = ALIGN_32 / sizeof(int32_t);
constexpr uint32_t DISPATCH_ROUTE_RANK_SHIFT = 24U;
constexpr uint32_t DISPATCH_ROUTE_TOPK_MASK = (1U << DISPATCH_ROUTE_RANK_SHIFT) - 1U;
constexpr uint32_t DISPATCH_ROUTE_RANK_CAPACITY = 1U << (32U - DISPATCH_ROUTE_RANK_SHIFT);
constexpr int32_t DISPATCH_GROUPED_TASK_INDEX_CAPACITY =
    static_cast<int32_t>(DISPATCH_ROUTE_RANK_CAPACITY);

// 每个本核任务保存一个紧凑 route word。queue 下标决定目的 row，word 中只保存
// source rank 和 source topkIndex，避免额外保存目的 row。
struct DispatchCoreRouteQueue {
    // 本核在当前专家分段中最终需要搬运的 token 数。
    int32_t taskCount = 0;
    // topkIndexTensor 前部为最终 route queue 预留的容量。
    int32_t taskCapacity = 0;
    // descriptor 区按 source rank 连续分组，保存 {最终taskIdx, compactOrdinal}。
    int32_t descriptorOffset = 0;
    // 临时 CreateVecIndex 区紧跟 descriptor，三者复用同一块 UB。
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
    // 当前 remoteRank 在原始活跃 rank 顺序中的位置。
    int32_t rankPosition = -1;
};

// 固定 ownership 下，同一 core 在专家 row 空间中始终跨 coreCount 领取下一行。
struct DispatchCoreRowCursor {
    int32_t expertRow = 0;
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
    // 当前 rank 在按 rank 连续排列的临时 descriptor 区中的起点。
    int32_t descriptorBegin = 0;
};

struct DispatchRankRouteSlice {
    int32_t segmentOrdinalBegin = 0;
    int32_t segmentOrdinalEnd = 0;
    int32_t routeUbBase = 0;
};

// 单调描述当前 expertRow 所在的 rank-round interval。一个 core 领取的
// expertRow 严格递增，因此 cursor 只向前推进，不再为每一行从 ordinal 0 重找。
struct DispatchSourceCursor {
    int32_t roundBegin = 0;
    int32_t roundEnd = 0;
    int32_t intervalRowBegin = 0;
    int32_t intervalRowEnd = 0;
    int32_t activeRankCount = 0;
};

// 临时 grouped descriptor 复用相同的 8+24 bit 布局：高8位保存最终 queue
// 下标，低24位保存 rank 内 compact ordinal；source rank 由所在分组隐含。
__aicore__ inline int32_t PackDispatchGroupedDescriptor(
    int32_t taskIdx, int32_t rankOrdinal)
{
    return PackDispatchRouteEntry(static_cast<uint32_t>(taskIdx), rankOrdinal);
}

__aicore__ inline int32_t GetDispatchGroupedTaskIndex(int32_t descriptor)
{
    return static_cast<int32_t>(GetDispatchRouteRank(descriptor));
}

__aicore__ inline int32_t GetDispatchGroupedRankOrdinal(int32_t descriptor)
{
    return GetDispatchRouteTopkIndex(descriptor);
}

__aicore__ inline DispatchCoreRowCursor GetFirstDispatchRowOwnedByCore(
    const TokenDispatchConfig &context, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd)
{
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t coreIdx = static_cast<int32_t>(context.blockJob.jobIndex);
    DispatchCoreRowCursor cursor{};
    int32_t firstGlobalRow = expertGlobalRowBegin + segmentRowBegin;
    int32_t ownerDelta = coreIdx - firstGlobalRow % coreCount;
    if (ownerDelta < 0) {
        ownerDelta += coreCount;
    }
    cursor.expertRow = segmentRowBegin + ownerDelta;
    if (cursor.expertRow >= segmentRowEnd) {
        cursor.expertRow = segmentRowEnd;
    }
    return cursor;
}

__aicore__ inline void AdvanceDispatchCoreRowCursor(
    DispatchCoreRowCursor &cursor, int32_t coreCount)
{
    cursor.expertRow += coreCount;
}

__aicore__ inline int32_t CountDispatchRowsOwnedByCore(
    const TokenDispatchConfig &context, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd)
{
    DispatchCoreRowCursor cursor = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    if (cursor.expertRow >= segmentRowEnd) {
        return 0;
    }
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    return (segmentRowEnd - 1 - cursor.expertRow) / coreCount + 1;
}

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

template <typename ActivationType, bool PrepareGroupedDescriptors = true>
__aicore__ inline void PrepareDispatchSegmentRankPlans(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch,
    int32_t segmentRowBegin, int32_t segmentRowEnd)
{
    // 每个 rank 的 32B scratch burst 同时保存 count、当前 core 的任务分组信息和
    // 当前 wave 分段的 compact ordinal 边界。先以“该 rank 已耗尽”作为安全默认值。
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_COUNT_LANE);
        if constexpr (PrepareGroupedDescriptors) {
            scratch.sendCntTensor.SetValue(
                rankScratchBase + DISPATCH_RANK_TASK_COUNT_LANE, 0);
            scratch.sendCntTensor.SetValue(
                rankScratchBase + DISPATCH_RANK_DESCRIPTOR_BEGIN_LANE, 0);
            scratch.sendCntTensor.SetValue(
                rankScratchBase + DISPATCH_RANK_DESCRIPTOR_CURSOR_LANE, 0);
        } else {
            scratch.sendCntTensor.SetValue(
                rankScratchBase + DISPATCH_RANK_ROUTE_USED_LANE, 0);
        }
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_BEGIN_LANE, rankTokenCount);
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_END_LANE, rankTokenCount);
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_BEGIN_RESOLVED_LANE, 0);
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_END_RESOLVED_LANE, 0);
    }

    // 所有 rank 共用同一组 rank-round interval。这里只正向遍历一次 interval，
    // 同时解析两个 wave 边界，替代过去“每个 rank 各自从 ordinal 0 重扫”的做法。
    int32_t roundBegin = 0;
    int32_t intervalRowBegin = 0;
    while (intervalRowBegin <= segmentRowEnd) {
        int32_t activeRankCount = 0;
        int32_t roundEnd = 0x7fffffff;
        for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
            int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
            int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
                rankScratchBase + DISPATCH_RANK_COUNT_LANE);
            if (rankTokenCount <= roundBegin) {
                continue;
            }
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
        int32_t rankPosition = 0;
        for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
            int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
            int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
                rankScratchBase + DISPATCH_RANK_COUNT_LANE);
            if (rankTokenCount <= roundBegin) {
                continue;
            }
            DispatchRankRoundInterval interval{
                roundBegin, roundEnd, intervalRowBegin, activeRankCount, rankPosition};
            if (scratch.sendCntTensor.GetValue(
                    rankScratchBase + DISPATCH_RANK_BEGIN_RESOLVED_LANE) == 0 &&
                segmentRowBegin <= intervalRowEnd) {
                scratch.sendCntTensor.SetValue(
                    rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_BEGIN_LANE,
                    GetDispatchRankOrdinalBeforeRow(interval, segmentRowBegin));
                scratch.sendCntTensor.SetValue(
                    rankScratchBase + DISPATCH_RANK_BEGIN_RESOLVED_LANE, 1);
            }
            if (scratch.sendCntTensor.GetValue(
                    rankScratchBase + DISPATCH_RANK_END_RESOLVED_LANE) == 0 &&
                segmentRowEnd <= intervalRowEnd) {
                scratch.sendCntTensor.SetValue(
                    rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_END_LANE,
                    GetDispatchRankOrdinalBeforeRow(interval, segmentRowEnd));
                scratch.sendCntTensor.SetValue(
                    rankScratchBase + DISPATCH_RANK_END_RESOLVED_LANE, 1);
            }
            ++rankPosition;
        }
        if (segmentRowEnd <= intervalRowEnd) {
            break;
        }
        intervalRowBegin = intervalRowEnd;
        roundBegin = roundEnd;
    }
}

template <typename ActivationType>
__aicore__ inline DispatchCoreRankPlan LoadDispatchCoreRankPlan(
    const TokenDispatchScratch<ActivationType> &scratch, uint32_t remoteRankIdx)
{
    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    DispatchCoreRankPlan plan{};
    plan.taskCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_TASK_COUNT_LANE);
    plan.segmentOrdinalBegin = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_BEGIN_LANE);
    plan.segmentOrdinalEnd = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_END_LANE);
    plan.descriptorBegin = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_DESCRIPTOR_BEGIN_LANE);
    return plan;
}

template <typename ActivationType>
__aicore__ inline DispatchRankRouteSlice LoadDispatchRankRouteSlice(
    const TokenDispatchScratch<ActivationType> &scratch, uint32_t remoteRankIdx)
{
    int32_t rankScratchBase =
        static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    return {
        scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_BEGIN_LANE),
        scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_END_LANE),
        scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_ROUTE_UB_BASE_LANE)};
}

template <typename ActivationType, bool CacheActiveRanks = false>
__aicore__ inline bool GetDispatchSourceForExpertRow(
    const TokenDispatchConfig &context, TokenDispatchScratch<ActivationType> &scratch,
    int32_t expertRow, DispatchSourceCursor &cursor,
    uint32_t &remoteRankIdx, int32_t &rankOrdinal)
{
    // expertRow 按本核 row cursor 严格递增；只有越过当前 interval 时才重新计算
    // 活跃 rank 集合，彻底消除逐 row 从 ordinal 0 开始的重复 interval 搜索。
    while (cursor.activeRankCount == 0 || expertRow >= cursor.intervalRowEnd) {
        if (cursor.activeRankCount != 0) {
            cursor.roundBegin = cursor.roundEnd;
            cursor.intervalRowBegin = cursor.intervalRowEnd;
        }
        cursor.activeRankCount = 0;
        cursor.roundEnd = 0x7fffffff;
        for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
            int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
            int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
                rankScratchBase + DISPATCH_RANK_COUNT_LANE);
            if (rankTokenCount <= cursor.roundBegin) {
                continue;
            }
            if constexpr (CacheActiveRanks) {
                int32_t activeRankCacheBase =
                    cursor.activeRankCount * DISPATCH_RANK_SCRATCH_STRIDE;
                scratch.sendCntTensor.SetValue(
                    activeRankCacheBase + DISPATCH_ACTIVE_RANK_CACHE_LANE,
                    static_cast<int32_t>(rankIdx));
            }
            ++cursor.activeRankCount;
            if (rankTokenCount < cursor.roundEnd) {
                cursor.roundEnd = rankTokenCount;
            }
        }
        if (cursor.activeRankCount == 0) {
            return false;
        }
        cursor.intervalRowEnd = cursor.intervalRowBegin +
            (cursor.roundEnd - cursor.roundBegin) * cursor.activeRankCount;
    }

    int32_t rowOffset = expertRow - cursor.intervalRowBegin;
    int32_t rankPosition = rowOffset % cursor.activeRankCount;
    rankOrdinal = cursor.roundBegin + rowOffset / cursor.activeRankCount;
    if constexpr (CacheActiveRanks) {
        int32_t activeRankCacheBase =
            rankPosition * DISPATCH_RANK_SCRATCH_STRIDE;
        remoteRankIdx = static_cast<uint32_t>(scratch.sendCntTensor.GetValue(
            activeRankCacheBase + DISPATCH_ACTIVE_RANK_CACHE_LANE));
        return remoteRankIdx < context.common.worldSize;
    }
    int32_t activePosition = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_COUNT_LANE);
        if (rankTokenCount <= cursor.roundBegin) {
            continue;
        }
        if (activePosition == rankPosition) {
            remoteRankIdx = rankIdx;
            return true;
        }
        ++activePosition;
    }
    return false;
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
    uint32_t remoteRankIdx, int32_t segmentRowBegin,
    const DispatchCoreRouteQueue &queue, const DispatchCoreRankPlan &rankPlan,
    WaveDispatchTraceMetrics *traceMetrics)
{
    if (rankPlan.taskCount <= 0) {
        return 0;
    }

    // descriptor 已按 source rank 连续分组，每项保存 {最终taskIdx, compactOrdinal}。
    // 本函数只扫描当前 rank 的 sendMask，并把对应 ordinal 回填到最终 queue 下标。
    // rankTokenCount 决定 compact ordinal 上界；sendTotalNum 决定物理 mask 上界。
    int32_t rankScratchBase = static_cast<int32_t>(remoteRankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
    int32_t rankTokenCount = scratch.sendCntTensor.GetValue(
        rankScratchBase + DISPATCH_RANK_COUNT_LANE);
    uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    // 每个 [expert][rank] mask 槽尾部还包含一个 32B 的 count 区域。
    uint32_t maskSlotSize = context.maskAlignSize + static_cast<uint32_t>(ALIGN_32);
    // topkIndexTensor 尾部临时生成当前 mask batch 的物理索引。
    LocalTensor<int32_t> routeIndexTensor = scratch.topkIndexTensor[queue.routeIndexOffset];
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
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
    // 同一 rank 分组内的 descriptor 按 compact ordinal 单调递增。cursor 跨
    // mask batch 持续推进，避免每个 batch 都从 descriptorBegin 重扫本 rank
    // 的全部任务；每个 descriptor 最多解析一次。
    int32_t descriptorCursor = rankPlan.descriptorBegin;
    int32_t descriptorEnd = rankPlan.descriptorBegin + rankPlan.taskCount;

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
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        uint32_t routeIndexTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
        if (traceMetrics != nullptr) {
            // DataCopy 为异步下发；这里只统计 API 下发区间，完成时间归入后面的 MTE2_V 等待。
            traceMetrics->maskCopyIssueCycles += routeIndexTraceBegin - maskTraceBegin;
        }
#endif
        // routeIndexTensor 与 mask 内容无关，因此可在 MTE2 搬 mask 时并行生成；
        // 只在 GatherMask 真正消费 mask 前等待 MTE2 完成。
        CreateVecIndex(routeIndexTensor, routeBegin, queue.routeCapacity);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        uint32_t maskIndexWaitTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
        if (traceMetrics != nullptr) {
            // CreateVecIndex 同样是异步下发；实际向量完成延迟由 MTE2_V 同步区间共同承接。
            traceMetrics->routeIndexIssueCycles += maskIndexWaitTraceBegin - routeIndexTraceBegin;
        }
#endif
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        uint32_t gatherMaskTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
        if (traceMetrics != nullptr) {
            traceMetrics->maskIndexReadyWaitCycles += gatherMaskTraceBegin - maskIndexWaitTraceBegin;
        }
#endif
        uint64_t batchMatchedRouteCount = 0U;
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (traceMetrics != nullptr) {
            ++traceMetrics->gatherMaskCallCount;
        }
#endif
        GatherMask(scratch.validTopkIndexTensor, routeIndexTensor, scratch.maskBatchU32Tensor, true,
                   static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        uint32_t gatherMaskWaitTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
        if (traceMetrics != nullptr) {
            traceMetrics->gatherMaskIssueCycles += gatherMaskWaitTraceBegin - gatherMaskTraceBegin;
        }
#endif
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (traceMetrics != nullptr) {
            uint32_t maskTraceEnd = MegaMoeWaveTraceDetail::Now();
            traceMetrics->gatherMaskReadyWaitCycles += maskTraceEnd - gatherMaskWaitTraceBegin;
            traceMetrics->maskLoadAndGatherCycles += maskTraceEnd - maskTraceBegin;
        }
#endif

        // GatherMask 返回本 batch 的有效项数量，由此得到该批在 rank compact
        // 序列中的 ordinal 范围 [batchOrdinalBegin, batchOrdinalEnd)。
        int32_t batchOrdinalBegin = matchedRouteCount;
        int32_t batchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);

        // descriptor 已按 ordinal 排序。先跳过恢复 cursor 之前已经覆盖的项，
        // 再只消费落入当前 batch 的连续子区间；遇到下一个 batch 的 ordinal
        // 立即停止，本 batch 不再扫描后续 descriptor。
        while (descriptorCursor < descriptorEnd) {
            int32_t descriptor = scratch.topkIndexTensor.GetValue(
                queue.descriptorOffset + descriptorCursor);
            int32_t rankOrdinal = GetDispatchGroupedRankOrdinal(descriptor);
            if (rankOrdinal < batchOrdinalBegin) {
                ++descriptorCursor;
                continue;
            }
            if (rankOrdinal >= batchOrdinalEnd) {
                break;
            }
            int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
                rankOrdinal - batchOrdinalBegin);
            int32_t taskIdx = GetDispatchGroupedTaskIndex(descriptor);
            scratch.topkIndexTensor.SetValue(
                taskIdx, PackDispatchRouteEntry(remoteRankIdx, topkIndex));
            ++plannedTaskCount;
            ++descriptorCursor;
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
    int32_t expectedTaskCount, int32_t taskCapacity,
    WaveDispatchTraceMetrics *traceMetrics)
{
    DispatchCoreRouteQueue queue{};
    // 同一 topkIndexTensor 分成三段：最终 queue、按 rank 分组的 descriptor，以及
    // CreateVecIndex 的 mask 扫描临时区。
    queue.taskCapacity = taskCapacity;
    queue.descriptorOffset = queue.taskCapacity;
    queue.routeIndexOffset = queue.taskCapacity * 2;
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
    // 一次性解析所有 rank 的当前 wave ordinal 边界，并清空本核 rank 任务计数。
    PrepareDispatchSegmentRankPlans(
        context, scratch, segmentRowBegin, segmentRowEnd);
    // 按本核固定 lane 的 row 序列正向构造 {rank, compactOrdinal}。逻辑 row 到
    // source 的映射保持不变，同一 core 在后续 round 中固定跨 coreCount 领取 row。sourceCursor
    // 随 expertRow 单调推进，不再为每一行从 ordinal 0 重新搜索 interval。
    DispatchCoreRowCursor rowCursor = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    DispatchSourceCursor sourceCursor{};
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t descriptorCount = 0;
    while (rowCursor.expertRow < segmentRowEnd && descriptorCount < expectedTaskCount) {
        uint32_t rankIdx = 0U;
        int32_t rankOrdinal = 0;
        if (!GetDispatchSourceForExpertRow(
                context, scratch, rowCursor.expertRow, sourceCursor,
                rankIdx, rankOrdinal)) {
            queue.taskCount = 0;
            return queue;
        }
        // 最终 queue 前部先暂存同序 {rank, ordinal}；构造时顺手累计 rank 任务数，
        // 消除后续 worldSize 次完整 descriptor 统计扫描。
        scratch.topkIndexTensor.SetValue(
            descriptorCount,
            PackDispatchRouteEntry(rankIdx, rankOrdinal));
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        int32_t rankTaskCount = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_TASK_COUNT_LANE);
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_TASK_COUNT_LANE, rankTaskCount + 1);
        ++descriptorCount;
        AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
    }
    if (descriptorCount != expectedTaskCount) {
        queue.taskCount = 0;
        return queue;
    }
    queue.taskCount = descriptorCount;

    // 根据各 rank 任务数生成连续分组的起点及写 cursor。
    int32_t descriptorGroupBegin = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        int32_t rankTaskCount = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_TASK_COUNT_LANE);
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_DESCRIPTOR_BEGIN_LANE, descriptorGroupBegin);
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_DESCRIPTOR_CURSOR_LANE, descriptorGroupBegin);
        descriptorGroupBegin += rankTaskCount;
    }
    if (descriptorGroupBegin != descriptorCount) {
        queue.taskCount = 0;
        return queue;
    }
    // 把前部同序 descriptor 散列到 descriptor 区的连续 rank 分组。分组项携带
    // 原 taskIdx，因此 mask 解析后仍可写回原顺序的最终 route queue。
    for (int32_t taskIdx = 0; taskIdx < descriptorCount; ++taskIdx) {
        int32_t descriptor = scratch.topkIndexTensor.GetValue(taskIdx);
        uint32_t rankIdx = GetDispatchRouteRank(descriptor);
        int32_t rankScratchBase = static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        int32_t descriptorCursor = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_DESCRIPTOR_CURSOR_LANE);
        scratch.topkIndexTensor.SetValue(
            queue.descriptorOffset + descriptorCursor,
            PackDispatchGroupedDescriptor(
                taskIdx, GetDispatchRouteTopkIndex(descriptor)));
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_DESCRIPTOR_CURSOR_LANE,
            descriptorCursor + 1);
    }

    // 分别扫描本核真正用到的 source rank，并回填同一个按目的 row 排序的 queue。
    int32_t plannedTotalTaskCount = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        DispatchCoreRankPlan rankPlan = LoadDispatchCoreRankPlan(scratch, rankIdx);
        int32_t plannedTaskCount = PlanDispatchCoreRankRoutes(
            context, params, scratch, localExpertId, rankIdx, segmentRowBegin,
            queue, rankPlan, traceMetrics);
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

/**
 * 从共享 Route Table 构造本核 route queue。
 *
 * 先把本核在当前专家分段实际访问的 source-rank compact slice 搬到
 * validTopkIndexTensor 的互不重叠区域；任务侧先按 logical row 生成原序
 * {rankId, ordinal}，搬入后再原地改写为 {rankId, topkIndex}。这样从三遍降为
 * 两遍，不再需要 grouped descriptor 和 rank scatter；不均衡场景仍由统一
 * rank-round source cursor 处理。
 */
template <typename ActivationType>
__aicore__ inline DispatchCoreRouteQueue PlanDispatchCoreRouteQueueFromTable(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, int32_t segmentRowBegin, int32_t segmentRowEnd,
    int32_t expectedTaskCount, int32_t taskCapacity,
    WaveDispatchTraceMetrics *traceMetrics)
{
    DispatchCoreRouteQueue queue{};
    queue.taskCapacity = taskCapacity;
    if (queue.taskCapacity <= 0 || queue.taskCapacity >
            context.bufferConfig.routeItemsPerBatch) {
        queue.taskCapacity = 0;
        return queue;
    }

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t planTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
    uint32_t routeTableLoadCyclesBefore =
        traceMetrics == nullptr ? 0U : traceMetrics->routeTableLoadCycles;
#endif
    PrepareDispatchSegmentRankPlans<ActivationType, false>(
        context, scratch, segmentRowBegin, segmentRowEnd);

    // 第一遍只按最终目的 row 顺序生成 {rank, ordinal}，并标记本核实际访问的
    // source rank。这里不统计 task count，也不再构造 grouped descriptor。
    DispatchCoreRowCursor rowCursor = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    DispatchSourceCursor sourceCursor{};
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t descriptorCount = 0;
    while (rowCursor.expertRow < segmentRowEnd &&
           descriptorCount < expectedTaskCount) {
        uint32_t rankIdx = 0U;
        int32_t rankOrdinal = 0;
        if (!GetDispatchSourceForExpertRow<ActivationType, true>(
                context, scratch, rowCursor.expertRow, sourceCursor,
                rankIdx, rankOrdinal)) {
            queue.taskCount = 0;
            return queue;
        }
        scratch.topkIndexTensor.SetValue(
            descriptorCount, PackDispatchRouteEntry(rankIdx, rankOrdinal));
        int32_t rankScratchBase =
            static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_ROUTE_USED_LANE, 1);
        ++descriptorCount;
        AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
    }
    if (descriptorCount != expectedTaskCount) {
        queue.taskCount = 0;
        return queue;
    }

    // 只为本核实际访问的 rank 规划 UB 布局，不下发 MTE2。每个 rank slice 从 32B 对齐地址开始，
    // 避免 DataCopyPad 的尾部 padding 覆盖下一个 rank。全部容量校验通过后
    // 才开始搬运，失败时可以安全返回而不会遗留未回收的 MTE2 事件。
    int32_t routeSliceEnd = 0;
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        const int32_t rankScratchBase =
            static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        if (scratch.sendCntTensor.GetValue(
                rankScratchBase + DISPATCH_RANK_ROUTE_USED_LANE) == 0) {
            continue;
        }
        const int32_t segmentOrdinalBegin = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_BEGIN_LANE);
        const int32_t segmentOrdinalEnd = scratch.sendCntTensor.GetValue(
            rankScratchBase + DISPATCH_RANK_SEGMENT_ORDINAL_END_LANE);
        const int32_t routeSliceCount = segmentOrdinalEnd - segmentOrdinalBegin;
        if (routeSliceCount <= 0) {
            queue.taskCount = 0;
            return queue;
        }
        const int32_t routeUbBase = Ops::Base::CeilAlign(
            routeSliceEnd, DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS);
        if (routeUbBase + routeSliceCount > context.bufferConfig.routeItemsPerBatch) {
            queue.taskCount = 0;
            return queue;
        }
        scratch.sendCntTensor.SetValue(
            rankScratchBase + DISPATCH_RANK_ROUTE_UB_BASE_LANE, routeUbBase);
        routeSliceEnd = routeUbBase + routeSliceCount;
    }

    GlobalTensor<int32_t> routeIndexGlobal;
    routeIndexGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.dispatchRouteIndexPtr));
    const uint32_t expertRankBase = localExpertId * context.common.worldSize;
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t routeTableLoadBegin =
        traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    bool hasRouteSlice = false;
    // 不在 rank 循环内等待：所有 GM->UB slice 先下发，最后统一执行一次
    // MTE2_S，使多 rank 搬运共享同一个完成边界。
    for (uint32_t rankIdx = 0U; rankIdx < context.common.worldSize; ++rankIdx) {
        const int32_t rankScratchBase =
            static_cast<int32_t>(rankIdx) * DISPATCH_RANK_SCRATCH_STRIDE;
        if (scratch.sendCntTensor.GetValue(
                rankScratchBase + DISPATCH_RANK_ROUTE_USED_LANE) == 0) {
            continue;
        }
        const DispatchRankRouteSlice rankPlan =
            LoadDispatchRankRouteSlice(scratch, rankIdx);
        const int32_t routeSliceCount =
            rankPlan.segmentOrdinalEnd - rankPlan.segmentOrdinalBegin;
        if (routeSliceCount <= 0) {
            continue;
        }

        const uint32_t pairIndex = expertRankBase + rankIdx;
        const int32_t rankGlobalRowBegin = pairIndex == 0U ? 0 :
            scratch.cumsumInfoTensor.GetValue(pairIndex - 1U);
        DataCopyPad(
            scratch.validTopkIndexTensor[rankPlan.routeUbBase],
            routeIndexGlobal[rankGlobalRowBegin + rankPlan.segmentOrdinalBegin],
            {1U, static_cast<uint32_t>(routeSliceCount * sizeof(int32_t)), 0U, 0U, 0U},
            {true, 0U, 0U, 0U});
        hasRouteSlice = true;
    }
    if (hasRouteSlice) {
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->routeTableLoadCycles +=
            MegaMoeWaveTraceDetail::Now() - routeTableLoadBegin;
    }
#endif

    // 所有本核所需 source slice 已在 UB 就绪。第二遍按原 task 顺序原地读取
    // {rank, ordinal}，直接改写为最终 {rank, topkIndex} queue。
    for (int32_t taskIdx = 0; taskIdx < descriptorCount; ++taskIdx) {
        const int32_t descriptor = scratch.topkIndexTensor.GetValue(taskIdx);
        const uint32_t rankIdx = GetDispatchRouteRank(descriptor);
        const int32_t rankOrdinal = GetDispatchRouteTopkIndex(descriptor);
        const DispatchRankRouteSlice rankPlan =
            LoadDispatchRankRouteSlice(scratch, rankIdx);
        if (rankOrdinal < rankPlan.segmentOrdinalBegin ||
            rankOrdinal >= rankPlan.segmentOrdinalEnd) {
            queue.taskCount = 0;
            return queue;
        }
        const int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
            rankPlan.routeUbBase + rankOrdinal - rankPlan.segmentOrdinalBegin);
        scratch.topkIndexTensor.SetValue(
            taskIdx, PackDispatchRouteEntry(rankIdx, topkIndex));
    }
    queue.taskCount = descriptorCount;
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        const uint32_t totalPlanCycles =
            MegaMoeWaveTraceDetail::Now() - planTraceBegin;
        const uint32_t routeTableLoadCycles =
            traceMetrics->routeTableLoadCycles - routeTableLoadCyclesBefore;
        traceMetrics->rankRoundPlanCycles +=
            totalPlanCycles > routeTableLoadCycles ?
                totalPlanCycles - routeTableLoadCycles : 0U;
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
    int32_t &currentRankIdx, WaveDispatchTraceMetrics *traceMetrics)
{
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t routePrepareBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    int32_t routeEntry = scratch.topkIndexTensor.GetValue(taskIdx);
    int32_t remoteRankIdx = static_cast<int32_t>(GetDispatchRouteRank(routeEntry));
    int32_t topkIndex = GetDispatchRouteTopkIndex(routeEntry);
    if (remoteRankIdx != currentRankIdx) {
        remoteRankGlobalTensor.SetGlobalBuffer(
            reinterpret_cast<__gm__ ActivationType *>(
                winRankAddr[remoteRankIdx] + context.quantWinOffset));
        currentRankIdx = remoteRankIdx;
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferRoutePrepareCycles += MegaMoeWaveTraceDetail::Now() - routePrepareBegin;
    }
#endif
    FetchDispatchTokenAndMetaInfo<IsBufferReuse, TopkWeightsPrefetch>(
        context, scratch, bufferIdx, topkIndex, remoteRankIdx,
        remoteRankGlobalTensor, traceMetrics);
    return topkIndex;
}

// 使用与紧凑基线一致的四阶段 MTE 环形流水消费本核 route queue：首任务预取、
// 首轮填充、稳态复用和尾部回收。route 规划与 rank 切换不参与事件同步决策，
// 因此热循环中没有逐 token 的流水状态分支。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyInterleavedDispatchRouteQueue(
    const TokenDispatchConfig &context, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd, int32_t taskCount,
    GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor,
    GlobalTensor<int32_t> &metaInfoGlobalTensor,
    WaveDispatchTraceMetrics *traceMetrics)
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
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    DispatchCoreRowCursor rowCursor = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    int32_t pendingTopkIndex =
        FetchInterleavedDispatchRouteTask<false, TopkWeightsPrefetch>(
            context, winRankAddr, scratch, 0, 0, remoteRankGlobalTensor,
            currentRankIdx, traceMetrics);

    // 单槽环形缓冲区无法重叠下一次读取与当前写出，单独走正确性路径，避免在
    // 正常多槽热路径中增加 bufferCount 判断。
    if (bufferCount == 1) {
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, 0, expertGlobalRowBegin + rowCursor.expertRow, tokenRevGlobalTensor,
            scaleRevGlobalTensor, metaInfoGlobalTensor, pendingTopkIndex, traceMetrics);
        AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
        for (int32_t taskIdx = 1; taskIdx < taskCount; ++taskIdx) {
            pendingTopkIndex =
                FetchInterleavedDispatchRouteTask<true, TopkWeightsPrefetch>(
                    context, winRankAddr, scratch, taskIdx, 0,
                    remoteRankGlobalTensor, currentRankIdx, traceMetrics);
            StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, scratch, 0,
                expertGlobalRowBegin + rowCursor.expertRow,
                tokenRevGlobalTensor, scaleRevGlobalTensor,
                metaInfoGlobalTensor, pendingTopkIndex, traceMetrics);
            AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
        }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        uint32_t drainWaitBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(0));
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
        if (traceMetrics != nullptr) {
            traceMetrics->transferMte3DrainWaitCycles += MegaMoeWaveTraceDetail::Now() - drainWaitBegin;
        }
#endif
        return;
    }

    int32_t firstUseEnd = taskCount < bufferCount ? taskCount : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t currentTopkIndex =
            FetchInterleavedDispatchRouteTask<false, TopkWeightsPrefetch>(
                context, winRankAddr, scratch, issueIdx, issueIdx,
                remoteRankGlobalTensor, currentRankIdx, traceMetrics);
        int32_t copyIdx = issueIdx - 1;
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyIdx,
            expertGlobalRowBegin + rowCursor.expertRow,
            tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            pendingTopkIndex, traceMetrics);
        AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
        pendingTopkIndex = currentTopkIndex;
    }

    for (int32_t issueIdx = bufferCount; issueIdx < taskCount; ++issueIdx) {
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t currentTopkIndex =
            FetchInterleavedDispatchRouteTask<true, TopkWeightsPrefetch>(
                context, winRankAddr, scratch, issueIdx, issueBufferIdx,
                remoteRankGlobalTensor, currentRankIdx, traceMetrics);
        int32_t copyIdx = issueIdx - 1;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyBufferIdx,
            expertGlobalRowBegin + rowCursor.expertRow,
            tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            pendingTopkIndex, traceMetrics);
        AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
        pendingTopkIndex = currentTopkIndex;
    }

    int32_t lastTaskIdx = taskCount - 1;
    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, lastTaskIdx % bufferCount,
        expertGlobalRowBegin + rowCursor.expertRow,
        tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
        pendingTopkIndex, traceMetrics);

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t drainWaitBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->transferMte3DrainWaitCycles += MegaMoeWaveTraceDetail::Now() - drainWaitBegin;
    }
#endif
}

// rank-round 路径在完整专家分段内维持同一条 MTE 流水。尾部统一回收后，再向
// 分段覆盖的每个 GMM1 tile 发布本核实际完成的 row 数。
__aicore__ inline void PublishInterleavedDispatchTilesReady(
    const TokenDispatchConfig &context, const Params &params,
    uint32_t localExpertId, int32_t expertGlobalRowBegin,
    int32_t segmentRowBegin, int32_t segmentRowEnd, int32_t gmm1TileRowCount,
    WaveDispatchTraceMetrics *traceMetrics)
{
    if (segmentRowBegin >= segmentRowEnd) {
        return;
    }

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t readySyncBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->readySyncWaitCycles += MegaMoeWaveTraceDetail::Now() - readySyncBegin;
    }
#endif

    __gm__ int32_t *gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * context.dispatchFlagSlotsPerExpert;

    // 按与搬运完全一致的固定 lane row cursor 精确累计 tile 内行数，
    // 避免 wave 在专家中部截断时出现边界误差。
    DispatchCoreRowCursor rowCursor = GetFirstDispatchRowOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    int32_t currentTileIdx = -1;
    int32_t readyRowCount = 0;
    while (rowCursor.expertRow < segmentRowEnd) {
        int32_t tileIdx = rowCursor.expertRow / gmm1TileRowCount;
        if (tileIdx != currentTileIdx) {
            if (readyRowCount > 0) {
                AtomicAdd(gmm1TileReadyCount +
                          static_cast<int64_t>(currentTileIdx) * INT_CACHELINE,
                          readyRowCount);
            }
            currentTileIdx = tileIdx;
            readyRowCount = 0;
        }
        ++readyRowCount;
        AdvanceDispatchCoreRowCursor(rowCursor, coreCount);
    }
    if (readyRowCount > 0) {
        AtomicAdd(gmm1TileReadyCount +
                  static_cast<int64_t>(currentTileIdx) * INT_CACHELINE,
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
 *       -> 按 globalRow % coreCount 对应的固定目的 core。
 *
 * 本函数只为当前 core 完成以下六个阶段：
 *   1. 确定当前专家分段及截断后的专家总行数；
 *   2. 为跨 wave 的专家维护每个 rank 的 mask 扫描 cursor；
 *   3. 每个 coreCount-sized round 将本核领取的 lane 右移一位；
 *   4. 保持逻辑 row 的 rank-round 映射不变，解析并回填本核 route queue；
 *   5. 复用同一 MTE 环形流水，按 queue 顺序读取远端 token/metaInfo 并写目的 row；
 *   6. MTE3 全部完成后，按 GMM1 tile 发布本核实际完成的 row 数。
 *
 * 分核不改变 expertRow 对应的 source rank/ordinal。以28核为例，
 * 同一专家内 core0 固定跨 28 行领取任务，不再逐 round 换卡。
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
    // 阶段2：旧 mask 回退路径维护跨 wave 扫描起点。共享 Route Table 路径
    // 直接按 compact ordinal 定位，不读取该 cursor。
    // 专家首段不能读取旧 cursor。只有确实跨 wave 的专家才需要所有核清空私有
    // cursor；即使某核在首段没有任务，也必须为后续专家续段建立一致初始状态。
    if (MEGA_MOE_USE_SHARED_DISPATCH_ROUTE_TABLE == 0 &&
        segmentRowBegin == 0 && segmentRowEnd < expertRowCount) {
        ResetDispatchRankScanCursors(context, scratch);
    }
    // 阶段3：划分本核目的 row，并为本地 route queue 预留 UB。
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    int32_t coreCount = static_cast<int32_t>(
        context.blockJob.totalJobs == 0U ? 1U : context.blockJob.totalJobs);
    // 当前分段触及的 core-sized round 数是所有核统一的安全任务上界；分段截断
    // 时个别核可能少一项，但不会超过该值。
    int32_t maxCoreTaskCount =
        (segmentRowEnd - 1) / coreCount - segmentRowBegin / coreCount + 1;
    int32_t maxTaskCapacity = Ops::Base::CeilAlign(
        maxCoreTaskCount, DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS);
    // 旧路径还要在 queue/descriptor 之后留出 CreateVecIndex 区；共享表路径
    // 的 topkIndexTensor 只保存最终 queue，validTopkIndexTensor 同时容纳本核所需 rank slice。
    int32_t legacyRouteCapacity = context.bufferConfig.routeItemsPerBatch - maxTaskCapacity * 2;
    legacyRouteCapacity = legacyRouteCapacity / static_cast<int32_t>(ALIGN_256) *
        static_cast<int32_t>(ALIGN_256);
    uint64_t sendTotalNum =
        static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;
    // 紧凑 route word 高8位保存 rank、低24位保存 topkIndex；超过编码范围时
    // 回退到直接按 row/rank 处理的兼容路径。
    bool canPackRouteEntry =
        context.common.worldSize <= DISPATCH_ROUTE_RANK_CAPACITY &&
        sendTotalNum <= static_cast<uint64_t>(DISPATCH_ROUTE_TOPK_MASK) + 1U;
    // 只有旧 grouped-descriptor 路径会把 taskIdx 编码进高8位。
    bool canPackGroupedDescriptor =
        maxTaskCapacity <= DISPATCH_GROUPED_TASK_INDEX_CAPACITY;
    // 每个 rank slice 在 UB 中按 32B 对齐。实际 padding 不超过
    // worldSize * (alignItems - 1)，入口先按上界保护，避免规划中途因容量不足返回。
    uint64_t sharedRouteSliceCapacityUpperBound =
        static_cast<uint64_t>(segmentRowEnd - segmentRowBegin) +
        static_cast<uint64_t>(context.common.worldSize) *
            static_cast<uint64_t>(DISPATCH_ROUTE_QUEUE_ALIGN_ITEMS - 1);
    const bool canUseSharedRouteTable =
        MEGA_MOE_USE_SHARED_DISPATCH_ROUTE_TABLE != 0 &&
        params.workspaceInfo.dispatchRouteIndexPtr != nullptr &&
        maxTaskCapacity <= context.bufferConfig.routeItemsPerBatch &&
        sharedRouteSliceCapacityUpperBound <=
            static_cast<uint64_t>(context.bufferConfig.routeItemsPerBatch);
    if (!canPackRouteEntry ||
        (!canUseSharedRouteTable &&
         (legacyRouteCapacity <= 0 || !canPackGroupedDescriptor))) {
        DispatchExpertTokensByRows<ActivationType, QuantScaleType, PipelineTileM,
                                   TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, localExpertId,
            expertGlobalRowBegin, dispatchRowCount, expertRowOffset);
        return;
    }

    // 使用与搬运、ready 发布完全相同的 row cursor 得到本核精确任务数。
    int32_t expectedTaskCount = CountDispatchRowsOwnedByCore(
        context, expertGlobalRowBegin, segmentRowBegin, segmentRowEnd);
    if (expectedTaskCount <= 0) {
        return;
    }

#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    uint32_t countTraceBegin = traceMetrics == nullptr ? 0U : MegaMoeWaveTraceDetail::Now();
#endif
    // 阶段4：把各 source rank 的 mask 解析为按本核目的 row 排序的 route queue。
    // 从专家 cumsum 计算每个 source rank 的独立 token 数，供原始 rank-round
    // 映射、活跃 rank 切分和 ordinal 边界计算共同使用。
    LoadDispatchExpertRankCounts(context, scratch, localExpertId, expertGlobalRowBegin);
#if MEGA_MOE_ENABLE_WAVE_TRACE != 0
    if (traceMetrics != nullptr) {
        traceMetrics->prepareDispatchCountsCycles += MegaMoeWaveTraceDetail::Now() - countTraceBegin;
    }
#endif
    // 规划结果按本核目的 row 顺序存放，每项仅包含 {sourceRank, topkIndex}。
    DispatchCoreRouteQueue queue = canUseSharedRouteTable ?
        PlanDispatchCoreRouteQueueFromTable(
            context, params, scratch, localExpertId, expertGlobalRowBegin,
            segmentRowBegin, segmentRowEnd, expectedTaskCount,
            maxTaskCapacity, traceMetrics) :
        PlanDispatchCoreRouteQueue(
            context, params, scratch, localExpertId, expertGlobalRowBegin,
            segmentRowBegin, segmentRowEnd, expectedTaskCount,
            maxTaskCapacity, traceMetrics);
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
    // queue 已按本核固定 lane 的目的 row 排序；搬运使用同一个 row cursor 写回。
    CopyInterleavedDispatchRouteQueue<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, winRankAddr, scratch, expertGlobalRowBegin,
        segmentRowBegin, segmentRowEnd, expectedTaskCount, tokenRevGlobalTensor,
        scaleRevGlobalTensor, metaInfoGlobalTensor, traceMetrics);
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
        segmentRowBegin, segmentRowEnd, gmm1TileRowCount, traceMetrics);
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
