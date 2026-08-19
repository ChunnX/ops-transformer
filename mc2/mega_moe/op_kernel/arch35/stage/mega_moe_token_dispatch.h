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
    int32_t copyStartIdx, int32_t copyIdx, int32_t copyIndexStride = 1)
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
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
            copyStartIdx + copyIdx * copyIndexStride);
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

// Wave Dispatch 的选中 token 和目标 row 都可能是等差序列。该版本仍复用原有
// 2~6 槽 MTE2/MTE3 ring，只把 source ordinal 与 destination row 的步长显式化。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyStridedTokensAndMetaForDispatch(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, int32_t firstDstRow, int32_t remoteRankIdx,
    int32_t firstBatchMatchIdx, int32_t tokenCount, int32_t sourceOrdinalStride,
    int32_t destinationRowStride)
{
    if (tokenCount <= 0) {
        return;
    }
    int32_t bufferCount = context.bufferConfig.bufferCount;
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params.workspaceInfo.dispatchRevDataPtr + firstDstRow * scratch.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(
        params.workspaceInfo.dispatchRevScalePtr + firstDstRow * scratch.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        winRankAddr[remoteRankIdx] + context.quantWinOffset));
    metaInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.metaInfoPtr + firstDstRow * INT32_PER_256B * sizeof(int32_t)));

    int32_t firstTopkIndex = scratch.validTopkIndexTensor.GetValue(firstBatchMatchIdx);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(
        context, scratch, 0, firstTopkIndex, remoteRankIdx, remoteRankGlobalTensor);

    int32_t firstUseEnd = tokenCount < bufferCount ? tokenCount : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
            firstBatchMatchIdx + issueIdx * sourceOrdinalStride);
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(
            context, scratch, issueIdx, topkIndex, remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyIdx, copyIdx * destinationRowStride, tokenRevGlobalTensor,
            scaleRevGlobalTensor, metaInfoGlobalTensor, firstBatchMatchIdx, copyIdx,
            sourceOrdinalStride);
    }
    for (int32_t issueIdx = bufferCount; issueIdx < tokenCount; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(
            firstBatchMatchIdx + issueIdx * sourceOrdinalStride);
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(
            context, scratch, issueBufferIdx, topkIndex, remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyBufferIdx, copyIdx * destinationRowStride, tokenRevGlobalTensor,
            scaleRevGlobalTensor, metaInfoGlobalTensor, firstBatchMatchIdx, copyIdx,
            sourceOrdinalStride);
    }
    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, (tokenCount - 1) % bufferCount,
        (tokenCount - 1) * destinationRowStride, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, firstBatchMatchIdx, tokenCount - 1, sourceOrdinalStride);
    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// 所有目标 row 已写回后，按 GMM1 的专家内 M-group 汇总发布 ready count。
__aicore__ inline void PublishStridedGmm1TileReady(
    const TokenDispatchConfig &context, const Params &params, uint32_t localExpertId,
    int32_t firstExpertRow, int32_t rowCount, int32_t rowStride, int32_t tileRowCount)
{
    if (rowCount <= 0) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    __gm__ int32_t *flagBase =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * context.dispatchFlagSlotsPerExpert;
    int32_t lastExpertRow = firstExpertRow + (rowCount - 1) * rowStride;
    int32_t firstTileIdx = firstExpertRow / tileRowCount;
    int32_t lastTileIdx = lastExpertRow / tileRowCount;
    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileRowBegin = tileIdx * tileRowCount;
        int32_t tileRowEnd = tileRowBegin + tileRowCount;
        int32_t firstSelected = tileRowBegin <= firstExpertRow ? 0 :
            Ops::Base::CeilDiv(tileRowBegin - firstExpertRow, rowStride);
        int32_t selectedEnd = tileRowEnd <= firstExpertRow ? 0 :
            Ops::Base::CeilDiv(tileRowEnd - firstExpertRow, rowStride);
        firstSelected = firstSelected < rowCount ? firstSelected : rowCount;
        selectedEnd = selectedEnd < rowCount ? selectedEnd : rowCount;
        int32_t tileReadyRows = selectedEnd - firstSelected;
        if (tileReadyRows > 0) {
            AtomicAdd(flagBase + static_cast<int64_t>(tileIdx) * INT_CACHELINE, tileReadyRows);
        }
    }
}

// Wave 专用二层路由：第一层的 core stride 跨专家连续，不在专家边界重置；
// 第二层在当前专家内按活跃 source rank 逐轮交织，rank 耗尽后从后续轮次移除。
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchWaveExpertSegmentInterleaved(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, int32_t expertRowBegin, int32_t expertRowEnd,
    int32_t expertWaveLogicalRowBegin)
{
    if (expertRowBegin >= expertRowEnd || context.blockJob.totalJobs == 0U) {
        return;
    }
    constexpr int32_t routeCacheStride = ALIGN_32 / sizeof(int32_t);
    constexpr int32_t matchedCountSlot = 0;
    constexpr int32_t tokenCountSlot = 1;
    constexpr int32_t layerBaseSlot = 2;
    constexpr int32_t lastSelectedSlot = 3;
    constexpr int32_t ordinalEndSlot = 4;
    constexpr int32_t firstSelectedSlot = 5;
    constexpr int32_t ordinalStrideSlot = 6;
    constexpr int32_t rowStrideSlot = 7;
    int32_t coreCount = static_cast<int32_t>(context.blockJob.totalJobs);
    int32_t coreLane = static_cast<int32_t>(context.blockJob.jobIndex);
    int32_t maskSlotSize = static_cast<int32_t>(context.maskAlignSize + ALIGN_32);
    int32_t routeItemsPerBatch = context.bufferConfig.routeItemsPerBatch;
    int32_t routeBatchCount = context.bufferConfig.routeBatchCount;
    uint64_t sendTotalNum = static_cast<uint64_t>(context.common.tokenNum) * context.common.topK;

    int32_t completedRound = 0;
    int32_t layerLogicalRowBegin = 0;
    while (layerLogicalRowBegin < expertRowEnd) {
        int32_t activeRankCount = 0;
        int32_t nextRound = 0x7fffffff;
        int32_t previousRankPrefix = expertGlobalRowBegin;
        for (int32_t rankIdx = 0; rankIdx < static_cast<int32_t>(context.common.worldSize); ++rankIdx) {
            int32_t rankPrefix = scratch.cumsumInfoTensor.GetValue(
                localExpertId * context.common.worldSize + static_cast<uint32_t>(rankIdx));
            int32_t rankTokenCount = rankPrefix - previousRankPrefix;
            previousRankPrefix = rankPrefix;
            if (rankTokenCount > completedRound) {
                ++activeRankCount;
                nextRound = rankTokenCount < nextRound ? rankTokenCount : nextRound;
            }
        }
        if (activeRankCount == 0 || nextRound <= completedRound) {
            break;
        }
        int32_t layerLogicalRowEnd = layerLogicalRowBegin +
            (nextRound - completedRound) * activeRankCount;
        int32_t segmentLayerBegin = expertRowBegin > layerLogicalRowBegin ?
            expertRowBegin : layerLogicalRowBegin;
        int32_t segmentLayerEnd = expertRowEnd < layerLogicalRowEnd ?
            expertRowEnd : layerLogicalRowEnd;

        if (segmentLayerEnd > segmentLayerBegin) {
            int32_t activeRankPosition = 0;
            previousRankPrefix = expertGlobalRowBegin;
            for (int32_t rankIdx = 0; rankIdx < static_cast<int32_t>(context.common.worldSize); ++rankIdx) {
                int32_t rankPrefix = scratch.cumsumInfoTensor.GetValue(
                    localExpertId * context.common.worldSize + static_cast<uint32_t>(rankIdx));
                int32_t rankTokenCount = rankPrefix - previousRankPrefix;
                previousRankPrefix = rankPrefix;
                int32_t cacheBase = rankIdx * routeCacheStride;
                scratch.sendCntTensor.SetValue(cacheBase + tokenCountSlot, rankTokenCount);
                scratch.sendCntTensor.SetValue(cacheBase + matchedCountSlot, 0);
                scratch.sendCntTensor.SetValue(cacheBase + firstSelectedSlot, 0x7fffffff);
                scratch.sendCntTensor.SetValue(cacheBase + ordinalEndSlot, 0);
                if (rankTokenCount <= completedRound) {
                    continue;
                }
                int32_t rankLayerLogicalBase = layerLogicalRowBegin + activeRankPosition++;
                if (rankLayerLogicalBase >= segmentLayerEnd) {
                    continue;
                }
                int32_t ordinalBegin = completedRound;
                if (rankLayerLogicalBase < segmentLayerBegin) {
                    ordinalBegin += Ops::Base::CeilDiv(
                        segmentLayerBegin - rankLayerLogicalBase, activeRankCount);
                }
                int32_t ordinalEnd = completedRound + Ops::Base::CeilDiv(
                    segmentLayerEnd - rankLayerLogicalBase, activeRankCount);
                ordinalEnd = ordinalEnd < nextRound ? ordinalEnd : nextRound;
                if (ordinalEnd <= ordinalBegin) {
                    continue;
                }
                int32_t firstSelected = ordinalEnd;
                int32_t probeEnd = ordinalBegin + coreCount;
                probeEnd = probeEnd < ordinalEnd ? probeEnd : ordinalEnd;
                for (int32_t ordinal = ordinalBegin; ordinal < probeEnd; ++ordinal) {
                    int32_t expertLogicalRow = rankLayerLogicalBase +
                        (ordinal - completedRound) * activeRankCount;
                    int32_t waveLogicalRow = expertWaveLogicalRowBegin +
                        expertLogicalRow - expertRowBegin;
                    if (waveLogicalRow % coreCount == coreLane) {
                        firstSelected = ordinal;
                        break;
                    }
                }
                if (firstSelected >= ordinalEnd) {
                    continue;
                }
                int32_t gcdValue = activeRankCount;
                int32_t gcdRemainder = coreCount;
                while (gcdRemainder != 0) {
                    int32_t nextRemainder = gcdValue % gcdRemainder;
                    gcdValue = gcdRemainder;
                    gcdRemainder = nextRemainder;
                }
                int32_t ordinalStride = coreCount / gcdValue;
                int32_t expertRowStride = activeRankCount * ordinalStride;
                int32_t lastSelected = firstSelected +
                    ((ordinalEnd - 1 - firstSelected) / ordinalStride) * ordinalStride;
                scratch.sendCntTensor.SetValue(cacheBase + layerBaseSlot, rankLayerLogicalBase);
                scratch.sendCntTensor.SetValue(cacheBase + lastSelectedSlot, lastSelected);
                scratch.sendCntTensor.SetValue(cacheBase + ordinalEndSlot, ordinalEnd);
                scratch.sendCntTensor.SetValue(cacheBase + firstSelectedSlot, firstSelected);
                scratch.sendCntTensor.SetValue(cacheBase + ordinalStrideSlot, ordinalStride);
                scratch.sendCntTensor.SetValue(cacheBase + rowStrideSlot, expertRowStride);
            }

            for (int32_t batchIdx = 0; batchIdx < routeBatchCount; ++batchIdx) {
                int32_t batchRouteBegin = batchIdx * routeItemsPerBatch;
                bool isLastBatch = batchIdx == routeBatchCount - 1;
                int32_t validRouteCount = routeItemsPerBatch;
                int32_t maskSliceBytes = routeItemsPerBatch / 8;
                if (isLastBatch) {
                    validRouteCount = static_cast<int32_t>(sendTotalNum - batchRouteBegin);
                    if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
                        maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - batchRouteBegin / 8;
                    }
                }
                CreateVecIndex(scratch.topkIndexTensor, batchRouteBegin, routeItemsPerBatch);
                for (int32_t rankIdx = 0; rankIdx < static_cast<int32_t>(context.common.worldSize); ++rankIdx) {
                    int32_t cacheBase = rankIdx * routeCacheStride;
                    int32_t rankTokenCount = scratch.sendCntTensor.GetValue(cacheBase + tokenCountSlot);
                    int32_t firstSelected = scratch.sendCntTensor.GetValue(cacheBase + firstSelectedSlot);
                    int32_t ordinalEnd = scratch.sendCntTensor.GetValue(cacheBase + ordinalEndSlot);
                    if (rankTokenCount <= completedRound || firstSelected >= ordinalEnd) {
                        continue;
                    }
                    int32_t matchedCount = scratch.sendCntTensor.GetValue(cacheBase + matchedCountSlot);
                    int32_t lastSelected = scratch.sendCntTensor.GetValue(cacheBase + lastSelectedSlot);
                    if (matchedCount > lastSelected) {
                        continue;
                    }
                    GlobalTensor<uint8_t> rankMaskGm;
                    rankMaskGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                        params.peermemInfo.maskRecvPtr +
                        (static_cast<uint64_t>(localExpertId) * context.common.worldSize +
                         static_cast<uint32_t>(rankIdx)) * maskSlotSize + batchRouteBegin / 8));
                    DataCopy(scratch.maskBatchTensor, rankMaskGm, static_cast<uint32_t>(maskSliceBytes));
                    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
                    uint64_t batchMatchedCount = 0U;
                    GatherMask(scratch.validTopkIndexTensor, scratch.topkIndexTensor,
                               scratch.maskBatchU32Tensor, true, static_cast<uint32_t>(validRouteCount),
                               {1, 1, 0, 0}, batchMatchedCount);
                    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                    int32_t batchOrdinalBegin = matchedCount;
                    int32_t batchOrdinalEnd = matchedCount + static_cast<int32_t>(batchMatchedCount);
                    int32_t ordinalStride = scratch.sendCntTensor.GetValue(cacheBase + ordinalStrideSlot);
                    int32_t expertRowStride = scratch.sendCntTensor.GetValue(cacheBase + rowStrideSlot);
                    int32_t layerBase = scratch.sendCntTensor.GetValue(cacheBase + layerBaseSlot);
                    if (firstSelected < batchOrdinalBegin) {
                        firstSelected += Ops::Base::CeilDiv(
                            batchOrdinalBegin - firstSelected, ordinalStride) * ordinalStride;
                    }
                    int32_t selectedEnd = ordinalEnd < batchOrdinalEnd ? ordinalEnd : batchOrdinalEnd;
                    if (firstSelected < selectedEnd) {
                        int32_t selectedCount = 1 +
                            (selectedEnd - 1 - firstSelected) / ordinalStride;
                        int32_t batchLocalMatchBegin = firstSelected - batchOrdinalBegin;
                        int32_t firstExpertRow = layerBase +
                            (firstSelected - completedRound) * activeRankCount;
                        int32_t firstDstRow = expertGlobalRowBegin + firstExpertRow;
                        CopyStridedTokensAndMetaForDispatch<ActivationType, QuantScaleType,
                                                           TopkWeightsPrefetch>(
                            context, params, winRankAddr, scratch, firstDstRow, rankIdx,
                            batchLocalMatchBegin, selectedCount, ordinalStride, expertRowStride);
                        PublishStridedGmm1TileReady(
                            context, params, localExpertId, firstExpertRow, selectedCount,
                            expertRowStride, static_cast<int32_t>(PipelineTileM));
                    }
                    scratch.sendCntTensor.SetValue(cacheBase + matchedCountSlot, batchOrdinalEnd);
                }
            }
        }
        layerLogicalRowBegin = layerLogicalRowEnd;
        completedRound = nextRound;
    }
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
            copyStartIdx, copyIdx);
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
            copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, (copyNum - 1) % bufferCount, copyNum - 1, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

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
    int32_t expertGlobalRowBegin, uint32_t dispatchRowCount)
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

    DispatchExpertRange expertRange{coreDispatchRowBegin, coreDispatchRowEnd};
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
