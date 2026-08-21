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
#include "mega_moe_token_quant.h"

namespace MegaMoeImpl {

using namespace AscendC;

// Token Dispatch 的阶段私有配置。任务分工(BlockJobContext)与 count 槽位(BlockWorkspaceContext)
// 与 GMM 阶段共用编排层的同一份实例，不在此重复持有。
struct TokenDispatchConfig {
    uint64_t maxOutputSize;
    // 全卡一致的单卡 token 数上界, 决定远端 mask 的扫描宽度(上界*topK);
    // common.tokenNum 保持本卡真实 bs, 两者在可变 bs 下不可混用。
    uint64_t numMaxTokensPerRank;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint32_t maskAlignSize;
    // 与 SendMaskConfig 同源的槽位尺寸（mask 位区 + 32B count 区），装配时算一次。
    uint32_t maskSlotSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
};

template <typename ActivationType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    LocalTensor<int32_t> topkIndexTensor;
    LocalTensor<uint8_t> maskBatchTensor;
    LocalTensor<uint32_t> maskBatchU32Tensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    // Contiguous runtime-sized ring. A slot view is built from this UB base on demand.
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
    uint32_t nextDispatchCoreIdx;
};

// Wave Dispatch 将专家有效行在全部物理 AIV1 上连续均分，并跨专家滚动首核。
struct WaveDispatchCoreRange {
    uint64_t expertRowCount;
    uint32_t expertGlobalRowBegin;
    uint32_t localExpertRowBegin;
    uint32_t localExpertRowEnd;
};

__aicore__ inline bool HasWaveDispatchRows(const WaveDispatchCoreRange &range)
{
    return range.localExpertRowBegin < range.localExpertRowEnd;
}

template <typename ActivationType>
__aicore__ inline WaveDispatchCoreRange PlanWaveExpertDispatch(
    const TokenDispatchConfig &context, const MoeStageCommonConfig &common, const BlockJobContext &blockJob,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId)
{
    uint32_t expertPrefixIndex = (localExpertId + 1U) * common.worldSize - 1U;
    int32_t expertGlobalRowEndValue = scratch.cumsumInfoTensor.GetValue(expertPrefixIndex);
    int32_t expertGlobalRowBeginValue = 0;
    if (localExpertId != 0U) {
        expertGlobalRowBeginValue = scratch.cumsumInfoTensor.GetValue(localExpertId * common.worldSize - 1U);
    }

    uint32_t expertGlobalRowBegin = static_cast<uint32_t>(expertGlobalRowBeginValue);
    uint32_t expertGlobalRowEnd = static_cast<uint32_t>(expertGlobalRowEndValue);
    uint64_t expertRowCount = expertGlobalRowEnd > expertGlobalRowBegin ?
                                  static_cast<uint64_t>(expertGlobalRowEnd - expertGlobalRowBegin) :
                                  0U;
    WaveDispatchCoreRange range{expertRowCount, expertGlobalRowBegin, 0U, 0U};

    uint32_t totalCoreCount = blockJob.totalJobs;
    if (expertRowCount == 0U || totalCoreCount == 0U || expertGlobalRowBegin >= context.maxOutputSize) {
        return range;
    }

    uint64_t validGlobalRowEnd =
        expertGlobalRowEnd < context.maxOutputSize ? expertGlobalRowEnd : context.maxOutputSize;
    uint32_t validExpertRowCount = static_cast<uint32_t>(validGlobalRowEnd - expertGlobalRowBegin);
    uint32_t activeCoreCount = validExpertRowCount < totalCoreCount ? validExpertRowCount : totalCoreCount;

    uint32_t firstPhysicalCoreIdx = scratch.nextDispatchCoreIdx;
    uint32_t logicalCoreIdx = blockJob.jobIndex >= firstPhysicalCoreIdx ?
                                  blockJob.jobIndex - firstPhysicalCoreIdx :
                                  blockJob.jobIndex + totalCoreCount - firstPhysicalCoreIdx;
    if (logicalCoreIdx < activeCoreCount) {
        if (validExpertRowCount < totalCoreCount) {
            range.localExpertRowBegin = logicalCoreIdx;
            range.localExpertRowEnd = logicalCoreIdx + 1U;
        } else {
            uint32_t rowsPerCore = validExpertRowCount / totalCoreCount;
            uint32_t extraRowCoreCount = validExpertRowCount % totalCoreCount;
            range.localExpertRowBegin = logicalCoreIdx * rowsPerCore +
                                        (logicalCoreIdx < extraRowCoreCount ? logicalCoreIdx : extraRowCoreCount);
            range.localExpertRowEnd =
                range.localExpertRowBegin + rowsPerCore + (logicalCoreIdx < extraRowCoreCount ? 1U : 0U);
        }
    }

    uint32_t nextPhysicalCoreIdx = firstPhysicalCoreIdx + activeCoreCount;
    scratch.nextDispatchCoreIdx =
        nextPhysicalCoreIdx >= totalCoreCount ? nextPhysicalCoreIdx - totalCoreCount : nextPhysicalCoreIdx;
    return range;
}

// 由 tiling/peermem/quant 配置装配 Token Dispatch 阶段配置（普通与 wave 编排模板共用）。
__aicore__ inline TokenDispatchConfig CreateTokenDispatchConfig(const Params &params,
                                                                const QuantProcessConfig &quantProcessConfig,
                                                                uint32_t maskAlignSize)
{
    uint64_t quantWinOffset =
        static_cast<uint64_t>(params.peermemInfo.quantTokenScalePtr - params.peermemInfo.rankSyncInWorldPtr);
    return {.maxOutputSize = params.tilingData->maxOutputSize,
            .numMaxTokensPerRank = params.tilingData->numMaxTokensPerRank,
            .bufferConfig = params.tilingData->dispatchBufferConfig,
            .maskAlignSize = maskAlignSize,
            .maskSlotSize = maskAlignSize + static_cast<uint32_t>(ALIGN_32),
            .quantWinOffset = quantWinOffset,
            .quantTokenAlignBytes = quantProcessConfig.quantTokenAlignBytes,
            .quantScaleAlignBytes = quantProcessConfig.quantScaleAlignBytes,
            .quantTokenScaleAlignBytes = quantProcessConfig.quantTokenScaleAlignBytes};
}

template <typename ActivationType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(const TokenDispatchConfig &context,
                                                                    const TokenDispatchScratch<ActivationType> &scratch,
                                                                    int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig &context, const Params &params,
                                                     TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
                                                     int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType> &remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(params.tilingData->topK);
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
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(
        TOPK_INDEX, topkIndex % static_cast<int32_t>(params.tilingData->topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// copyIdx 同时是段内目标行号：三个调用点的 ring 排空顺序保证写出行与匹配序恒一一对应。
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig &context, const Params &params,
                                                     TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
                                                     GlobalTensor<ActivationType> &tokenRevGlobalTensor,
                                                     GlobalTensor<QuantScaleType> &scaleRevGlobalTensor,
                                                     GlobalTensor<int32_t> &metaInfoGlobalTensor, int32_t copyStartIdx,
                                                     int32_t copyIdx)
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
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx);
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(params.tilingData->topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    DataCopyPad(tokenRevGlobalTensor[copyIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[copyIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[copyIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Prototype: MegaMoe::CopyGMToGMPerToken. Copies one dispatch shard and emits its metadata.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(const TokenDispatchConfig &context, const Params &params,
                                                    GM_ADDR *winRankAddr, TokenDispatchScratch<ActivationType> &scratch,
                                                    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx,
                                                    int32_t copyNum)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t bufferCount = bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params.workspaceInfo.dispatchRevDataPtr + rowDstOffset * scratch.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(
        params.workspaceInfo.dispatchRevScalePtr + rowDstOffset * scratch.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    metaInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.metaInfoPtr + rowDstOffset * INT32_PER_256B * sizeof(int32_t)));

    // The only caller enters this function for a non-empty match interval, so copyNum is at least one.
    // GatherMask's V-to-S synchronization covers the first scalar read, and the drain below covers
    // cross-call reuse of the copy and metadata rings.
    int32_t firstTopkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, 0, firstTopkIndex,
                                                              remoteRankIdx, remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, issueIdx, topkIndex,
                                                                  remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, params, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyBufferIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, scratch, (copyNum - 1) % bufferCount, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// 发布一个 source-rank 段覆盖到的 GMM1 tile ready 计数。
__aicore__ inline void PublishGmm1TileReady(const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                            uint32_t localExpertId, int32_t gmm1TileRowCount,
                                            int32_t segmentExpertRowBegin, int32_t segmentExpertRowEnd)
{
    if (segmentExpertRowBegin >= segmentExpertRowEnd) {
        return;
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
    __gm__ int32_t *gmm1TileReadyCount =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr) +
        static_cast<uint64_t>(localExpertId) * syncLayout.dispatchFlagSlotCountPerExpert;
    int32_t firstTileIdx = segmentExpertRowBegin / gmm1TileRowCount;
    int32_t lastTileIdx = (segmentExpertRowEnd - 1) / gmm1TileRowCount;
    for (int32_t tileIdx = firstTileIdx; tileIdx <= lastTileIdx; ++tileIdx) {
        int32_t tileExpertRowBegin = tileIdx * gmm1TileRowCount;
        int32_t tileExpertRowEnd = tileExpertRowBegin + gmm1TileRowCount;
        int32_t overlapRowBegin =
            segmentExpertRowBegin > tileExpertRowBegin ? segmentExpertRowBegin : tileExpertRowBegin;
        int32_t overlapRowEnd = segmentExpertRowEnd < tileExpertRowEnd ? segmentExpertRowEnd : tileExpertRowEnd;
        AtomicAdd(gmm1TileReadyCount + static_cast<int64_t>(tileIdx) * INT_CACHELINE, overlapRowEnd - overlapRowBegin);
    }
}

/**
 * Dispatch 第 3 层：扫描单个 source rank 的 route mask，搬运选中的 token
 * 和 metadata，并发布 GMM1 tile ready count。
 */
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchRankTokens(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                          const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                          GM_ADDR *winRankAddr, TokenDispatchScratch<ActivationType> &scratch,
                                          uint32_t localExpertId, int32_t expertGlobalRowBegin, uint32_t remoteRankIdx,
                                          int32_t rankSegmentRowBegin, int32_t segmentMatchOrdinalBegin,
                                          int32_t segmentMatchOrdinalEnd, int32_t gmm1TileRowCount)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    uint32_t maskSlotSize = context.maskSlotSize;
    GlobalTensor<uint8_t> remoteRankMaskGlobal;
    int32_t matchedRouteCount = 0;
    int32_t dispatchedRowCount = 0;
    for (int32_t batchIdx = 0; batchIdx < bufferConfig.routeBatchCount && matchedRouteCount < segmentMatchOrdinalEnd;
         ++batchIdx) {
        int32_t batchRouteBegin = batchIdx * bufferConfig.routeItemsPerBatch;
        bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
        int32_t validRouteCount = bufferConfig.routeItemsPerBatch;
        int32_t maskSliceBytes = bufferConfig.routeItemsPerBatch / 8;
        if (isLastBatch) {
            uint64_t sendTotalNum = context.numMaxTokensPerRank * common.topK;
            validRouteCount = static_cast<int32_t>(sendTotalNum - static_cast<uint64_t>(batchRouteBegin));
            if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
                maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - batchRouteBegin / 8;
            }
        }

        remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
            params.peermemInfo.maskRecvPtr +
            (static_cast<uint64_t>(localExpertId) * common.worldSize + remoteRankIdx) * maskSlotSize +
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
            int32_t dispatchDstRowBegin = rankSegmentRowBegin + expertGlobalRowBegin + dispatchMatchOrdinalBegin;
            CopyTokensAndMetaForDispatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, params, winRankAddr, scratch, dispatchDstRowBegin, remoteRankIdx, batchLocalMatchBegin,
                batchDispatchRowCount);
            dispatchedRowCount += batchDispatchRowCount;
        }
        matchedRouteCount = batchMatchOrdinalEnd;
    }

    if (dispatchedRowCount > 0) {
        int32_t segmentExpertRowBegin = rankSegmentRowBegin + segmentMatchOrdinalBegin;
        int32_t segmentExpertRowEnd = segmentExpertRowBegin + dispatchedRowCount;
        PublishGmm1TileReady(syncLayout, params, localExpertId, gmm1TileRowCount, segmentExpertRowBegin,
                             segmentExpertRowEnd);
    }
}

template <typename ActivationType>
__aicore__ inline uint32_t FindDispatchSourceRank(const MoeStageCommonConfig &common,
                                                  const TokenDispatchScratch<ActivationType> &scratch,
                                                  uint32_t localExpertId, uint32_t globalRowIdx)
{
    uint32_t expertRankBegin = localExpertId * common.worldSize;
    uint32_t searchBegin = expertRankBegin;
    uint32_t searchEnd = expertRankBegin + common.worldSize;
    while (searchBegin < searchEnd) {
        uint32_t middle = searchBegin + (searchEnd - searchBegin) / 2U;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(middle));
        if (rankGlobalRowEnd <= globalRowIdx) {
            searchBegin = middle + 1U;
        } else {
            searchEnd = middle;
        }
    }
    return searchBegin - expertRankBegin;
}

// Wave 专用路径：本核只扫描与连续专家行区间相交的 source rank，未分到行的核直接返回。
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void DispatchWaveExpertRows(const TokenDispatchConfig &context,
                                              const MoeStageCommonConfig &common,
                                              const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                              GM_ADDR *winRankAddr, TokenDispatchScratch<ActivationType> &scratch,
                                              uint32_t localExpertId, const WaveDispatchCoreRange &coreRange)
{
    if (!HasWaveDispatchRows(coreRange)) {
        return;
    }

    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t coreGlobalRowBegin = coreRange.expertGlobalRowBegin + coreRange.localExpertRowBegin;
    uint32_t coreGlobalRowEnd = coreRange.expertGlobalRowBegin + coreRange.localExpertRowEnd;
    uint32_t sourceRankIdx = FindDispatchSourceRank(common, scratch, localExpertId, coreGlobalRowBegin);

    while (sourceRankIdx < common.worldSize) {
        uint32_t prefixIndex = localExpertId * common.worldSize + sourceRankIdx;
        uint32_t rankGlobalRowEnd = static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex));
        uint32_t rankGlobalRowBegin =
            prefixIndex == 0U ? 0U : static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(prefixIndex - 1U));
        if (rankGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }

        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > rankGlobalRowBegin ? coreGlobalRowBegin : rankGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < rankGlobalRowEnd ? coreGlobalRowEnd : rankGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            DispatchRankTokens<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, common, syncLayout, params, winRankAddr, scratch, localExpertId,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                static_cast<int32_t>(rankGlobalRowBegin - coreRange.expertGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin), gmm1TileRowCount);
        }
        ++sourceRankIdx;
    }
}

// 非 layered 路径统一入口：使用紧凑 count 表规划跨专家滚动分核，并发送当前专家的有效行。
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline uint32_t RunWaveExpertDispatchStage(
    const TokenDispatchConfig &context, const MoeStageCommonConfig &common, const BlockJobContext &blockJob,
    const MoeSyncWorkspaceLayout &syncLayout, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId)
{
    if constexpr (g_coreType == AIC) {
        return 0U;
    }
    if (GetSubBlockIdx() != 1U) {
        return 0U;
    }
    WaveDispatchCoreRange coreRange =
        PlanWaveExpertDispatch(context, common, blockJob, scratch, localExpertId);
    if (HasWaveDispatchRows(coreRange)) {
        DispatchWaveExpertRows<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
            context, common, syncLayout, params, winRankAddr, scratch, localExpertId, coreRange);
    }
    return static_cast<uint32_t>(coreRange.expertRowCount);
}

// 非 layered 路径统一在 GMM pipeline 前完成全部专家 Dispatch。
template <typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM, bool TopkWeightsPrefetch>
__aicore__ inline void RunAllWaveExpertDispatchStages(
    const TokenDispatchConfig &context, const MoeStageCommonConfig &common, const BlockJobContext &blockJob,
    const MoeSyncWorkspaceLayout &syncLayout, const Params &params, GM_ADDR *winRankAddr,
    TokenDispatchScratch<ActivationType> &scratch)
{
    for (uint32_t expertIdx = 0U; expertIdx < common.moeExpertPerRank; ++expertIdx) {
        RunWaveExpertDispatchStage<ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
            context, common, blockJob, syncLayout, params, winRankAddr, scratch, expertIdx);
    }
}

// 每个寄存器内执行 Hillis-Steele int32 prefix-scan，再用 carry 串接相邻寄存器；随后提取
// 每个 expert 的累计尾值并作差。高阶 CumSum 在 A5 上只支持 FP16/FP32；这里保持整数计算，
// 避免累计路由数超过 2^24 后丢精度。
static __aicore__ inline void ComputeCompactExpertCountTablesVF(__local_mem__ int32_t *count,
                                                                __local_mem__ int32_t *expertCounts,
                                                                uint32_t elementCount, uint32_t expertCount,
                                                                uint32_t worldSize)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = GetVecLen() / sizeof(int32_t);
    {
        MicroAPI::RegTensor<int32_t> prefixReg;
        MicroAPI::RegTensor<int32_t> shiftedReg;
        MicroAPI::RegTensor<int32_t> carryReg;
        MicroAPI::RegTensor<int32_t> laneIndexReg;
        MicroAPI::RegTensor<int32_t> shiftedIndexReg;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<int32_t>();

        MicroAPI::Arange(laneIndexReg, 0);
        MicroAPI::Duplicate(carryReg, 0, fullMask);
        for (uint32_t vectorOffset = 0U; vectorOffset < elementCount; vectorOffset += ELEMENTS_PER_VECTOR) {
            uint32_t validCount = elementCount - vectorOffset;
            validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
            uint32_t maskCount = validCount;
            MicroAPI::MaskReg activeMask = MicroAPI::UpdateMask<int32_t>(maskCount);
            MicroAPI::DataCopy(prefixReg, count + vectorOffset);

            for (uint32_t shift = 1U; shift < validCount; shift <<= 1U) {
                MicroAPI::Adds(shiftedIndexReg, laneIndexReg, -static_cast<int32_t>(shift), fullMask);
                MicroAPI::Maxs(shiftedIndexReg, shiftedIndexReg, 0, fullMask);
                MicroAPI::Gather(shiftedReg, prefixReg,
                                 reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(shiftedIndexReg));
                MicroAPI::MaskReg shiftedLaneMask;
                MicroAPI::Compares<int32_t, CMPMODE::GE>(shiftedLaneMask, laneIndexReg, static_cast<int32_t>(shift),
                                                         activeMask);
                MicroAPI::Add<int32_t, MicroAPI::MaskMergeMode::MERGING>(prefixReg, prefixReg, shiftedReg,
                                                                         shiftedLaneMask);
            }

            MicroAPI::Add<int32_t, MicroAPI::MaskMergeMode::MERGING>(prefixReg, prefixReg, carryReg, activeMask);
            MicroAPI::DataCopy(count + vectorOffset, prefixReg, activeMask);

            MicroAPI::Duplicate(shiftedIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
            MicroAPI::Gather(carryReg, prefixReg, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(shiftedIndexReg));
        }
    }

    // 前半段把 prefix 写回 UB；后半段将重新从同一区域 Gather，显式建立 VEC_STORE -> VEC_LOAD 可见性。
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    // 在同一次 VF 中提取各专家尾值，避免第二次 VF 启动与外部 PIPE_V。
    {
        MicroAPI::RegTensor<int32_t> expertEndPrefixReg;
        MicroAPI::RegTensor<int32_t> previousEndPrefixReg;
        MicroAPI::RegTensor<int32_t> expertCountReg;
        MicroAPI::RegTensor<int32_t> previousVectorEndReg;
        MicroAPI::RegTensor<int32_t> laneIndexReg;
        MicroAPI::RegTensor<int32_t> gatherIndexReg;
        MicroAPI::RegTensor<int32_t> previousLaneIndexReg;
        MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<int32_t>();
        MicroAPI::MaskReg firstLaneMask = MicroAPI::CreateMask<int32_t, MicroAPI::MaskPattern::VL1>();

        MicroAPI::Arange(laneIndexReg, 0);
        MicroAPI::Duplicate(previousVectorEndReg, 0, fullMask);
        for (uint32_t expertOffset = 0U; expertOffset < expertCount; expertOffset += ELEMENTS_PER_VECTOR) {
            uint32_t validCount = expertCount - expertOffset;
            validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
            uint32_t maskCount = validCount;
            MicroAPI::MaskReg activeMask = MicroAPI::UpdateMask<int32_t>(maskCount);

            MicroAPI::Adds(gatherIndexReg, laneIndexReg, static_cast<int32_t>(expertOffset), fullMask);
            MicroAPI::Muls(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize), fullMask);
            MicroAPI::Adds(gatherIndexReg, gatherIndexReg, static_cast<int32_t>(worldSize - 1U), fullMask);
            MicroAPI::Gather(expertEndPrefixReg, count,
                             reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(gatherIndexReg), activeMask);

            MicroAPI::Adds(previousLaneIndexReg, laneIndexReg, -1, fullMask);
            MicroAPI::Maxs(previousLaneIndexReg, previousLaneIndexReg, 0, fullMask);
            MicroAPI::Gather(previousEndPrefixReg, expertEndPrefixReg,
                             reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(previousLaneIndexReg));
            MicroAPI::Select(previousEndPrefixReg, previousVectorEndReg, previousEndPrefixReg, firstLaneMask);
            MicroAPI::Sub(expertCountReg, expertEndPrefixReg, previousEndPrefixReg, activeMask);
            MicroAPI::DataCopy(expertCounts + expertOffset, expertCountReg, activeMask);

            MicroAPI::Duplicate(previousLaneIndexReg, static_cast<int32_t>(validCount - 1U), fullMask);
            MicroAPI::Gather(previousVectorEndReg, expertEndPrefixReg,
                             reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(previousLaneIndexReg));
        }
    }
}

__aicore__ inline void ComputeCompactExpertCountTables(LocalTensor<int32_t> countTensor,
                                                       LocalTensor<int32_t> expertCountTensor, uint32_t expertCount,
                                                       uint32_t worldSize)
{
    __local_mem__ int32_t *count = reinterpret_cast<__local_mem__ int32_t *>(countTensor.GetPhyAddr());
    __local_mem__ int32_t *expertCounts = reinterpret_cast<__local_mem__ int32_t *>(expertCountTensor.GetPhyAddr());
    VF_CALL<ComputeCompactExpertCountTablesVF>(count, expertCounts, expertCount * worldSize, expertCount, worldSize);
    PipeBarrier<PIPE_V>();
}

// Wave 进入逐专家流水前一次准备完整 count 表；每个物理 block 只发布一次 ready。
template <typename ActivationType>
__aicore__ inline void PrepareMoeExpertTokenCountTable(const MoeStageCommonConfig &common,
                                                       const BlockWorkspaceContext &countWorkspace,
                                                       const Params &params,
                                                       TokenDispatchScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;
    GlobalTensor<int32_t> compactCountGlobal;
    compactCountGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.peermemInfo.compactCountRecvPtr));
    DataCopyPad(scratch.cumsumInfoTensor, compactCountGlobal,
                {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();

    ComputeCompactExpertCountTables(scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor,
                                    common.moeExpertPerRank, common.worldSize);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    uint64_t countOffset =
        GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    DataCopyPad(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();

    __gm__ int32_t *countTableReady =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WriteGmByPassDCache(countTableReady, static_cast<int32_t>(1));
}

template <typename ActivationType>
__aicore__ inline void ExportCompactExpertTokenCounts(const MoeStageCommonConfig &common,
                                                      const BlockWorkspaceContext &countWorkspace,
                                                      const Params &params,
                                                      TokenDispatchScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || countWorkspace.blockIdx != 0U) {
        return;
    }

    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    GlobalTensor<int32_t> expertRevTokenNums;
    expertRevTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.expertRevTokenNumsPtr));
    DataCopyPad(scratch.expertTokenNumsOutTensor, expertRevTokenNums[countOffset],
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_MTE3, SYNC_EVENT_ID2>();

    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_H
