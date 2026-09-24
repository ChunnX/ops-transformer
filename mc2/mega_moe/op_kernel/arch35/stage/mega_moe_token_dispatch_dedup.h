/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_DISPATCH_DEDUP_H
#define MEGA_MOE_TOKEN_DISPATCH_DEDUP_H

#include "mega_moe_token_dispatch.h"

namespace MegaMoeImpl {

// 去重 dispatch 的 UB/GM 视图集合：一次构造，批循环各段共用。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
struct DedupDispatchViews {
    LocalTensor<int32_t> descBatchTensor;  // 本批 rowDesc 装载区
    LocalTensor<int32_t> fanMetaTensor;    // 扇出 metaInfo 拼装区（按 buffer 槽分块）
    LocalTensor<int32_t> activeListTensor; // 非 SKIP 条目的批内序号压紧表
    GlobalTensor<ActivationType> tokenRevAbsTensor;
    GlobalTensor<QuantScaleType> scaleRevAbsTensor;
    GlobalTensor<int32_t> metaInfoAbsTensor;
    GlobalTensor<int32_t> rowDescGm;
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
};

template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> CreateDedupDispatchViews(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr, uint32_t remoteRankIdx,
    uint32_t dedupUbBaseAddr)
{
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> views;
    views.descBatchTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, dedupUbBaseAddr, DEDUP_DESC_BATCH_UB_BYTES / sizeof(int32_t));
    views.fanMetaTensor = LocalTensor<int32_t>(TPosition::VECCALC, dedupUbBaseAddr + DEDUP_DESC_BATCH_UB_BYTES,
                                               DEDUP_FAN_META_UB_BYTES / sizeof(int32_t));
    views.activeListTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, dedupUbBaseAddr + DEDUP_DESC_BATCH_UB_BYTES + DEDUP_FAN_META_UB_BYTES,
                             DEDUP_ACTIVE_LIST_UB_BYTES / sizeof(int32_t));
    views.remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    views.tokenRevAbsTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.dispatchRevDataPtr));
    views.scaleRevAbsTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.dispatchRevScalePtr));
    views.metaInfoAbsTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.metaInfoPtr));
    views.rowDescGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(context.dedup.rowDescPtr));
    return views;
}

// 组内每个成员写一份数据/scale，并在 UB 拼装各成员的 metaInfo（按成员自己的 topkIndex/权重位）。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline void FanoutDedupDataAndMeta(DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views,
                                              TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                              const LocalTensor<ActivationType> &tokenScaleBuffer,
                                              const LocalTensor<QuantScaleType> &scaleBuffer,
                                              const LocalTensor<int32_t> &weightBitsTensor, bool recordHasWeights,
                                              int32_t topK, uint32_t remoteRankIdx, int32_t selfRow,
                                              int32_t selfTopkIndex, int32_t descBase, int32_t memberCount,
                                              uint32_t fanMetaSlotBase)
{
    const int32_t tokenIndex = selfTopkIndex / topK;
    for (int32_t m = 0; m < memberCount; ++m) {
        int32_t row = selfRow;
        int32_t topkIndex = selfTopkIndex;
        if (memberCount > 1) {
            uint32_t tripleBase =
                static_cast<uint32_t>(descBase) + DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(m) * 3U;
            row = views.descBatchTensor.GetValue(tripleBase);
            topkIndex = views.descBatchTensor.GetValue(tripleBase + 1U);
        }
        DataCopyPad(views.tokenRevAbsTensor[static_cast<uint64_t>(row) * scratch.revTokenElemCnt], tokenScaleBuffer,
                    {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(views.scaleRevAbsTensor[static_cast<uint64_t>(row) * scratch.revScaleElemCnt], scaleBuffer,
                    {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
        uint32_t metaBase = fanMetaSlotBase + static_cast<uint32_t>(m) * static_cast<uint32_t>(INT32_PER_256B);
        views.fanMetaTensor.SetValue(metaBase + RANK_ID, static_cast<int32_t>(remoteRankIdx));
        views.fanMetaTensor.SetValue(metaBase + TOKEN_ID, tokenIndex);
        views.fanMetaTensor.SetValue(metaBase + TOPK_INDEX, topkIndex % topK);
        views.fanMetaTensor.SetValue(metaBase + WEIGHT_INDEX,
                                     recordHasWeights ?
                                         weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % topK)) :
                                         0);
    }
}

// tkw=0 的 combine 去重：FIRST 行扇出时把整 token 的 topK 权重段落到 LAST 成员行的 revWeights 槽
//（组成员按行号升序、末位即 LAST；merge 在 LAST 行驱动，一次连续读回按成员 topkIndex 乘权）。
// 条件按 GROUP 标志而非 memberCount>1：clamp 裁剪后 survivorCount==1 的单成员组仍走 merge 乘权路径，
// 漏写即读到未初始化权重。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline void StoreDedupGroupWeights(DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views,
                                              const Params &params, const LocalTensor<int32_t> &weightBitsTensor,
                                              int32_t topK, int32_t descBase, int32_t memberCount, int32_t flags)
{
    if ((flags & DEDUP_ROW_FLAG_GROUP) == 0) {
        return;
    }
    const uint32_t weightAlignBytes = Ops::Base::CeilAlign(
        static_cast<uint32_t>(topK) * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    const int32_t lastRow = views.descBatchTensor.GetValue(
        static_cast<uint32_t>(descBase) + DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(memberCount - 1) * 3U);
    GlobalTensor<int32_t> revWeightsGm;
    revWeightsGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.dispatchRevWeightsPtr +
                                           static_cast<uint64_t>(static_cast<uint32_t>(lastRow)) * weightAlignBytes));
    DataCopyPad(revWeightsGm, weightBitsTensor,
                {1U, static_cast<uint32_t>(topK) * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U});
}

/*
 * 去重版单条目落地：把 ring 槽内一条已拉取记录写到其所属行；该行是重复组首行（FIRST）时，
 * 同一份 UB 数据额外扇出写到组内全部成员行。事件契约与基线 Store 对齐：消费 Fetch 的
 * MTE2_MTE3（以及 prefetch 的 MTE2_S / 非 prefetch 的 S_MTE3），结束时释放 MTE3_MTE2 与 MTE3_S。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDedupEntry(const TokenDispatchConfig &context, const Params &params,
                                       TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                       DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views,
                                       int32_t bufferIdx, uint32_t remoteRankIdx, int32_t selfRow,
                                       int32_t selfTopkIndex, int32_t descBase)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    const int32_t topK = static_cast<int32_t>(params.tilingData->topK);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    // 与 Fetch 侧事件分叉同条件：记录带权重段（prefetch 或 tkw=0 combine 去重）走 MTE2_S。
    const bool recordHasWeights = TopkWeightsPrefetch || (params.tilingData->topkWeightsPrefetch == 1 || IsCombineDedupOn(params.tilingData->dedupMode));
    LocalTensor<int32_t> weightBitsTensor;
    if (recordHasWeights) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
    } else {
        // Fetch 在无权重段布局下为 ring meta 置了 S_MTE3；本路径不消费 ring meta，在此配平。
        WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    const int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(descBase));
    const int32_t memberCount = (flags & DEDUP_ROW_FLAG_FIRST) != 0 ? (flags & DEDUP_ROW_MEMBER_COUNT_MASK) : 1;

    const uint32_t fanMetaSlotBase =
        static_cast<uint32_t>(bufferIdx) * static_cast<uint32_t>(topK) * static_cast<uint32_t>(INT32_PER_256B);
    FanoutDedupDataAndMeta(views, scratch, tokenScaleBuffer, scaleBuffer, weightBitsTensor, recordHasWeights, topK,
                           remoteRankIdx, selfRow, selfTopkIndex, descBase, memberCount, fanMetaSlotBase);
    if constexpr (!TopkWeightsPrefetch) { // tkw=1 合并纯加不需要 revWeights 区，编译期折叠
        if (recordHasWeights) {
            StoreDedupGroupWeights(views, params, weightBitsTensor, topK, descBase, memberCount, flags);
        }
    }
    // metaInfo 的 UB 拼装（标量）先于 MTE3 写回。
    SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    for (int32_t m = 0; m < memberCount; ++m) {
        int32_t row = selfRow;
        if (memberCount > 1) {
            row = views.descBatchTensor.GetValue(static_cast<uint32_t>(descBase) + DEDUP_ROW_DESC_HEADER_INT32 +
                                                 static_cast<uint32_t>(m) * 3U);
        }
        DataCopy(views.metaInfoAbsTensor[static_cast<uint64_t>(row) * INT32_PER_256B],
                 views.fanMetaTensor[fanMetaSlotBase + static_cast<uint32_t>(m) * static_cast<uint32_t>(INT32_PER_256B)],
                 INT32_PER_256B);
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// 本批 desc 覆盖的 prepass 工作项 = (srcRank, [首token,末token] 的 chunk 区间)；只等自己需要的项。
template <typename TopkIndexType, typename ActivationType>
__aicore__ inline void WaitDedupPrepassForBatch(const TokenDispatchConfig &context, const Params &params,
                                                TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                uint32_t localExpertId, uint32_t remoteRankIdx, int32_t batchCount,
                                                int32_t &confirmedChunkEnd)
{
    const int32_t topK = static_cast<int32_t>(params.tilingData->topK);
    const int32_t chunkTokens = static_cast<int32_t>(context.dedup.prepassChunkTokens);
    const int32_t tokenFirst = scratch.validTopkIndexTensor.GetValue(0U) / topK;
    const int32_t tokenLast = scratch.validTopkIndexTensor.GetValue(static_cast<uint32_t>(batchCount - 1)) / topK;
    const int32_t chunkLast = tokenLast / chunkTokens;
    if (chunkLast <= confirmedChunkEnd) {
        return;
    }
    __gm__ int32_t *prepassReadyBase = reinterpret_cast<__gm__ int32_t *>(context.dedup.prepassReadyPtr);
    int32_t chunkFirst = tokenFirst / chunkTokens;
    chunkFirst = chunkFirst > confirmedChunkEnd + 1 ? chunkFirst : confirmedChunkEnd + 1;
    for (int32_t c = chunkFirst; c <= chunkLast; ++c) {
        const uint32_t item = remoteRankIdx * context.dedup.prepassChunkCount + static_cast<uint32_t>(c);
        WaitUntilGmFlagEquals(prepassReadyBase + (1U + item) * INT_CACHELINE, 1);
    }
    confirmedChunkEnd = chunkLast;
}

// 装载本批 route 条目与 rowDesc；装载 desc 前按需等待其覆盖的 prepass 工作项 ready。
// confirmedChunkEnd 跨批递增：已确认 ready 的 chunk 不再重查。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline void LoadDedupRouteAndDescBatch(
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views, const TokenDispatchConfig &context,
    const MoeStageCommonConfig &common, const Params &params,
    TokenDispatchScratch<ActivationType, TopkIndexType> &scratch, uint32_t localExpertId, uint32_t remoteRankIdx,
    int32_t routeOrdinal, int32_t batchRowBegin, int32_t batchCount, int32_t &confirmedChunkEnd)
{
    const uint32_t strideInt32 = context.dedup.rowDescStrideInt32;
    // 上一批的标量读（desc/activeList）先于本批 MTE2 覆盖完成。
    SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID4>();
    uint64_t slotOffset =
        (static_cast<uint64_t>(localExpertId) * common.worldSize + remoteRankIdx) * context.routeIndexAlignSize;
    // route 槽元素类型随 numMaxTokensPerRank*topK 在 int16/int32 间切换（UseInt16TopkIndex），与基线同式读取。
    GlobalTensor<TopkIndexType> remoteRouteIndexGlobal;
    remoteRouteIndexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ TopkIndexType *>(
        params.peermemInfo.maskRecvPtr + slotOffset + static_cast<uint64_t>(routeOrdinal) * sizeof(TopkIndexType)));
    DataCopyExtParams routeCopyParams{1U, static_cast<uint32_t>(batchCount * sizeof(TopkIndexType)), 0U, 0U, 0U};
    DataCopyPadExtParams<TopkIndexType> routeCopyPad{false, 0U, 0U, 0U};
    DataCopyPad(scratch.validTopkIndexTensor, remoteRouteIndexGlobal, routeCopyParams, routeCopyPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID4>();
    WaitDedupPrepassForBatch(context, params, scratch, localExpertId, remoteRankIdx, batchCount, confirmedChunkEnd);
    SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID4>();
    DataCopyPad(views.descBatchTensor, views.rowDescGm[static_cast<uint64_t>(batchRowBegin) * strideInt32],
                {1U, static_cast<uint32_t>(batchCount * strideInt32 * sizeof(int32_t)), 0U, 0U, 0U},
                {false, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID4>();
}

// 压紧非 SKIP 条目（PLAIN 或 FIRST）的批内序号到活跃表，返回活跃条数。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline int32_t CompactDedupActiveEntries(
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views, uint32_t strideInt32, int32_t batchCount)
{
    int32_t activeCount = 0;
    for (int32_t batchEntryIdx = 0; batchEntryIdx < batchCount; ++batchEntryIdx) {
        int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(batchEntryIdx) * strideInt32);
        if ((flags & DEDUP_ROW_FLAG_GROUP) != 0 && (flags & DEDUP_ROW_FLAG_FIRST) == 0) {
            continue;
        }
        views.activeListTensor.SetValue(static_cast<uint32_t>(activeCount), batchEntryIdx);
        ++activeCount;
    }
    return activeCount;
}

// 对活跃条目执行基线同款 fetch/store 软流水：先发满 buffer 深度的 fetch，之后每发一个新 fetch
// 落一个旧 store，尾部落最后一个 store 并收割各 buffer 的完成事件。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void RunDedupFetchStorePipeline(
    const TokenDispatchConfig &context, const Params &params,
    TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views, uint32_t remoteRankIdx,
    int32_t batchRowBegin, uint32_t strideInt32, int32_t activeCount)
{
    const int32_t bufferCount = context.bufferConfig.bufferCount;
    int32_t entry = views.activeListTensor.GetValue(0U);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(
        context, params, scratch, 0, scratch.validTopkIndexTensor.GetValue(static_cast<uint32_t>(entry)),
        static_cast<int32_t>(remoteRankIdx), views.remoteRankGlobalTensor);
    const int32_t firstUseEnd = activeCount < bufferCount ? activeCount : bufferCount;
    for (int32_t a = 1; a < activeCount; ++a) {
        int32_t issueEntry = views.activeListTensor.GetValue(static_cast<uint32_t>(a));
        if (a < bufferCount) {
            FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(
                context, params, scratch, a, scratch.validTopkIndexTensor.GetValue(static_cast<uint32_t>(issueEntry)),
                static_cast<int32_t>(remoteRankIdx), views.remoteRankGlobalTensor);
        } else {
            FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(
                context, params, scratch, a % bufferCount,
                scratch.validTopkIndexTensor.GetValue(static_cast<uint32_t>(issueEntry)),
                static_cast<int32_t>(remoteRankIdx), views.remoteRankGlobalTensor);
        }
        int32_t storeEntry = views.activeListTensor.GetValue(static_cast<uint32_t>(a - 1));
        StoreDedupEntry<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, views, (a - 1) % bufferCount, remoteRankIdx, batchRowBegin + storeEntry,
            scratch.validTopkIndexTensor.GetValue(static_cast<uint32_t>(storeEntry)),
            storeEntry * static_cast<int32_t>(strideInt32));
    }
    int32_t lastEntry = views.activeListTensor.GetValue(static_cast<uint32_t>(activeCount - 1));
    StoreDedupEntry<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, scratch, views, (activeCount - 1) % bufferCount, remoteRankIdx, batchRowBegin + lastEntry,
        scratch.validTopkIndexTensor.GetValue(static_cast<uint32_t>(lastEntry)),
        lastEntry * static_cast<int32_t>(strideInt32));
    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// 发布本批已写行的 tile-ready：连续已写行聚合为 run 一次发布（AtomicAdd 语义允许分段累加），
// GMM1 等值判据不变——每个物理行恒被其写入者发布一次。
template <uint32_t PipelineTileM, typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline void PublishDedupTileReadyRuns(
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views,
    const MoeSyncWorkspaceLayout &syncLayout, const Params &params, uint32_t localExpertId,
    int32_t expertGlobalRowBegin, int32_t batchRowBegin, uint32_t strideInt32, int32_t batchCount)
{
    int32_t runBegin = -1;
    for (int32_t batchEntryIdx = 0; batchEntryIdx <= batchCount; ++batchEntryIdx) {
        bool written = false;
        if (batchEntryIdx < batchCount) {
            int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(batchEntryIdx) * strideInt32);
            written = ((flags & DEDUP_ROW_FLAG_GROUP) == 0) || ((flags & DEDUP_ROW_FLAG_FIRST) != 0);
        }
        if (written) {
            runBegin = runBegin < 0 ? batchEntryIdx : runBegin;
            continue;
        }
        if (runBegin >= 0) {
            int32_t segRowBegin = batchRowBegin - expertGlobalRowBegin + runBegin;
            int32_t segRowEnd = batchRowBegin - expertGlobalRowBegin + batchEntryIdx;
            PublishGmm1TileReady(syncLayout, params, localExpertId, static_cast<int32_t>(PipelineTileM), segRowBegin,
                                 segRowEnd);
            runBegin = -1;
        }
    }
}

// 扇出成员行（组内 j>=1，必在其它专家）逐行发布 tile-ready；flag 槽偏移由 prepass 预计算。
// 首次发布前插一次 MTE3_S：确保本批全部扇出 GM 写完成后 GMM1 才可能读到 ready。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType>
__aicore__ inline void PublishDedupFanoutMembers(
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> &views, const Params &params,
    uint32_t strideInt32, int32_t batchCount)
{
    __gm__ int32_t *flagBaseAll = reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagDispatchToGmm1Ptr);
    bool memberSyncDone = false;
    for (int32_t batchEntryIdx = 0; batchEntryIdx < batchCount; ++batchEntryIdx) {
        int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(batchEntryIdx) * strideInt32);
        if ((flags & DEDUP_ROW_FLAG_FIRST) == 0) {
            continue;
        }
        int32_t memberCount = flags & DEDUP_ROW_MEMBER_COUNT_MASK;
        for (int32_t memberIdx = 1; memberIdx < memberCount; ++memberIdx) {
            int32_t flagSlotOffset = views.descBatchTensor.GetValue(
                static_cast<uint32_t>(batchEntryIdx) * strideInt32 + DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(memberIdx) * 3U +
                2U);
            if (!memberSyncDone) {
                SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
                memberSyncDone = true;
            }
            AtomicAdd(flagBaseAll + flagSlotOffset, 1);
        }
    }
}

/*
 * DispatchRankTokens 的去重版本：按批装载 route 条目与 rowDesc；SKIP 行（组内非首行）不产生任何
 * 拉取与写入（由其组首行扇出写入）；PLAIN/FIRST 行沿用基线的 fetch/store 软流水（对活跃条目
 * 压紧后流水）。批尾统一发布 tile-ready。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchRankTokensDedup(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                               const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                               GM_ADDR *winRankAddr,
                                               TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                               uint32_t localExpertId, int32_t expertGlobalRowBegin,
                                               uint32_t remoteRankIdx, int32_t rankSegmentRowBegin,
                                               int32_t segmentMatchOrdinalBegin, int32_t segmentMatchOrdinalEnd)
{
    DedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType> views =
        CreateDedupDispatchViews<TopkIndexType, ActivationType, QuantScaleType>(context, params, winRankAddr,
                                                                                remoteRankIdx, scratch.dedupUbBaseAddr);
    const uint32_t strideInt32 = context.dedup.rowDescStrideInt32;
    const int32_t descRowsCap = static_cast<int32_t>(DEDUP_DESC_BATCH_UB_BYTES / (strideInt32 * sizeof(int32_t)));
    const int32_t batchCap =
        context.bufferConfig.routeItemsPerBatch < descRowsCap ? context.bufferConfig.routeItemsPerBatch : descRowsCap;
    // prepass 按需等待：本段（单一 srcRank）批序 token 单调递增，已确认 ready 的 chunk 不再重查。
    int32_t confirmedChunkEnd = -1;
    int32_t processed = 0;
    const int32_t segmentTotal = segmentMatchOrdinalEnd - segmentMatchOrdinalBegin;
    while (processed < segmentTotal) {
        const int32_t remaining = segmentTotal - processed;
        const int32_t batchCount = remaining < batchCap ? remaining : batchCap;
        const int32_t routeOrdinal = segmentMatchOrdinalBegin + processed;
        const int32_t batchRowBegin = rankSegmentRowBegin + expertGlobalRowBegin + routeOrdinal;
        LoadDedupRouteAndDescBatch(views, context, common, params, scratch, localExpertId, remoteRankIdx, routeOrdinal,
                                   batchRowBegin, batchCount, confirmedChunkEnd);
        const int32_t activeCount = CompactDedupActiveEntries(views, strideInt32, batchCount);
        if (activeCount > 0) {
            RunDedupFetchStorePipeline<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                context, params, scratch, views, remoteRankIdx, batchRowBegin, strideInt32, activeCount);
        }
        PublishDedupTileReadyRuns<PipelineTileM>(views, syncLayout, params, localExpertId, expertGlobalRowBegin,
                                                 batchRowBegin, strideInt32, batchCount);
        // 去重收益计数：本批实拉/跳过行数（跳过数≈被省掉的跨片拉取数，扇出写在 Store 内计）。
        PublishDedupFanoutMembers(views, params, strideInt32, batchCount);
        processed += batchCount;
    }
}
/*
 * DispatchOwnedExpertRows 的去重版本：range 拆段逻辑与基线一致，逐段调用去重版搬运；
 * tile-ready 已在去重批内完成发布，此处不再做段级 Publish。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchOwnedExpertRowsDedup(const TokenDispatchConfig &context,
                                                    const MoeStageCommonConfig &common,
                                                    const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                                    GM_ADDR *winRankAddr,
                                                    TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                    uint32_t localExpertId, const ExpertDispatchCoreRange &coreRange)
{
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
            DispatchRankTokensDedup<TopkIndexType, ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
                context, common, syncLayout, params, winRankAddr, scratch, localExpertId,
                static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                static_cast<int32_t>(rankGlobalRowBegin - coreRange.expertGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin));
        }
        ++sourceRankIdx;
    }
}
__aicore__ inline void WaitDedupPrepassPublishGate(const TokenDispatchConfig &context, const Params &params)
{
    if (context.dedup.dedupMode == 0 || params.tilingData->hiddenDim > DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM) {
        return;
    }
    // AIV0 参与建表后，其 UB 建表区会被 AIC 的首批 GMM1 tile 覆写（覆写发生在 dispatch 首批
    // 发布之后；GMM1 problem 抢全局 tile，任一 AIV1 的发布都可能触发任一 block 的 AIC 写其
    // AIV0 UB）。发布任何 tile 前只等 done 计数高 16 位=全体 AIV0（本 AIV1 自己已建完，AIV1
    // 之间不互等，保住流水交叠）；combine 单开同样经由本入口。首个 wave 之后瞬时通过。
    __gm__ int32_t *dedupDoneSlot = reinterpret_cast<__gm__ int32_t *>(context.dedup.prepassReadyPtr);
    const int32_t expectAiv0Done = static_cast<int32_t>(GetBlockNum());
    while ((ReadGmBypassDCache(dedupDoneSlot) >> 16) != expectAiv0Done) {
        int64_t gateBackoffStart = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - gateBackoffStart < GM_FLAG_POLL_BACKOFF_CYCLES) {
        }
    }
}

// 按 dispatch 去重开关把一个专家的本核区间路由到去重/基线 dispatch 实现。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchOwnedExpertRowsRouted(const TokenDispatchConfig &context,
                                                     const MoeStageCommonConfig &common,
                                                     const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                                     GM_ADDR *winRankAddr,
                                                     TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                     uint32_t expertIdx, const ExpertDispatchCoreRange &expertRange)
{
    if (IsDispatchDedupOn(context.dedup.dedupMode)) {
        DispatchOwnedExpertRowsDedup<TopkIndexType, ActivationType, QuantScaleType, PipelineTileM,
                                     TopkWeightsPrefetch>(context, common, syncLayout, params, winRankAddr, scratch,
                                                          expertIdx, expertRange);
    } else {
        DispatchOwnedExpertRows<TopkIndexType, ActivationType, QuantScaleType, PipelineTileM, TopkWeightsPrefetch>(
            context, common, syncLayout, params, winRankAddr, scratch, expertIdx, expertRange);
    }
}

template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchTokenRange(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                          const BlockJobContext &blockJob, const MoeSyncWorkspaceLayout &syncLayout,
                                          const Params &params, GM_ADDR *winRankAddr,
                                          TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                          const ExpertTokenRange &range)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || blockJob.totalJobs == 0U) {
        return;
    }
    WaitDedupPrepassPublishGate(context, params);
    uint32_t coreGlobalRowBegin = 0U;
    uint32_t coreGlobalRowEnd = 0U;
    if (!CalcDispatchCoreRowRange(context, blockJob, range, coreGlobalRowBegin, coreGlobalRowEnd)) {
        return;
    }

    // end 位于专家内部时需要包含该专家；恰好位于专家边界时 end.expertIdx 已指向下一专家。
    uint32_t lastExpertExclusive = range.end.expertIdx + (range.end.tokenIndexInExpert == 0U ? 0U : 1U);
    if (lastExpertExclusive > common.moeExpertPerRank) {
        lastExpertExclusive = common.moeExpertPerRank;
    }
    for (uint32_t expertIdx = range.begin.expertIdx; expertIdx < lastExpertExclusive; ++expertIdx) {
        uint32_t expertGlobalRowBegin =
            expertIdx == 0U ?
                0U :
                static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue(expertIdx * common.worldSize - 1U));
        uint32_t expertGlobalRowEnd =
            static_cast<uint32_t>(scratch.cumsumInfoTensor.GetValue((expertIdx + 1U) * common.worldSize - 1U));
        if (expertGlobalRowEnd <= coreGlobalRowBegin) {
            continue;
        }
        if (expertGlobalRowBegin >= coreGlobalRowEnd) {
            break;
        }
        uint32_t overlapGlobalRowBegin =
            coreGlobalRowBegin > expertGlobalRowBegin ? coreGlobalRowBegin : expertGlobalRowBegin;
        uint32_t overlapGlobalRowEnd = coreGlobalRowEnd < expertGlobalRowEnd ? coreGlobalRowEnd : expertGlobalRowEnd;
        if (overlapGlobalRowBegin < overlapGlobalRowEnd) {
            // 将本核的全局连续区间投影为当前专家内的局部 row 区间，按去重开关走对应 dispatch 路径。
            ExpertDispatchCoreRange expertRange{expertGlobalRowBegin, overlapGlobalRowBegin - expertGlobalRowBegin,
                                                overlapGlobalRowEnd - expertGlobalRowBegin};
            DispatchOwnedExpertRowsRouted<TopkIndexType, ActivationType, QuantScaleType, PipelineTileM,
                                          TopkWeightsPrefetch>(context, common, syncLayout, params, winRankAddr,
                                                               scratch, expertIdx, expertRange);
        }
    }
}
/*
 * AIV0 参与去重 prepass 建表的本地准备：AIV0 的 dispatch UB 布局从未初始化（DispatchBuffInit 仅
 * AIV1 执行），在本子核 UB 自建 cumsum / expertTokenNumsOut 视图与 dedupUbBaseAddr——两子核 UB
 * 物理独立，且 P0 探针实测本窗口 AIV0 UB [0,200KB) 无人写（AIC 首批 GMM1 tile 的覆写发生在
 * dispatch 首批发布之后，而发布侧有 done-gate 挡到全体建表完成）。随后本地复算 count 前缀表
 *（发布仍由 AIV1 独担；count 槽只被读，无并发写）。必须在 BuildDedupRowDescTable 之前调用；
 * 去重关闭或非 AIV0 为空操作。
 * 到达性判据：与 AIV1 相同走 WaitForCountTable（count 携带本轮 epoch tag，整表校验+退避重读；
 * 输入准备阶段每个物理 AIV 含 AIV0 都推进自己的同步计数，期望 tag 可用）。不可改等 countTableReady
 * 残留判据：入口无跨卡握手，且轮尾 count 接收区被清零，而 countTableReady 无人清零——
 * 首轮=0 时等待成立，第 2 轮起残留非零使 AIV0 冲线，读到"已清零、本轮未到达"的 count 槽即建出
 * 全 0 前缀表，消费者死等（多轮连续 launch 场景必现）。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void PrepareDedupPrepassAiv0Local(const DedupConfig &dedup, const MoeStageCommonConfig &common,
                                                    const Params &params,
                                                    TokenDispatchScratch<ActivationType, TopkIndexType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (dedup.dedupMode == 0 || GetSubBlockIdx() != 0U ||
        params.tilingData->hiddenDim > DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM) {
        return;
    }
    const uint32_t cumsumInfoTensorSize = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(common.worldSize * common.moeExpertPerRank * sizeof(int32_t)),
                             static_cast<int64_t>(ALIGN_32)));
    scratch.cumsumInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, 0U, cumsumInfoTensorSize / sizeof(int32_t));
    const uint32_t expertTokenNumsOutTensorSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(common.moeExpertPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32)));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorSize,
                                                            expertTokenNumsOutTensorSize / sizeof(int32_t));
    scratch.dedupUbBaseAddr = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(cumsumInfoTensorSize + expertTokenNumsOutTensorSize),
                             static_cast<uint64_t>(ALIGN_512)));
    LoadAndComputeExpertCountTable(common, params, scratch);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_DEDUP_H
