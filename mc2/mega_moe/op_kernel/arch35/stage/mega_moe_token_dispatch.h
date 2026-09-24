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

#include "mega_moe_expert_count.h"
#include "../common/mega_moe_utils.h"
#include "mega_moe_token_quant.h"
#include "mega_moe_token_dedup.h"

namespace MegaMoeImpl {

using namespace AscendC;

// Token Dispatch 的阶段私有配置。任务分工(BlockJobContext)与 count 槽位(BlockWorkspaceContext)
// 与 GMM 阶段共用编排层的同一份实例，不在此重复持有。
struct TokenDispatchConfig {
    uint64_t maxOutputSize;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint64_t routeIndexAlignSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    DedupConfig dedup; // dedupMode==0 时所有去重字段为空，走基线路径
};

template <typename ActivationType, typename TopkIndexType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<TopkIndexType> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
    // 去重暂存区动态基址：紧跟 dispatch 固定 UB 区尾（host 已按 DEDUP_DISPATCH_UB_RESERVE_BYTES 预留）。
    uint32_t dedupUbBaseAddr = 0U;
};

// 当前物理 AIV1 在单个专家内负责的左闭右开 row 区间。
struct ExpertDispatchCoreRange {
    uint32_t expertGlobalRowBegin;
    uint32_t localExpertRowBegin;
    uint32_t localExpertRowEnd;
};

// 由 tiling/peermem/quant 配置装配 Token Dispatch 阶段配置（普通与 wave 编排模板共用）。
template <typename TopkIndexType>
__aicore__ inline TokenDispatchConfig CreateTokenDispatchConfig(const Params &params,
                                                                const QuantProcessConfig &quantProcessConfig)
{
    uint64_t quantWinOffset =
        static_cast<uint64_t>(params.peermemInfo.quantTokenScalePtr - params.peermemInfo.rankSyncInWorldPtr);
    return {.maxOutputSize = params.tilingData->maxOutputSize,
            .bufferConfig = params.tilingData->dispatchBufferConfig,
            .routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData)),
            .quantWinOffset = quantWinOffset,
            .quantTokenAlignBytes = quantProcessConfig.quantTokenAlignBytes,
            .quantScaleAlignBytes = quantProcessConfig.quantScaleAlignBytes,
            .quantTokenScaleAlignBytes = quantProcessConfig.quantTokenScaleAlignBytes,
            .dedup = CreateDedupConfig(params)};
}

template <typename ActivationType, typename TopkIndexType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(
    const TokenDispatchConfig &context, const TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
    int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType, typename TopkIndexType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig &context, const Params &params,
                                                     TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                     int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
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
    // 事件分叉与 Store 侧的消费条件同式：记录带权重段（prefetch 或 tkw=0 combine 去重）时
    // Store 要标量读权重，走 MTE2_S；否则直接放行 S_MTE3。
    if (TopkWeightsPrefetch || MegaMoeRecordHasTopkWeights(params.tilingData)) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// copyIdx 同时是段内目标行号：三个调用点的 ring 排空顺序保证写出行与匹配序恒一一对应。
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig &context, const Params &params,
                                                     TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                     int32_t bufferIdx,
                                                     GlobalTensor<ActivationType> &tokenRevGlobalTensor,
                                                     GlobalTensor<QuantScaleType> &scaleRevGlobalTensor,
                                                     GlobalTensor<int32_t> &metaInfoGlobalTensor,
                                                     GlobalTensor<int32_t> &revWeightsGlobalTensor,
                                                     int32_t copyStartIdx, int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if (TopkWeightsPrefetch || MegaMoeRecordHasTopkWeights(params.tilingData)) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx));
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(params.tilingData->topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
        // dc 单开（combine 去重 + dispatch 去重关）时本路径是唯一落地入口：每行把整 token
        // 权重段落到自己的 revWeights 槽，供 merge 在 LAST 行一次连续读回乘权。
        if constexpr (!TopkWeightsPrefetch) {
            if ((params.tilingData->dedupMode & DEDUP_MODE_COMBINE) != 0) {
                const uint32_t topK = params.tilingData->topK;
                const uint32_t weightAlignInt32 =
                    Ops::Base::CeilAlign(topK * static_cast<uint32_t>(sizeof(float)),
                                         static_cast<uint32_t>(ALIGN_32)) /
                    static_cast<uint32_t>(sizeof(int32_t));
                DataCopyPad(revWeightsGlobalTensor[static_cast<uint64_t>(copyIdx) * weightAlignInt32],
                            weightBitsTensor,
                            {1U, topK * static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U});
            }
        }
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

// 基线 dispatch 落地目标的 GM 视图集合（按段起始行 rowDstOffset 偏置）。
template <typename ActivationType, typename QuantScaleType>
struct DispatchCopyTargets {
    GlobalTensor<ActivationType> remoteRank;
    GlobalTensor<ActivationType> tokenRev;
    GlobalTensor<QuantScaleType> scaleRev;
    GlobalTensor<int32_t> metaInfo;
    GlobalTensor<int32_t> revWeights;
};

template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline DispatchCopyTargets<ActivationType, QuantScaleType> CreateDispatchCopyTargets(
    const TokenDispatchConfig &context, const Params &params, GM_ADDR *winRankAddr,
    const TokenDispatchScratch<ActivationType, TopkIndexType> &scratch, int32_t rowDstOffset, int32_t remoteRankIdx)
{
    DispatchCopyTargets<ActivationType, QuantScaleType> targets;
    targets.tokenRev.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params.workspaceInfo.dispatchRevDataPtr + rowDstOffset * scratch.revTokenElemCnt));
    targets.scaleRev.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleType *>(
        params.workspaceInfo.dispatchRevScalePtr + rowDstOffset * scratch.revScaleElemCnt));
    targets.remoteRank.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(winRankAddr[remoteRankIdx] + context.quantWinOffset));
    targets.metaInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params.workspaceInfo.metaInfoPtr + rowDstOffset * INT32_PER_256B * sizeof(int32_t)));
    // tkw=0 的 combine 去重且 dispatch 去重未开（dc 单开）：本基线路径是唯一落地入口，每行把
    // 整 token 权重段写到自己的 revWeights 槽（merge 在 LAST 行上读，LAST 行也经本路径写过）。
    // full 模式走 DispatchOwnedExpertRowsDedup/StoreDedupEntry，由那边写 LAST 槽，与此互斥。
    if constexpr (!TopkWeightsPrefetch) {
        if ((params.tilingData->dedupMode & DEDUP_MODE_COMBINE) != 0) {
            uint32_t weightAlignInt32 =
                Ops::Base::CeilAlign(static_cast<uint32_t>(params.tilingData->topK) *
                                         static_cast<uint32_t>(sizeof(float)),
                                     static_cast<uint32_t>(ALIGN_32)) /
                static_cast<uint32_t>(sizeof(int32_t));
            targets.revWeights.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                params.workspaceInfo.dispatchRevWeightsPtr +
                static_cast<uint64_t>(rowDstOffset) * weightAlignInt32 * sizeof(int32_t)));
        }
    }
    return targets;
}

template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(const TokenDispatchConfig &context, const Params &params,
                                                    GM_ADDR *winRankAddr,
                                                    TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                    int32_t rowDstOffset, int32_t remoteRankIdx, int32_t copyStartIdx,
                                                    int32_t copyNum)
{
    int32_t bufferCount = context.bufferConfig.bufferCount;
    DispatchCopyTargets<ActivationType, QuantScaleType> targets =
        CreateDispatchCopyTargets<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, rowDstOffset, remoteRankIdx);
    GlobalTensor<ActivationType> &remoteRankGlobalTensor = targets.remoteRank;
    GlobalTensor<ActivationType> &tokenRevGlobalTensor = targets.tokenRev;
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor = targets.scaleRev;
    GlobalTensor<int32_t> &metaInfoGlobalTensor = targets.metaInfo;
    GlobalTensor<int32_t> &revWeightsGlobalTensor = targets.revWeights;

    int32_t firstTopkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx));
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, 0, firstTopkIndex,
                                                              remoteRankIdx, remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx));
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, params, scratch, issueIdx, topkIndex,
                                                                  remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            revWeightsGlobalTensor, copyStartIdx, copyIdx);
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = static_cast<int32_t>(scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx));
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, params, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, scratch, copyBufferIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            revWeightsGlobalTensor, copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, params, scratch, (copyNum - 1) % bufferCount, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, revWeightsGlobalTensor, copyStartIdx, copyNum - 1);

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

/*
 * 按单个 source rank 的 compact route-index ordinal 区间搬运 token 和 metadata。
 * compact index 已按 [expert][source rank] 分槽保存，因此这里只需按 ordinal 分批顺序读取，无需再次筛选 mask。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline int32_t DispatchRankTokens(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                             const Params &params, GM_ADDR *winRankAddr,
                                             TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                             uint32_t localExpertId, int32_t expertGlobalRowBegin,
                                             uint32_t remoteRankIdx, int32_t rankSegmentRowBegin,
                                             int32_t segmentMatchOrdinalBegin, int32_t segmentMatchOrdinalEnd)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t processedRouteCount = 0;
    int32_t dispatchedRowCount = 0;
    GlobalTensor<TopkIndexType> remoteRouteIndexGlobal;
    while (processedRouteCount < segmentMatchOrdinalEnd - segmentMatchOrdinalBegin) {
        int32_t remainingRouteCount = segmentMatchOrdinalEnd - segmentMatchOrdinalBegin - processedRouteCount;
        int32_t batchDispatchRowCount = remainingRouteCount < bufferConfig.routeItemsPerBatch ?
                                            remainingRouteCount :
                                            bufferConfig.routeItemsPerBatch;
        int32_t routeOrdinal = segmentMatchOrdinalBegin + processedRouteCount;
        // 直接定位当前专家、source rank 的 route ordinal，不再扫描完整 topK mask。
        uint64_t slotOffset =
            (static_cast<uint64_t>(localExpertId) * common.worldSize + remoteRankIdx) * context.routeIndexAlignSize;
        remoteRouteIndexGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ TopkIndexType *>(
            params.peermemInfo.maskRecvPtr + slotOffset + static_cast<uint64_t>(routeOrdinal) * sizeof(TopkIndexType)));
        DataCopyExtParams routeCopyParams{1U, static_cast<uint32_t>(batchDispatchRowCount * sizeof(TopkIndexType)), 0U,
                                          0U, 0U};
        DataCopyPadExtParams<TopkIndexType> routeCopyPad{false, 0U, 0U, 0U};
        DataCopyPad(scratch.validTopkIndexTensor, remoteRouteIndexGlobal, routeCopyParams, routeCopyPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID4>();
        // route ordinal 同时决定专家输出行位置，保证分批搬运后仍保持专家内连续布局。
        int32_t dispatchDstRowBegin = rankSegmentRowBegin + expertGlobalRowBegin + routeOrdinal;
        CopyTokensAndMetaForDispatch<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, params, winRankAddr, scratch, dispatchDstRowBegin, remoteRankIdx, 0, batchDispatchRowCount);
        processedRouteCount += batchDispatchRowCount;
        dispatchedRowCount += batchDispatchRowCount;
    }

    return dispatchedRowCount;
}

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
    const bool recordHasWeights = TopkWeightsPrefetch || MegaMoeRecordHasTopkWeights(params.tilingData);
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
    for (int32_t i = 0; i < batchCount; ++i) {
        int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(i) * strideInt32);
        if ((flags & DEDUP_ROW_FLAG_GROUP) != 0 && (flags & DEDUP_ROW_FLAG_FIRST) == 0) {
            continue;
        }
        views.activeListTensor.SetValue(static_cast<uint32_t>(activeCount), i);
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
    for (int32_t i = 0; i <= batchCount; ++i) {
        bool written = false;
        if (i < batchCount) {
            int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(i) * strideInt32);
            written = ((flags & DEDUP_ROW_FLAG_GROUP) == 0) || ((flags & DEDUP_ROW_FLAG_FIRST) != 0);
        }
        if (written) {
            runBegin = runBegin < 0 ? i : runBegin;
            continue;
        }
        if (runBegin >= 0) {
            int32_t segRowBegin = batchRowBegin - expertGlobalRowBegin + runBegin;
            int32_t segRowEnd = batchRowBegin - expertGlobalRowBegin + i;
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
    for (int32_t i = 0; i < batchCount; ++i) {
        int32_t flags = views.descBatchTensor.GetValue(static_cast<uint32_t>(i) * strideInt32);
        if ((flags & DEDUP_ROW_FLAG_FIRST) == 0) {
            continue;
        }
        int32_t memberCount = flags & DEDUP_ROW_MEMBER_COUNT_MASK;
        for (int32_t j = 1; j < memberCount; ++j) {
            int32_t flagSlotOffset = views.descBatchTensor.GetValue(
                static_cast<uint32_t>(i) * strideInt32 + DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(j) * 3U +
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
 * 在当前专家的 source-rank 累计行数中二分查找 globalRowIdx 所属的首个 rank。
 * 返回值作为 DispatchOwnedExpertRows 的扫描起点，后续只访问与本核范围相交的连续 rank 段。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline uint32_t FindDispatchSourceRank(const MoeStageCommonConfig &common,
                                                  const TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
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

/*
 * 搬运当前物理 AIV1 在一个专家内负责的连续 row。
 * 该范围先按 source-rank prefix 拆段，再从 compact index 槽搬运 token/metadata；全部 rank 段完成后，
 * 按实际覆盖行数累加 GMM1 tile ready。AtomicAdd 允许同一 tile 被多个 range 分段发布。
 */
template <typename TopkIndexType, typename ActivationType, typename QuantScaleType, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchOwnedExpertRows(const TokenDispatchConfig &context, const MoeStageCommonConfig &common,
                                               const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                               GM_ADDR *winRankAddr,
                                               TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                               uint32_t localExpertId, const ExpertDispatchCoreRange &coreRange)
{
    constexpr int32_t gmm1TileRowCount = static_cast<int32_t>(PipelineTileM);
    uint32_t coreGlobalRowBegin = coreRange.expertGlobalRowBegin + coreRange.localExpertRowBegin;
    uint32_t coreGlobalRowEnd = coreRange.expertGlobalRowBegin + coreRange.localExpertRowEnd;
    // prefix 单调递增，二分找到与本核首行相交的第一个 source rank，避免从 rank 0 线性扫描。
    uint32_t sourceRankIdx = FindDispatchSourceRank(common, scratch, localExpertId, coreGlobalRowBegin);
    int32_t dispatchedRowCount = 0;

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
            int32_t rankDispatchedRowCount =
                DispatchRankTokens<TopkIndexType, ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                    context, common, params, winRankAddr, scratch, localExpertId,
                    static_cast<int32_t>(coreRange.expertGlobalRowBegin), sourceRankIdx,
                    static_cast<int32_t>(rankGlobalRowBegin - coreRange.expertGlobalRowBegin),
                    static_cast<int32_t>(overlapGlobalRowBegin - rankGlobalRowBegin),
                    static_cast<int32_t>(overlapGlobalRowEnd - rankGlobalRowBegin));
            dispatchedRowCount += rankDispatchedRowCount;
        }
        ++sourceRankIdx;
    }
    // 所有 source-rank 子段完成后统一发布本核贡献；GMM1 按 tile 累计到完整行数后即可启动。
    PublishGmm1TileReady(syncLayout, params, localExpertId, gmm1TileRowCount,
                         static_cast<int32_t>(coreRange.localExpertRowBegin),
                         static_cast<int32_t>(coreRange.localExpertRowBegin) + dispatchedRowCount);
}

/*
 * Dispatch 唯一执行入口：搬运调用方给定的左闭右开专家 token 范围，并发布 GMM1 tile ready。
 * 调用方负责提前规划范围、推进 WAVE/专家位置以及在必要时恢复 prefix；本函数只做范围裁剪、AIV1 分工、
 * 专家/source-rank 拆段和数据发送。A8W8 可传整 WAVE，W4 也可传单个专家 slice，执行路径完全一致。
 */
// 计算本核负责的全局行区间：先按 maxOutputSize（Dispatch workspace 硬边界）统一裁剪，再对整个
// 输入范围一次连续均分；按全局起点轮转首 owner，避免余数长期集中在低编号 AIV1。
// 返回 false = 本核无行可做。
__aicore__ inline bool CalcDispatchCoreRowRange(const TokenDispatchConfig &context, const BlockJobContext &blockJob,
                                                const ExpertTokenRange &range, uint32_t &coreGlobalRowBegin,
                                                uint32_t &coreGlobalRowEnd)
{
    uint32_t dispatchGlobalRowBegin = static_cast<uint32_t>(
        range.begin.globalTokenIndex < context.maxOutputSize ? range.begin.globalTokenIndex : context.maxOutputSize);
    uint32_t dispatchGlobalRowEnd = static_cast<uint32_t>(
        range.end.globalTokenIndex < context.maxOutputSize ? range.end.globalTokenIndex : context.maxOutputSize);
    if (dispatchGlobalRowEnd <= dispatchGlobalRowBegin) {
        return false;
    }
    uint32_t dispatchRowCount = dispatchGlobalRowEnd - dispatchGlobalRowBegin;
    WorkRange ownedRange =
        GetRotatedBalancedWorkRange(dispatchRowCount, blockJob.jobIndex, blockJob.totalJobs, dispatchGlobalRowBegin);
    if (ownedRange.count == 0U) {
        return false;
    }
    coreGlobalRowBegin = dispatchGlobalRowBegin + ownedRange.start;
    coreGlobalRowEnd = coreGlobalRowBegin + ownedRange.count;
    return true;
}

// dispatch 发布 gate：AIV0 参与建表档，发布任何 tile 前等全体 AIV0 建表完成。
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
    if ((context.dedup.dedupMode & DEDUP_MODE_DISPATCH) != 0) {
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
 * count 表的装载与本地计算段（从 PrepareMoeExpertTokenCountTable 提取，语义逐字不变）：
 * 装载原始 count → 逐槽 epoch 到达校验 → 前缀和。不含 GM 发布与 ready 标志。
 * AIV1 经 PrepareMoeExpertTokenCountTable 调用；AIV0 参与去重 prepass 建表时直接调用本函数，
 * 在自己的 UB 里得到一份本地前缀表（发布仍由 AIV1 独担；count 槽只被读，无并发写）。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void LoadAndComputeExpertCountTable(const MoeStageCommonConfig &common, const Params &params,
                                                      TokenDispatchScratch<ActivationType, TopkIndexType> &scratch)
{
    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;

    /*
     * count 的本轮 epoch 代替入口跨卡握手：发送前已等待全部 MoE 量化和对应 index 写回。
     * 本核已在输入准备阶段更新同步计数，直接读取本核计数生成预期标签。
     * 整表搬入 UB 后批量检查本轮标签，包括零 count；失败退避重试，成功快照直接用于 cumsum。
     */
    const int32_t syncCount = GetSyncCount(params.peermemInfo.rankSyncInWorldPtr, GetBlockIdx());
    const uint32_t expectedSyncRoundTag = GetSyncRoundTag(syncCount);
    GlobalTensor<int32_t> expertCountGlobal;
    expertCountGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.peermemInfo.expertCountRecvPtr));
    // 共享计算之后已有 MTE3_MTE2，保护此处 MTE2 对同一 UB 的复用。
    // expertTokenNumsOutTensor 尚未保存输出，首项暂存整表检查结果，不增加 UB 分配。
    WaitForCountTable(expertCountGlobal, scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor,
                      rawCountElementCount, expectedSyncRoundTag);
    ComputeExpertCountTables(scratch.cumsumInfoTensor, scratch.expertTokenNumsOutTensor, common.moeExpertPerRank,
                             common.worldSize, params.tilingData->maxOutputSize);
    // 后续只使用 UB 结果及 workspace 副本，不再读取 peermem 原始 count 表。
    CrossCoreSetFlag<ALL_AICORE_SYNC_MODE, PIPE_V>(COUNT_TABLE_READ_DONE_FLAG);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
}

// Wave 进入逐专家流水前一次准备完整 count 表；每个物理 block 只发布一次 ready。
// W4 Wave-ahead 路径同时将 prefix 持久化到 GM，避免后续 Activation 覆盖 UB 后丢失 Dispatch 状态。
template <bool NeedCumsumReload = false, typename ActivationType, typename TopkIndexType>
__aicore__ inline void PrepareMoeExpertTokenCountTable(const MoeStageCommonConfig &common,
                                                       const BlockWorkspaceContext &countWorkspace,
                                                       const Params &params,
                                                       TokenDispatchScratch<ActivationType, TopkIndexType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        // AIV0 不读取 count，但 mode 0 需要全部 AIV 每轮各上报一次；参与去重建表的 AIV0
        // 经 PrepareDedupPrepassAiv0Local -> LoadAndComputeExpertCountTable 已上报，此处不得重复。
        // 条件必须与 PrepareDedupPrepassAiv0Local 的启用门严格互补（含 hiddenDim 档位），
        // 否则计算主导档的 AIV0 无人上报，ExportAndResetExpertCounts 等 read-done 挂死。
        if (params.tilingData->dedupMode == 0 ||
            params.tilingData->hiddenDim > DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM) {
            CrossCoreSetFlag<ALL_AICORE_SYNC_MODE, PIPE_V>(COUNT_TABLE_READ_DONE_FLAG);
        }
        return;
    }
    LoadAndComputeExpertCountTable(common, params, scratch);
    uint32_t rawCountElementCount = common.worldSize * common.moeExpertPerRank;
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    DataCopyPad(scratch.expertRevNumsGlobalTensor[countOffset], scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    if constexpr (NeedCumsumReload) {
        DataCopyPad(scratch.cumsumInfoGlobalTensor, scratch.cumsumInfoTensor,
                    {1U, rawCountElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    }
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID2>();

    __gm__ int32_t *countTableReady =
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.flagSendCntCalToUpdParamsPtr) +
        static_cast<uint64_t>(countWorkspace.blockIdx) * INT_CACHELINE;
    WriteGmBypassDCache(countTableReady, static_cast<int32_t>(1));
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

/*
 * W4 的 Activation/SwiGLU 会覆盖 AIV1 UB 0 开始的 prefix 表。调用方在执行 DispatchTokenRange 前，
 * 从当前物理 block 的 GM 备份恢复“首专家前缀 + 本次范围覆盖的专家前缀”。
 */
template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void ReloadDispatchCumsumRange(const MoeStageCommonConfig &common,
                                                 TokenDispatchScratch<ActivationType, TopkIndexType> &scratch,
                                                 uint32_t firstExpertIdx, uint32_t lastExpertIdx)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U) {
        return;
    }

    if (firstExpertIdx >= common.moeExpertPerRank || firstExpertIdx > lastExpertIdx) {
        return;
    }
    if (lastExpertIdx >= common.moeExpertPerRank) {
        lastExpertIdx = common.moeExpertPerRank - 1U;
    }
    constexpr uint32_t INT32_PER_32B = ALIGN_32 / sizeof(int32_t);
    uint32_t requiredBeginIndex = firstExpertIdx == 0U ? 0U : firstExpertIdx * common.worldSize - 1U;
    uint32_t alignedBeginIndex = requiredBeginIndex / INT32_PER_32B * INT32_PER_32B;
    uint32_t requiredEndIndex = (lastExpertIdx + 1U) * common.worldSize;
    uint32_t reloadElementCount = requiredEndIndex - alignedBeginIndex;
    DataCopyPad(scratch.cumsumInfoTensor[alignedBeginIndex], scratch.cumsumInfoGlobalTensor[alignedBeginIndex],
                {1U, reloadElementCount * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
}

template <typename ActivationType, typename TopkIndexType>
__aicore__ inline void ExportExpertTokenCounts(const MoeStageCommonConfig &common,
                                               const BlockWorkspaceContext &countWorkspace, const Params &params,
                                               TokenDispatchScratch<ActivationType, TopkIndexType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || countWorkspace.blockIdx != 0U) {
        return;
    }

    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, common.moeExpertPerRank, 0U, true);
    GlobalTensor<int32_t> expertRecvTokenCount;
    expertRecvTokenCount.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params.workspaceInfo.expertRecvTokenCountPtr));
    DataCopyPad(scratch.expertTokenNumsOutTensor, expertRecvTokenCount[countOffset],
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U}, {true, 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE2_MTE3, SYNC_EVENT_ID2>();

    GlobalTensor<int32_t> expertTokenNumsOut;
    expertTokenNumsOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.expertTokenNumsOutGmAddr));
    DataCopyPad(expertTokenNumsOut, scratch.expertTokenNumsOutTensor,
                {1U, common.moeExpertPerRank * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U});
    SyncFuncStatic<HardEvent::MTE3_S, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_H
