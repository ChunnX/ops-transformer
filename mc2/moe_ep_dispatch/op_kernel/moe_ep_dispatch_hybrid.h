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
 * \file moe_ep_dispatch_hybrid.h
 * \brief Hybrid MoE EP dispatch kernel for intra-server scaleup communication.
 */

#ifndef MOE_EP_DISPATCH_HYBRID_H
#define MOE_EP_DISPATCH_HYBRID_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#include "adv_api/hcomm/hcomm.h"
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch_base.h"

#if __has_include("../common/moe_distribute_base.h")
#include "../common/moe_distribute_base.h"
#include "../common/mc2_kernel_utils.h"
#include "../common/mc2_moe_context.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

namespace MoeEpDispatchHybridImpl {

#define TemplateMoeEpDispatchHybridTypeClass                                                                           \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchHybridTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t MAX_TOPK = 32U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t NETWORK_DIRECT = 0U;
constexpr uint32_t NETWORK_HYBRID = 1U;
constexpr uint32_t META_INFO = 32U;
constexpr uint32_t TOPK_INFO_SIZE = 4U; // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;      // UB_ALIGN/sizeof(int32_t)=8
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 50 * 1024U; // count 50KB per group
constexpr uint32_t COUNTER_STRIDE = 2U;         // state and dstRankRecvNum
constexpr uint32_t CORE_NUM_RATIO = 2U;         // per-rank/per-expert count core ratio
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint64_t CQE_MAX_WRITE_SIZE = 256UL * 1024UL * 1024UL;

template <TemplateMoeEpDispatchHybridTypeClass>
class MoeEpDispatchHybrid {
public:
    __aicore__ inline MoeEpDispatchHybrid(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR cachedRouteCount, GM_ADDR cachedRouteDstScaleout,
                                GM_ADDR cachedRouteScaleoutSlot, GM_ADDR cachedRouteScaleupMask,
                                GM_ADDR cachedRouteScaleupSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout,
                                GM_ADDR routeScaleoutSlot, GM_ADDR routeScaleupMask, GM_ADDR routeScaleupSlot,
                                GM_ADDR debugOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessHybridIntraServerOnly();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCntPerRank(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                             uint32_t startRankId, uint32_t endRankId);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                               uint32_t startExpertId, uint32_t endExpertId);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void ExchangeCount();
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline uint32_t DedupAndSend(uint32_t tokenId, LocalTensor<int32_t> hitPerRankTensor,
                                            LocalTensor<int32_t> seenScaleoutTensor,
                                            LocalTensor<int32_t> seenScaleoutSlotTensor);
    __aicore__ inline uint32_t AllocateScaleupSlot(uint32_t dstRankId);
    __aicore__ inline uint32_t AllocateScaleoutSlot(uint32_t dstScaleoutIndex);
    __aicore__ inline uint32_t GetDstScaleoutIndex(uint32_t dstRankId);
    __aicore__ inline uint32_t GetDstScaleupIndex(uint32_t dstRankId);
    __aicore__ inline uint32_t FindOrCreateRemoteRouteEntry(uint32_t tokenId, uint32_t dstScaleoutIndex,
                                                            LocalTensor<int32_t> seenScaleoutTensor,
                                                            LocalTensor<int32_t> seenScaleoutSlotTensor,
                                                            uint32_t &routeEntryCount,
                                                            uint32_t &scaleoutSlot);
    __aicore__ inline void UpdateRemoteRouteEntryTarget(uint32_t tokenId, uint32_t routeEntryIndex,
                                                        uint32_t dstScaleupIndex, uint32_t scaleupSlot);
    __aicore__ inline uint32_t CopyCachedRouteForToken(uint32_t tokenId);
    __aicore__ inline GM_ADDR GetScaleoutSlotAddr(GM_ADDR baseAddr, uint32_t scaleoutIndex, uint32_t slot);
    __aicore__ inline GM_ADDR GetScaleoutStatusAddr(GM_ADDR baseAddr, uint32_t scaleoutIndex);
    __aicore__ inline uint32_t GetCurrentScaleupIndex();
    __aicore__ inline uint32_t GetProxyRankId(uint32_t dstScaleoutIndex);
    __aicore__ inline void WriteScaleoutPayloadFromRoute(uint32_t tokenId, uint32_t routeEntryCount);
    __aicore__ inline void InitPostSendHcomm();
    __aicore__ inline void Stage1ScaleoutSend();
    __aicore__ inline void Stage2ScaleoutProxy();
    __aicore__ inline void FanoutScaleoutPayload(uint32_t srcRankId, uint32_t srcScaleoutIndex, uint32_t scaleoutSlot);
    __aicore__ inline void WriteFinalReadyStatusForRemoteSource(uint32_t srcRankId);
    __aicore__ inline void WriteSlotToLocal(uint32_t dstRankId, uint32_t slot);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void SendPhase();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline bool IsActiveRank(uint32_t rankId);
    __aicore__ inline void MarkRemoteRouteUnsupported();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum, bool isFront);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::MoeCommContext *ctx, uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)ctx->epHcclBuffer[rankId] + offset;
    }

    __aicore__ inline uint64_t GetCommHandle(__gm__ Mc2Aclnn::MoeCommContext *ctx, uint32_t localRankId,
                                             uint32_t rankId)
    {
        uint32_t index = rankId > localRankId ? rankId - 1 : rankId;
        return ctx->hcommHandle[index];
    }

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::MoeCommContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // 通信上下文

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> cachedRouteCountGMTensor_;
    GlobalTensor<int32_t> cachedRouteDstScaleoutGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleoutSlotGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleupMaskGMTensor_;
    GlobalTensor<int32_t> cachedRouteScaleupSlotGMTensor_;
    GlobalTensor<int32_t> routeCountGMTensor_;
    GlobalTensor<int32_t> routeDstScaleoutGMTensor_;
    GlobalTensor<int32_t> routeScaleoutSlotGMTensor_;
    GlobalTensor<int32_t> routeScaleupMaskGMTensor_;
    GlobalTensor<int32_t> routeScaleupSlotGMTensor_;
    GlobalTensor<uint8_t> debugOutGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> scaleoutCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntGMTensor_;
    GlobalTensor<XType> slotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<uint8_t> hcommTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> perSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dstSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> numRecvBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;
    TBuf<> recvCntBuf_;
    TBuf<> recvTempBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR scaleoutCounterAddr_{nullptr};
    GM_ADDR scaleoutNotifyWorkspaceAddr_{nullptr};
    GM_ADDR readyStatusWorkspaceAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR slotWorkspaceAddr_{nullptr};
    GM_ADDR scaleoutSendWorkspaceAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    GM_ADDR localWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t scaleoutSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t numScaleupRanks_{0};
    uint32_t numScaleoutRanks_{0};
    uint32_t numAivStage1_{0};
    uint32_t numAivStage2_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t rankSizePerServer_{0};
    uint32_t currentServerIndex_{0};
    uint32_t serverStartRank_{0};
    uint32_t serverEndRank_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t hAlignSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t scaleoutMaskWordCount_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t cntPerRankSizeAlign_{0};
    uint32_t cntPerRankSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t scaleoutCounterCnt_{0};
    uint32_t scaleoutNotifyCnt_{0};
    uint32_t metaOffset_{0};
    uint32_t calRankCoreNum_{0};
    uint32_t calExpertCoreNum_{0};
    int64_t actualACnt_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t scaleoutRecvDataOffset_{0};
    uint64_t scaleoutRecvStatusOffset_{0};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR cachedRouteCount, GM_ADDR cachedRouteDstScaleout, GM_ADDR cachedRouteScaleoutSlot,
    GM_ADDR cachedRouteScaleupMask, GM_ADDR cachedRouteScaleupSlot, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR dstBufferSlotIdx, GM_ADDR routeCount, GM_ADDR routeDstScaleout, GM_ADDR routeScaleoutSlot,
    GM_ADDR routeScaleupMask, GM_ADDR routeScaleupSlot, GM_ADDR debugOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    epRankId_ = mc2Context_->epRankId;

    const auto &info = tilingData->moeEpDispatchInfo;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.cfg.scalesBytes;
    perSlotBytes_ = info.cfg.perSlotBytes;
    scaleoutSlotBytes_ = info.scaleoutSlotAlignedBytes;
    numScaleupRanks_ = info.numScaleupRanks;
    numScaleoutRanks_ = info.numScaleoutRanks;
    numAivStage1_ = info.numAivStage1;
    numAivStage2_ = info.numAivStage2;
    aivNum_ = info.aivNum;
    rankSizePerServer_ = info.rankSizePerServer;
    scaleoutMaskWordCount_ = info.scaleoutMaskWordCount;
    currentServerIndex_ = MoeEpDispatchBase::GetCurrentServerIndex(epRankId_, rankSizePerServer_);
    serverStartRank_ = MoeEpDispatchBase::GetServerStartRank(epRankId_, rankSizePerServer_);
    serverEndRank_ = MoeEpDispatchBase::GetServerEndRank(epRankId_, rankSizePerServer_, epWorldSize_);
    cntWinStateOffset_ = info.cntWinStateOffset;
    slotWinStateOffset_ = info.slotWinStateOffset;
    winDataOffset_ = info.winDataOffset;
    scaleoutRecvDataOffset_ = info.scaleoutRecvDataOffset;
    scaleoutRecvStatusOffset_ = info.scaleoutRecvStatusOffset;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);
    moeExpertNum_ = moeExpertNumPerRank_ * epWorldSize_;

    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaOffset_ = hAlignSize_;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    // perRank state+count
    cntPerRankSizeAlign_ = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    scaleoutCounterCnt_ = Ceil(numScaleoutRanks_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    scaleoutNotifyCnt_ = Ceil(numScaleoutRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN / sizeof(uint64_t);
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    cntPerRankSizeAlign512_ = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeNumPerRankAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;

    uint32_t numPerCore = (moeExpertNum_ + CORE_NUM_RATIO * epWorldSize_) / aivNum_;
    calRankCoreNum_ =
        CORE_NUM_RATIO * numPerCore >= epWorldSize_ ? Ceil(CORE_NUM_RATIO * epWorldSize_, numPerCore) : epWorldSize_;
    calRankCoreNum_ = (calRankCoreNum_ >= aivNum_) ? (aivNum_ - 2U) : calRankCoreNum_;
    calExpertCoreNum_ = aivNum_ - calRankCoreNum_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        metaOffset_ += Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
        cachedRouteCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteCount);
        cachedRouteDstScaleoutGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteDstScaleout);
        cachedRouteScaleoutSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleoutSlot);
        cachedRouteScaleupMaskGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleupMask);
        cachedRouteScaleupSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedRouteScaleupSlot);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);
    routeCountGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeCount);
    routeDstScaleoutGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeDstScaleout);
    routeScaleoutSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleoutSlot);
    routeScaleupMaskGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleupMask);
    routeScaleupSlotGMTensor_.SetGlobalBuffer((__gm__ int32_t *)routeScaleupSlot);
    debugOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t *)debugOut);

    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U,
                         static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U,
                              static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
    padParams_ = {true, 0, 0, 0};

    scaleupCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + epWorldSizeAlign512_;
    scaleoutCounterAddr_ = workspaceGM_ + info.routeWorkspaceOffset;
    scaleoutNotifyWorkspaceAddr_ =
        scaleoutCounterAddr_ + Ceil(numScaleoutRanks_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    readyStatusWorkspaceAddr_ =
        scaleoutNotifyWorkspaceAddr_ + Ceil(numScaleoutRanks_ * sizeof(uint64_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    slotWorkspaceAddr_ = workspaceGM_ + info.scaleupSendWorkspaceOffset;
    scaleoutSendWorkspaceAddr_ = workspaceGM_ + info.scaleoutSendWorkspaceOffset;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, info.scaleupFinalRecvDataOffset);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleupCounterAddr_);
    scaleoutCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleoutCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr_);
    slotGMTensor_.SetGlobalBuffer((__gm__ XType *)slotWorkspaceAddr_);
    sendCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);

    if (aivId_ == 0U) {
        debugOutGMTensor_.SetValue(MoeEpDispatchBase::DEBUG_OFFSET_PATH_MARKER, MoeEpDispatchBase::DEBUG_PATH_HYBRID);
        debugOutGMTensor_.SetValue(MoeEpDispatchBase::DEBUG_OFFSET_NETWORK_MODE, NetworkMode);
        debugOutGMTensor_.SetValue(MoeEpDispatchBase::DEBUG_OFFSET_SUPER_NODE_COUNT,
                                   static_cast<uint8_t>(numScaleoutRanks_ & 0xFFU));
        debugOutGMTensor_.SetValue(MoeEpDispatchBase::DEBUG_OFFSET_RANK_SIZE_PER_SERVER,
                                   static_cast<uint8_t>(rankSizePerServer_ & 0xFFU));
        debugOutGMTensor_.SetValue(MoeEpDispatchBase::DEBUG_OFFSET_CURRENT_SERVER_INDEX,
                                   static_cast<uint8_t>(currentServerIndex_ & 0xFFU));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline bool MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::IsActiveRank(uint32_t rankId)
{
    return rankId >= serverStartRank_ && rankId < serverEndRank_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::MarkRemoteRouteUnsupported()
{
    MoeEpDispatchBase::IncrementDebugUint32(debugOutGMTensor_, MoeEpDispatchBase::DEBUG_OFFSET_REMOTE_ROUTE_COUNT);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId, uint32_t &sendNum, bool isFront)
{
    sendNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的数量
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId = aivId_;
    if (!isFront) {
        newAivId -= calRankCoreNum_; // 由于是后面的核做perExpert计算，因此需要换算
    }
    startId = sendNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendNum++;
        startId += newAivId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ResetCounters()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }
    // workspace 计数器清0
    LocalTensor<int32_t> counterLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
    Duplicate<int32_t>(counterLocalTensor, 0, counterCnt_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(scaleupCounterGMTensor_, counterLocalTensor, counterCnt_);
    if (numScaleoutRanks_ > 0U) {
        Duplicate<int32_t>(counterLocalTensor, 0, scaleoutCounterCnt_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(scaleoutCounterGMTensor_, counterLocalTensor, scaleoutCounterCnt_);

        GlobalTensor<uint64_t> scaleoutNotifyTensor;
        scaleoutNotifyTensor.SetGlobalBuffer((__gm__ uint64_t *)scaleoutNotifyWorkspaceAddr_);
        LocalTensor<uint64_t> notifyLocalTensor = counterLocalTensor.template ReinterpretCast<uint64_t>();
        Duplicate<uint64_t>(notifyLocalTensor, 0UL, scaleoutNotifyCnt_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(scaleoutNotifyTensor, notifyLocalTensor, scaleoutNotifyCnt_);

        GlobalTensor<uint64_t> readyStatusTensor;
        readyStatusTensor.SetGlobalBuffer((__gm__ uint64_t *)readyStatusWorkspaceAddr_);
        readyStatusTensor.SetValue(0, 1UL);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(readyStatusTensor);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t startRankId, uint32_t endRankId)
{
    if (startRankId >= epWorldSize_) {
        return;
    }

    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t tokenCnt = calCnt / axisK_;
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    uint32_t mask = tokenCnt;
    uint32_t shape[2] = {tokenCnt, axisK_};
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();

    Duplicate<int16_t>(dstTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    Div(tempTensorInt16, expertIdsTensor, dstTensorInt16, calCnt);
    // 筛选无效expert id 消除影响
    CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    Select(dstTensorInt16, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);

    for (uint32_t dstRankId = startRankId; dstRankId < endRankId; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        Subs(expertIdsTensor, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        Abs(tempTensorInt16, expertIdsTensor, calCnt);
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(expertIdsTensor, tempTensorInt16, shape, false); // 0为目标
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign); // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
        GatherMask(tempTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        uint32_t offset = (dstRankId - startRankId) * COUNTER_STRIDE + 1;
        int32_t curRankCnt = sendCntTensor_.GetValue(offset) + static_cast<int32_t>(rsvdCnt);
        sendCntTensor_.SetValue(offset, curRankCnt);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t startExpertId, uint32_t endExpertId)
{
    if (startExpertId >= moeExpertNum_) {
        return;
    }

    LocalTensor<uint8_t> gatherMaskTensorInt8 = topkIdsBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.Get<uint16_t>();
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    uint32_t mask = calCnt;
    uint32_t calCntAlign = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);

    for (uint32_t dstExpertId = startExpertId; dstExpertId < endExpertId; dstExpertId++) {
        uint64_t rsvdCnt = 0;
        Duplicate<uint16_t>(maskTensorInt16, 0, calCntAlign); // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(dstExpertId), AscendC::CMPMODE::EQ,
                      calCntAlign);
        GatherMask(dstTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curExpertCnt = sendCntTensor_.GetValue(dstExpertId - startExpertId);
        sendCntTensor_.SetValue(dstExpertId - startExpertId, curExpertCnt + static_cast<int32_t>(rsvdCnt));
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CalSendCnt()
{
    uint32_t startId = 0;
    uint32_t endId = 0;
    uint32_t numPerCore = 0;
    uint32_t perGroupTokenNum = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
    uint32_t groupCnt = Ceil(axisBS_, perGroupTokenNum);
    uint32_t calCnt = perGroupTokenNum * axisK_;
    uint32_t perGroupSizeAlign = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t cntPerRankSizeAlign256 =
        Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t maxSizePerCore = moeExpertNumAlign_ > cntPerRankSizeAlign256 ? moeExpertNumAlign_ : cntPerRankSizeAlign256;
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(numRecvBuf_, maxSizePerCore);
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    sendCntTensor_ = numRecvBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendCntTensor_, 0, maxSizePerCore / sizeof(int32_t));

    if (aivId_ < calRankCoreNum_) { // 前面的核做per rank count计算
        SplitToCore(epWorldSize_, calRankCoreNum_, startId, endId, numPerCore, true);
        if (startId >= epWorldSize_) {
            return;
        }
        // 两两一组，每组第一个位置填充1，作为状态位
        LocalTensor<int64_t> sendCntTensorInt64 = sendCntTensor_.template ReinterpretCast<int64_t>();
        Duplicate<int64_t>(sendCntTensorInt64, static_cast<int64_t>(1), cntPerRankSizeAlign256 / sizeof(int64_t));
    } else { // 后面的核做per expert count计算
        SplitToCore(moeExpertNum_, calExpertCoreNum_, startId, endId, numPerCore, false);
        if (startId >= moeExpertNum_) {
            return;
        }
    }

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (axisBS_ - group * perGroupTokenNum) * axisK_;
        }
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        uint32_t topkIdxOffset = group * perGroupTokenNum * axisK_;
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    topkIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);

        if (aivId_ >= calRankCoreNum_) { // 后面的核统计发送给各专家的token count
            CalSendCntPerExpert(tempTensorInt16, calCnt, startId, endId);
        } else { // 前面的核统计发送到各卡token count
            CalSendCntPerRank(tempTensorInt16, calCnt, startId, endId);
        }
    }

    uint32_t gmOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + startId;
    uint32_t blockLen = numPerCore * sizeof(int32_t);
    if (aivId_ < calRankCoreNum_) {
        gmOffset = startId * COUNTER_STRIDE;
        blockLen = numPerCore * sizeof(int32_t) * COUNTER_STRIDE;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyParams cntCopyParams = {1U, static_cast<uint16_t>(blockLen), 0U, 0U};
    DataCopyPad(sendCntGMTensor_[gmOffset], sendCntTensor_, cntCopyParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ExchangeCount()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_, true);
    if (startRankId_ >= epWorldSize_) {
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    GlobalTensor<uint64_t> numSendGMTensorInt64;
    numSendGMTensorInt64.SetGlobalBuffer((__gm__ uint64_t *)sendCntWorkspaceAddr_);
    LocalTensor<uint64_t> sendCntPerRankInt64 = dstExpBuf_.Get<uint64_t>();
    DataCopyParams cntCopyParams = {1U, static_cast<uint16_t>(rankNumPerCore_ * sizeof(uint64_t)), 0U, 0U};
    DataCopyPad(sendCntPerRankInt64, numSendGMTensorInt64[startRankId_], cntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>(); // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint64_t notifyVal = sendCntPerRankInt64.GetValue(dstRankId - startRankId_);
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        GM_ADDR remoteCountAddr = remoteStateAddr + epWorldSize_ * WIN_ADDR_ALIGN + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr =
            sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_ + dstRankId * moeExpertNumPerRank_ * sizeof(int32_t);

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            hcomm_.WriteNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteCountAddr, srcWorkspaceAddr,
                            static_cast<uint64_t>(moeExpertNumPerRank_ * sizeof(int32_t)));
            hcomm_.Drain(GetCommHandle(mc2Context_, epRankId_, dstRankId));
            hcomm_.WriteNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), notifyAddr,
                            sendCntWorkspaceAddr_ + dstRankId * sizeof(uint64_t), sizeof(uint64_t));
            hcomm_.Drain(GetCommHandle(mc2Context_, epRankId_, dstRankId));
        } else { // 本端
            GlobalTensor<int32_t> countGMTensor;
            GlobalTensor<uint64_t> notifyGMTensor;
            countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
            notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
            LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.Get<int32_t>();
            DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U,
                                                  0U};
            uint32_t srcOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + dstRankId * moeExpertNumPerRank_;
            DataCopyPad(cntPerExpertTensor, sendCntGMTensor_[srcOffset], expertCntCopyParams, padParams_);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>(); // perExpert 写完再写notifyVal
            notifyGMTensor.SetValue(0, notifyVal);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(notifyGMTensor);
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetRecvCount()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    uint32_t serverRankCount = epWorldSize_;
    uint32_t mask = 1U;
    int32_t sumOfFlag = -1;
    int32_t compareFlag = static_cast<int32_t>(serverRankCount);
    LocalTensor<int32_t> recvCounterTensor = topkIdsBuf_.GetWithOffset<int32_t>(serverRankCount * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 =
        topkIdsBuf_.GetWithOffset<float>(serverRankCount * UB_STRIDE, serverRankCount * UB_ALIGN);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 等待前面尾核计数器清0(buf 复用)
    while (sumOfFlag != compareFlag) {      // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, serverRankCount, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, serverRankCount * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvExpertSizeAlign = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t recvExpertAlign = recvExpertSizeAlign / sizeof(int32_t);
    uint32_t serverRankCount = epWorldSize_;
    tpipe_->InitBuffer(recvCntBuf_, serverRankCount * recvExpertSizeAlign);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, moeNumPerRankAlign_);
    tpipe_->InitBuffer(recvTempBuf_, UB_ALIGN);
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int32_t> recvTensorInt32 = recvCntBuf_.Get<int32_t>();
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = recvTempBuf_.Get<int64_t>();
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t shape[] = {serverRankCount, recvExpertAlign};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(serverRankCount),
                                      static_cast<uint16_t>(recvExpertSizeAlign),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - recvExpertSizeAlign), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    uint32_t recvExpertOffset = epWorldSize_ * WIN_ADDR_ALIGN / sizeof(int32_t);
    DataCopyPad(recvTensorInt32, recvCounterGMTensor_[recvExpertOffset], inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, serverRankCount * recvExpertAlign);
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(hostPinnedCounterTensor, recvCntTensor, UB_ALIGN / sizeof(int64_t));
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor)
{
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};

    LocalTensor<uint32_t> gatherIndexTensor =
        topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    gatherIndexTensor.SetValue(0, 2U); // Extract element index 1 from each UB-aligned rank status block.
    uint32_t compactMask = 2U;
    uint64_t reservedCount = 0UL;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherIndexTensor, true, compactMask, recvMaskParams,
               reservedCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::AllocateScaleupSlot(
    uint32_t dstRankId)
{
    __gm__ int32_t *dstCounterAddr =
        reinterpret_cast<__gm__ int32_t *>(scaleupCounterAddr_ + dstRankId * sizeof(int32_t));
    return AscendC::AtomicAdd(dstCounterAddr, 1);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::AllocateScaleoutSlot(
    uint32_t dstScaleoutIndex)
{
    __gm__ int32_t *dstCounterAddr =
        reinterpret_cast<__gm__ int32_t *>(scaleoutCounterAddr_ + dstScaleoutIndex * sizeof(int32_t));
    return AscendC::AtomicAdd(dstCounterAddr, 1);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetDstScaleoutIndex(
    uint32_t dstRankId)
{
    return rankSizePerServer_ == 0U ? 0U : dstRankId / rankSizePerServer_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetDstScaleupIndex(
    uint32_t dstRankId)
{
    return rankSizePerServer_ == 0U ? dstRankId : dstRankId % rankSizePerServer_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FindOrCreateRemoteRouteEntry(
    uint32_t tokenId, uint32_t dstScaleoutIndex, LocalTensor<int32_t> seenScaleoutTensor,
    LocalTensor<int32_t> seenScaleoutSlotTensor, uint32_t &routeEntryCount, uint32_t &scaleoutSlot)
{
    for (uint32_t routeEntryIndex = 0; routeEntryIndex < routeEntryCount; routeEntryIndex++) {
        if (static_cast<uint32_t>(seenScaleoutTensor.GetValue(routeEntryIndex)) == dstScaleoutIndex) {
            scaleoutSlot = static_cast<uint32_t>(seenScaleoutSlotTensor.GetValue(routeEntryIndex));
            return routeEntryIndex;
        }
    }

    uint32_t routeEntryIndex = routeEntryCount;
    if (routeEntryIndex >= axisK_) {
        MoeEpDispatchBase::WriteUint32ToDebug(debugOutGMTensor_, MoeEpDispatchBase::DEBUG_OFFSET_ERROR_FLAGS,
                                              MoeEpDispatchBase::DEBUG_FLAG_REMOTE_ROUTE_UNSUPPORTED);
        routeEntryIndex = axisK_ - 1U;
    } else {
        routeEntryCount++;
    }

    scaleoutSlot = AllocateScaleoutSlot(dstScaleoutIndex);
    seenScaleoutTensor.SetValue(routeEntryIndex, static_cast<int32_t>(dstScaleoutIndex));
    seenScaleoutSlotTensor.SetValue(routeEntryIndex, static_cast<int32_t>(scaleoutSlot));

    uint32_t routeTwoDimOffset = tokenId * axisK_ + routeEntryIndex;
    routeDstScaleoutGMTensor_.SetValue(routeTwoDimOffset, static_cast<int32_t>(dstScaleoutIndex));
    routeScaleoutSlotGMTensor_.SetValue(routeTwoDimOffset, static_cast<int32_t>(scaleoutSlot));
    return routeEntryIndex;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::UpdateRemoteRouteEntryTarget(
    uint32_t tokenId, uint32_t routeEntryIndex, uint32_t dstScaleupIndex, uint32_t scaleupSlot)
{
    uint32_t routeMaskBase = (tokenId * axisK_ + routeEntryIndex) * scaleoutMaskWordCount_;
    uint32_t maskWordIndex = dstScaleupIndex / 32U;
    uint32_t maskBitIndex = dstScaleupIndex % 32U;
    uint32_t routeMaskOffset = routeMaskBase + maskWordIndex;
    uint32_t currentMask = static_cast<uint32_t>(routeScaleupMaskGMTensor_.GetValue(routeMaskOffset));
    currentMask |= (1U << maskBitIndex);
    routeScaleupMaskGMTensor_.SetValue(routeMaskOffset, static_cast<int32_t>(currentMask));

    uint32_t routeSlotOffset = (tokenId * axisK_ + routeEntryIndex) * rankSizePerServer_ + dstScaleupIndex;
    routeScaleupSlotGMTensor_.SetValue(routeSlotOffset, static_cast<int32_t>(scaleupSlot));
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t
MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::CopyCachedRouteForToken(uint32_t tokenId)
{
    uint32_t routeEntryCount = static_cast<uint32_t>(cachedRouteCountGMTensor_.GetValue(tokenId));
    routeCountGMTensor_.SetValue(tokenId, static_cast<int32_t>(routeEntryCount));
    for (uint32_t routeEntryIndex = 0; routeEntryIndex < axisK_; routeEntryIndex++) {
        uint32_t routeTwoDimOffset = tokenId * axisK_ + routeEntryIndex;
        routeDstScaleoutGMTensor_.SetValue(routeTwoDimOffset, cachedRouteDstScaleoutGMTensor_.GetValue(routeTwoDimOffset));
        routeScaleoutSlotGMTensor_.SetValue(routeTwoDimOffset,
                                            cachedRouteScaleoutSlotGMTensor_.GetValue(routeTwoDimOffset));

        for (uint32_t maskWordIndex = 0; maskWordIndex < scaleoutMaskWordCount_; maskWordIndex++) {
            uint32_t routeMaskOffset = routeTwoDimOffset * scaleoutMaskWordCount_ + maskWordIndex;
            routeScaleupMaskGMTensor_.SetValue(routeMaskOffset, cachedRouteScaleupMaskGMTensor_.GetValue(routeMaskOffset));
        }
        for (uint32_t dstScaleupIndex = 0; dstScaleupIndex < rankSizePerServer_; dstScaleupIndex++) {
            uint32_t routeSlotOffset = routeTwoDimOffset * rankSizePerServer_ + dstScaleupIndex;
            routeScaleupSlotGMTensor_.SetValue(routeSlotOffset,
                                               cachedRouteScaleupSlotGMTensor_.GetValue(routeSlotOffset));
        }
    }
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::DedupAndSend(
    uint32_t tokenId, LocalTensor<int32_t> hitPerRankTensor, LocalTensor<int32_t> seenScaleoutTensor,
    LocalTensor<int32_t> seenScaleoutSlotTensor)
{
    uint32_t routeEntryCount = 0U;
    for (uint32_t topkIndex = 0; topkIndex < axisK_; topkIndex++) {
        int32_t expertId = topkIdxTensor_.GetValue(topkIndex);
        if (expertId < 0) {
            dstSlotIdxTensor_.SetValue(topkIndex, -1);
            continue;
        }

        uint32_t dstRankId = static_cast<uint32_t>(expertId) / moeExpertNumPerRank_;
        int32_t hit = hitPerRankTensor.GetValue(dstRankId);
        uint32_t scaleupSlot = 0U;
        if (hit >= 0) {
            scaleupSlot = static_cast<uint32_t>(hit);
            dstSlotIdxTensor_.SetValue(topkIndex, hit);
        } else {
            scaleupSlot = AllocateScaleupSlot(dstRankId);
            dstSlotIdxTensor_.SetValue(topkIndex, static_cast<int32_t>(scaleupSlot));
            hitPerRankTensor.SetValue(dstRankId, static_cast<int32_t>(scaleupSlot));
            if (IsActiveRank(dstRankId)) {
                WriteSlotToLocal(dstRankId, scaleupSlot);
            }
        }

        if (IsActiveRank(dstRankId)) {
            continue;
        }

        uint32_t dstScaleoutIndex = GetDstScaleoutIndex(dstRankId);
        uint32_t dstScaleupIndex = GetDstScaleupIndex(dstRankId);
        uint32_t scaleoutSlot = 0U;
        uint32_t routeEntryIndex = FindOrCreateRemoteRouteEntry(tokenId, dstScaleoutIndex, seenScaleoutTensor,
                                                                seenScaleoutSlotTensor, routeEntryCount, scaleoutSlot);
        (void)scaleoutSlot;
        UpdateRemoteRouteEntryTarget(tokenId, routeEntryIndex, dstScaleupIndex, scaleupSlot);
        MarkRemoteRouteUnsupported();
    }
    routeCountGMTensor_.SetValue(tokenId, static_cast<int32_t>(routeEntryCount));
    return routeEntryCount;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline GM_ADDR MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetScaleoutSlotAddr(
    GM_ADDR baseAddr, uint32_t scaleoutIndex, uint32_t slot)
{
    return baseAddr +
           (static_cast<uint64_t>(scaleoutIndex) * axisMaxBS_ + static_cast<uint64_t>(slot)) * scaleoutSlotBytes_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline GM_ADDR MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetScaleoutStatusAddr(
    GM_ADDR baseAddr, uint32_t scaleoutIndex)
{
    return baseAddr + static_cast<uint64_t>(scaleoutIndex) * axisMaxBS_ * WIN_ADDR_ALIGN;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetCurrentScaleupIndex()
{
    return rankSizePerServer_ == 0U ? epRankId_ : epRankId_ % rankSizePerServer_;
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline uint32_t MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::GetProxyRankId(
    uint32_t dstScaleoutIndex)
{
    return dstScaleoutIndex * rankSizePerServer_ + GetCurrentScaleupIndex();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteScaleoutPayloadFromRoute(
    uint32_t tokenId, uint32_t routeEntryCount)
{
    DataCopyParams scaleoutPayloadCopyParams = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    for (uint32_t routeEntryIndex = 0; routeEntryIndex < routeEntryCount; routeEntryIndex++) {
        uint32_t routeTwoDimOffset = tokenId * axisK_ + routeEntryIndex;
        int32_t dstScaleoutValue = routeDstScaleoutGMTensor_.GetValue(routeTwoDimOffset);
        int32_t scaleoutSlotValue = routeScaleoutSlotGMTensor_.GetValue(routeTwoDimOffset);
        if (dstScaleoutValue < 0 || scaleoutSlotValue < 0) {
            continue;
        }

        uint32_t dstScaleoutIndex = static_cast<uint32_t>(dstScaleoutValue);
        uint32_t scaleoutSlot = static_cast<uint32_t>(scaleoutSlotValue);
        GM_ADDR scaleoutSlotAddr = GetScaleoutSlotAddr(scaleoutSendWorkspaceAddr_, dstScaleoutIndex, scaleoutSlot);
        GlobalTensor<XType> scaleoutPayloadTensor;
        scaleoutPayloadTensor.SetGlobalBuffer((__gm__ XType *)scaleoutSlotAddr);
        DataCopyPad(scaleoutPayloadTensor, tokenSlotTensor_, scaleoutPayloadCopyParams);

        GlobalTensor<int32_t> scaleoutRouteTensor;
        scaleoutRouteTensor.SetGlobalBuffer((__gm__ int32_t *)(scaleoutSlotAddr + perSlotBytes_));
        for (uint32_t maskWordIndex = 0; maskWordIndex < scaleoutMaskWordCount_; maskWordIndex++) {
            uint32_t routeMaskOffset = routeTwoDimOffset * scaleoutMaskWordCount_ + maskWordIndex;
            scaleoutRouteTensor.SetValue(maskWordIndex, routeScaleupMaskGMTensor_.GetValue(routeMaskOffset));
        }
        for (uint32_t dstScaleupIndex = 0; dstScaleupIndex < rankSizePerServer_; dstScaleupIndex++) {
            uint32_t routeSlotOffset = routeTwoDimOffset * rankSizePerServer_ + dstScaleupIndex;
            scaleoutRouteTensor.SetValue(scaleoutMaskWordCount_ + dstScaleupIndex,
                                         routeScaleupSlotGMTensor_.GetValue(routeSlotOffset));
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteSlotToLocal(uint32_t dstRankId,
                                                                                                  uint32_t slot)
{
    GlobalTensor<XType> slotGMTensor;
    GM_ADDR rankGM = (dstRankId == epRankId_) ? localWinAddr_ : slotWorkspaceAddr_;
    GM_ADDR slotAddr = rankGM + (static_cast<uint64_t>(dstRankId) * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor.SetGlobalBuffer((__gm__ XType *)slotAddr);
    DataCopyParams slotCopyParams = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    DataCopyPad(slotGMTensor, tokenSlotTensor_, slotCopyParams);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::InitPostSendHcomm()
{
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteToRemoteWindow()
{
    uint32_t localStartRankOffset = 0U;
    uint32_t localEndRankOffset = 0U;
    SplitToCore(rankSizePerServer_, aivNum_, localStartRankOffset, localEndRankOffset, rankNumPerCore_, true);
    startRankId_ = serverStartRank_ + localStartRankOffset;
    endRankId_ = serverStartRank_ + localEndRankOffset;
    if (startRankId_ >= serverEndRank_ || startRankId_ >= endRankId_) {
        return;
    }
    if (endRankId_ > serverEndRank_) {
        endRankId_ = serverEndRank_;
        rankNumPerCore_ = endRankId_ - startRankId_;
    }

    uint32_t sendCntSize = rankNumPerCore_ * COUNTER_STRIDE * sizeof(int32_t);
    tpipe_->InitBuffer(recvTempBuf_, Ceil(sendCntSize, UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> sendNumPerRankTensor = recvTempBuf_.Get<int32_t>();
    DataCopyParams sendCntCopyParams = {1U, static_cast<uint16_t>(sendCntSize), 0U, 0U};
    DataCopyPad(sendNumPerRankTensor, sendCntGMTensor_[startRankId_ * COUNTER_STRIDE], sendCntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>(); // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - startRankId_) * COUNTER_STRIDE + 1);
        GM_ADDR notifyAddr =
            GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) + epRankId_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            GlobalTensor<int32_t> statusGMTensor;
            statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)(notifyAddr));
            statusGMTensor.SetValue(0, static_cast<int32_t>(1));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGMTensor);
            continue; //  本端slot已经写入win
        }

        uint64_t commHandle = GetCommHandle(mc2Context_, epRankId_, dstRankId);
        if (sendTokenNum > 0) {
            uint64_t sendDataSize = static_cast<uint64_t>(perSlotBytes_) * sendTokenNum;
            uint64_t srcWorkspaceOffset = static_cast<uint64_t>(dstRankId) * axisMaxBS_ * perSlotBytes_;
            uint64_t dstRankWinOffset =
                static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_; // 计算目标窗口地址偏移
            GM_ADDR remoteWinAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset;
            GM_ADDR localWorkspaceAddr = slotWorkspaceAddr_ + srcWorkspaceOffset;
            MoeEpDispatchBase::WriteRemoteWindowByChunks(hcomm_, commHandle, remoteWinAddr, localWorkspaceAddr,
                                                         sendDataSize);
        }
        hcomm_.WriteNbi(commHandle, notifyAddr, sendCntWorkspaceAddr_ + dstRankId * sizeof(uint64_t),
                        static_cast<uint64_t>(sizeof(uint64_t)));
        hcomm_.Drain(commHandle);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Stage1ScaleoutSend()
{
    uint32_t startScaleoutIndex = 0U;
    uint32_t endScaleoutIndex = 0U;
    uint32_t scaleoutNumPerCore = 0U;
    SplitToCore(numScaleoutRanks_, aivNum_, startScaleoutIndex, endScaleoutIndex, scaleoutNumPerCore, true);
    if (startScaleoutIndex >= numScaleoutRanks_) {
        return;
    }

    GlobalTensor<int32_t> scaleoutCounterTensor;
    scaleoutCounterTensor.SetGlobalBuffer((__gm__ int32_t *)scaleoutCounterAddr_);
    PipeBarrier<PIPE_MTE3>();

    for (uint32_t dstScaleoutIndex = startScaleoutIndex; dstScaleoutIndex < endScaleoutIndex; dstScaleoutIndex++) {
        if (dstScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint32_t proxyRankId = GetProxyRankId(dstScaleoutIndex);
        if (proxyRankId >= epWorldSize_) {
            continue;
        }

        uint32_t scaleoutSlotCount = static_cast<uint32_t>(scaleoutCounterTensor.GetValue(dstScaleoutIndex));
        GlobalTensor<int32_t> notifyTensor;
        notifyTensor.SetGlobalBuffer(
            (__gm__ int32_t *)(scaleoutNotifyWorkspaceAddr_ + dstScaleoutIndex * sizeof(uint64_t)));
        notifyTensor.SetValue(0, 1);
        notifyTensor.SetValue(1, static_cast<int32_t>(scaleoutSlotCount));
        PipeBarrier<PIPE_MTE3>();
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(notifyTensor);

        uint64_t commHandle = GetCommHandle(mc2Context_, epRankId_, proxyRankId);
        if (scaleoutSlotCount > 0U) {
            uint64_t sendDataSize = static_cast<uint64_t>(scaleoutSlotBytes_) * scaleoutSlotCount;
            GM_ADDR localScaleoutAddr = GetScaleoutSlotAddr(scaleoutSendWorkspaceAddr_, dstScaleoutIndex, 0U);
            GM_ADDR remoteScaleoutAddr =
                GetWinAddrByRankId(mc2Context_, proxyRankId, scaleoutRecvDataOffset_) +
                (static_cast<uint64_t>(currentServerIndex_) * axisMaxBS_) * scaleoutSlotBytes_;
            MoeEpDispatchBase::WriteRemoteWindowByChunks(hcomm_, commHandle, remoteScaleoutAddr, localScaleoutAddr,
                                                         sendDataSize);
        }

        GM_ADDR remoteNotifyAddr =
            GetScaleoutStatusAddr(GetWinAddrByRankId(mc2Context_, proxyRankId, scaleoutRecvStatusOffset_),
                                  currentServerIndex_);
        hcomm_.WriteNbi(commHandle, remoteNotifyAddr,
                        scaleoutNotifyWorkspaceAddr_ + dstScaleoutIndex * sizeof(uint64_t), sizeof(uint64_t));
        hcomm_.Drain(commHandle);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::FanoutScaleoutPayload(
    uint32_t srcRankId, uint32_t srcScaleoutIndex, uint32_t scaleoutSlot)
{
    GM_ADDR localScaleoutBase = GetWinAddrByRankId(mc2Context_, epRankId_, scaleoutRecvDataOffset_);
    GM_ADDR scaleoutSlotAddr = GetScaleoutSlotAddr(localScaleoutBase, srcScaleoutIndex, scaleoutSlot);
    GlobalTensor<int32_t> scaleoutRouteTensor;
    scaleoutRouteTensor.SetGlobalBuffer((__gm__ int32_t *)(scaleoutSlotAddr + perSlotBytes_));

    for (uint32_t maskWordIndex = 0U; maskWordIndex < scaleoutMaskWordCount_; maskWordIndex++) {
        uint32_t maskValue = static_cast<uint32_t>(scaleoutRouteTensor.GetValue(maskWordIndex));
        if (maskValue == 0U) {
            continue;
        }
        for (uint32_t maskBitIndex = 0U; maskBitIndex < 32U; maskBitIndex++) {
            uint32_t dstScaleupIndex = maskWordIndex * 32U + maskBitIndex;
            if (dstScaleupIndex >= rankSizePerServer_) {
                break;
            }
            if ((maskValue & (1U << maskBitIndex)) == 0U) {
                continue;
            }

            int32_t scaleupSlotValue = scaleoutRouteTensor.GetValue(scaleoutMaskWordCount_ + dstScaleupIndex);
            if (scaleupSlotValue < 0) {
                continue;
            }
            uint32_t dstRankId = currentServerIndex_ * rankSizePerServer_ + dstScaleupIndex;
            if (dstRankId >= epWorldSize_) {
                continue;
            }

            GM_ADDR finalSlotAddr =
                GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) +
                (static_cast<uint64_t>(srcRankId) * axisMaxBS_ + static_cast<uint32_t>(scaleupSlotValue)) *
                    perSlotBytes_;
            DataCopyParams payloadCopyParams = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
            if (dstRankId == epRankId_) {
                GlobalTensor<XType> scaleoutPayloadTensor;
                GlobalTensor<XType> finalPayloadTensor;
                scaleoutPayloadTensor.SetGlobalBuffer((__gm__ XType *)scaleoutSlotAddr);
                finalPayloadTensor.SetGlobalBuffer((__gm__ XType *)finalSlotAddr);
                LocalTensor<XType> localPayloadTensor = perSlotQueue_.AllocTensor<XType>();
                DataCopyPad(localPayloadTensor, scaleoutPayloadTensor, payloadCopyParams, padParams_);
                perSlotQueue_.EnQue(localPayloadTensor);
                LocalTensor<XType> localPayloadOutTensor = perSlotQueue_.DeQue<XType>();
                DataCopyPad(finalPayloadTensor, localPayloadOutTensor, payloadCopyParams);
                perSlotQueue_.FreeTensor<XType>(localPayloadOutTensor);
            } else {
                uint64_t commHandle = GetCommHandle(mc2Context_, epRankId_, dstRankId);
                MoeEpDispatchBase::WriteRemoteWindowByChunks(hcomm_, commHandle, finalSlotAddr, scaleoutSlotAddr,
                                                             perSlotBytes_);
            }
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WriteFinalReadyStatusForRemoteSource(
    uint32_t srcRankId)
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    for (uint32_t dstRankId = serverStartRank_; dstRankId < serverEndRank_; dstRankId++) {
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                             static_cast<uint64_t>(srcRankId) * WIN_ADDR_ALIGN;
        if (dstRankId == epRankId_) {
            GlobalTensor<int32_t> statusTensor;
            statusTensor.SetGlobalBuffer((__gm__ int32_t *)notifyAddr);
            statusTensor.SetValue(0, 1);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusTensor);
        } else {
            uint64_t commHandle = GetCommHandle(mc2Context_, epRankId_, dstRankId);
            hcomm_.WriteNbi(commHandle, notifyAddr, readyStatusWorkspaceAddr_, sizeof(uint64_t));
            hcomm_.Drain(commHandle);
        }
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Stage2ScaleoutProxy()
{
    if (aivId_ != 0U) {
        return;
    }

    GM_ADDR localScaleoutStatusBase = GetWinAddrByRankId(mc2Context_, epRankId_, scaleoutRecvStatusOffset_);
    GlobalTensor<int32_t> statusTensor;
    statusTensor.SetGlobalBuffer((__gm__ int32_t *)localScaleoutStatusBase);
    LocalTensor<int32_t> stageStatusTensor = numRecvPerRankBuf_.Get<int32_t>();
    DataCopyParams stageStatusCopyParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};

    uint32_t currentScaleupIndex = GetCurrentScaleupIndex();
    for (uint32_t srcScaleoutIndex = 0U; srcScaleoutIndex < numScaleoutRanks_; srcScaleoutIndex++) {
        if (srcScaleoutIndex == currentServerIndex_) {
            continue;
        }
        uint32_t srcRankId = srcScaleoutIndex * rankSizePerServer_ + currentScaleupIndex;
        if (srcRankId >= epWorldSize_) {
            continue;
        }

        uint32_t statusOffset =
            static_cast<uint32_t>(srcScaleoutIndex * axisMaxBS_ * WIN_ADDR_ALIGN / sizeof(int32_t));
        int32_t readyValue = 0;
        while (readyValue != 1) {
            DataCopyPad(stageStatusTensor, statusTensor[statusOffset], stageStatusCopyParams, padParams_);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            readyValue = stageStatusTensor.GetValue(0);
        }
        uint32_t scaleoutSlotCount = static_cast<uint32_t>(stageStatusTensor.GetValue(1));
        for (uint32_t scaleoutSlot = 0U; scaleoutSlot < scaleoutSlotCount; scaleoutSlot++) {
            FanoutScaleoutPayload(srcRankId, srcScaleoutIndex, scaleoutSlot);
        }
        WriteFinalReadyStatusForRemoteSource(srcRankId);
        GlobalTensor<int32_t> currentStatusTensor;
        currentStatusTensor.SetGlobalBuffer(
            (__gm__ int32_t *)(localScaleoutStatusBase + statusOffset * sizeof(int32_t)));
        currentStatusTensor.SetValue(0, 0);
        currentStatusTensor.SetValue(1, 0);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(currentStatusTensor);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::SendPhase()
{
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_, true); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }

    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, 1, kAlignSize_);
    uint32_t routeStateBytes = Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_ + routeStateBytes * 2U);
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams topkCopyParams = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    DataCopyParams scalesCopyParams = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    LocalTensor<int32_t> hitPerRankTensor = numRecvPerRankBuf_.Get<int32_t>();
    LocalTensor<int32_t> seenScaleoutTensor =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(axisKAlign_, epWorldSizeAlign_);
    LocalTensor<int32_t> seenScaleoutSlotTensor =
        numRecvPerRankBuf_.GetWithOffset<int32_t>(axisKAlign_, epWorldSizeAlign_ + routeStateBytes);

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        if (tokenId > startTokenId_) {
            SyncFunc<AscendC::HardEvent::S_V>();
        }
        Duplicate<int32_t>(hitPerRankTensor, -1, epWorldSize_);
        Duplicate<int32_t>(seenScaleoutTensor, -1, axisKAlign_);
        Duplicate<int32_t>(seenScaleoutSlotTensor, -1, axisKAlign_);
        xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
        metaLocalTensor_ = xLocalTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, padParams_);
        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            DataCopyPad(xLocalTensor_[hAlignSize_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                        scalesGMTensor_[tokenId * scalesBytes_ / sizeof(ScalesType)], scalesCopyParams, padParams_);
        }
        DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
        if constexpr (IsTopkWeights) {
            DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(),
                        topkWeightsGMTensor_[topkOffset], topkCopyParams, padParams_);
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
        metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
        perSlotQueue_.EnQue(xLocalTensor_);
        tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
        topkIdxTensor_ = tokenSlotTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();
        SyncFunc<AscendC::HardEvent::V_S>();

        if constexpr (!IsCached) {
            uint32_t routeEntryCount =
                DedupAndSend(tokenId, hitPerRankTensor, seenScaleoutTensor, seenScaleoutSlotTensor);
            WriteScaleoutPayloadFromRoute(tokenId, routeEntryCount);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
        } else {
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            uint32_t routeEntryCount = CopyCachedRouteForToken(tokenId);
            for (uint32_t topkIndex = 0; topkIndex < axisK_; topkIndex++) {
                int32_t slot = dstSlotIdxTensor_.GetValue(topkIndex);
                if (slot == -1) {
                    continue;
                }
                int32_t expertId = topkIdxTensor_.GetValue(topkIndex);
                uint32_t dstRankId = expertId / moeExpertNumPerRank_;
                if (!IsActiveRank(dstRankId)) {
                    MarkRemoteRouteUnsupported();
                    continue;
                }
                if (hitPerRankTensor.GetValue(dstRankId) == slot) {
                    continue;
                }
                WriteSlotToLocal(dstRankId, slot);
                hitPerRankTensor.SetValue(dstRankId, slot);
            }
            WriteScaleoutPayloadFromRoute(tokenId, routeEntryCount);
        }
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor_, topkCopyParams);
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
    }
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::WaitDispatch()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    uint32_t serverRankCount = epWorldSize_;
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(serverRankCount);
    const uint32_t shape[] = {serverRankCount, UB_ALIGN / sizeof(int32_t)};
    GlobalTensor<int32_t> statusGMTensor;
    statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)localSlotStateWinAddr_);
    tpipe_->InitBuffer(recvCntBuf_, serverRankCount * UB_ALIGN);
    tpipe_->InitBuffer(tempBuf_, UB_ALIGN);
    LocalTensor<int32_t> statusLocalTensor = recvCntBuf_.Get<int32_t>();
    LocalTensor<int32_t> sumTensor = tempBuf_.Get<int32_t>();
    while (sumOfFlag != compareFlag) { // 状态位check
        DataCopy(statusLocalTensor, statusGMTensor, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(sumTensor, statusLocalTensor, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = sumTensor.GetValue(0);
    }
    // status clear
    Duplicate<int32_t>(statusLocalTensor, 0, serverRankCount * UB_ALIGN / sizeof(int32_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusGMTensor, statusLocalTensor, clearStatusCopyParams_);
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::ProcessHybridIntraServerOnly()
{
    ResetCounters(); // 计数器清0 为了后续发送token计算atomicAdd
    CalSendCnt();
    SyncAll<true>();
    ExchangeCount();
    GetRecvCount();
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    SendPhase();
    SyncAll<true>();
    InitPostSendHcomm();
    WriteToRemoteWindow();
    Stage1ScaleoutSend();
    Stage2ScaleoutProxy();
    WaitDispatch();
}

template <TemplateMoeEpDispatchHybridTypeClass>
__aicore__ inline void MoeEpDispatchHybrid<TemplateMoeEpDispatchHybridTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        ProcessHybridIntraServerOnly();
    }
}
} // namespace MoeEpDispatchHybridImpl

#endif // MOE_EP_DISPATCH_HYBRID_H
