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
 * \file moe_ep_dispatch.h
 * \brief
 */

#ifndef MOE_EP_DISPATCH_H
#define MOE_EP_DISPATCH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/hcomm/hcomm.h"
#include "moe_ep_dispatch_tiling.h"
#include "moe_ep_dispatch_base.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"

namespace MoeEpDispatchImpl {

#define TemplateMoeEpDispatchTypeClass typename XType, uint32_t DoCpuSync, uint32_t IsCached, uint8_t NetworkMode
#define TemplateMoeEpDispatchTypeFunc XType, DoCpuSync, IsCached, NetworkMode

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t MAX_TOPK = 8;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t NO_CPU_SYNC = 0U;
constexpr uint32_t NO_CACHED = 0U;
constexpr uint32_t NETWORK_DIRECT = 0U; 
constexpr uint32_t NETWORK_HYBRID = 1U; 
constexpr uint32_t META_INFO = 32U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;   // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t HCOMM_INIT_SIZE = 512UL
constexpr uint32_t PER_GROUP_TOKEN_NUM = 256U;
constexpr uint32_t PER_GROUP_EXPERT_NUM = 12800U;
constexpr uint8_t BUFFER_NUM = 2;

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp,
        GM_ADDR dstBufferSlotIdx, GM_ADDR recvSrcMetadataInternal, GM_ADDR workspaceGM, TPipe *pipe,
        const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessDirect();
    __aicore__ inline void ProcessHybrid();
    __aicore__ inline void HcclBarrier();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void SendPhaseFirstTime();
    __aicore__ inline void DedupAndSendDirect(uint32_t t);
    __aicore__ inline void WriteSlotToRemoteWindow(uint32_t dstRank, uint32_t slot, uint32_t t);
    __aicore__ inline void SendPhaseCached();
    __aicore__ inline void RecvPhase();
    __aicore__ inline void WriteActualAToHostPinned();
    __aicore__ inline void SendPhaseScaleoutFirstTime();
    __aicore__ inline void DedupAndSendHybridStage1(uint32_t t);
    __aicore__ inline void WriteSlotToRemoteScaleoutWindow(uint32_t proxyRank, uint32_t slot, uint32_t t, uint32_t mask);
    __aicore__ inline void SendPhaseScaleoutCached();
    __aicore__ inline void SendPhaseScaleupFanout();
    __aicore__ inline void SplitToCore(uint32_t totalCnt, uint32_t &startId, uint32_t &endId, uint32_t &perCoreCnt);

    TPipe *tpipe_{nullptr};
    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // todo: 通信上下文

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> recvSrcMetaGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankHpGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertHpGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<float> topkWeightsTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> metaTensorInt32_;
    LocalTensor<float> metaTensorFloat_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<uint8_t> hcommTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> perSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dstSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> hcommBuf_;   // 通信
    TBuf<> cachedSlotIdxBuf_;
    TBuf<> dstSlotIdxBuf_;
    TBuf<> metaBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;
    TBuf<> recvCounterBuf_;

    // URMA
    TBuf<> urmaSqInfoBuf_;  // 80B
    TBuf<> urmaCqInfoBuf_;  // 48B
    TBuf<> sqHeadAddrBuf_;  // 32B
    TBuf<> sqTailAddrBuf_;  // 32B
    TBuf<> cqHeadAddrBuf_;  // 32B
    TBuf<> cqTailAddrBuf_;  // 32B
    TBuf<> jfsDoorBellBuf_; // 32B
    TBuf<> jfcDoorBellBuf_; // 32B
    TBuf<> cqeBuf_;         // 32B * CQ_DEPTH_256
    TBuf<> templateSqeBuf_; // 64B
    TBuf<> tokenSqeBuf_;    // 64B * moeExpertNumPerRank，发给同一个对端的WQE批量下发
    TBuf<> statusSqeBuf_;   // 64B * moeExpertNumPerRank，发给同一个对端的WQE批量下发

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR slotWorspaceAddr_{nullptr};
    GM_ADDR localWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t perSlotBytes_{0};
    uint32_t numScaleupRanks_{0};
    uint32_t numScaleoutRanks_{0};
    uint32_t numAivStage1_{0};
    uint32_t numAivStage2_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t actualACnt_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t groupCnt_{0};
    uint32_t hAlignUbSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlignWin_{0};
    uint32_t localExpertSizeAlign_{0};
    uint32_t metaAlignUbSize_{0};
    uint32_t metaOffset_{0};

    // URMA通信变量
    uint32_t startRankId_ {0};
    uint32_t endRankId_ {0};
    uint32_t sendRankNum_ {0};
    uint32_t sqPi_ {0};
    uint32_t sqCi_ {0};
    uint32_t cqPi_ {0};
    uint32_t cqCi_ {0};
    uint32_t sqPiLinear_ {0};
    uint32_t cqCiLinear_ {0};

    DataCopyParams xCopyParams_;
    DataCopyParams topkCopyParams_;
    DataCopyParams countCopyParams_;
    DataCopyParams metaCopyOutParams_;
    DataCopyParams slotCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRankHp, GM_ADDR numRecvPerExpertHp, GM_ADDR dstBufferSlotIdx,
    GM_ADDR recvSrcMetadataInternal, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeEpDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    workspaceGM_ = workspaceGM;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    epRankId_ = mc2Context_->epRankId;

    const auto& info = tilingData->moeEpDispatchInfo;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    perSlotBytes_ = info.cfg.perSlotBytes;
    numScaleupRanks_ = info.numScaleupRanks;
    numScaleoutRanks_ = info.numScaleoutRanks;
    numAivStage1_ = info.numAivStage1;
    numAivStage2_ = info.numAivStage2;
    aivNum_ = info.aivNum;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);

    SplitToCore(axisBS_, startTokenId_, endTokenId_, sendTokenNum_);    // 按Toekn总数分核
    SplitToCore(epWorldSize_, startRankId_, endRankId_, sendRankNum_); // 按卡分核

    // xGMTensor_.SetGlobalBuffer((__gm__ XType*)x[xOffset], sendTokenNum_ * axisH_);
    // topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx[topkOffset], sendTokenNum_ * axisK_);
        // topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights[topkOffset], sendTokenNum_ * axisK_);
    // dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx[topkOffset], sendTokenNum_ * axisK_);
    // // recvSrcMetaGMTensor_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal);
    // // numRecvPerRankHpGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp);
    // // numRecvPerExpertHpGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp);
    // recvSrcMetaGMTensor_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal, epWorldSize_ * axisMaxBS_ * META_INFO);
    // numRecvPerRankHpGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp, epWorldSize_);
    // numRecvPerExpertHpGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp, moeExpertNumPerRank_);

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx);
    recvSrcMetaGMTensor_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal);
    numRecvPerRankHpGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp);
    numRecvPerExpertHpGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp);

    hAlignUbSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    uint32_t metaBytes = kAlignSize_ + 2 * sizeof(int32_t);
    metaAlignUbSize_ = Ceil(metaBytes, UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlignWin_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * _;
    localExpertSizeAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    // tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hAlignUbSize_);
    // tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    // if constexpr (IsCached == NO_CACHED) {
    //     // topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx[topkOffset], sendTokenNum_ * axisK_);
    //     // topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights[topkOffset], sendTokenNum_ * axisK_);
    //     topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    //     topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);
    // } else {
    //     // cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx[topkOffset], sendTokenNum_ * axisK_);
    //     cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
    //     // tpipe_->InitBuffer(cachedSlotIdxBuf_, kAlignSize_);
    // }
    if constexpr (IsCached > NO_CACHED) {
        // cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx[topkOffset], sendTokenNum_ * axisK_);
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
    }

    // tpipe_->InitBuffer(dstSlotIdxBuf_, kAlignSize_);
    // tpipe_->InitBuffer(metaBuf_, kAlignSize_ * 2 + UB_ALIGN);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, epWorldSizeAlign_);
    // dstSlotIdxTensor_ = dstSlotIdxBuf_.Get<int32_t>();
    // metaLocalTensor_ = metaBuf_.Get<int32_t>();
    // metaTensorFloat_ = metaBuf_.Get<float>();
    // numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int32_t>();

    xCopyParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    // countCopyParams_ = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    countCopyParams_ = {1U, static_cast<uint16_t>((1 + moeExpertNumPerRank_) * sizeof(int32_t)), 0U, 0U};
    metaCopyOutParams_ = {1U, static_cast<uint16_t>(metaBytes), 0U, 0U};
    slotCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U}
    padParams_ = {false, 0, 0, 0};

    scaleupCounterAddr_ = workspaceGM;
    expertCounterAddr_ = scaleupCounterAddr_ + AlignUp(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN);
    slotWorspaceAddr_ = expertCounterAddr_ + AlignUp(epWorldSize_ * localExpertSizeAlign_, WIN_ADDR_ALIGN);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_);
    // metaOffset_ = AlignUp(axisH_ * sizeof(XType), WIN_ADDR_ALIGN);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(
    uint32_t totalCnt, uint32_t &startId, uint32_t &endId, uint32_t &perCoreCnt)
{
    perCoreCnt = totalCnt / aivNum_;
    uint32_t remainder = totalCnt % aivNum_;
    startId = perCoreCnt * aivId_;
    if (aivId_ < remainder) {
        perCoreCnt += 1;
        startId += aivId_;
    } else {
        startId += remainder;
    }
    endId = startId + perCoreCnt;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::HcclBarrier()
{
    
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    if (aivId_ == 0) {
        return;
    }
    // 计数器 workspace
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scaleupCounterAddr_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    LocalTensor<int32_t> counterLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
    Duplicate<int32_t>(counterLocalTensor, 0, epWorldSize_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(scaleupCounterGMTensor_, counterLocalTensor, countCopyParams_);

    // if (startRankId_ > epWorldSize_) {
    //     return;
    // }
    // LocalTensor<int32_t> counterLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
    // Duplicate<int32_t>(counterLocalTensor, 0, 1 + moeExpertNumPerRank_);
    // SyncFunc<AscendC::HardEvent::V_MTE3>();
    // for (uint32_t rankId = startRankId_; rankId < endRankId_; ++rankId) {
    //     scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scaleupCounterAddr_ + rankId * counterSize_);
    //     DataCopyPad(scaleupCounterGMTensor_, counterLocalTensor, countCopyParams_);
    // }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t perGroupTokenNum = PER_GROUP_EXPERT_NUM / axisK_;
    uint32_t perGroupSizeAlign = Ceil(perGroupTokenNum * axisK_ * int32_t, UB_ALIGN) * UB_ALIGN;
    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum);
    uint32_t calCnt = perGroupTokenNum * axisK_;
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};

    tpipe_->InitBuffer(topkIdsBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(subExpBuf_, perGroupSizeAlign);
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> tempTensorInt32 = tempBuf_.Get<int32_t>();
    LocalTensor<int32_t> dstTensorInt32 = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subTensorInt32 = subExpBuf_.Get<int32_t>();
    LocalTensor<float> tmpoutFp32 = dstExpBuf_.Get<float>();
    LocalTensor<float> tmpFp32 = subExpBuf_.Get<float>();
    LocalTensor<float> workLocalTensorFp32 = topkIdsBuf_.Get<float>();

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (sendTokenNum_ - group * perGroupTokenNum) * axisK_;
        }
        uint32_t expertIdsAlignSize = Ceil(calCnt * int32_t, UB_ALIGN) * UB_ALIGN;
        DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        
        uint32_t topkIdxOffset = (startTokenId_ + group * perGroupTokenNum) * axisK_;
        DataCopyPad(topkIdsGroupTensor, topkIdsGMTensor_[topkIdxOffset], expertIdsCntParams, expertIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Divs(tempTensorInt32, topkIdsGroupTensor, static_cast<int32_t>(moeExpertNumPerRank_), calCnt);    //  计算本组所有topkId对应dstRankId
        for (uint32_t dstRankId = 0; dstRankId < epWorldSize_; dstRankId++) {
            Subs(subTensorInt32, tempTensorInt32, dstRankId, calCnt);
            PipeBarrier<PIPE_V>();
            Abs(dstTensorInt32, subTensorInt32, calCnt);
            PipeBarrier<PIPE_V>();
            Mins(subTensorInt32, dstTensorInt32, 1, calCnt);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensorFp32, calCnt);
            SyncFunc<AscendC::HardEvent::V_S>();
            int32_t curOtherRankCnt = dstTensorInt32(0);
            if (dstRankId == 0) {
                SyncFunc<AscendC::HardEvent::MTE3_S>(); // 等待计数器reset
            }
            if (calCnt >= curOtherRankCnt) {
                uint32_t curRankCnt = AscendC::AtomicAdd<int32_t>(scaleupCounterAddr_ + dstRankId * sizeof(int32_t), calCnt - curOtherRankCnt);
            }
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetTokenRecvNum()
{
    if (aivId_ = aivNum_ - 1) {  // 最后一个核处理
        GlobalTensor<int32_t> recvWinTensor;
        GM_ADDR localWinAddr = GetWinAddrByRankId(mc2Context_, epRankId_);
        recvWinTensor.SetGlobalBuffer((__gm__ int32_t*)(localWinAddr + epRankId_ * sizeof(int32_t)));
        LocalTensor<int32_t> counterLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
        LocalTensor<float> counterFp32 = numRecvPerRankBuf_.Get<float>();
        DataCopyParams recvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(sizeof(int32_t)),
                                        static_cast<uint16_t>(epWorldSizeAlignWin_ - sizeof(int32_t)), 0U, 0U};
        DataCopyPadParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad<int32_t, PaddingMode::Compact>(counterLocalTensor, recvWinTensor, recvCntParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum<float>(tmpoutFp32, counterFp32, workLocalTensorFp32, epWorldSize_);
        actualACnt_ = dstTensorInt32(0);
        WriteActualAToHostPinned();
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSendDirect(uint32_t srcTokenId)
{
    int32_t  seenDst[axisK_];
    int32_t  seenSlot[axisK_];
    uint32_t nSeen = 0;
    for (uint32_t k = 0; k < axisK_; k++) {
        uint32_t expertId = topkIdxTensor_.GetValue(k);
        if (expertId == -1) {
            dstSlotIdxTensor_.SetValue(k, -1);
            continue;
        }
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
        GM_ADDR dstExpertCounterAddr = scaleupCounterAddr_ + dstRankId * counterSize_ +
                                       (1 + expertId % moeExpertNumPerRank_) * sizeof(int32_t);
        uint32_t expertRecvNum = AscendC::AtomicAdd<int32_t>(dstExpertCounterAddr, 1);
        // 查 seen list
        int32_t hit = -1;
        for (uint32_t j = 0; j < nSeen; ++j) {
            if (seenDst[j] == (int32_t)dstRankId) {
                hit = j;
                break;
            }
        }
        if (hit >= 0) {
            dstSlotIdxTensor_.SetValue(k, seenSlot[hit]);
            continue;
        }

        // atomicAdd 计算 计算本卡发送到dstRankId的token数量  预留槽位  该token放在workspace哪个位置？
        uint32_t slot = AscendC::AtomicAdd(scaleupCounterAddr_ + dstRankId * counterSize_, 1);
        dstSlotIdxTensor_.SetValue(k, slot);
        seenDst[nSeen]  = dstRankId;
        seenSlot[nSeen] = slot;
        nSeen++;

        // 发送该token slot到dstRank
        WriteSlotToRemoteWindow(dstRankId, slot);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteSlotToRemoteWindow(uint32_t dstRankId,
    uint32_t slot)
{
    GlobalTensor<XType> slotGMTensor;
    GM_ADDR rankGM = (dstRankId == epRankId_) ? localWinAddr_ : slotWorspaceAddr_;
    GM_ADDR slotAddr = rankGM + (dstRankId * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor.SetGlobalBuffer((__gm__ XType*)slotAddr);
    DataCopyPad(slotGMTensor, tokenSlotTensor_, slotCopyParams_);
    xQueue_.FreeTensor<XType>(tokenSlotTensor_);
    if (dstRankId != epRankId_) {
        //printf("START WRITR TOKEN");
        GM_ADDR remoteAddr = GetWinAddrByRankId(mc2Context_, dstRankId) + (epRankId_ * axisMaxBS_ + slot) * perSlotBytes_;
        hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteAddr, slotAddr, perSlotBytes_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }

    topkIdxTensor_ = metaBuf_.GetWithOffset<uint32_t>(kAlignSize_ / sizeof(uint32_t), 0);
    topkWeightsTensor_ = metaBuf_.GetWithOffset<float>(kAlignSize_ / sizeof(float), kAlignSize_);
    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, kAlignSize_);
    // tpipe_->InitBuffer(dstSlotIdxBuf_, kAlignSize_);
    // dstSlotIdxTensor_ = dstSlotIdxBuf_.Get<int32_t>();
    uint32_t axisKAlign = kAlignSize_ / sizeof(int32_t);

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        // if (tokenId > startTokenId_) {
        //     SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
        //     SyncFunc<AscendC::HardEvent::MTE3_S>();
        // }
        xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
        metaDataTensor_ = xLocalTensor_[hAlignUbSize_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams_, padParams_);
        DataCopyPad(metaDataTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
        DataCopyPad(metaDataTensor_[axisKAlign].template ReinterpretCast<float>(), topkWeightsGMTensor_[topkOffset],
                    topkCopyParams_, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaDataTensor_.SetValue(2 * axisKAlign epRankId_);
        metaDataTensor_.SetValue(2 * axisKAlign + 1, tokenId);
        perSlotQueue_.EnQue(xLocalTensor_);
        tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
        topkIdxTensor_ = tokenSlotTensor_[hAlignUbSize_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();

        if constexpr (IsCached == NO_CACHED) {
            // SyncFunc<AscendC::HardEvent::MTE2_S>();
            // if (tokenId > startTokenId_) {
            //     SyncFunc<AscendC::HardEvent::MTE3_S>();
            // }
            DedupAndSendDirect(tokenId, topkOffset);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
        } else {
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            for (uint32_t k = 0; k < axisK_; k++) {
                int32_t slot = cachedSlotIdxGMTensor_.GetValue(k);
                if (slot == -1) {
                    continue;
                }
                int32_t expertId = topkIdxTensor_.GetValue(k);
                uint32_t dstRankId = expertId / moeExpertNumPerRank_;
                WriteSlotToRemoteWindow(dstRankId, slot);
            }
        }
        DataCopyPad(dstSlotIdxGMTensor_[tokenId], dstSlotIdxTensor_, topkCopyParams_);
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_, dstRankId++) {
        // 计算目标窗口地址:
        GM_ADDR remoteAddr = (GM_ADDR)(GetWinAddrByRankId(dstRankId) + epRankId_ * epWorldSizeAlignWin_);
        if (dstRankId != epRankId_) {
            //printf("START WRITR TOKEN");
            hcomm_.WriteNbi(GetCommHandle(dstRankId), remoteAddr, scaleupCounterAddr_, epWorldSizeAlign_);
        } else {    // 拷贝本端
            GlobalTensor<int32_t> localWinTensor;
            localWinTensor.SetGlobalBuffer((__gm__ int32_t*)remoteAddr);
            DataCopyPad(counterLocalTensor, scaleupCounterGMTensor_, countCopyParams_, padParams_);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(localWinTensor, counterLocalTensor, countCopyParams_);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::RecvPhase()
{
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        return;
    }

    GlobalTensor<int32_t> recvCounterGMTensor;
    LocalTensor<int32_t> recvCounterTensor;
    GM_ADDR recvCounterAddr = GetWinAddrByRankId(mc2Context_, epRankId_) + startRankId_ * counterSize_;
    recvCounterGMTensor.SetGlobalBuffer((__gm__ int32_t*)recvCounterAddr);
    DataCopyExtParams counterCopyParams = {1, static_cast<uint32_t>(sendRankNum_ * counterSize_), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    tpipe_->InitBuffer(recvCounterBuf_, sendRankNum_ * counterSize_);
    recvCounterTensor = recvCounterBuf_.Get<int32_t>();
    DataCopyPad(recvCounterTensor, recvCounterGMTensor[startRankId_], counterCopyParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<float> recvCounterTensorFloat = recvCounterTensor_.ReinterpretCast<float>();
    LocalTensor<float> sumTensorFloat = sumTensor_.ReinterpretCast<float>();
    const uint32_t shape[] = {sendRankNum_, static_cast<uint32_t>(counterSize_ / sizeof(float))};
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(sumTensorFloat, recvCounterTensorFloat, shape, true);
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t srcRankId = startRankId_; srcRankId < endRankId_; srcRankId++) {
        uint32_t offset = (srcRankId - startRankId_) * counterSize_ / sizeof(int32_t)
        int32_t numRecvFromSrcRank = recvCounterTensor.GetValue(offset);
        numRecvPerRankTensor.SetValue(srcRankId, numRecvFromSrcRank);
        expertRecvTensor = sumTensor_[offset];
        for (uint32_t localExpert = 0; localExpert < moeExpertNumPerRank_; ++localExpert) {
            uint32_t count = expertRecvTensor.GetValue(localExpert + 1);
            uint32_t expertRecvNum = AscendC::AtomicAdd(scaleupCounterAddr_ + localExpert, count);
        }
    }

    GM_ADDR expertRecvNumAddr = scaleupCounterAddr_;

    SyncAll<true>();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteActualAToHostPinned()
{
    GlobalTensor<int32_t> hostPinnedCounterTensor;
    hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int32_t*)hostPinnedCounterAddrGM_);
    hostPinnedCounterTensor.SetValue(0, actualACnt_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(hostPinnedCounterTensor);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessDirect()
{
    ResetCounters();
    SyncAll<true>();
    if constexpr (DoCpuSync > NO_CPU_SYNC) {
        CalSendCnt();
        SyncAll<true>();
        Communication();
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>();
        tpipe_->Reset();
        ResetCounters();
        SetTokenRecvNum();
    }
    SendPhase();
    HcclBarrier();  //TODO
    RecvPhase();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessHybrid()
{
    // const uint32_t aiv = aivId_;
    // const uint32_t s1End = numAivStage1_;
    // const uint32_t s2End = s1End + numAivStage2_;

    // if (aiv < s1End) {
    //     if constexpr (IsCached == NO_CACHED) {
    //         SendPhaseScaleoutFirstTime();
    //     } else {
    //         SendPhaseScaleoutCached();
    //     }
    // } else if (aiv < s2End) {
    //     SendPhaseScaleupFanout();
    // } else {
    //     RecvPhase();
    // }

    // SyncAll<true>();
    // HcclBarrier();

    // if constexpr (DoCpuSync > NO_CPU_SYNC) {
    //     if (aivId_ == 0) {
    //         WriteActualAToHostPinned();
    //     }
    // }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    // if constexpr (NetworkMode == NETWORK_DIRECT) {
    //     ProcessDirect();
    // } else {
    //     ProcessHybrid();
    // }
    if ASCEND_IS_AIV { // 全aiv处理
        ProcessDirect();
    }
    AscendC::printf("End MoeEpDispatch kernel Process\n");
}

}  // namespace MoeEpDispatchImpl

#endif  // MOE_EP_DISPATCH_H