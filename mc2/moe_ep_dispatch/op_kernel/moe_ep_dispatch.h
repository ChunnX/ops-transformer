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
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
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

constexpr uint32_t MAX_TOPK = 32U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t NETWORK_DIRECT = 0U; 
constexpr uint32_t NETWORK_HYBRID = 1U; 
constexpr uint32_t META_INFO = 32U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;   // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;   // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_TOKEN_NUM = 256U;
constexpr uint32_t PER_GROUP_EXPERT_NUM = 12800U;
constexpr uint32_t COUNTER_STRIDE = 2U;    // counter前两个数为 state、dstRankRecvNum
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
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void SetTokenRecvNumAndClear();
    __aicore__ inline void DedupAndSendDirect(uint32_t srcTokenId);
    __aicore__ inline void WriteSlotToLocal(uint32_t dstRankId, uint32_t slot);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void SendPhase();
    __aicore__ inline void RecvPhase();
    __aicore__ inline void WriteActualAToHostPinned(LocalTensor<int32_t> tempTensor);
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
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<XType> slotGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<float> topkWeightsTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> metaTensorInt32_;
    LocalTensor<float> metaTensorFloat_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> statusTensor_;
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
    TBuf<> numRecvBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;
    TBuf<> recvCounterBuf_;
    TBuf<> counterBuf_;
    TBuf<> recvTempBuf_;
    TBuf<> numRecvBuf;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR slotWorspaceAddr_{nullptr};
    GM_ADDR slotLocalAddr_{nullptr};
    GM_ADDR localStateWinAddr_{nullptr};
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
    uint32_t dataState_{0};
    uint32_t actualACnt_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t groupCnt_{0};
    uint32_t hAlignUbSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t perSlotBytesAlign_{0};
    uint32_t counterLenPerRank_{0};
    uint32_t counterSizeAlign_{0};
    uint32_t counterCnt_{0};
    uint32_t metaAlignUbSize_{0};
    uint32_t metaOffset_{0};

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
    counterLenPerRank_ = moeExpertNumPerRank_ + COUNTER_STRIDE;

    SplitToCore(axisBS_, startTokenId_, endTokenId_, sendTokenNum_);    // 按Toekn总数分核
    SplitToCore(epWorldSize_, startRankId_, endRankId_, rankNumPerCore_); // 按卡分核

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx);
    recvSrcMetaGMTensor_.SetGlobalBuffer((__gm__ int32_t*)recvSrcMetadataInternal);
    numRecvPerRankHpGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRankHp);
    numRecvPerExpertHpGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpertHp);

    hAlignUbSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    uint32_t metaBytes = kAlignSize_ + 2 * sizeof(int32_t);
    metaAlignUbSize_ = Ceil(metaBytes, UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    counterSizeAlign_ = Ceil(counterLenPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    counterCnt_ = counterSizeAlign_ / sizeof(int32_t);
    perSlotBytesAlign_ = Ceil(perSlotBytes_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeNumPerRankAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    // tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hAlignUbSize_);
    // tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);

    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
    }

    tpipe_->InitBuffer(counterBuf_, counterSizeAlign_);
    // tpipe_->InitBuffer(dstSlotIdxBuf_, kAlignSize_);
    // tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);

    // dstSlotIdxTensor_ = dstSlotIdxBuf_.Get<int32_t>();
    // metaLocalTensor_ = metaBuf_.Get<int32_t>();
    // metaTensorFloat_ = metaBuf_.Get<float>();
    // numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();

    xCopyParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    // countCopyParams_ = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    countCopyParams_ = {1U, static_cast<uint16_t>(counterLenPerRank_ * sizeof(int32_t)), 0U, 0U};
    metaCopyOutParams_ = {1U, static_cast<uint16_t>(metaBytes), 0U, 0U};
    slotCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    padParams_ = {false, 0, 0, 0};

    dataState_ = 0;
    scaleupCounterAddr_ = workspaceGM;
    slotWorspaceAddr_ = scaleupCounterAddr_ + AlignUp(epWorldSize_ * counterSizeAlign_, WIN_ADDR_ALIGN);
    localStateWinAddr_ = GetStateAddrByRankId(mc2Context_, epRankId_, dataState_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scaleupCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)localStateWinAddr_);
    slotGMTensor_.SetGlobalBuffer((__gm__ XType*)slotWorspaceAddr_);
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    // if (startRankId_ >= epWorldSize_) {
    //     return;
    // }
    if (aivId_ != 0) {
        return;
    }
    // workspace 计数器重置
    LocalTensor<int32_t> counterLocalTensor = counterBuf_.Get<int32_t>();
    Duplicate<int32_t>(counterLocalTensor, 0, counterCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
    counterLocalTensor.SetValue(0, 1);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyParams clearCopyParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(counterSizeAlign_), 0U, 0U};
    // DataCopyParams clearCopyParams = {static_cast<uint16_t>(rankNumPerCore_), static_cast<uint16_t>(counterSizeAlign_), 0U, 0U};
    DataCopyPad(scaleupCounterGMTensor_, counterLocalTensor, clearCopyParams);
    // for (uint32_t rankId = startRankId_; rankId < endRankId_, rankId++) {
    //     DataCopy(scaleupCounterGMTensor_[rankId * counterCnt_], counterLocalTensor, counterCnt_);
    // }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }

    uint32_t perGroupTokenNum = PER_GROUP_EXPERT_NUM / axisK_;
    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum);
    uint32_t calCnt = perGroupTokenNum * axisK_;
    uint32_t perGroupSizeAlign = Ceil(calCnt * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

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
    Duplicate<int32_t>(subTensorInt32, static_cast<int32_t>(moeExpertNumPerRank_), calCnt);

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (sendTokenNum_ - group * perGroupTokenNum) * axisK_;
        }
        uint32_t topkIdsAlignSize = Ceil(calCnt * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        uint32_t topkIdxOffset = (startTokenId_ + group * perGroupTokenNum) * axisK_;
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams, topkIdsCntCopyPadParams); // copy topkId
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        Div(tempTensorInt32, topkIdsGroupTensor, subTensorInt32, calCnt);    //  计算本组所有topkId对应dstRankId
        PipeBarrier<PIPE_V>();
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
                int32_t dstOffset = dstRankId * counterSizeAlign_ + sizeof(int32_t);
                int32_t curRankCnt = AscendC::AtomicAdd<int32_t>(
                    reinterpret_cast<__gm__ int32_t*>(scaleupCounterAddr_ + dstOffset), calCnt - curOtherRankCnt);
            }
        }
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

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = (GM_ADDR)(GetStateAddrByRankId(mc2Context_, dstRankId, dataState_) + epRankId_ * counterSizeAlign_);
        if (likely(dstRankId != epRankId_)) {
            //printf("START WRITR TOKEN");
            hcomm_.WriteNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteStateAddr,
                            scaleupCounterAddr_ + dstRankId * counterSizeAlign_, UB_ALIGN);
            hcomm_.Drain(GetCommHandle(mc2Context_, epRankId_, dstRankId));
        } else {    // 本端
            LocalTensor<int32_t> counterLocalTensor = counterBuf_.GetWithOffset<int32_t>(UB_STRIDE, 0);
            DataCopy(counterLocalTensor, scaleupCounterGMTensor_[dstRankId * counterCnt_], UB_STRIDE);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopy(recvCounterGMTensor_[dstRankId * counterCnt_], counterLocalTensor, UB_STRIDE);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetTokenRecvNumAndClear()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    // 最后一个核处理recvCount 复位计数器
    uint32_t maskState = 1;
    uint32_t maskCnt = 2;  // 每个datablock处理第2个元素
    int32_t sumOfFlag = -1;
    tpipe_->InitBuffer(recvTempBuf_, epWorldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(numRecvBuf_, UB_ALIGN);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    LocalTensor<int32_t> recvCounterTensor = recvTempBuf_.Get<int32_t>();
    LocalTensor<int32_t> numRecvTensor = numRecvBuf_.Get<int32_t>();
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvTensorFp32 = numRecvTensor.template ReinterpretCast<float>();
    LocalTensor<float> tempTensorFp32 = tempBuf_.Get<float>();
    // DataCopyParams recvCntCopyParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(COUNTER_STRIDE * sizeof(int32_t)),
    //                                     static_cast<uint16_t>(counterSizeAlign_ - COUNTER_STRIDE * sizeof(int32_t)), 0U};
    DataCopyParams clearCopyParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                      static_cast<uint16_t>(counterSizeAlign_ - UB_ALIGN)};
    DataCopyParams clearCounterParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(sizeof(int32_t)), 0U,
                                        static_cast<uint16_t>(counterSizeAlign_ - sizeof(int32_t))};
    DataCopyParams recvCntCopyParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN),
                                        static_cast<uint16_t>(counterSizeAlign_ - UB_ALIGN), 0U};
    while(sumOfFlag != epWorldSize_) {  //状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, recvCntCopyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        // ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(numRecvTensor, recvCounterTensor, shape);
        ReduceSum(numRecvTensorFp32, recvCounterTensorFp32, tempTensorFp32, maskState, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvTensor.GetValue(0);
    }

    // 本端接收actualA计算  写入host pin
    ReduceSum(numRecvTensorFp32, recvCounterTensorFp32, tempTensorFp32, maskCnt, epWorldSize_, 1);
    SyncFunc<AscendC::HardEvent::V_S>();
    actualACnt_ = numRecvTensor.GetValue(0);
    WriteActualAToHostPinned(numRecvTensor);

    // 计数器复位
    Duplicate<int32_t>(recvCounterTensor, 0, epWorldSizeAlign_ * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearCopyParams);   // 复位Window 计数器
    DataCopy(scaleupCounterGMTensor_[1], recvCounterTensor, clearCounterParams);   // 复位Workspace 计数器
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSendDirect(uint32_t srcTokenId)
{
    int32_t  seenDst[MAX_TOPK];
    int32_t  seenSlot[MAX_TOPK];
    uint32_t nSeen = 0;
    for (uint32_t k = 0; k < axisK_; k++) {
        uint32_t expertId = topkIdxTensor_.GetValue(k);
        if (expertId == -1) {
            dstSlotIdxTensor_.SetValue(k, -1);
            continue;
        }
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
        uint32_t dstExpertOffset = (COUNTER_STRIDE + expertId % moeExpertNumPerRank_) * sizeof(int32_t);
        __gm__ int32_t* dstCounterBaseAddr = reinterpret_cast<__gm__ int32_t*>(scaleupCounterAddr_ + dstRankId * counterSizeAlign_);
        uint32_t sendNum = AscendC::AtomicAdd<int32_t>(dstCounterBaseAddr + dstExpertOffset, 1);
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
        uint32_t slot = AscendC::AtomicAdd(dstCounterBaseAddr + sizeof(int32_t), 1);
        dstSlotIdxTensor_.SetValue(k, slot);
        seenDst[nSeen]  = dstRankId;
        seenSlot[nSeen] = slot;
        nSeen++;

        // 发送该token slot到本端workspace dstRank
        WriteSlotToLocal(dstRankId, slot);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteSlotToLocal(uint32_t dstRankId,
    uint32_t slot)
{
    GlobalTensor<XType> slotGMTensor;
    GM_ADDR rankGM = (dstRankId == epRankId_) ? localWinAddr_ : slotWorspaceAddr_;
    GM_ADDR slotAddr = rankGM + (dstRankId * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor.SetGlobalBuffer((__gm__ XType*)slotAddr);
    DataCopyPad(slotGMTensor, tokenSlotTensor_, slotCopyParams_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(recvTempBuf_, rankNumPerCore_ * UB_ALIGN);
    LocalTensor<int32_t> sendNumPerRankTensor = recvTempBuf_.Get<int32_t>();
    DataCopyParams recvCntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), static_cast<uint16_t>(UB_ALIGN),
                                        static_cast<uint16_t>(counterSizeAlign_ - UB_ALIGN), 0U};
    DataCopy(sendNumPerRankTensor, scaleupCounterGMTensor_[startRankId_ * counterCnt_], recvCntCopyParams);
    // DataCopyParams recvCntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), static_cast<uint16_t>(sizeof(int32_t)),
    //                                     static_cast<uint16_t>(counterSizeAlign_ - sizeof(int32_t)), 0U};
    // DataCopyPad(sendNumPerRankTensor, scaleupCounterGMTensor_[startRankId_ * counterCnt_ + 1], recvCntCopyParams, padParams_);
    uint32_t tokenNumPerGroup = 100;  // TODO
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        if (unlikely(dstRankId == epRankId_)) {
            continue;   //  本端已经写入win
        }

        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - startRankId_) * UB_STRIDE + 1);
        uint32_t loopTimes = Ceil(sendTokenNum, tokenNumPerGroup);    // 分组多token发送到对端
        uint32_t lastTokenNum = sendTokenNum - tokenNumPerGroup * (loopTimes - 1);
        // 计算目标窗口地址:
        uint32_t dstRankOffset = dstRankId * axisMaxBS_ * perSlotBytes_;
        GM_ADDR remoteWinAddr = (GM_ADDR)(GetWinAddrByRankId(mc2Context_, dstRankId) + dstRankOffset);
        GM_ADDR remoteStateAddr = (GM_ADDR)(GetStateAddrByRankId(mc2Context_, dstRankId, dataState_) + epRankId_ * counterSizeAlign_);
        uint64_t commHandle = GetCommHandle(mc2Context_, epRankId_, dstRankId);
        // 发送count
        hcomm_.WriteNbi(commHandle, remoteStateAddr, scaleupCounterAddr_ + dstRankId * counterSizeAlign_, counterSizeAlign_);
        hcomm_.Drain(commHandle);

        // 发送slot
        for (uint32_t loop = 0; loop < loopTimes - 1; loop++) {
            //printf("START WRITR TOKEN");
            hcomm_.WriteNbi(commHandle, remoteWinAddr, slotWorspaceAddr_ + dstRankOffset, perSlotBytes_ * tokenNumPerGroup);
            remoteWinAddr += perSlotBytes_ * tokenNumPerGroup;
            hcomm_.Drain(commHandle);
        }
        hcomm_.WriteNbi(commHandle, remoteWinAddr, slotWorspaceAddr_ + dstRankOffset, perSlotBytes_ * lastTokenNum);

        // hcomm_.Drain(commHandle);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    if (startTokenId_ >= axisBS_) {
        return;
    }

    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, 1, kAlignSize_);
    // tpipe_->InitBuffer(dstSlotIdxBuf_, kAlignSize_);
    // dstSlotIdxTensor_ = dstSlotIdxBuf_.Get<int32_t>();

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
        metaLocalTensor_ = xLocalTensor_[hAlignUbSize_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams_, padParams_);
        DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
        DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(), topkWeightsGMTensor_[topkOffset],
                    topkCopyParams_, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
        metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
        perSlotQueue_.EnQue(xLocalTensor_);
        tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
        topkIdxTensor_ = tokenSlotTensor_[hAlignUbSize_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        slotLocalAddr_ = (__gm__ uint8_t*)(slotWorspaceAddr_ + tokenId * perSlotBytesAlign_);
        slotGMTensor_.SetGlobalBuffer((__gm__ XType*)(slotLocalAddr_));
        DataCopyPad(slotGMTensor_, tokenSlotTensor_, slotCopyParams_);

        dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();

        if constexpr (!IsCached) {
            // SyncFunc<AscendC::HardEvent::MTE2_S>();
            // if (tokenId > startTokenId_) {
            //     SyncFunc<AscendC::HardEvent::MTE3_S>();
            // }
            DedupAndSendDirect(tokenId);
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
                WriteSlotToLocal(dstRankId, slot);
            }
        }
        DataCopyPad(dstSlotIdxGMTensor_[tokenId], dstSlotIdxTensor_, topkCopyParams_);
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::RecvPhase()
{
    if (aivId_ != aivNum_ - 1) { // 空闲核，直接返回
        return;
    }

    DataCopyParams inCopyParams = {static_cast<uint16_t>(epWorldSize_), 1,
                                   static_cast<uint16_t>(counterSizeAlign_ / UB_ALIGN - 1), 0U};
    // DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyParams recvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(sizeof(int32_t)),
                                    static_cast<uint16_t>(counterSizeAlign_ - sizeof(int32_t)), 0U};
    // DataCopyPadParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    tpipe_->InitBuffer(recvCounterBuf_, epWorldSize_ * counterSizeAlign_);
    tpipe_->InitBuffer(numRecvBuf_, counterSizeAlign_);
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, moeNumPerRankAlign_);
    LocalTensor<int32_t> recvCounterTensor = recvCounterBuf_.Get<int32_t>();
    LocalTensor<int32_t> numRecvTensor = numRecvBuf_.Get<int32_t>();
    LocalTensor<int32_t> numRecvPerRankTensor = recvCounterBuf_.Get<int32_t>();
    LocalTensor<int32_t> numRecvPerExpertTensorInt32 = numRecvPerExpertBuf_.Get<int32_t>();
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvTensorFp32 = numRecvTensor.template ReinterpretCast<float>();
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    int32_t sumOfFlag = -1;
    uint32_t mask = 1;
    const uint32_t shape[] = {epWorldSize_, static_cast<uint32_t>(moeNumPerRankAlign_ / sizeof(int32_t))};
    while(sumOfFlag != epWorldSize_) {
        DataCopy(recvCounterTensor, recvCounterGMTensor_, inCopyParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum<float>(numRecvPerRankTensorFp32, recvCounterTensorFp32, numRecvTensorFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor.GetValue(0);
    }

    // copy out numRecv
    DataCopyParams expertRecvCntParams = {static_cast<uint16_t>(epWorldSize_),
        static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)),
        static_cast<uint16_t>(counterSizeAlign_ - moeExpertNumPerRank_ * sizeof(int32_t)), 0U};
    DataCopyParams numPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U, 0U};
    DataCopyParams numPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopyPad<int32_t, PaddingMode::Compact>(numRecvPerRankTensor, recvCounterGMTensor_, recvCntParams, padParams_);
    DataCopyPad(recvCounterTensor, recvCounterGMTensor_[2], expertRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
    DataCopyPad(numRecvPerRankHpGMTensor_, numRecvPerRankTensor, numPerRankParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(numRecvTensor, recvCounterTensor, shape, true);
    PipeBarrier<PIPE_V>();
    Cast(numRecvPerExpertTensor_, numRecvTensor, RoundMode::CAST_NONE, moeExpertNumPerRank_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertHpGMTensor_, numRecvPerExpertTensor_, numPerExpertParams);
    
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteActualAToHostPinned(
    LocalTensor<int32_t> tempTensor)
{
    GlobalTensor<int32_t> hostPinnedCounterTensor;
    hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int32_t*)hostPinnedCounterAddrGM_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(hostPinnedCounterTensor, tempTensor, UB_STRIDE);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessDirect()
{
    ResetCounters();
    SyncAll<true>();
    if constexpr (DoCpuSync) {
        CalSendCnt();
        SyncAll<true>();
        Communication();
        SetTokenRecvNumAndClear();
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>();
        tpipe_->Reset();
    }
    SendPhase();
    SyncAll<true>();
    WriteToRemoteWindow();
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
        // ProcessDirect();
        if (aivId_ == 0) {
            tpipe_->InitBuffer(numRecvBuf_, UB_ALIGN);
            LocalTensor<int32_t> numRecvTensor = numRecvBuf_.Get<int32_t>();
            numRecvTensor.SetValue(0, 10);
            WriteActualAToHostPinned(numRecvTensor);
        }
        AscendC::printf("End MoeEpDispatch kernel Process\n");
    }
}

}  // namespace MoeEpDispatchImpl

#endif  // MOE_EP_DISPATCH_H