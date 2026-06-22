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

#define TemplateMoeEpDispatchTypeClass typename XType, typename ScalesType, uint32_t DoCpuSync, uint32_t IsCached, uint8_t NetworkMode
#define TemplateMoeEpDispatchTypeFunc XType, ScalesType, DoCpuSync, IsCached, NetworkMode

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t MAX_TOPK = 32U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t ALIGNED_LEN_128 = 128U;
constexpr uint32_t NETWORK_DIRECT = 0U; 
constexpr uint32_t NETWORK_HYBRID = 1U; 
constexpr uint32_t MAX_DTYPE_SIZE = 2U;     //  sizeof(bf16)
constexpr uint32_t META_INFO = 32U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;   // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;   // UB_ALIGN/sizeof(int32_t)=8
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_TOKEN_NUM = 256U;
constexpr uint32_t PER_GROUP_EXPERT_NUM = 12800U; // bs*k
constexpr uint32_t COUNTER_STRIDE = 2U;    // counter前两个数为 state、dstRankRecvNum
constexpr uint8_t BUFFER_NUM = 2;

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights,
        GM_ADDR scales, GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
        GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, TPipe *pipe,
        const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessDirect();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCntPerRank(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                             uint32_t startRankId, uint32_t endRankId);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                               uint32_t startExpertId, uint32_t endExpertId);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void DedupAndSendDirect(uint32_t srcTokenId);
    __aicore__ inline void WriteSlotToLocal(uint32_t dstRankId, uint32_t slot);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void SendPhase();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId,
                                       uint32_t &endId, uint32_t &sendNum, bool isFront);

    TPipe *tpipe_{nullptr};
    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_; // todo: 通信上下文

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> topkIdxGMTensor_;
    GlobalTensor<float> topkWeightsGMTensor_;
    GlobalTensor<int32_t> dstSlotIdxGMTensor_;
    GlobalTensor<int32_t> numRecvPerRankGMTensor_;
    GlobalTensor<int64_t> numRecvPerExpertGMTensor_;
    GlobalTensor<int32_t> cachedSlotIdxGMTensor_;
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntGMTensor_;
    GlobalTensor<XType> slotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

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
    LocalTensor<int32_t> sendCntTensor_;
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
    TBuf<> recvCntBuf_;
    TBuf<> recvTempBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR slotWorkspaceAddr_{nullptr};
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
    uint32_t moeExpertNum_{0};
    uint32_t numScaleupRanks_{0};
    uint32_t numScaleoutRanks_{0};
    uint32_t numAivStage1_{0};
    uint32_t numAivStage2_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t dataState_{0};
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
    uint32_t scalesAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t perSlotBytesAlign_{0};
    uint32_t counterLenPerRank_{0};
    uint32_t cntPerRankSizeAlign_{0};
    uint32_t cntPerRankSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t metaAlignUbSize_{0};
    uint32_t metaOffset_{0};
    uint32_t calRankCoreNum_{0};
    uint32_t calExpertCoreNum_{0};
    int64_t actualACnt_{0};

    DataCopyParams xCopyParams_;
    DataCopyParams topkCopyParams_;
    DataCopyParams countCopyParams_;
    DataCopyParams metaCopyOutParams_;
    DataCopyParams slotCopyParams_;
    DataCopyParams scalesCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
    GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
    GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, TPipe *pipe,
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
    moeExpertNum_ = moeExpertNumPerRank_ * epWorldSize_;

    uint32_t numPerCore = (moeExpertNum_ + epWorldSize_) / aivNum_;
    calRankCoreNum_ =  Ceil(epWorldSize_, numPerCore);
    calExpertCoreNum_ = aivNum_ - calRankCoreNum_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);   // TODO
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx);

    hAlignUbSize_ = Ceil(axisH_ * MAX_DTYPE_SIZE, UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    // uint32_t metaBytes = kAlignSize_ + 2 * sizeof(int32_t);
    // metaAlignUbSize_ = Ceil(metaBytes, UB_ALIGN) * UB_ALIGN;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    cntPerRankSizeAlign_ = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // perRank state+count
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    cntPerRankSizeAlign512_ = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    perSlotBytesAlign_ = Ceil(perSlotBytes_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeNumPerRankAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    // tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hAlignUbSize_);
    // tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);

    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();

    if constexpr (Std::IsSame<ScalesType, float>::value) {
        scalesAlign_ = Ceil(axisH_, ALIGNED_LEN_128);
    } else if constexpr (Std::IsSame<ScalesType, fp8_e8m0_t>::value) {
        scalesAlign_ = Ceil(axisH_, UB_ALIGN);
    }
    xCopyParams_ = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    topkCopyParams_ = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    scalesCopyParams_ = {1U, static_cast<uint16_t>(scalesAlign_ * sizeof(ScalesType)), 0U, 0U};
    // countCopyParams_ = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    countCopyParams_ = {1U, static_cast<uint16_t>(counterLenPerRank_ * sizeof(int32_t)), 0U, 0U};
    // metaCopyOutParams_ = {1U, static_cast<uint16_t>(metaBytes), 0U, 0U};
    slotCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    padParams_ = {true, 0, 0, 0};

    dataState_ = 0;
    scaleupCounterAddr_ = workspaceGM;
    // slotWorkspaceAddr_ = scaleupCounterAddr_ + AlignUp(epWorldSize_ * cntPerRankSizeAlign_, WIN_ADDR_ALIGN);
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + epWorldSizeAlign512_;
    slotWorkspaceAddr_ = sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_ + moeExpertNumAlign512_;
    localStateWinAddr_ = GetStateAddrByRankId(mc2Context_, epRankId_, dataState_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scaleupCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)localStateWinAddr_);
    slotGMTensor_.SetGlobalBuffer((__gm__ XType*)slotWorkspaceAddr_);
    sendCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sendCntWorkspaceAddr_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId,
    uint32_t &endId, uint32_t &sendNum, bool isFront)
{
    sendNum = curSendCnt / curUseAivNum;                // 每个aiv需要发送的数量
    uint32_t remainderNum = curSendCnt % curUseAivNum;  // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - calExpertCoreNum_;  // 由于是后面的核做perRank计算，因此需要换算
    }
    startId = sendNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendNum += 1;
        startId += newAivId;
    } else {
        startId += remainderNum;
    }
    endId = startId + sendNum;
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }
    // workspace 计数器清0
    LocalTensor<int32_t> counterLocalTensor = numRecvPerRankBuf_.Get<int32_t>();
    Duplicate<int32_t>(counterLocalTensor, 0, counterCnt_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // DataCopyParams clearCopyParams = {1U, static_cast<uint16_t>(epWorldSizeAlign_), 0U, 0U};
    // DataCopyPad(scaleupCounterGMTensor_, counterLocalTensor, clearCopyParams);    // workspace
    DataCopy(scaleupCounterGMTensor_, counterLocalTensor, counterCnt_);
    // DataCopyPad(recvCounterGMTensor_, counterLocalTensor, clearCopyParams);    // window buffer
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t startRankId, uint32_t endRankId)
{
    if (startRankId > epWorldSize_) {
        return;
    }

    LocalTensor<int16_t> subTensorInt16 = subExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.Get<int16_t>();
    LocalTensor<int16_t> invalidMaskInt16 = tempBuf_.Get<int16_t>();
    LocalTensor<half> tempTensor = topkIdsBuf_.Get<half>();
    LocalTensor<half> subTensorHalf = subExpBuf_.Get<half>();
    LocalTensor<half> tmpHalf = subTensorHalf[Ceil(calCnt / axisK_, UB_STRIDE)];
    LocalTensor<uint8_t> gatherMaskTensorInt8 = topkIdsBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.Get<uint16_t>();
    uint32_t mask = calCnt / axisK_;
    uint32_t countAlign = Ceil(calCnt * sizeof(int16_t), UB_ALIGN) * UB_ALIGN / sizeof(int16_t);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half);
    SumParams axisKSumParams{calCnt / axisK_, innerAlign, axisK_};

    Duplicate<int16_t>(subTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    Div(dstTensorInt16, expertIdsTensor, subTensorInt16, calCnt);
    // 筛选无效expert id
    Mins(tempTensorInt16, expertIdsTensor, static_cast<int16_t>(0), calCnt); // 有效->0 无效->负数
    Abs(subTensorInt16, tempTensorInt16, calCnt);
    Mins(invalidMaskInt16, subTensorInt16, static_cast<int16_t>(1), calCnt);

    for (uint32_t dstRankId = startRankId; dstRankId < endRankId; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        Subs(subTensorInt16, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        Abs(tempTensorInt16, subTensorInt16, calCnt);
        Mins(subTensorInt16, tempTensorInt16, static_cast<int16_t>(1), calCnt); // 目标 0 非目标 1
        Max(tempTensorInt16, subTensorInt16, invalidMaskInt16, calCnt);  // 去除无效值影响
        Cast(tempTensor, tempTensorInt16, RoundMode::CAST_NONE, calCnt);
        Sum(subTensorHalf, tempTensor, axisKSumParams);
        Duplicate<uint16_t>(maskTensorInt16, 0, countAlign);    // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, subTensorHalf, static_cast<half>(axisK_), AscendC::CMPMODE::LT, calCnt / axisK_);
        GatherMask(tmpHalf, subTensorHalf, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        uint32_t offset = (dstRankId - startRankId) * COUNTER_STRIDE + 1;
        printf("dstRankId %d, count %d, rsvdCnt %d, epRankId %d, aivId %d\n", dstRankId, sendCntTensor_.GetValue(offset), rsvdCnt, epRankId_, aivId_);
        int32_t curRankCnt = sendCntTensor_.GetValue(offset) + static_cast<int32_t>(rsvdCnt);
        sendCntTensor_.SetValue(offset, curRankCnt);
        // sendCntTensor_.SetValue(offset - 1, 1);   // 状态位
        printf("dstRankId %d, count %d, rsvdCnt %d, epRankId %d, aivId %d\n", dstRankId, curRankCnt, rsvdCnt, epRankId_, aivId_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t startExpertId, uint32_t endExpertId)
{
    if (startExpertId > moeExpertNum_) {
        return;
    }

    LocalTensor<uint8_t> gatherMaskTensorInt8 = subExpBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> maskTensorInt16 = subExpBuf_.Get<uint16_t>();
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    uint32_t mask = calCnt;
    uint32_t countAlign = Ceil(calCnt * sizeof(int16_t), UB_ALIGN) * UB_ALIGN / sizeof(int16_t);

    for (uint32_t dstExpertId = startExpertId; dstExpertId < endExpertId; dstExpertId++) {
        uint64_t rsvdCnt = 0;
        Duplicate<uint16_t>(maskTensorInt16, 0, countAlign);    // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(dstExpertId), AscendC::CMPMODE::EQ, calCnt);
        GatherMask(dstTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        int32_t curExpertCnt = sendCntTensor_.GetValue(dstExpertId - startExpertId);
        sendCntTensor_.SetValue(dstExpertId - startExpertId, curExpertCnt + static_cast<int32_t>(rsvdCnt));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    uint32_t startId = 0;
    uint32_t endId = 0;
    uint32_t numPerCore = 0;
    uint32_t perGroupTokenNum = 2 * PER_GROUP_EXPERT_NUM / axisK_;
    uint32_t groupCnt = Ceil(axisBS_, perGroupTokenNum);
    uint32_t calCnt = perGroupTokenNum * axisK_;
    uint32_t perGroupSizeAlign = Ceil(calCnt * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint32_t copyInCountInt32 = perGroupSizeAlign / sizeof(int32_t);
    uint32_t cntPerRankSizeAlign256 = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    uint32_t maxSizePerCore = moeExpertNumAlign_ > cntPerRankSizeAlign256 ? moeExpertNumAlign_ : cntPerRankSizeAlign256;
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};

    tpipe_->InitBuffer(topkIdsBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(subExpBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign);
    tpipe_->InitBuffer(numRecvBuf_, maxSizePerCore);
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> dstExpertTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    sendCntTensor_ = numRecvBuf_.Get<int32_t>();
    Duplicate<int32_t>(sendCntTensor_, 0, maxSizePerCore / sizeof(int32_t));

    if (aivId_ < calExpertCoreNum_) {  // 前面的核做per expert count计算
        SplitToCore(moeExpertNum_, calExpertCoreNum_, startId, endId, numPerCore, true);
    } else {    // 后面的核做per rank count计算
        SplitToCore(epWorldSize_, calRankCoreNum_, startId, endId, numPerCore, false);
        // 两两一组，每组第一个位置填充1，作为状态位
        uint64_t mask[2] = { 0x5555555555555555, 0 };
        Duplicate<int32_t>(sendCntTensor_, 1, mask, maxSizePerCore / 64, 1, 8);
        // LocalTensor<int64_t> sendCntTensorInt64 = sendCntTensor_.template ReinterpretCast<int64_t>();
        // Duplicate<int64_t>(sendCntTensorInt64, static_cast<int64_t>(1), cntPerRankSizeAlign256 / sizeof(int64_t));
    }

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            calCnt = (axisBS_ - group * perGroupTokenNum) * axisK_;
        }
        if (group > 0) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();
        }
        uint32_t copyInCountFirst = calCnt > copyInCountInt32 ? copyInCountInt32 : calCnt;
        uint32_t copyInCountSecond = calCnt > copyInCountInt32 ? (calCnt - copyInCountInt32) : 0;
        uint32_t topkIdxOffset = group * perGroupTokenNum * axisK_;
        DataCopyExtParams topkIdsCntParamsFirst = {1U, static_cast<uint32_t>(copyInCountFirst * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParamsFirst, topkIdsCntCopyPadParams); // copy topkId first
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, copyInCountFirst);
        if(copyInCountSecond > 0) {
            DataCopyExtParams topkIdsCntParamsSecond = {1U, static_cast<uint32_t>(copyInCountSecond * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(dstExpertTensor, topkIdxGMTensor_[topkIdxOffset + copyInCountFirst], topkIdsCntParamsSecond, topkIdsCntCopyPadParams); // copy topkId
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(tempTensorInt16[copyInCountFirst], dstExpertTensor, RoundMode::CAST_NONE, copyInCountSecond);
        }

        if (aivId_ < calExpertCoreNum_) {   // 前面的核统计发送给各专家的token count
            // CalSendCntPerExpert(tempTensorInt16, calCnt, startExpertId, endExpertId);
            CalSendCntPerExpert(tempTensorInt16, calCnt, startId, endId);
            // continue;
        } else {    // 后面的核统计发送到各卡token count
            // CalSendCntPerRank(tempTensorInt16, calCnt, startRankId, endRankId);
            CalSendCntPerRank(tempTensorInt16, calCnt, startId, endId);
        }
    }

    uint32_t gmOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + startId;
    uint32_t blockLen = numPerCore * sizeof(int32_t);
    if (aivId_ >= calExpertCoreNum_) {
        gmOffset = startId * COUNTER_STRIDE;
        blockLen = numPerCore * sizeof(int32_t) * COUNTER_STRIDE;
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyParams cntCopyParams = {1U, static_cast<uint16_t>(blockLen), 0U, 0U};
    DataCopyPad(sendCntGMTensor_[gmOffset], sendCntTensor_, cntCopyParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_, true); // 按卡分核
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    GlobalTensor<uint64_t> numSendGMTensorInt64;
    numSendGMTensorInt64.SetGlobalBuffer((__gm__ uint64_t*)sendCntWorkspaceAddr_);
    LocalTensor<uint64_t> sendCntPerRankInt64 = numRecvBuf_.Get<uint64_t>();
    DataCopyParams cntCopyParams = {1U, static_cast<uint16_t>(rankNumPerCore_ * sizeof(uint64_t)), 0U, 0U};
    DataCopyPad(sendCntPerRankInt64, numSendGMTensorInt64[startRankId_], cntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>();   // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint64_t notifyVal = sendCntPerRankInt64.GetValue(dstRankId - startRankId_);
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetStateAddrByRankId(mc2Context_, dstRankId, dataState_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * sizeof(uint64_t);
        GM_ADDR remoteCountAddr = remoteStateAddr + cntPerRankSizeAlign512_ + epRankId_ * moeExpertNumPerRank_ * sizeof(int32_t);
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_ + dstRankId * moeExpertNumPerRank_ * sizeof(int32_t);
        // GM_ADDR remoteStateAddr = (GM_ADDR)(GetStateAddrByRankId(mc2Context_, dstRankId, dataState_) + epRankId_ * cntPerRankSizeAlign_);
        
        if (dstRankId != epRankId_) {   // 远端 使用URMA发送 count + state
            // hcomm_.WriteNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteStateAddr,
            //                 scaleupCounterAddr_ + dstRankId * cntPerRankSizeAlign_, cntPerRankSizeAlign_);
            hcomm_.WriteWithNotifyNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteCountAddr, srcWorkspaceAddr,
                                      static_cast<uint64_t>(moeExpertNumPerRank_ * sizeof(int32_t)), notifyAddr, notifyVal);
            hcomm_.Drain(GetCommHandle(mc2Context_, epRankId_, dstRankId));
        } else {    // 本端
            GlobalTensor<uint64_t> notifyGMTensor;
            notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t*)notifyAddr);
            LocalTensor<uint64_t> notifyTensor = tempBuf_.GetWithOffset<uint64_t>(UB_ALIGN / sizeof(uint64_t), 0);
            LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.GetWithOffset<int32_t>(moeExpertNumAlign_ / sizeof(int32_t), UB_ALIGN);
            DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U, 0U};
            DataCopyParams notifyValCopyParams = {1U, static_cast<uint16_t>(sizeof(uint64_t)), 0U, 0U};
            uint32_t gmOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + epRankId_ * moeExpertNumPerRank_;
            DataCopyPad(cntPerExpertTensor, sendCntGMTensor_[gmOffset], expertCntCopyParams, padParams_);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(recvCounterGMTensor_[gmOffset], cntPerExpertTensor, expertCntCopyParams);
            notifyTensor.SetValue(0, notifyVal);
            PipeBarrier<PIPE_MTE3>();   // perExpert 写完再写notifyVal
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            DataCopyPad(notifyGMTensor, notifyTensor, notifyValCopyParams);
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetRecvCount()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    // 最后一个核处理recvCount
    uint32_t mask = 1;
    int32_t sumOfFlag = -1;
    int32_t commpareFlag = static_cast<int32_t>(epWorldSize_);
    DataCopyParams recvCntCopyParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(COUNTER_STRIDE * sizeof(int32_t)),
                                        0U, 0U};
    LocalTensor<int32_t> recvCounterTensor = subExpBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 = dstExpBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    while (sumOfFlag != commpareFlag) {  //状态位check
        DataCopyPad(recvCounterTensor, recvCounterGMTensor_, recvCntCopyParams, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
        printf("flag %d, rank0 count %d, ranId %d, aivId %d\n", sumOfFlag, recvCounterTensor(1), epRankId_, aivId_);
    }
    // printf("sumOfFlag %d, epWorldSize_ %d, aivId %d\n", sumOfFlag, epWorldSize_, aivId_);
    SetRecvNumPerExpert();  // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor);   // 计算本端接收来自各卡的token总数
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvExpertSizeAlign = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(recvCntBuf_, epWorldSize_ * recvExpertSizeAlign);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, moeNumPerRankAlign_);
    tpipe_->InitBuffer(recvTempBuf_, UB_ALIGN);
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int32_t> recvTensorInt32 = recvCntBuf_.Get<int32_t>();
    LocalTensor<int32_t> tempRecvTensor = dstExpBuf_.GetWithOffset<int32_t>(recvExpertSizeAlign / sizeof(int32_t), 0);
    LocalTensor<int64_t> recvCntTensor = recvTempBuf_.Get<int64_t>();
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();

    const uint32_t shape[] = {epWorldSize_, static_cast<uint32_t>(recvExpertSizeAlign / sizeof(int32_t))};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(epWorldSize_),
                                      static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U, 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(recvTensorInt32, recvCounterGMTensor_[cntPerRankSizeAlign512_ / sizeof(int32_t)], inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(tempRecvTensor, recvTensorInt32, shape, true);
    Cast(numRecvPerExpertTensor_, tempRecvTensor, RoundMode::CAST_NONE, moeExpertNumPerRank_);

    printf("local0 count %d, epRankId_ %d, aivId %d\n", tempRecvTensor(0), epRankId_, aivId_);

    if constexpr (DoCpuSync) {   // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t*)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        // SyncFunc<AscendC::HardEvent::V_S>();
        // actualACnt_ = recvCntTensor.GetValue(0);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(hostPinnedCounterTensor, recvCntTensor, UB_ALIGN / sizeof(int64_t));
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerRank(
    LocalTensor<int32_t> recvTmpTensor)
{
    LocalTensor<uint32_t> gatherTmpTensor = subExpBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    uint64_t rsvdCnt = 0;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
    printf("rank0 count %d, epRankId_ %d, aivId %d\n", recvTmpTensor(1), epRankId_, aivId_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSendDirect(uint32_t srcTokenId)
{
    int32_t  seenDst[MAX_TOPK];
    int32_t  seenSlot[MAX_TOPK];
    uint32_t nSeen = 0;
    for (uint32_t k = 0; k < axisK_; k++) {
        int32_t expertId = topkIdxTensor_.GetValue(k);
        if (expertId < 0) {
            dstSlotIdxTensor_.SetValue(k, -1);
            continue;
        }
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
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

        // atomicAdd 计算 计算本卡发送到dstRankId的token数量  预留槽位
        __gm__ int32_t* dstCounterAddr = reinterpret_cast<__gm__ int32_t*>(scaleupCounterAddr_ + dstRankId * sizeof(int32_t));
        uint32_t slot = AscendC::AtomicAdd(dstCounterAddr, 1);
        printf("cur 643 dstRankId %d, slot %d, epRankId_ %d, aivId %d\n", dstRankId, slot, epRankId_, aivId_);
        dstSlotIdxTensor_.SetValue(k, slot);
        seenDst[nSeen]  = dstRankId;
        seenSlot[nSeen] = slot;
        nSeen++;

        // 写入workspace对应区域
        WriteSlotToLocal(dstRankId, slot);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteSlotToLocal(uint32_t dstRankId,
    uint32_t slot)
{
    GlobalTensor<XType> slotGMTensor;
    GM_ADDR rankGM = (dstRankId == epRankId_) ? localWinAddr_ : slotWorkspaceAddr_;
    GM_ADDR slotAddr = rankGM + (dstRankId * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor.SetGlobalBuffer((__gm__ XType*)slotAddr);
    DataCopyPad(slotGMTensor, tokenSlotTensor_, slotCopyParams_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_, true); // 按卡分核
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        return;
    }

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);

    uint32_t tokenNumPerGroup = 100;  // TODO
    uint32_t sendNumAlign = Ceil(rankNumPerCore_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(recvTempBuf_, sendNumAlign);
    LocalTensor<int32_t> sendNumPerRankTensor = recvTempBuf_.Get<int32_t>();
    DataCopyParams sendCntCopyParams = {1U, static_cast<uint16_t>(rankNumPerCore_ * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(sendNumPerRankTensor, scaleupCounterGMTensor_[startRankId_], sendCntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>();   // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue(dstRankId - startRankId_);
        printf("cur 694 dstRankId %d, count %d, epRankId_ %d, aivId %d\n", dstRankId, sendTokenNum, epRankId_, aivId_);
        if (unlikely(dstRankId == epRankId_)) {
            GlobalTensor<uint64_t> localCntGMTensor;
            localCntGMTensor.SetGlobalBuffer((__gm__ uint64_t*)(GetStateAddrByRankId(mc2Context_, dstRankId, dataState_) +
                                                                cntPerRankSizeAlign512_ + moeExpertNumAlign512_));
            localCntGMTensor.SetValue(epRankId_ * axisMaxBS_, static_cast<uint64_t>(sendTokenNum));
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(localCntGMTensor[epRankId_ * axisMaxBS_]);
            continue;   //  本端slot已经写入win
        }
        uint32_t loopTimes = sendTokenNum / tokenNumPerGroup;    // 分组多token发送到对端
        uint32_t lastTokenNum = sendTokenNum - tokenNumPerGroup * loopTimes;
        uint32_t srcWorkspaceOffset = dstRankId * axisMaxBS_ * perSlotBytes_;
        uint32_t dstRankWinOffset = epRankId_ * axisMaxBS_ * perSlotBytes_;    // 计算目标窗口地址偏移
        GM_ADDR remoteWinAddr = GetWinAddrByRankId(mc2Context_, dstRankId);
        GM_ADDR notifyAddr = GetStateAddrByRankId(mc2Context_, dstRankId, dataState_) + cntPerRankSizeAlign512_ + 
                                moeExpertNumAlign512_ + epRankId_ * axisMaxBS_ * sizeof(uint64_t);
        for (uint32_t loop = 0; loop < loopTimes; loop++) {     // 发送slot
            printf("START WRITR TOKEN");
            // hcomm_.WriteNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteWinAddr + dstRankWinOffset,
            //                 slotWorkspaceAddr_ + srcWorkspaceOffset, perSlotBytes_ * tokenNumPerGroup);
            hcomm_.WriteWithNotifyNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteWinAddr + dstRankWinOffset,
                                      slotWorkspaceAddr_ + srcWorkspaceOffset, perSlotBytes_ * tokenNumPerGroup,
                                      notifyAddr + loop * sizeof(uint64_t), static_cast<uint64_t>(tokenNumPerGroup));
            srcWorkspaceOffset += perSlotBytes_ * tokenNumPerGroup;
            dstRankWinOffset += perSlotBytes_ * tokenNumPerGroup;
            hcomm_.Drain(GetCommHandle(mc2Context_, epRankId_, dstRankId));   // 疑问？
        }
        // hcomm_.WriteNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteWinAddr + dstRankWinOffset,
        //                 slotWorkspaceAddr_ + srcWorkspaceOffset, perSlotBytes_ * lastTokenNum);
        hcomm_.WriteWithNotifyNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteWinAddr + dstRankWinOffset,
                                  slotWorkspaceAddr_ + srcWorkspaceOffset, perSlotBytes_ * lastTokenNum,
                                  notifyAddr + loopTimes * sizeof(uint64_t), static_cast<uint64_t>(lastTokenNum));
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
    uint32_t hAlignBytes = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
        metaLocalTensor_ = xLocalTensor_[hAlignUbSize_ / MAX_DTYPE_SIZE].template ReinterpretCast<int32_t>();
        DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams_, padParams_);
        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            DataCopyPad(xLocalTensor_[hAlignBytes / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                        scalesGMTensor_[tokenId * scalesAlign_], scalesCopyParams_, padParams_);
        }
        DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
        DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(), topkWeightsGMTensor_[topkOffset],
                    topkCopyParams_, padParams_);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
        metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
        perSlotQueue_.EnQue(xLocalTensor_);
        tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
        topkIdxTensor_ = tokenSlotTensor_[hAlignUbSize_ / MAX_DTYPE_SIZE].template ReinterpretCast<int32_t>();
        dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();

        if constexpr (!IsCached) {
            DedupAndSendDirect(tokenId);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
        } else {
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams_, padParams_);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            for (uint32_t k = 0; k < axisK_; k++) {
                int32_t slot = dstSlotIdxTensor_.GetValue(k);
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
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessDirect()
{
    PipeBarrier<PIPE_ALL>();
    int64_t start = AscendC::GetSystemCycle();
    ResetCounters();    //计数器清0 为了后续发送token计算atomicAdd
    CalSendCnt();
    SyncAll<true>();
    Communication();
    GetRecvCount();
    PipeBarrier<PIPE_ALL>();
    int64_t end = AscendC::GetSystemCycle();
    int64_t duration = (end - start) / 1000;
    printf("duration %ld [us], ranId %d, aivId %d\n", duration, epRankId_, aivId_);
    // SyncAll<true>();
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    SendPhase();
    SyncAll<true>();
    WriteToRemoteWindow();
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