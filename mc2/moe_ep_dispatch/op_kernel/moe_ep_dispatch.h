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

#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#include "../common/op_kernel/mc2_moe_context.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#endif

namespace MoeEpDispatchImpl {

#define TemplateMoeEpDispatchTypeClass typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, \
    bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t MAX_TOPK = 32U;
constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t NETWORK_DIRECT = 0U;
constexpr uint32_t NETWORK_HYBRID = 1U;
constexpr uint32_t META_INFO = 32U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;   // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;   // UB_ALIGN/sizeof(int32_t)=8
constexpr int32_t BITS_PER_BYTE = 8;
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 50 * 1024U; // 计算count 50KB per group
constexpr uint32_t COUNTER_STRIDE = 2U;    // counter前两个数为 state、dstRankRecvNum
constexpr uint32_t CORE_NUM_RATIO = 2U;     // 计算count perrank/perexpert 核数比例
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
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId,
                                       uint32_t &endId, uint32_t &sendNum, bool isFront);
    __aicore__ inline GM_ADDR GetWinAddrByRankId(__gm__ Mc2Aclnn::Mc2MoeContext *ctx, uint32_t rankId, uint64_t offset)
    {
        return (GM_ADDR)ctx->epHcclBuffer_[rankId] + offset;
    }

    __aicore__ inline uint64_t GetCommHandle(__gm__ Mc2Aclnn::Mc2MoeContext *ctx, uint32_t localRankId, uint32_t rankId)
    {
        uint32_t index = rankId > localRankId ? rankId - 1 : rankId;
        return ctx->hcommHandle_[index];
    }

    TPipe *tpipe_{nullptr};
    __gm__ Mc2Aclnn::Mc2MoeContext *mc2Context_{nullptr};
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
    TBuf<> hcommBuf_;   // 通信
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
    uint32_t moeExpertNum_{0};
    uint32_t numScaleupRanks_{0};
    uint32_t numScaleoutRanks_{0};
    uint32_t numAivStage1_{0};
    uint32_t numAivStage2_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t hAlignSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t cntPerRankSizeAlign_{0};
    uint32_t cntPerRankSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t metaOffset_{0};
    uint32_t calRankCoreNum_{0};
    uint32_t calExpertCoreNum_{0};
    int64_t actualACnt_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
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
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::Mc2MoeContext*>(context);
    epRankId_ = mc2Context_->epRankId;

    const auto& info = tilingData->moeEpDispatchInfo;
    axisBS_ = info.cfg.numTokens;
    axisH_ = info.cfg.hidden;
    axisK_ = info.cfg.topK;
    epWorldSize_ = info.cfg.epWorldSize;
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    scalesBytes_ = info.cfg.scalesBytes;
    perSlotBytes_ = info.cfg.perSlotBytes;
    numScaleupRanks_ = info.numScaleupRanks;
    numScaleoutRanks_ = info.numScaleoutRanks;
    numAivStage1_ = info.numAivStage1;
    numAivStage2_ = info.numAivStage2;
    aivNum_ = info.aivNum;
    cntWinStateOffset_ = info.cntWinStateOffset;
    slotWinStateOffset_ = info.slotWinStateOffset;
    winDataOffset_ = info.winDataOffset;
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);
    moeExpertNum_ = moeExpertNumPerRank_ * epWorldSize_;

    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    metaOffset_ = hAlignSize_;
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    cntPerRankSizeAlign_ = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    // perRank state+count
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    cntPerRankSizeAlign512_ = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeNumPerRankAlign_ = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    moeNumPerRankAlign512_ = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;

    uint32_t numPerCore = (moeExpertNum_ + CORE_NUM_RATIO * epWorldSize_) / aivNum_;
    calRankCoreNum_ = CORE_NUM_RATIO * numPerCore >= epWorldSize_ ?
                      Ceil(CORE_NUM_RATIO * epWorldSize_, numPerCore) : epWorldSize_;
    calRankCoreNum_ = (calRankCoreNum_ >= aivNum_) ? (aivNum_ - 2U) : calRankCoreNum_;
    calExpertCoreNum_ = aivNum_ - calRankCoreNum_;

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)topkIdx);
    if constexpr (IsTopkWeights) {
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float*)topkWeights);
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType*)scales);
        metaOffset_ += Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
    }
    if constexpr (IsCached) {
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)cachedSlotIdx);
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t*)numRecvPerRank);
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t*)numRecvPerExpert);
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t*)dstBufferSlotIdx);

    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U,
                         static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN), 0U};
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U,
                              static_cast<uint16_t>((WIN_ADDR_ALIGN - UB_ALIGN) / UB_ALIGN)};
    padParams_ = {true, 0, 0, 0};

    scaleupCounterAddr_ = workspaceGM;
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + epWorldSizeAlign512_;
    slotWorkspaceAddr_ = sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_ + moeExpertNumAlign512_;
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)scaleupCounterAddr_);
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t*)localCntStateWinAddr_);
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
    uint32_t newAivId = aivId_;
    if (!isFront) {
        newAivId -= calRankCoreNum_;  // 由于是后面的核做perExpert计算，因此需要换算
    }
    startId = sendNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderNum) {      // 前remainderRankNum个aiv需要多发1个卡的数据
        sendNum ++;
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
    DataCopy(scaleupCounterGMTensor_, counterLocalTensor, counterCnt_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t startRankId, uint32_t endRankId)
{
    if (startRankId > epWorldSize_) {
        return;
    }

    uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint32_t mask = calCnt / axisK_;
    uint32_t sumSizeAlign = Ceil(calCnt / axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t shape[2] = {calCnt / axisK_, axisK_};
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(tmpOffset/ sizeof(int16_t), 0);
    LocalTensor<int32_t> tempTensorInt32 = topkIdsBuf_.Get<int32_t>();
    LocalTensor<int32_t> sumTensorInt32 = tempBuf_.GetWithOffset<int32_t>(sumSizeAlign / sizeof(int32_t), 0);
    LocalTensor<uint32_t> maskTensorInt32 = tempBuf_.GetWithOffset<uint32_t>(sumSizeAlign / sizeof(uint32_t),
        sumSizeAlign);
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt32.template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskTensorInt8 = topkIdsBuf_.GetWithOffset<uint8_t>(tmpOffset / sizeof(uint8_t), tmpOffset);

    Duplicate<int16_t>(dstTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    Div(tempTensorInt16, expertIdsTensor, dstTensorInt16, calCnt);
    // 筛选无效expert id 消除影响
    CompareScalar(maskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::LT, calCnt);
    Select(dstTensorInt16, maskTensorInt8, expertIdsTensor, tempTensorInt16,
        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, calCnt);

    for (uint32_t dstRankId = startRankId; dstRankId < endRankId; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        Subs(expertIdsTensor, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        Abs(tempTensorInt16, expertIdsTensor, calCnt);
        Mins(expertIdsTensor, tempTensorInt16, static_cast<int16_t>(1), calCnt); // 目标 0 非目标 1
        Cast(tempTensorInt32, expertIdsTensor, RoundMode::CAST_NONE, calCnt);
        ReduceSum<int32_t, Pattern::Reduce::AR, true>(sumTensorInt32, tempTensorInt32, shape, false);
        Duplicate<uint32_t>(maskTensorInt32, 0, sumSizeAlign / sizeof(uint32_t));    // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, sumTensorInt32, static_cast<int32_t>(axisK_), AscendC::CMPMODE::LT,
            calCnt / axisK_);
        GatherMask(tempTensorInt32, sumTensorInt32, maskTensorInt32, true, mask, {1, 1, 0, 0}, rsvdCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        uint32_t offset = (dstRankId - startRankId) * COUNTER_STRIDE + 1;
        int32_t curRankCnt = sendCntTensor_.GetValue(offset) + static_cast<int32_t>(rsvdCnt);
        sendCntTensor_.SetValue(offset, curRankCnt);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t startExpertId, uint32_t endExpertId)
{
    if (startExpertId > moeExpertNum_) {
        return;
    }

    LocalTensor<uint8_t> gatherMaskTensorInt8 = topkIdsBuf_.Get<uint8_t>();
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.Get<uint16_t>();
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    uint32_t mask = calCnt;
    uint32_t countAlign = Ceil(calCnt * sizeof(int16_t), UB_ALIGN) * UB_ALIGN / sizeof(int16_t);

    for (uint32_t dstExpertId = startExpertId; dstExpertId < endExpertId; dstExpertId++) {
        uint64_t rsvdCnt = 0;
        Duplicate<uint16_t>(maskTensorInt16, 0, countAlign);    // GatherMask前清0
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(dstExpertId),
            AscendC::CMPMODE::EQ, calCnt);
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
    uint32_t perGroupTokenNum = PER_GROUP_SIZE / sizeof(int16_t) / axisK_;
    uint32_t groupCnt = Ceil(axisBS_, perGroupTokenNum);
    uint32_t calCnt = perGroupTokenNum * axisK_;
    uint32_t perGroupSizeAlign = Ceil(calCnt * sizeof(int16_t), UB_ALIGN) * UB_ALIGN;
    uint32_t cntPerRankSizeAlign256 = Ceil(epWorldSize_ * COUNTER_STRIDE * sizeof(int32_t),
        ALIGNED_LEN_256) * ALIGNED_LEN_256;
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

    if (aivId_ < calRankCoreNum_) {  // 前面的核做per rank count计算
        SplitToCore(epWorldSize_, calRankCoreNum_, startId, endId, numPerCore, true);
        if (startId > epWorldSize_) {
            return;
        }
        // 两两一组，每组第一个位置填充1，作为状态位
        LocalTensor<int64_t> sendCntTensorInt64 = sendCntTensor_.template ReinterpretCast<int64_t>();
        Duplicate<int64_t>(sendCntTensorInt64, static_cast<int64_t>(1), cntPerRankSizeAlign256 / sizeof(int64_t));
    } else {    // 后面的核做per expert count计算
        SplitToCore(moeExpertNum_, calExpertCoreNum_, startId, endId, numPerCore, false);
        if (startId > moeExpertNum_) {
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

        if (aivId_ >= calRankCoreNum_) {   // 后面的核统计发送给各专家的token count
            CalSendCntPerExpert(tempTensorInt16, calCnt, startId, endId);
        } else {    // 前面的核统计发送到各卡token count
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
    LocalTensor<uint64_t> sendCntPerRankInt64 = dstExpBuf_.Get<uint64_t>();
    DataCopyParams cntCopyParams = {1U, static_cast<uint16_t>(rankNumPerCore_ * sizeof(uint64_t)), 0U, 0U};
    DataCopyPad(sendCntPerRankInt64, numSendGMTensorInt64[startRankId_], cntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>();   // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint64_t notifyVal = sendCntPerRankInt64.GetValue(dstRankId - startRankId_);
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        GM_ADDR remoteCountAddr = remoteStateAddr + epWorldSize_ * WIN_ADDR_ALIGN
            + epRankId_ * moeNumPerRankAlign512_;
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + cntPerRankSizeAlign512_
            + dstRankId * moeExpertNumPerRank_ * sizeof(int32_t);

        if (dstRankId != epRankId_) {   // 远端 使用URMA发送 count + state
            hcomm_.WriteWithNotifyNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteCountAddr,
                                      srcWorkspaceAddr, static_cast<uint64_t>(moeExpertNumPerRank_ * sizeof(int32_t)),
                                      notifyAddr, notifyVal);
        } else {    // 本端
            GlobalTensor<int32_t> countGMTensor;
            GlobalTensor<uint64_t> notifyGMTensor;
            countGMTensor.SetGlobalBuffer((__gm__ int32_t*)remoteCountAddr);
            notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t*)notifyAddr);
            LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.Get<int32_t>();
            DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int32_t)),
                                                  0U, 0U};
            uint32_t srcOffset = cntPerRankSizeAlign512_ / sizeof(int32_t) + dstRankId * moeExpertNumPerRank_;
            DataCopyPad(cntPerExpertTensor, sendCntGMTensor_[srcOffset], expertCntCopyParams, padParams_);
            SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
            DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
            SyncFunc<AscendC::HardEvent::MTE3_S>();   // perExpert 写完再写notifyVal
            notifyGMTensor.SetValue(0, notifyVal);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(notifyGMTensor);
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
    LocalTensor<int32_t> recvCounterTensor = topkIdsBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, epWorldSize_ * UB_ALIGN);
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // 等待前面尾核计数器清0(buf 复用)
    while (sumOfFlag != commpareFlag) {  // 状态位check
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
    }
    SetRecvNumPerExpert();  // 计算本卡上各专家接收的token总数
    SetRecvNumPerRank(recvCounterTensor);   // 计算本端接收来自各卡的token总数
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, epWorldSize_ * UB_STRIDE);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerExpert()
{
    uint32_t recvExpertSizeAlign = Ceil(moeExpertNumPerRank_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t recvExpertAlign = recvExpertSizeAlign / sizeof(int32_t);
    tpipe_->InitBuffer(recvCntBuf_, epWorldSize_ * recvExpertSizeAlign);
    tpipe_->InitBuffer(numRecvPerExpertBuf_, moeNumPerRankAlign_);
    tpipe_->InitBuffer(recvTempBuf_, UB_ALIGN);
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    LocalTensor<int32_t> recvTensorInt32 = recvCntBuf_.Get<int32_t>();
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    LocalTensor<int64_t> recvCntTensor = recvTempBuf_.Get<int64_t>();
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();

    const uint32_t shape[] = {epWorldSize_, recvExpertAlign};
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(recvExpertSizeAlign),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - recvExpertSizeAlign), 0U};
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    DataCopyPad(recvTensorInt32, recvCounterGMTensor_[epWorldSize_ * WIN_ADDR_ALIGN / sizeof(int32_t)],
                inRecvCntParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, epWorldSize_ * recvExpertAlign);
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);

    if constexpr (DoCpuSync) {   // 计算actualA，并写入host pin
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t*)hostPinnedCounterAddrGM_);
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
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
    LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    gatherTmpTensor.SetValue(0, 2);  // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;               // 源操作数每个datablock只需要处理两个元素
    uint64_t rsvdCnt = 0;
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    SyncFunc<AscendC::HardEvent::S_V>();
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
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
        __gm__ int32_t* dstCounterAddr = reinterpret_cast<__gm__ int32_t*>(
            scaleupCounterAddr_ + dstRankId * sizeof(int32_t));
        uint32_t slot = AscendC::AtomicAdd(dstCounterAddr, 1);
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
    GM_ADDR slotAddr = rankGM + (static_cast<uint64_t>(dstRankId) * axisMaxBS_ + slot) * perSlotBytes_;
    slotGMTensor.SetGlobalBuffer((__gm__ XType*)slotAddr);
    DataCopyParams slotCopyParams = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    DataCopyPad(slotGMTensor, tokenSlotTensor_, slotCopyParams);
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

    uint32_t sendCntSize = rankNumPerCore_ * COUNTER_STRIDE * sizeof(int32_t);
    tpipe_->InitBuffer(recvTempBuf_, Ceil(sendCntSize, UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> sendNumPerRankTensor = recvTempBuf_.Get<int32_t>();
    DataCopyParams sendCntCopyParams = {1U, static_cast<uint16_t>(sendCntSize), 0U, 0U};
    DataCopyPad(sendNumPerRankTensor, sendCntGMTensor_[startRankId_ * COUNTER_STRIDE], sendCntCopyParams, padParams_);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_MTE3>();   // 数据拷贝到workspace后再发远端

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - startRankId_) * COUNTER_STRIDE + 1);
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                             epRankId_ * WIN_ADDR_ALIGN;
        if (unlikely(dstRankId == epRankId_)) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            GlobalTensor<uint64_t> statusGMTensor;
            statusGMTensor.SetGlobalBuffer((__gm__ uint64_t*)(notifyAddr));
            statusGMTensor.SetValue(0, static_cast<uint64_t>(1));
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusGMTensor);
            continue;   //  本端slot已经写入win
        }

        uint64_t sendDataSize = sendTokenNum > 0 ? perSlotBytes_ * sendTokenNum : UB_ALIGN;
        uint32_t srcWorkspaceOffset = dstRankId * axisMaxBS_ * perSlotBytes_;
        uint32_t dstRankWinOffset = epRankId_ * axisMaxBS_ * perSlotBytes_;    // 计算目标窗口地址偏移
        GM_ADDR remoteWinAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset;
        hcomm_.WriteWithNotifyNbi(GetCommHandle(mc2Context_, epRankId_, dstRankId), remoteWinAddr,
                                  slotWorkspaceAddr_ + srcWorkspaceOffset, sendDataSize,
                                  notifyAddr, static_cast<uint64_t>(1));
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_, true); // 按token数量分核
    if (startTokenId_ >= axisBS_) {
        return;
    }

    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    tpipe_->InitBuffer(dstSlotQueue_, 1, kAlignSize_);
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams topkCopyParams = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    DataCopyParams scalesCopyParams = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
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

        if constexpr (!IsCached) {
            DedupAndSendDirect(tokenId);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
        } else {
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            SyncFunc<AscendC::HardEvent::MTE2_S>();
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
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor_, topkCopyParams);
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WaitDispatch()
{
    if (aivId_ != aivNum_ - 1) {
        return;
    }

    // 最后一个核check slot 发送是否完成
    uint32_t mask = 1;
    uint64_t sumOfFlag = 0;
    uint64_t commpareFlag = static_cast<uint64_t>(epWorldSize_);
    const uint32_t shape[] = {epWorldSize_, UB_ALIGN / sizeof(uint64_t)};
    GlobalTensor<uint64_t> statusGMTensor;
    statusGMTensor.SetGlobalBuffer((__gm__ uint64_t*)localSlotStateWinAddr_);
    tpipe_->InitBuffer(recvCntBuf_, epWorldSize_ * UB_ALIGN);
    tpipe_->InitBuffer(tempBuf_, UB_ALIGN);
    LocalTensor<uint64_t> statusLocalTensor = recvCntBuf_.Get<uint64_t>();
    LocalTensor<uint64_t> sumTensor = tempBuf_.Get<uint64_t>();

    while (sumOfFlag != commpareFlag) {  // 状态位check
        DataCopy(statusLocalTensor, statusGMTensor, statusCopyParams_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum<uint64_t, AscendC::Pattern::Reduce::RA, true>(sumTensor, statusLocalTensor, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = sumTensor.GetValue(0);
    }
    // status clear
    Duplicate<uint64_t>(statusLocalTensor, 0, epWorldSize_ * UB_ALIGN / sizeof(uint64_t));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusGMTensor, statusLocalTensor, clearStatusCopyParams_);
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ProcessDirect()
{
    ResetCounters();    // 计数器清0 为了后续发送token计算atomicAdd
    CalSendCnt();
    SyncAll<true>();
    Communication();
    GetRecvCount();
    PipeBarrier<PIPE_ALL>();
    tpipe_->Reset();
    SendPhase();
    SyncAll<true>();
    WriteToRemoteWindow();
    WaitDispatch();
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        ProcessDirect();
    }
}
}  // namespace MoeEpDispatchImpl

#endif  // MOE_EP_DISPATCH_H