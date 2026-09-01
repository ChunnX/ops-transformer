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

#if __has_include("version/asc_devkit_version.h") && __has_include("version/hcomm_version.h")
#include "version/asc_devkit_version.h"
#include "version/hcomm_version.h"

#if (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0)) && \
    (HCOMM_MAJOR > 9 || (HCOMM_MAJOR == 9 && HCOMM_MINOR > 0))
#define ENABLE_MOE_EP_KERNEL
#endif

#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/reduce.h"
#include "adv_api/reduce/sum.h"
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
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

namespace MoeEpDispatchImpl {

#define TemplateMoeEpDispatchTypeClass \
    typename XType, typename ScalesType, bool DoCpuSync, bool IsCached, bool IsTopkWeights, uint8_t NetworkMode
#define TemplateMoeEpDispatchTypeFunc XType, ScalesType, DoCpuSync, IsCached, IsTopkWeights, NetworkMode

#if defined(ENABLE_MOE_EP_KERNEL)

using namespace AscendC;
using namespace Mc2Kernel;

constexpr uint32_t UB_ALIGN = 32U;
constexpr uint32_t WIN_ADDR_ALIGN = 512U;
constexpr uint32_t ALIGNED_LEN_256 = 256U;
constexpr uint32_t TOPK_INFO_SIZE = 4U;  // sizeof(int32_t)=sizeof(float)=4B
constexpr uint32_t UB_STRIDE = 8U;       // UB_ALIGN/sizeof(int32_t)=8
constexpr uint32_t INT64_UB_STRIDE = 4U; // UB_ALIGN/sizeof(int64_t)=4
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr uint32_t STATE_STRIDE = 15U; // (WIN_ADDR_ALIGN-UB_ALIGN)/UB_ALIGN=15
constexpr uint32_t HCOMM_INIT_SIZE = 512U;
constexpr uint32_t PER_GROUP_SIZE = 40 * 1024U; // 计算count 40KB per group
constexpr uint32_t EXPERT_NUM_PER_GROUP = 256U; // 直方图每次计算0-255
constexpr uint32_t DATA_BLOCK_NUM = 8U;         // 256B/32B=8
constexpr uint8_t BUFFER_NUM = 2;
static constexpr struct UrmaWqeEntry DEFAULT_WQE_CONFIG = {
    .odr = 5,
    .fence = 1,
    .se = 0,
    .cqe = 0,
    .inlineEn = 0};

template <TemplateMoeEpDispatchTypeClass>
class MoeEpDispatch {
public:
    __aicore__ inline MoeEpDispatch(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales,
                                GM_ADDR cachedSlotIdx, GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert,
                                GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeEpDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void BufferInit();
    __aicore__ inline void ResetCounters();
    __aicore__ inline void CalSendCntPerRank(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t tmpOffset);
    __aicore__ inline void CalSendCntPerExpert(LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt,
                                               uint32_t tmpOffset);
    __aicore__ inline void CalSendCnt();
    __aicore__ inline void Communication();
    __aicore__ inline void GetRecvCount();
    __aicore__ inline void SetRecvNumPerExpert();
    __aicore__ inline void SetRecvNumPerRank(LocalTensor<int32_t> recvTmpTensor);
    __aicore__ inline void DedupAndSendDirect(uint32_t hitStride);
    __aicore__ inline void WriteSlotToLocal(uint32_t dstRankId, uint32_t slot);
    __aicore__ inline void WriteToRemoteWindow();
    __aicore__ inline void SendPhase();
    __aicore__ inline void GetSlotStartNum();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startId, uint32_t &endId,
                                       uint32_t &sendNum);

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
    GlobalTensor<int32_t> scaleupCounterGMTensor_;
    GlobalTensor<int32_t> recvCounterGMTensor_;
    GlobalTensor<int32_t> sendCntPerRankGMTensor_;
    GlobalTensor<int32_t> sendCntPerExpertGMTensor_;
    GlobalTensor<XType> slotGMTensor_;
    GlobalTensor<ScalesType> scalesGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<XType> tokenSlotTensor_;
    LocalTensor<int32_t> topkIdxTensor_;
    LocalTensor<int32_t> dstSlotIdxTensor_;
    LocalTensor<int32_t> metaLocalTensor_;
    LocalTensor<int32_t> numRecvPerRankTensor_;
    LocalTensor<int64_t> numRecvPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerExpertTensor_;
    LocalTensor<int32_t> sendCntPerRankTensor_;
    LocalTensor<int32_t> slotIdxPerRankTensor_;
    LocalTensor<int32_t> hitPerRankTensor_;
    LocalTensor<uint8_t> hcommTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> perSlotQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> dstSlotQueue_;
    TBuf<> topkIdsBuf_;
    TBuf<> tempBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> hcommBuf_; // 通信
    TBuf<> sendCntRankBuf_;
    TBuf<> sendCntExpertBuf_;
    TBuf<> numRecvPerRankBuf_;
    TBuf<> numRecvPerExpertBuf_;

    GM_ADDR workspaceGM_{nullptr};
    GM_ADDR hostPinnedCounterAddrGM_{nullptr};
    GM_ADDR scaleupCounterAddr_{nullptr};
    GM_ADDR sendCntWorkspaceAddr_{nullptr};
    GM_ADDR localCntStateWinAddr_{nullptr};
    GM_ADDR localSlotStateWinAddr_{nullptr};
    GM_ADDR localWinAddr_{nullptr};
    GM_ADDR payloadStashWinAddr_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t epWorldSize_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t axisMaxBS_{0};
    uint32_t scalesBytes_{0};
    uint32_t perSlotBytes_{0};
    uint32_t moeExpertNum_{0};
    uint32_t aivNum_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendTokenNum_{0};
    uint32_t perGroupTokenNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t rankNumPerCore_{0};
    uint32_t hAlignSize_{0};
    uint32_t kAlignSize_{0};
    uint32_t axisKAlign_{0};
    uint32_t epWorldSizeAlign_{0};
    uint32_t epWorldSizeAlign512_{0};
    uint32_t counterCnt_{0};
    uint32_t counterAlign512_{0};
    uint32_t perGroupSizeAlign_{0};
    uint32_t moeNumPerRankSize_{0};
    uint32_t moeNumPerRankAlign_{0};
    uint32_t moeExpertNumAlign_{0};
    uint32_t moeNumPerRankAlign512_{0};
    uint32_t moeExpertNumAlign512_{0};
    uint32_t cntFlagWinSize_{0};
    uint32_t dispatchNotifyCount_{1};
    uint32_t metaOffset_{0};
    uint64_t cntWinStateOffset_{0};
    uint64_t slotWinStateOffset_{0};
    uint64_t winDataOffset_{0};
    uint64_t payloadStashWinOffset_{0};
    uint64_t dstRankWinOffset_{0};

    DataCopyParams statusCopyParams_;
    DataCopyParams clearStatusCopyParams_;
    DataCopyParams sendPerExpertParams_;
    DataCopyParams slotCopyParams_;
    DataCopyPadParams padParams_;
};

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIdx, GM_ADDR topkWeights, GM_ADDR scales, GM_ADDR cachedSlotIdx,
    GM_ADDR numRecvPerRank, GM_ADDR numRecvPerExpert, GM_ADDR dstBufferSlotIdx, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeEpDispatchTilingData *tilingData)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_ = pipe;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    aivId_ = GetBlockIdx();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    workspaceGM_ = workspaceGM;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    mc2Context_ = reinterpret_cast<__gm__ Mc2Aclnn::MoeCommContext *>(context);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    epRankId_ = mc2Context_->epRankId;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    const auto &info = tilingData->moeEpDispatchInfo;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisBS_ = info.cfg.numTokens;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisH_ = info.cfg.hidden;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisK_ = info.cfg.topK;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    epWorldSize_ = info.cfg.epWorldSize;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeExpertNum_ = info.cfg.numExperts;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeExpertNumPerRank_ = info.cfg.numLocalExperts;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisMaxBS_ = info.cfg.numMaxTokensPerRank;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    scalesBytes_ = info.scalesBytes;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    perSlotBytes_ = info.perSlotBytes;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    aivNum_ = info.aivNum;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    cntWinStateOffset_ = info.window.cntWinStateOffset;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    slotWinStateOffset_ = info.window.slotWinStateOffset;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    winDataOffset_ = info.window.winDataOffset;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    payloadStashWinOffset_ = info.window.payloadStashWinOffset;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    dispatchNotifyCount_ = info.dispatchNotifyCount;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    hostPinnedCounterAddrGM_ = reinterpret_cast<GM_ADDR>(info.hostPinnedCounterAddr);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    kAlignSize_ = Ceil(axisK_ * TOPK_INFO_SIZE, UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    axisKAlign_ = kAlignSize_ / TOPK_INFO_SIZE;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    metaOffset_ = hAlignSize_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    epWorldSizeAlign_ = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    perGroupTokenNum_ = PER_GROUP_SIZE / sizeof(int16_t) / axisK_; // token num per group
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    perGroupSizeAlign_ = Ceil(perGroupTokenNum_ * axisK_ * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    epWorldSizeAlign512_ = Ceil(epWorldSize_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    counterCnt_ = epWorldSizeAlign_ / sizeof(int32_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    counterAlign512_ = epWorldSizeAlign512_ / sizeof(int32_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeNumPerRankSize_ = moeExpertNumPerRank_ * sizeof(int32_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeNumPerRankAlign_ = Ceil(moeNumPerRankSize_, UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeExpertNumAlign_ = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeNumPerRankAlign512_ = Ceil(moeNumPerRankSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    moeExpertNumAlign512_ = Ceil(moeExpertNum_ * sizeof(int32_t), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    cntFlagWinSize_ = epWorldSize_ * WIN_ADDR_ALIGN;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    dstRankWinOffset_ = static_cast<uint64_t>(epRankId_) * axisMaxBS_ * perSlotBytes_; // 目标窗口地址偏移
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    topkIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)topkIdx);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if constexpr (IsTopkWeights) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        topkWeightsGMTensor_.SetGlobalBuffer((__gm__ float *)topkWeights);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        scalesGMTensor_.SetGlobalBuffer((__gm__ ScalesType *)scales);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        metaOffset_ += Ceil(scalesBytes_, UB_ALIGN) * UB_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    if constexpr (IsCached) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        cachedSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)cachedSlotIdx);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    numRecvPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)numRecvPerRank);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numRecvPerExpertGMTensor_.SetGlobalBuffer((__gm__ int64_t *)numRecvPerExpert);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    dstSlotIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)dstBufferSlotIdx);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    statusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, static_cast<uint16_t>(STATE_STRIDE), 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    clearStatusCopyParams_ = {static_cast<uint16_t>(epWorldSize_), 1U, 0U, static_cast<uint16_t>(STATE_STRIDE)};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendPerExpertParams_ = {1U, static_cast<uint16_t>(moeExpertNum_ * sizeof(int32_t)), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    slotCopyParams_ = {1U, static_cast<uint16_t>(perSlotBytes_), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    padParams_ = {true, 0, 0, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    scaleupCounterAddr_ = workspaceGM;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendCntWorkspaceAddr_ = scaleupCounterAddr_ + aivNum_ * epWorldSizeAlign512_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    localCntStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, cntWinStateOffset_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    localSlotStateWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, slotWinStateOffset_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    localWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, winDataOffset_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    payloadStashWinAddr_ = GetWinAddrByRankId(mc2Context_, epRankId_, payloadStashWinOffset_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    scaleupCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)scaleupCounterAddr_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    recvCounterGMTensor_.SetGlobalBuffer((__gm__ int32_t *)localCntStateWinAddr_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendCntPerRankGMTensor_.SetGlobalBuffer((__gm__ int32_t *)sendCntWorkspaceAddr_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendCntPerExpertGMTensor_.SetGlobalBuffer(
        (__gm__ int32_t *)(sendCntWorkspaceAddr_ + epWorldSize_ * WIN_ADDR_ALIGN));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                 uint32_t curUseAivNum,
                                                                                 uint32_t &startId, uint32_t &endId,
                                                                                 uint32_t &sendNum)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的数量
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t remainderNum = curSendCnt % curUseAivNum; // 余数
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    startId = sendNum * aivId_;                        // 每个aiv发送时的起始rankid
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (aivId_ < remainderNum) {                       // 前remainderRankNum个aiv需要多发1个卡的数据
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        sendNum++;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        startId += aivId_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    } else {
        startId += remainderNum;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    endId = startId + sendNum;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::BufferInit()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t sendCntRankSizeAlign = Ceil(epWorldSize_ * UB_ALIGN, ALIGNED_LEN_256) * ALIGNED_LEN_256;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    // 通信初始化
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    hcommTensor_ = hcommBuf_.Get<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    hcomm_.Init(hcommTensor_, HCOMM_INIT_SIZE);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    tpipe_->InitBuffer(perSlotQueue_, BUFFER_NUM, perSlotBytes_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(dstSlotQueue_, BUFFER_NUM, kAlignSize_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(topkIdsBuf_, 2 * perGroupSizeAlign_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(tempBuf_, perGroupSizeAlign_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(dstExpBuf_, perGroupSizeAlign_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(sendCntRankBuf_, sendCntRankSizeAlign);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(sendCntExpertBuf_, moeExpertNumAlign_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(numRecvPerExpertBuf_, 2 * moeNumPerRankAlign_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    tpipe_->InitBuffer(numRecvPerRankBuf_, epWorldSizeAlign_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numRecvPerRankTensor_ = numRecvPerRankBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendCntPerRankTensor_ = sendCntRankBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    sendCntPerExpertTensor_ = sendCntExpertBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Duplicate<int32_t>(sendCntPerRankTensor_, 0, epWorldSize_ * UB_STRIDE);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Duplicate<int32_t>(sendCntPerExpertTensor_, 0, moeExpertNumAlign_ / sizeof(int32_t));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (aivId_ == 0) {                             // 状态位仅做1次累加
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint64_t mask[2] = {0x101010101010101, 0}; // 一次性操作256字节，也是8个datablock，每8个数将首个设置为1
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Duplicate<int32_t>(sendCntPerRankTensor_, 1, mask, Ceil(epWorldSize_, DATA_BLOCK_NUM), 1, DATA_BLOCK_NUM);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    ResetCounters();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncAll<true>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::ResetCounters()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (aivId_ != aivNum_ - 1) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopy(sendCntPerRankGMTensor_, sendCntPerRankTensor_, clearStatusCopyParams_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerRank(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t tmpOffset)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t tokenCnt = calCnt / axisK_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t tokenCntAlign = Ceil(tokenCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int16_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t mask = tokenCnt;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t shape[2] = {tokenCnt, axisK_};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int16_t> tempTensorInt16 = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, 0);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint16_t> maskTensorInt16 = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint8_t> gatherMaskTensorInt8 = maskTensorInt16.template ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    Duplicate<int16_t>(dstTensorInt16, static_cast<int16_t>(moeExpertNumPerRank_), calCnt);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Div(tempTensorInt16, expertIdsTensor, dstTensorInt16, calCnt);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    // 筛选无效expert id 消除影响
    CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Select(dstTensorInt16, gatherMaskTensorInt8, tempTensorInt16, static_cast<int16_t>(-1),
           AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, calCnt);
           PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t dstRankId = 0; dstRankId < epWorldSize_; dstRankId++) {
        // 筛选出发送到目标卡的token
        uint64_t rsvdCnt = 0;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t offset = dstRankId * UB_STRIDE + 1;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Subs(expertIdsTensor, dstTensorInt16, static_cast<int16_t>(dstRankId), calCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Abs(tempTensorInt16, expertIdsTensor, calCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ReduceMin<int16_t, Pattern::Reduce::AR, true>(expertIdsTensor, tempTensorInt16, gatherMaskTensorInt8,
                                                      shape, false); // 0为目标
                                                      PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Duplicate<uint16_t>(maskTensorInt16, 0, tokenCntAlign);      // GatherMask前清0
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CompareScalar(gatherMaskTensorInt8, expertIdsTensor, static_cast<int16_t>(0), AscendC::CMPMODE::EQ,
                      tokenCntAlign);
                      PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GatherMask(tempTensorInt16, expertIdsTensor, maskTensorInt16, true, mask, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (rsvdCnt == 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        SyncFunc<AscendC::HardEvent::V_S>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        int32_t curRankCnt = sendCntPerRankTensor_.GetValue(offset);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        sendCntPerRankTensor_.SetValue(offset, curRankCnt + static_cast<int32_t>(rsvdCnt));
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCntPerExpert(
    LocalTensor<int16_t> expertIdsTensor, uint32_t calCnt, uint32_t tmpOffset)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t calCntAlign = tmpOffset / sizeof(int16_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t groupNum = Ceil(moeExpertNum_, EXPERT_NUM_PER_GROUP);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t maskU16Cnt = Ceil(calCntAlign, BITS_PER_BYTE * sizeof(uint16_t));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int16_t> tempTensor = topkIdsBuf_.GetWithOffset<int16_t>(calCntAlign, tmpOffset);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int16_t> dstTensorInt16 = dstExpBuf_.Get<int16_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint8_t> dstTensorU8 = dstExpBuf_.Get<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint32_t> dstTensorU32 = dstExpBuf_.Get<uint32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint16_t> gatherMaskTensorU16 = topkIdsBuf_.Get<uint16_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint16_t> gatherMaskU16GE = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, 0);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint16_t> gatherMaskU16LT = topkIdsBuf_.GetWithOffset<uint16_t>(calCntAlign, tmpOffset);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint8_t> gatherMaskU8GE = gatherMaskU16GE.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint8_t> gatherMaskU8LT = gatherMaskU16LT.ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t group = 0; group < groupNum; group++) {
        uint32_t baseExpertId = group * EXPERT_NUM_PER_GROUP;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t curGroupSize = (group == groupNum - 1) ? (moeExpertNum_ - baseExpertId) : EXPERT_NUM_PER_GROUP;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint64_t rsvdCnt = 0;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Subs(dstTensorInt16, expertIdsTensor, static_cast<int16_t>(baseExpertId), calCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Duplicate<uint16_t>(gatherMaskTensorU16, 0, calCntAlign * 2);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CompareScalar(gatherMaskU8GE, dstTensorInt16, static_cast<int16_t>(0), AscendC::CMPMODE::GE, calCntAlign);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CompareScalar(gatherMaskU8LT, dstTensorInt16, static_cast<int16_t>(curGroupSize - 1), AscendC::CMPMODE::LE,
                      calCntAlign);
                      PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        And(gatherMaskU16GE, gatherMaskU16GE, gatherMaskU16LT, maskU16Cnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GatherMask(tempTensor, dstTensorInt16, gatherMaskU16GE, true, calCnt, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::V_S>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (rsvdCnt == 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        Cast(dstTensorU8, tempTensor, RoundMode::CAST_NONE, rsvdCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GetExpertFreq(gatherMaskTensorU16, dstTensorU8, rsvdCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Cast(dstTensorU32, gatherMaskTensorU16, RoundMode::CAST_NONE, curGroupSize);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Add(sendCntPerExpertTensor_[baseExpertId], sendCntPerExpertTensor_[baseExpertId],
            dstTensorU32.ReinterpretCast<int32_t>(), static_cast<int32_t>(curGroupSize));
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::CalSendCnt()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SplitToCore(axisBS_, aivNum_, startTokenId_, endTokenId_, sendTokenNum_); // 按token数量分核
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (startTokenId_ >= axisBS_) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    uint32_t groupCnt = Ceil(sendTokenNum_, perGroupTokenNum_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t calCnt = perGroupTokenNum_ * axisK_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int32_t> topkIdsGroupTensor = topkIdsBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int16_t> tempTensorInt16 = tempBuf_.Get<int16_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams sendPerRankParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(UB_ALIGN), 0U,
                                        static_cast<uint16_t>(WIN_ADDR_ALIGN - UB_ALIGN)};
                                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPadExtParams<int32_t> topkIdsCntCopyPadParams{false, 0U, 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t group = 0; group < groupCnt; group++) {
        if (group == groupCnt - 1) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            calCnt = (sendTokenNum_ - group * perGroupTokenNum_) * axisK_;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        uint32_t topkIdxOffset = (startTokenId_ + group * perGroupTokenNum_) * axisK_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t tmpOffset = Ceil(calCnt * sizeof(int16_t), ALIGNED_LEN_256) * ALIGNED_LEN_256;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyExtParams topkIdsCntParams = {1U, static_cast<uint32_t>(calCnt * sizeof(int32_t)), 0U, 0U, 0U};
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (group > 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        DataCopyPad(topkIdsGroupTensor, topkIdxGMTensor_[topkIdxOffset], topkIdsCntParams,
                    topkIdsCntCopyPadParams); // copy topkId
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Cast(tempTensorInt16, topkIdsGroupTensor, RoundMode::CAST_NONE, calCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        // perExpert 和 perRank count 计算
        CalSendCntPerExpert(tempTensorInt16, calCnt, tmpOffset);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CalSendCntPerRank(tempTensorInt16, calCnt, tmpOffset);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    SyncFunc<AscendC::HardEvent::S_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SetAtomicAdd<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(sendCntPerRankGMTensor_, sendCntPerRankTensor_, sendPerRankParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(sendCntPerExpertGMTensor_, sendCntPerExpertTensor_, sendPerExpertParams_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    if constexpr (!IsCached) { // 非cache模式下计算每个核slot起始值
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        gatherTmpTensor.SetValue(0, 2); // 设置掩码，取源操作数每个datablock中的第2个元素
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint64_t rsvdCnt = 0;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::S_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GatherMask(numRecvPerRankTensor_, sendCntPerRankTensor_, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopy(scaleupCounterGMTensor_[aivId_ * counterAlign512_], numRecvPerRankTensor_, counterCnt_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Communication()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SplitToCore(epWorldSize_, aivNum_, startRankId_, endRankId_, rankNumPerCore_); // 按卡分核
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (startRankId_ >= epWorldSize_) {                                            // 空闲核，直接返回
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    GlobalTensor<uint64_t> numSendGMTensorU64;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numSendGMTensorU64.SetGlobalBuffer((__gm__ uint64_t *)(sendCntWorkspaceAddr_ + startRankId_ * WIN_ADDR_ALIGN));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint64_t> sendCntPerRankU64 = sendCntRankBuf_.Get<uint64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams cntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), 1U,
                                    static_cast<uint16_t>(STATE_STRIDE), 0U};
                                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopy(sendCntPerRankU64, numSendGMTensorU64, cntCopyParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t notifyStride = (dstRankId - startRankId_) * INT64_UB_STRIDE;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t srcOffset = dstRankId * moeExpertNumPerRank_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint64_t notifyVal = sendCntPerRankU64.GetValue(notifyStride);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        // 计算目标窗口地址:
        GM_ADDR remoteStateAddr = GetWinAddrByRankId(mc2Context_, dstRankId, cntWinStateOffset_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GM_ADDR notifyAddr = remoteStateAddr + epRankId_ * WIN_ADDR_ALIGN;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GM_ADDR remoteCountAddr = remoteStateAddr + cntFlagWinSize_ + epRankId_ * moeNumPerRankAlign512_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GM_ADDR srcWorkspaceAddr = sendCntWorkspaceAddr_ + cntFlagWinSize_ + srcOffset * sizeof(int32_t);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        if (dstRankId != epRankId_) { // 远端 使用URMA发送 count + state
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(commHandle, remoteCountAddr,
                                                                                   srcWorkspaceAddr, moeNumPerRankSize_,
                                                                                   notifyAddr, notifyVal);
                                                                                   PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        // 本端
        GlobalTensor<int32_t> countGMTensor;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GlobalTensor<uint64_t> notifyGMTensor;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        countGMTensor.SetGlobalBuffer((__gm__ int32_t *)remoteCountAddr);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        notifyGMTensor.SetGlobalBuffer((__gm__ uint64_t *)notifyAddr);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        LocalTensor<int32_t> cntPerExpertTensor = tempBuf_.Get<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyParams expertCntCopyParams = {1U, static_cast<uint16_t>(moeNumPerRankSize_), 0U, 0U};
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(cntPerExpertTensor, sendCntPerExpertGMTensor_[srcOffset], expertCntCopyParams, padParams_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(countGMTensor, cntPerExpertTensor, expertCntCopyParams);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        PipeBarrier<PIPE_MTE3>(); // perExpert 写完再写notifyVal
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopy(notifyGMTensor, sendCntPerRankU64[notifyStride], INT64_UB_STRIDE);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetRecvCount()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (aivId_ != aivNum_ - 1) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    // 最后一个核处理recvCount
    uint32_t mask = 1;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t sumOfFlag = -1;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    int32_t commpareFlag = static_cast<int32_t>(epWorldSize_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int32_t> recvCounterTensor = topkIdsBuf_.GetWithOffset<int32_t>(epWorldSize_ * UB_STRIDE, 0);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<float> tempFp32 = topkIdsBuf_.GetWithOffset<float>(epWorldSize_ * UB_STRIDE, epWorldSize_ * UB_ALIGN);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<float> recvCounterTensorFp32 = recvCounterTensor.template ReinterpretCast<float>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<float> numRecvPerRankTensorFp32 = numRecvPerRankTensor_.template ReinterpretCast<float>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE3_V>(); // buf 复用
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    while (sumOfFlag != commpareFlag) {     // 状态位check
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopy(recvCounterTensor, recvCounterGMTensor_, statusCopyParams_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ReduceSum(numRecvPerRankTensorFp32, recvCounterTensorFp32, tempFp32, mask, epWorldSize_, 1);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::V_S>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        sumOfFlag = numRecvPerRankTensor_.GetValue(0);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    SetRecvNumPerExpert();                // 计算本卡上各专家接收的token总数
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SetRecvNumPerRank(recvCounterTensor); // 计算本端接收来自各卡的token总数
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    // status clear
    Duplicate<int32_t>(recvCounterTensor, 0, epWorldSize_ * UB_STRIDE);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopy(recvCounterGMTensor_, recvCounterTensor, clearStatusCopyParams_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerExpert()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t recvExpertAlign = moeNumPerRankAlign_ / sizeof(int32_t);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    GlobalTensor<int32_t> localExpertRecvGMTensor;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    localExpertRecvGMTensor.SetGlobalBuffer((__gm__ int32_t *)(localCntStateWinAddr_ + cntFlagWinSize_));
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    numRecvPerExpertTensor_ = numRecvPerExpertBuf_.Get<int64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int64_t> recvCntTensor = tempBuf_.GetWithOffset<int64_t>(INT64_UB_STRIDE, 0);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int32_t> recvTensorInt32 = tempBuf_.GetWithOffset<int32_t>(epWorldSize_ * recvExpertAlign, UB_ALIGN);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int64_t> tempRecvTensorInt64 = dstExpBuf_.Get<int64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int64_t> sharedTmp = recvTensorInt32.template ReinterpretCast<int64_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint8_t> sharedTmpInt8 = recvTensorInt32.template ReinterpretCast<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    const uint32_t shape[] = {epWorldSize_, recvExpertAlign};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams inRecvCntParams = {static_cast<uint16_t>(epWorldSize_), static_cast<uint16_t>(moeNumPerRankAlign_),
                                      static_cast<uint16_t>(moeNumPerRankAlign512_ - moeNumPerRankAlign_), 0U};
                                      PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams recvPerExpertParams = {1U, static_cast<uint16_t>(moeExpertNumPerRank_ * sizeof(int64_t)), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(recvTensorInt32, localExpertRecvGMTensor, inRecvCntParams, padParams_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Cast(tempRecvTensorInt64, recvTensorInt32, RoundMode::CAST_NONE, epWorldSize_ * recvExpertAlign);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    ReduceSum<int64_t, AscendC::Pattern::Reduce::RA, true>(numRecvPerExpertTensor_, tempRecvTensorInt64, sharedTmpInt8,
                                                           shape, true);
                                                           PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    if constexpr (DoCpuSync) { // 计算actualA，并写入host pin
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GlobalTensor<int64_t> hostPinnedCounterTensor;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        hostPinnedCounterTensor.SetGlobalBuffer((__gm__ int64_t *)hostPinnedCounterAddrGM_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ReduceSum(recvCntTensor, numRecvPerExpertTensor_, sharedTmp, moeExpertNumPerRank_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopy(hostPinnedCounterTensor, recvCntTensor, INT64_UB_STRIDE);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(numRecvPerExpertGMTensor_, numRecvPerExpertTensor_, recvPerExpertParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SetRecvNumPerRank(
    LocalTensor<int32_t> recvTmpTensor)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint32_t> gatherTmpTensor = topkIdsBuf_.GetWithOffset<uint32_t>(UB_STRIDE, epWorldSize_ * UB_ALIGN);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    gatherTmpTensor.SetValue(0, 2); // 设置掩码，取源操作数每个datablock中的第2个元素
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint64_t rsvdCnt = 0;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    GatherMaskParams recvMaskParams = {1, static_cast<uint16_t>(epWorldSize_), 1, 0};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams recvPerRankParams = {1U, static_cast<uint16_t>(epWorldSize_ * sizeof(int32_t)), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::S_V>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    GatherMask(numRecvPerRankTensor_, recvTmpTensor, gatherTmpTensor, true, mask, recvMaskParams, rsvdCnt);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(numRecvPerRankGMTensor_, numRecvPerRankTensor_, recvPerRankParams);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::DedupAndSendDirect(uint32_t hitStride)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    for (uint32_t k = 0; k < axisK_; k++) {
        int32_t expertId = topkIdxTensor_.GetValue(k);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t dstRankId = expertId / moeExpertNumPerRank_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        int32_t hit = (expertId < 0) ? -1 : hitPerRankTensor_.GetValue(hitStride + dstRankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if ((hit >= 0) || (expertId < 0)) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            dstSlotIdxTensor_.SetValue(k, hit);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        int32_t slot = slotIdxPerRankTensor_.GetValue(dstRankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        dstSlotIdxTensor_.SetValue(k, slot);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        hitPerRankTensor_.SetValue(hitStride + dstRankId, slot);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        slotIdxPerRankTensor_.SetValue(dstRankId, slot + 1);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        // 本卡 token 直接写 receive window，远端 token 写本卡 send window。
        WriteSlotToLocal(dstRankId, slot);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteSlotToLocal(uint32_t dstRankId, uint32_t slot)
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    GM_ADDR rankGM = (dstRankId == epRankId_) ? localWinAddr_ : payloadStashWinAddr_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    GM_ADDR slotAddr = rankGM + (static_cast<uint64_t>(dstRankId) * axisMaxBS_ + slot) * perSlotBytes_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    slotGMTensor_.SetGlobalBuffer((__gm__ XType *)slotAddr);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyPad(slotGMTensor_, tokenSlotTensor_, slotCopyParams_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::WriteToRemoteWindow()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (startRankId_ >= epWorldSize_) { // 空闲核，直接返回
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    LocalTensor<int32_t> sendNumPerRankTensor = sendCntRankBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams sendCntCopyParams = {static_cast<uint16_t>(rankNumPerCore_), 1U,
                                        static_cast<uint16_t>(STATE_STRIDE), 0U};
                                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopy(sendNumPerRankTensor, sendCntPerRankGMTensor_[startRankId_ * WIN_ADDR_ALIGN / sizeof(int32_t)],
             sendCntCopyParams);
             PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t dstRankId = startRankId_; dstRankId < endRankId_; dstRankId++) {
        uint32_t sendTokenNum = sendNumPerRankTensor.GetValue((dstRankId - startRankId_) * UB_STRIDE + 1);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GM_ADDR notifyAddr = GetWinAddrByRankId(mc2Context_, dstRankId, slotWinStateOffset_) +
                             static_cast<uint64_t>(epRankId_) * dispatchNotifyCount_ * WIN_ADDR_ALIGN;
                             PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (unlikely(dstRankId == epRankId_)) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            uint32_t elemCnt = dispatchNotifyCount_ * UB_STRIDE;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            LocalTensor<int32_t> statusTensor = tempBuf_.GetWithOffset<int32_t>(elemCnt, 0);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            GlobalTensor<int32_t> statusGMTensor;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            statusGMTensor.SetGlobalBuffer((__gm__ int32_t *)(notifyAddr));
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            Duplicate<int32_t>(statusTensor, 1, elemCnt);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyParams copyParams = {static_cast<uint16_t>(dispatchNotifyCount_), 1U, 0U,
                                         static_cast<uint16_t>(STATE_STRIDE)};
                                         PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopy(statusGMTensor, statusTensor, copyParams);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            continue; //  本端slot已经写入win
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }

        uint64_t commHandle = GetCommHandle(mc2Context_, dstRankId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t slotCntPerNotify = sendTokenNum / dispatchNotifyCount_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t remainder = sendTokenNum % dispatchNotifyCount_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GM_ADDR remoteWinAddr = GetWinAddrByRankId(mc2Context_, dstRankId, winDataOffset_) + dstRankWinOffset_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GM_ADDR localSendWinAddr = payloadStashWinAddr_ + static_cast<uint64_t>(dstRankId) * axisMaxBS_ * perSlotBytes_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        for (uint32_t index = 0; index < dispatchNotifyCount_; index++) {
            uint32_t slotCnt = (index < remainder) ? slotCntPerNotify + 1 : slotCntPerNotify;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            uint64_t dataSize = (slotCnt > 0) ? static_cast<uint64_t>(perSlotBytes_) * slotCnt : UB_ALIGN;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            hcomm_.WriteWithNotifyNbi<true, PIPE_S, PIPE_MTE3, DEFAULT_WQE_CONFIG>(commHandle, remoteWinAddr,
                                                                                   localSendWinAddr, dataSize,
                                                                                   notifyAddr, 1);
                                                                                   PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            remoteWinAddr += dataSize;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            localSendWinAddr += dataSize;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            notifyAddr += WIN_ADDR_ALIGN;
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::GetSlotStartNum()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    slotIdxPerRankTensor_ = sendCntExpertBuf_.GetWithOffset<int32_t>(epWorldSizeAlign_ / sizeof(int32_t), 0);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    Duplicate<int32_t>(slotIdxPerRankTensor_, 0, epWorldSize_);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (aivId_ == 0) { // 0核起始id 均为0
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
    uint32_t groupCnt = Ceil(aivId_ * epWorldSizeAlign_, perGroupSizeAlign_ * 2);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    uint32_t copyNumPerGroup = perGroupSizeAlign_ * 2 / epWorldSizeAlign_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int32_t> counterTmpTensor = topkIdsBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<int32_t> hitTmpTensor = dstExpBuf_.GetWithOffset<int32_t>(epWorldSizeAlign_ / sizeof(int32_t), 0);
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    LocalTensor<uint8_t> sharedTmpTensor = tempBuf_.Get<uint8_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    for (uint32_t i = 0; i < groupCnt; i++) {
        uint32_t copyNum = (i == groupCnt - 1) ? (aivId_ - copyNumPerGroup * i) : copyNumPerGroup;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t gmOffset = i * copyNumPerGroup * counterAlign512_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyParams counterCopyParams = {static_cast<uint16_t>(copyNum), static_cast<uint16_t>(epWorldSizeAlign_),
                                            static_cast<uint16_t>(epWorldSizeAlign512_ - epWorldSizeAlign_), 0U};
                                            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(counterTmpTensor, scaleupCounterGMTensor_[gmOffset], counterCopyParams, padParams_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        const uint32_t shape[] = {copyNum, static_cast<uint32_t>(counterCnt_)};
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        ReduceSum<int32_t, AscendC::Pattern::Reduce::RA, true>(hitTmpTensor, counterTmpTensor, sharedTmpTensor,
                                                               shape, false);
                                                               PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Add(slotIdxPerRankTensor_, slotIdxPerRankTensor_, hitTmpTensor, epWorldSize_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (i + 1 < groupCnt) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::SendPhase()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if (startTokenId_ >= axisBS_) {
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        return;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }

    uint32_t groupCnt = perGroupSizeAlign_ / epWorldSizeAlign_;
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams topkCopyParams = {1U, static_cast<uint16_t>(axisK_ * TOPK_INFO_SIZE), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    DataCopyParams scalesCopyParams = {1U, static_cast<uint16_t>(scalesBytes_), 0U, 0U};
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    hitPerRankTensor_ = dstExpBuf_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

    for (uint32_t tokenId = startTokenId_; tokenId < endTokenId_; ++tokenId) {
        uint32_t topkOffset = tokenId * axisK_;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t groupIdx = (tokenId - startTokenId_) % groupCnt;
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        uint32_t hitStride = groupIdx * epWorldSizeAlign_ / sizeof(int32_t);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if (groupIdx == 0) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::S_V>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            Duplicate<int32_t>(hitPerRankTensor_, -1, perGroupSizeAlign_ / sizeof(int32_t));
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::V_S>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        xLocalTensor_ = perSlotQueue_.AllocTensor<XType>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        metaLocalTensor_ = xLocalTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        DataCopyPad(xLocalTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, padParams_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if constexpr (Std::IsSame<XType, fp8_e5m2_t>::value || Std::IsSame<XType, fp8_e4m3fn_t>::value) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPad(xLocalTensor_[hAlignSize_ / sizeof(XType)].template ReinterpretCast<ScalesType>(),
                        scalesGMTensor_[tokenId * scalesBytes_ / sizeof(ScalesType)], scalesCopyParams, padParams_);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        DataCopyPad(metaLocalTensor_, topkIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if constexpr (IsTopkWeights) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DataCopyPad(metaLocalTensor_[axisKAlign_].template ReinterpretCast<float>(),
                        topkWeightsGMTensor_[topkOffset], topkCopyParams, padParams_);
                        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        metaLocalTensor_.SetValue(2 * axisKAlign_, epRankId_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        metaLocalTensor_.SetValue(2 * axisKAlign_ + 1, tokenId);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        perSlotQueue_.EnQue(xLocalTensor_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        tokenSlotTensor_ = perSlotQueue_.DeQue<XType>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        topkIdxTensor_ = tokenSlotTensor_[metaOffset_ / sizeof(XType)].template ReinterpretCast<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        dstSlotIdxTensor_ = dstSlotQueue_.AllocTensor<int32_t>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement

        if constexpr (!IsCached) {
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            DedupAndSendDirect(hitStride);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        } else {
            DataCopyPad(dstSlotIdxTensor_, cachedSlotIdxGMTensor_[topkOffset], topkCopyParams, padParams_);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            dstSlotQueue_.EnQue(dstSlotIdxTensor_);
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            dstSlotIdxTensor_ = dstSlotQueue_.DeQue<int32_t>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            for (uint32_t k = 0; k < axisK_; k++) {
                int32_t slot = dstSlotIdxTensor_.GetValue(k);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                if (slot == -1) {
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    continue;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }
                int32_t expertId = topkIdxTensor_.GetValue(k);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                uint32_t dstRankId = expertId / moeExpertNumPerRank_;
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                if (hitPerRankTensor_.GetValue(hitStride + dstRankId) == slot) {
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                    continue;
                    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                }
                WriteSlotToLocal(dstRankId, slot);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
                hitPerRankTensor_.SetValue(hitStride + dstRankId, slot);
                PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            }
        }
        DataCopyPad(dstSlotIdxGMTensor_[topkOffset], dstSlotIdxTensor_, topkCopyParams);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        perSlotQueue_.FreeTensor<XType>(tokenSlotTensor_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        dstSlotQueue_.FreeTensor<int32_t>(dstSlotIdxTensor_);
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

template <TemplateMoeEpDispatchTypeClass>
__aicore__ inline void MoeEpDispatch<TemplateMoeEpDispatchTypeFunc>::Process()
{
    PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    if ASCEND_IS_AIV { // 全aiv处理
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        BufferInit();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        CalSendCnt();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        Communication();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        GetRecvCount();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        if constexpr (!IsCached) { // 计算起始slot id
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
            GetSlotStartNum();
            PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        }
        SendPhase();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        SyncAll<true>();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
        WriteToRemoteWindow();
        PipeBarrier<PIPE_ALL>(); // DEBUG: serialize every statement
    }
}

#endif

} // namespace MoeEpDispatchImpl

#endif // MOE_EP_DISPATCH_H
