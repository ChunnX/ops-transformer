/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_ARCH35_H
#define ENGRAM_FETCH_GRAD_ARCH35_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_grad_tiling_data.h"
#include "../engram_fetch_grad_utils.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace Mc2Kernel {

using namespace AscendC;

template <AscendC::HardEvent event>
__aicore__ inline void EngramFetchGradSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

constexpr uint32_t ENGRAM_GRAD_TIMEOUT_US = 5U * 1000U * 1000U;
constexpr uint32_t ENGRAM_GRAD_CYCLES_PER_US = 1000U;

class EngramFetchGradArch35 {
public:
    __aicore__ inline EngramFetchGradArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR permOut, GM_ADDR sendCountsOut,
                                GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut,
                                GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const EngramFetchGradTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void DrainChecked(uint64_t handle);
    __aicore__ inline void TimeoutCheck(uint64_t startTime);
    __aicore__ inline void UnsortGrad();
    __aicore__ inline void ExchangeGrad();
    __aicore__ inline void SendGradToPeers();
    __aicore__ inline void SendGradRemote(uint32_t dstRank, uint32_t senderIdx, int32_t sendCount,
                                          int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void RecvGradFromPeers();
    __aicore__ inline void UniqueScatterAdd();
    __aicore__ inline void MarkBitmap(uint32_t numRecv, uint32_t segStart, uint32_t segEnd);
    __aicore__ inline void ExtractUniqueEntries(uint32_t segStart, uint32_t segLen);
    __aicore__ inline void GatherUniqueEntries(uint32_t segLen);
    __aicore__ inline void ScatterAddGrad(uint32_t numRecv, uint32_t segStart, uint32_t segLen);
    __aicore__ inline void AccumulateGradForEntry(int32_t entryVal, uint32_t numRecv, LocalTensor<int32_t> &entryUb,
                                                  LocalTensor<uint8_t> &gradRaw, LocalTensor<float> &gradSumFp32,
                                                  LocalTensor<float> &gradTmp, GlobalTensor<int32_t> &recvLocalEntryGM);
    __aicore__ inline void WriteGradResult(int32_t outOffset, LocalTensor<float> &gradSumFp32,
                                           GlobalTensor<uint8_t> &gradUniqueGM);
    __aicore__ inline void InitFlagsAndDispls();
    __aicore__ inline void ClearWinCounters();
    __aicore__ inline void CrossRankBarrier();

    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank, uint32_t senderIdx);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                               uint32_t srcRank, uint32_t senderIdx);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value,
                                              uint32_t senderIdx);
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);

    TPipe *tpipe_{nullptr};
    GM_ADDR gradFetchedGM_{nullptr};
    GM_ADDR permOutGM_{nullptr};
    GM_ADDR sendCountsOutGM_{nullptr};
    GM_ADDR recvCountsOutGM_{nullptr};
    GM_ADDR recvLocalEntryOutGM_{nullptr};
    GM_ADDR numRecvOutGM_{nullptr};
    GM_ADDR gradUniqueOutGM_{nullptr};
    GM_ADDR uniqueLocalEntryOutGM_{nullptr};
    GM_ADDR numUniqueOutGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    uint32_t channelsPerRank_{1};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t hiddenDim_{0};
    int64_t totalRecv_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    uint64_t winSize_{0};
    uint64_t ubSize_{0};
    uint32_t tileBytes_{0};

    uint64_t barrierFlagOffset_{0};
    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t tokenDataOffset_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxTokensPerSlot_{0};

    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
    uint32_t sendersPerRank_{1};
    bool isSender_{false};
    bool isReceiver_{false};
    bool isFlagCore_{false};

    GM_ADDR gradSortedGM_{0};
    GM_ADDR recvGradGM_{0};
    GM_ADDR sendCountsGM_{0};
    GM_ADDR recvCountsGM_{0};
    GM_ADDR sdisplsGM_{0};
    GM_ADDR rdisplsGM_{0};
    GM_ADDR recvLocalEntryGM_{0};
    GM_ADDR counterScratchGM_{0};
    GM_ADDR flagScratchGM_{0};
    GM_ADDR segUniqueEntryGM_{0};
    GM_ADDR segCountGM_{0};

    uint64_t exchangeSendBytes_{0};
    uint64_t exchangeRecvBytes_{0};
    uint64_t exchangeSendCycles_{0};
    uint64_t exchangeRecvCycles_{0};

    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> hcommBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> indicesBuf_;
    TBuf<> gradSumBuf_;
    TBuf<> bitmapBuf_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchGradArch35::WriteNbiChecked(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    int32_t ret = hcomm_.WriteNbi(handle, dst, src, len);
    ascendc_assert(ret == 0, "WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
}

__aicore__ inline void EngramFetchGradArch35::DrainChecked(uint64_t handle)
{
    int32_t ret = hcomm_.Drain(handle);
}

__aicore__ inline void EngramFetchGradArch35::TimeoutCheck(uint64_t startTime)
{
    uint64_t nowUs = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
    ascendc_assert((nowUs - startTime) < ENGRAM_GRAD_TIMEOUT_US,
                   "timeout, rankId=%u, aivId=%u, elapsed=%llu us", rankId_, aivId_, nowUs - startTime);
}

__aicore__ inline GM_ADDR EngramFetchGradArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchGradArch35::GetCommHandle(uint32_t dstRank, uint32_t senderIdx)
{
    uint32_t roleOffset = isSender_ ? SENDER_CHANNEL_IDX : RECEIVER_CHANNEL_IDX;
    uint32_t channelIdx = senderIdx * 2U + roleOffset;
    if (channelIdx >= channelsPerRank_) {
        channelIdx = roleOffset;
    }
    return ctxPtr_->hcommHandle[dstRank * channelsPerRank_ + channelIdx];
}

__aicore__ inline int32_t EngramFetchGradArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                  uint32_t srcRank, uint32_t senderIdx)
{
    GM_ADDR counterAddr = winBase + counterOffset +
                          (static_cast<uint64_t>(srcRank) * sendersPerRank_ + senderIdx) * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counterGM);
    return counterGM.GetValue(0);
}

__aicore__ inline void EngramFetchGradArch35::WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                 int32_t value, uint32_t senderIdx)
{
    if (dstRank == rankId_)
        return;

    LocalTensor<int32_t> valLocal = statusBuf_.Get<int32_t>();
    valLocal.SetValue(0, value);
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();

    GM_ADDR srcAddr = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    GlobalTensor<int32_t> srcGM;
    srcGM.SetGlobalBuffer((__gm__ int32_t *)srcAddr);
    DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(srcGM, valLocal, valParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    uint64_t handle = GetCommHandle(dstRank, senderIdx);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) +
                                (static_cast<uint64_t>(rankId_) * sendersPerRank_ + senderIdx) * STATE_OFFSET;
    WriteNbiChecked(handle, remoteCounterAddr, srcAddr, sizeof(int32_t));
}

__aicore__ inline void EngramFetchGradArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(expectCount);
    uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
    while (sumOfFlag != compareFlag) {
        sumOfFlag = 0;
        for (uint32_t i = 0; i < numRanks_; i++) {
            GlobalTensor<int32_t> slotGM;
            slotGM.SetGlobalBuffer((__gm__ int32_t *)(statusWinBase + i * STATE_OFFSET));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(slotGM);
            int32_t flagVal = slotGM.GetValue(0);
            sumOfFlag += flagVal;
        }
        TimeoutCheck(startTime);
    }
}

__aicore__ inline void EngramFetchGradArch35::ClearStatusFlags(GM_ADDR statusWinBase)
{
    LocalTensor<int32_t> cleanTensor = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanTensor, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> statusGM;
    statusGM.SetGlobalBuffer((__gm__ int32_t *)statusWinBase);
    DataCopy(statusGM, cleanTensor, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    if (len == 0) {
        return;
    }

    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    LocalTensor<uint8_t> buf0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = pongBuf_.Get<uint8_t>();

    uint64_t off = 0;
    uint32_t tileIdx = 0;

    while (off < len) {
        uint32_t thisLen = (len - off > tileBytes_) ? tileBytes_ : static_cast<uint32_t>(len - off);
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> &buf = (bufIdx == 0U) ? buf0 : buf1;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }

        DataCopyExtParams mte2Params{1U, thisLen, 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(buf, srcGm[off], mte2Params, mte2Pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);

        DataCopyExtParams mte3Params{1U, thisLen, 0U, 0U, 0U};
        DataCopyPad(dstGm[off], buf, mte3Params);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);

        off += thisLen;
        tileIdx++;
    }

    uint32_t last = (tileIdx - 1U) % 2U;
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[last]);
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[last ^ 1U]);
    }
}

__aicore__ inline void EngramFetchGradArch35::Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR permOut,
                                                   GM_ADDR sendCountsOut, GM_ADDR recvCountsOut,
                                                   GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut, GM_ADDR gradUniqueOut,
                                                   GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
                                                   GM_ADDR workspaceGM, TPipe *pipe,
                                                   const EngramFetchGradTilingData *tilingData)
{
    tpipe_ = pipe;
    gradFetchedGM_ = gradFetched;
    permOutGM_ = permOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCountsOutGM_ = recvCountsOut;
    recvLocalEntryOutGM_ = recvLocalEntryOut;
    numRecvOutGM_ = numRecvOut;
    gradUniqueOutGM_ = gradUniqueOut;
    uniqueLocalEntryOutGM_ = uniqueLocalEntryOut;
    numUniqueOutGM_ = numUniqueOut;
    workspaceGM_ = workspaceGM;
    aivId_ = GetBlockIdx();
    totalBlocks_ = GetBlockNum();

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    channelsPerRank_ = ctxPtr_->channelsPerRank;
    if (channelsPerRank_ == 0) {
        channelsPerRank_ = 1;
    }

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    hiddenDim_ = tilingData->hiddenDim;
    totalRecv_ = tilingData->totalRecv;
    inputDtype_ = tilingData->inputDtype;
    outputDtype_ = tilingData->outputDtype;
    ubSize_ = tilingData->ubSize;
    tileBytes_ = tilingData->tileBytes;

    winSize_ = tilingData->commBufferSize;

    uint32_t halfBlocks = totalBlocks_ / 2U;
    if (halfBlocks == 0U) {
        halfBlocks = 1U;
    }
    sendersPerRank_ = halfBlocks / numRanks_;
    if (sendersPerRank_ < 1U) {
        sendersPerRank_ = 1U;
    }
    numSendCores_ = sendersPerRank_ * numRanks_;
    numRecvCores_ = totalBlocks_ - numSendCores_;
    if (numRecvCores_ < 1U) {
        numRecvCores_ = 1U;
        numSendCores_ = totalBlocks_ - 1U;
    }
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    uint64_t totalCounterEntries = static_cast<uint64_t>(numRanks_) * static_cast<uint64_t>(sendersPerRank_);
    uint64_t totalRanksOffset = totalCounterEntries * STATE_OFFSET;
    barrierFlagOffset_ = 0;
    tokenWriteOffset_ = totalRanksOffset;
    tokenReadOffset_ = tokenWriteOffset_ + totalRanksOffset;
    tokenDataOffset_ = 3U * totalRanksOffset;

    uint64_t tokenAreaRaw = (winSize_ > tokenDataOffset_) ? (winSize_ - tokenDataOffset_) : 0U;
    uint64_t tokenArea = tokenAreaRaw;
    if (hiddenBytes_ > 0) {
        uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
        tokenArea = Ceil(tokenAreaRaw, hiddenBytes) * hiddenBytes;
        if (tokenArea > tokenAreaRaw) {
            tokenArea = (tokenAreaRaw / hiddenBytes) * hiddenBytes;
        }
    }
    tokenSlotSize_ = tokenArea / numRanks_ / sendersPerRank_ / NUM_SLOTS;
    maxTokensPerSlot_ = static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_));

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(pingBuf_, tileBytes_);
    tpipe_->InitBuffer(pongBuf_, tileBytes_);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    sendCountsGM_ = sendCountsOutGM_;
    recvCountsGM_ = recvCountsOutGM_;
    recvLocalEntryGM_ = recvLocalEntryOutGM_;

    uint64_t wsOffset = 0;
    gradSortedGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
    int64_t totalRecvUB = totalRecv_;
    recvGradGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalRecvUB) * static_cast<uint64_t>(hiddenBytes_);
    sdisplsGM_ = workspaceGM_ + wsOffset;
    wsOffset += numRanks_ * UB_ALIGN;
    rdisplsGM_ = workspaceGM_ + wsOffset;
    wsOffset += numRanks_ * UB_ALIGN;
    counterScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    flagScratchGM_ = workspaceGM_ + wsOffset;
    wsOffset += UB_ALIGN;
    uint32_t segLen = (static_cast<uint32_t>(numEntriesPerRank_) + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t segLenAlign = Ceil(segLen * sizeof(int32_t), UB_ALIGN) * UB_ALIGN / sizeof(int32_t);
    segUniqueEntryGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalBlocks_) * static_cast<uint64_t>(segLenAlign) * sizeof(int32_t);
    segCountGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t);

    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tempBuf_, statusBufSize);
    uint32_t gradSumBufSize = Ceil(static_cast<uint32_t>(hiddenDim_) * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gradSumBuf_, gradSumBufSize);
    uint32_t indicesBufSize = 4U * 1024U;
    if (indicesBufSize < statusBufSize) {
        indicesBufSize = statusBufSize;
    }
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    uint32_t bitmapBufSize = Ceil(Ceil(segLen, BITS_PER_BYTE), ALIGNED_LEN_256) * ALIGNED_LEN_256;
    tpipe_->InitBuffer(bitmapBuf_, bitmapBufSize);

    uint64_t ubUsed = HCOMM_INIT_SIZE + 2U * tileBytes_ + statusBufSize + statusBufSize +
                      gradSumBufSize + indicesBufSize + bitmapBufSize;
    ascendc_assert(ubUsed <= ubSize_,
                   "UB overflow: used=%llu, ubSize=%llu, rankId=%u, hiddenDim=%ld, tileBytes=%u",
                   ubUsed, ubSize_, rankId_, hiddenDim_, tileBytes_);
}

__aicore__ inline void EngramFetchGradArch35::UnsortGrad()
{
    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t *)permOutGM_);

    for (int64_t i = static_cast<int64_t>(aivId_); i < numTokens_; i += static_cast<int64_t>(totalBlocks_)) {
        int32_t origIdx = permGM.GetValue(static_cast<uint32_t>(i));
        GM_ADDR src = gradFetchedGM_ + static_cast<uint64_t>(origIdx) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = gradSortedGM_ + static_cast<uint64_t>(i) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::ExchangeGrad()
{
    exchangeSendBytes_ = 0;
    exchangeRecvBytes_ = 0;
    exchangeSendCycles_ = 0;
    exchangeRecvCycles_ = 0;
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    uint64_t sendStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
    SendGradToPeers();
    uint64_t sendEnd = static_cast<uint64_t>(AscendC::GetSystemCycle());
    exchangeSendCycles_ = sendEnd - sendStart;

    uint64_t recvStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
    RecvGradFromPeers();
    uint64_t recvEnd = static_cast<uint64_t>(AscendC::GetSystemCycle());
    exchangeRecvCycles_ = recvEnd - recvStart;

    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::SendGradToPeers()
{
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t dstRank = aivId_ % numRanks_;
    uint32_t senderIdx = aivId_ / numRanks_;
    if (senderIdx >= sendersPerRank_) {
        return;
    }

    GM_ADDR sSlot = sendCountsGM_ + dstRank * UB_ALIGN;
    GlobalTensor<int32_t> sSlotGM;
    sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
    int32_t totalSendCount = sSlotGM.GetValue(0);

    GM_ADDR sDisplSlot = sdisplsGM_ + dstRank * UB_ALIGN;
    GlobalTensor<int64_t> sDisplSlotGM;
    sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
    int64_t sdispl = sDisplSlotGM.GetValue(0);

    int32_t baseShare = totalSendCount / static_cast<int32_t>(sendersPerRank_);
    int32_t mySendCount = baseShare;
    int64_t mySdispl = sdispl + static_cast<int64_t>(senderIdx) * baseShare;
    if (senderIdx == sendersPerRank_ - 1U) {
        mySendCount = totalSendCount - static_cast<int32_t>(senderIdx) * baseShare;
    }

    if (mySendCount <= 0) {
        return;
    }

    if (dstRank == rankId_) {
        GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * UB_ALIGN;
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);
        int64_t myRdispl = rdispl + static_cast<int64_t>(senderIdx) * baseShare;
        LocalCopySlice(recvGradGM_ + myRdispl * hiddenBytes_, gradSortedGM_ + mySdispl * hiddenBytes_,
                       static_cast<uint64_t>(mySendCount) * hiddenBytes_);
        exchangeSendBytes_ += static_cast<uint64_t>(mySendCount) * static_cast<uint64_t>(hiddenBytes_);
        return;
    }
    SendGradRemote(dstRank, senderIdx, mySendCount, mySdispl, localWinBase);
    exchangeSendBytes_ += static_cast<uint64_t>(mySendCount) * static_cast<uint64_t>(hiddenBytes_);
}

__aicore__ inline void EngramFetchGradArch35::SendGradRemote(uint32_t dstRank, uint32_t senderIdx,
                                                             int32_t sendCount, int64_t sdispl,
                                                             GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank, senderIdx);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank, senderIdx);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank, senderIdx);
                TimeoutCheck(startTime);
            }
        }

        uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        ascendc_assert(chunkLen != 0U, "ExchangeGrad chunkLen is 0");

        uint64_t slotBase = static_cast<uint64_t>(rankId_) * sendersPerRank_ * NUM_SLOTS * tokenSlotSize_ +
                            static_cast<uint64_t>(senderIdx) * NUM_SLOTS * tokenSlotSize_;
        uint64_t slotOffset = tokenDataOffset_ + slotBase + (localWriteCnt % NUM_SLOTS) * tokenSlotSize_;
        GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
        GM_ADDR srcAddr = gradSortedGM_ + (sdispl + totalSent) * hiddenBytes_;
        uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

        GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, tokenWriteOffset_) +
                                    (static_cast<uint64_t>(rankId_) * sendersPerRank_ + senderIdx) * STATE_OFFSET;
        int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                static_cast<uint64_t>(localWriteCnt + 1));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExTok_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);

        localWriteCnt++;
        totalSent += chunkLen;
    }

    DrainChecked(handle);
}

__aicore__ inline void EngramFetchGradArch35::RecvGradFromPeers()
{
    if (!isReceiver_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t recvIdx = aivId_ - numSendCores_;

    for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
        if (srcRank == rankId_)
            continue;

        GM_ADDR rSlot = recvCountsGM_ + srcRank * sizeof(int32_t);
        GlobalTensor<int32_t> rSlotGM;
        rSlotGM.SetGlobalBuffer((__gm__ int32_t *)rSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
        int32_t recvCount = rSlotGM.GetValue(0);
        exchangeRecvBytes_ += static_cast<uint64_t>(recvCount) * static_cast<uint64_t>(hiddenBytes_);

        GM_ADDR rDisplSlot = rdisplsGM_ + srcRank * UB_ALIGN;
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        int32_t baseShare = recvCount / static_cast<int32_t>(sendersPerRank_);
        uint32_t totalReceived = 0;
        uint32_t totalExpected = static_cast<uint32_t>(recvCount);

        uint32_t senderRecv[16];
        uint32_t senderReceived[16];
        uint32_t senderReadCnt[16];
        int64_t senderRecvBase[16];
        for (uint32_t si = 0; si < sendersPerRank_ && si < 16U; si++) {
            senderRecv[si] = (si == sendersPerRank_ - 1U) ? static_cast<uint32_t>(recvCount - static_cast<int32_t>(si) * baseShare) : static_cast<uint32_t>(baseShare);
            senderReceived[si] = 0;
            senderReadCnt[si] = 0;
            senderRecvBase[si] = rdispl + static_cast<int64_t>(si) * baseShare;
        }

        uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;

        while (totalReceived < totalExpected) {
            bool progressed = false;
            for (uint32_t si = 0; si < sendersPerRank_; si++) {
                if (senderReceived[si] >= senderRecv[si]) {
                    continue;
                }

                int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank, si);
                if (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= senderReadCnt[si]) {
                    continue;
                }

                uint32_t remaining = senderRecv[si] - senderReceived[si];
                uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;

                uint64_t slotBase = static_cast<uint64_t>(srcRank) * sendersPerRank_ * NUM_SLOTS * tokenSlotSize_ +
                                    static_cast<uint64_t>(si) * NUM_SLOTS * tokenSlotSize_;
                uint64_t slotOffset =
                    tokenDataOffset_ + slotBase + (senderReadCnt[si] % NUM_SLOTS) * tokenSlotSize_;
                GM_ADDR localSlotAddr = localWinBase + slotOffset;
                uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

                LocalCopySlice(recvGradGM_ + (senderRecvBase[si] + senderReceived[si]) * hiddenBytes_,
                               localSlotAddr, dataBytes);

                senderReadCnt[si]++;
                senderReceived[si] += chunkLen;
                totalReceived += chunkLen;
                WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(senderReadCnt[si]), si);
                progressed = true;
            }

            if (!progressed) {
                TimeoutCheck(startTime);
            }
        }
    }
}

__aicore__ inline void EngramFetchGradArch35::UniqueScatterAdd()
{
    GlobalTensor<int32_t> numRecvGM;
    numRecvGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(numRecvGM);
    int32_t numRecv = numRecvGM.GetValue(0);
    if (numRecv == 0) {
        if (aivId_ == 0) {
            LocalTensor<int32_t> tmp = statusBuf_.Get<int32_t>();
            tmp.SetValue(0, 0);
            EngramFetchGradSyncFunc<HardEvent::S_MTE3>();

            GlobalTensor<int32_t> numUniqueGM;
            numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM_);
            DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
            DataCopyPad(numUniqueGM, tmp, params);
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        return;
    }

    uint32_t segLen = (static_cast<uint32_t>(numEntriesPerRank_) + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t segStart = aivId_ * segLen;
    uint32_t segEnd = segStart + segLen;
    if (segEnd > static_cast<uint32_t>(numEntriesPerRank_)) {
        segEnd = static_cast<uint32_t>(numEntriesPerRank_);
    }

    MarkBitmap(static_cast<uint32_t>(numRecv), segStart, segEnd);
    ExtractUniqueEntries(segStart, segLen);
    SyncAll<true>();

    GatherUniqueEntries(segLen);
    SyncAll<true>();

    ScatterAddGrad(static_cast<uint32_t>(numRecv), segStart, segLen);
    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::MarkBitmap(uint32_t numRecv, uint32_t segStart, uint32_t segEnd)
{
    LocalTensor<uint8_t> bitmap = bitmapBuf_.Get<uint8_t>();
    uint32_t segLen = segEnd - segStart;
    uint32_t bitmapBytes = Ceil(segLen, BITS_PER_BYTE);
    uint32_t bitmapAlignBytes = Ceil(bitmapBytes, ALIGNED_LEN_256) * ALIGNED_LEN_256;
    Duplicate<uint8_t>(bitmap, 0, bitmapAlignBytes);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> entryUb = indicesBuf_.Get<int32_t>();
    constexpr uint32_t entryBufCap = ENTRY_BATCH_CAP;

    GlobalTensor<int32_t> recvLocalEntryGM;
    recvLocalEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);

    uint32_t batchStart = 0;
    while (batchStart < numRecv) {
        uint32_t batchLen = numRecv - batchStart;
        if (batchLen > entryBufCap) {
            batchLen = entryBufCap;
        }

        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(entryUb, recvLocalEntryGM[batchStart], cpParams, cpPad);
        EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t globalIdx = entryUb.GetValue(i);
            int64_t localIdx = static_cast<int64_t>(globalIdx) -
                               static_cast<int64_t>(rankId_) * static_cast<int64_t>(numEntriesPerRank_);
            if (localIdx < 0) {
                continue;
            }
            uint32_t uLocalIdx = static_cast<uint32_t>(localIdx);
            if (uLocalIdx < segStart || uLocalIdx >= segEnd) {
                continue;
            }
            uint32_t bitIdx = uLocalIdx - segStart;
            uint32_t byteIdx = bitIdx / BITS_PER_BYTE;
            uint32_t bitOffset = bitIdx % BITS_PER_BYTE;
            uint8_t mask = static_cast<uint8_t>(1U << bitOffset);
            uint8_t curVal = bitmap.GetValue(byteIdx);
            bitmap.SetValue(byteIdx, curVal | mask);
        }
        batchStart += batchLen;
    }
}

__aicore__ inline void EngramFetchGradArch35::ExtractUniqueEntries(uint32_t segStart, uint32_t segLen)
{
    LocalTensor<uint8_t> bitmap = bitmapBuf_.Get<uint8_t>();

    GM_ADDR segUniqueAddr = segUniqueEntryGM_ + static_cast<uint64_t>(aivId_) *
                                                    Ceil(segLen * sizeof(int32_t), UB_ALIGN) * UB_ALIGN /
                                                    sizeof(int32_t) * sizeof(int32_t);
    GlobalTensor<int32_t> segUniqueGM;
    segUniqueGM.SetGlobalBuffer((__gm__ int32_t *)segUniqueAddr);

    uint32_t segCount = 0;
    LocalTensor<int32_t> entryTmpUb = indicesBuf_.Get<int32_t>();
    for (uint32_t bitIdx = 0; bitIdx < segLen; bitIdx++) {
        uint32_t byteIdx = bitIdx / BITS_PER_BYTE;
        uint32_t bitOffset = bitIdx % BITS_PER_BYTE;
        uint8_t curVal = bitmap.GetValue(byteIdx);
        if (curVal & (1U << bitOffset)) {
            int32_t entryVal = static_cast<int32_t>(segStart + bitIdx);
            entryTmpUb.SetValue(0, entryVal);
            EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
            DataCopyParams entryParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
            DataCopyPad(segUniqueGM[segCount], entryTmpUb, entryParams);
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
            segCount++;
        }
    }

    LocalTensor<int32_t> segCountUb = statusBuf_.Get<int32_t>();
    segCountUb.SetValue(0, static_cast<int32_t>(segCount));
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    GM_ADDR segCountAddr = segCountGM_ + static_cast<uint64_t>(aivId_) * sizeof(int32_t);
    GlobalTensor<int32_t> segCountGM;
    segCountGM.SetGlobalBuffer((__gm__ int32_t *)segCountAddr);
    DataCopyParams segCountParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(segCountGM, segCountUb, segCountParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::GatherUniqueEntries(uint32_t segLen)
{
    if (aivId_ != 0) {
        return;
    }

    LocalTensor<int32_t> segCountsUb = indicesBuf_.Get<int32_t>();
    DataCopyExtParams cpParams{1U, static_cast<uint16_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    GlobalTensor<int32_t> segCountGmRead;
    segCountGmRead.SetGlobalBuffer((__gm__ int32_t *)segCountGM_);
    DataCopyPad(segCountsUb, segCountGmRead, cpParams, cpPad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    GlobalTensor<int32_t> uniqueLocalEntryGM;
    uniqueLocalEntryGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM_);

    int32_t globalOffset = 0;
    for (uint32_t core = 0; core < totalBlocks_; core++) {
        int32_t count = segCountsUb.GetValue(core);
        if (count > 0) {
            GM_ADDR src = segUniqueEntryGM_ + static_cast<uint64_t>(core) *
                                                  Ceil(segLen * sizeof(int32_t), UB_ALIGN) * UB_ALIGN /
                                                  sizeof(int32_t) * sizeof(int32_t);
            GM_ADDR dst = uniqueLocalEntryOutGM_ + static_cast<uint64_t>(globalOffset) * sizeof(int32_t);
            LocalCopySlice(dst, src, static_cast<uint64_t>(count) * sizeof(int32_t));
        }
        globalOffset += count;
    }

    LocalTensor<int32_t> tmp32 = statusBuf_.Get<int32_t>();
    tmp32.SetValue(0, globalOffset);
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> numUniqueOutGM;
    numUniqueOutGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM_);
    DataCopyParams numParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(numUniqueOutGM, tmp32, numParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::ScatterAddGrad(uint32_t numRecv, uint32_t segStart, uint32_t segLen)
{
    int32_t prefixCount = 0;
    {
        GlobalTensor<int32_t> segCountGmRead;
        segCountGmRead.SetGlobalBuffer((__gm__ int32_t *)segCountGM_);
        for (uint32_t core = 0; core < aivId_; core++) {
            GM_ADDR countAddr = segCountGM_ + static_cast<uint64_t>(core) * sizeof(int32_t);
            GlobalTensor<int32_t> countGM;
            countGM.SetGlobalBuffer((__gm__ int32_t *)countAddr);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(countGM);
            prefixCount += countGM.GetValue(0);
        }
    }

    LocalTensor<uint8_t> bitmap = bitmapBuf_.Get<uint8_t>();
    LocalTensor<int32_t> entryUb = indicesBuf_.Get<int32_t>();
    LocalTensor<float> gradSumFp32 = gradSumBuf_.Get<float>();
    LocalTensor<float> gradTmp = pingBuf_.Get<float>();
    LocalTensor<uint8_t> gradRaw = pongBuf_.Get<uint8_t>();

    GlobalTensor<int32_t> recvLocalEntryGM;
    recvLocalEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
    GlobalTensor<uint8_t> gradUniqueGM;
    gradUniqueGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM_);

    uint32_t uniqueIdx = 0;

    for (uint32_t bitIdx = 0; bitIdx < segLen; bitIdx++) {
        uint32_t byteIdx = bitIdx / BITS_PER_BYTE;
        uint32_t bitOffset = bitIdx % BITS_PER_BYTE;
        uint8_t curVal = bitmap.GetValue(byteIdx);
        if (!(curVal & (1U << bitOffset))) {
            continue;
        }

        int32_t entryVal = static_cast<int32_t>(segStart + bitIdx);
        int32_t outOffset = prefixCount + static_cast<int32_t>(uniqueIdx);

        Duplicate<float>(gradSumFp32, 0.0f, static_cast<uint32_t>(hiddenDim_));
        PipeBarrier<PIPE_V>();

        AccumulateGradForEntry(entryVal, numRecv, entryUb, gradRaw, gradSumFp32, gradTmp, recvLocalEntryGM);

        PipeBarrier<PIPE_V>();
        WriteGradResult(outOffset, gradSumFp32, gradUniqueGM);

        uniqueIdx++;
    }
}

__aicore__ inline void
EngramFetchGradArch35::AccumulateGradForEntry(int32_t entryVal, uint32_t numRecv, LocalTensor<int32_t> &entryUb,
                                              LocalTensor<uint8_t> &gradRaw, LocalTensor<float> &gradSumFp32,
                                              LocalTensor<float> &gradTmp, GlobalTensor<int32_t> &recvLocalEntryGM)
{
    uint32_t batchStart = 0;
    while (batchStart < numRecv) {
        uint32_t batchLen = numRecv - batchStart;
        if (batchLen > ENTRY_BATCH_CAP) {
            batchLen = ENTRY_BATCH_CAP;
        }

        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(entryUb, recvLocalEntryGM[batchStart], cpParams, cpPad);
        EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t globalIdx = entryUb.GetValue(i);
            int64_t localIdx = static_cast<int64_t>(globalIdx) -
                               static_cast<int64_t>(rankId_) * static_cast<int64_t>(numEntriesPerRank_);
            if (localIdx != entryVal) {
                continue;
            }

            uint32_t recvIdx = batchStart + i;
            GM_ADDR gradAddr = recvGradGM_ + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
            DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
            DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
            GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
            DataCopyPad(gradRaw, gradSrcGM, gradParams, gradPad);
            EngramFetchGradSyncFunc<HardEvent::MTE2_V>();

            if (inputDtype_ == ENGRAM_DT_BFLOAT16) {
                LocalTensor<bfloat16_t> inTensor = pongBuf_.Get<bfloat16_t>();
                Cast(gradTmp, inTensor, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(hiddenDim_));
                PipeBarrier<PIPE_V>();
                Add(gradSumFp32, gradSumFp32, gradTmp, static_cast<uint32_t>(hiddenDim_));
            } else if (inputDtype_ == ENGRAM_DT_FLOAT16) {
                LocalTensor<half> inTensor = pongBuf_.Get<half>();
                Cast(gradTmp, inTensor, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(hiddenDim_));
                PipeBarrier<PIPE_V>();
                Add(gradSumFp32, gradSumFp32, gradTmp, static_cast<uint32_t>(hiddenDim_));
            } else {
                LocalTensor<float> inTensor = pongBuf_.Get<float>();
                Add(gradSumFp32, gradSumFp32, inTensor, static_cast<uint32_t>(hiddenDim_));
            }
            EngramFetchGradSyncFunc<HardEvent::V_MTE2>();
        }
        batchStart += batchLen;
    }
}

__aicore__ inline void EngramFetchGradArch35::WriteGradResult(int32_t outOffset, LocalTensor<float> &gradSumFp32,
                                                              GlobalTensor<uint8_t> &gradUniqueGM)
{
    if (outputDtype_ == ENGRAM_DT_BFLOAT16) {
        LocalTensor<bfloat16_t> outTensor = pingBuf_.Get<bfloat16_t>();
        Cast(outTensor, gradSumFp32, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(hiddenDim_));
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(gradUniqueGM[static_cast<uint64_t>(outOffset) * hiddenDim_ * sizeof(bfloat16_t)],
                    outTensor.ReinterpretCast<uint8_t>(),
                    {1U, static_cast<uint16_t>(hiddenDim_ * sizeof(bfloat16_t)), 0U, 0U});
    } else if (outputDtype_ == ENGRAM_DT_FLOAT16) {
        LocalTensor<half> outTensor = pingBuf_.Get<half>();
        Cast(outTensor, gradSumFp32, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(hiddenDim_));
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(gradUniqueGM[static_cast<uint64_t>(outOffset) * hiddenDim_ * sizeof(half)],
                    outTensor.ReinterpretCast<uint8_t>(),
                    {1U, static_cast<uint16_t>(hiddenDim_ * sizeof(half)), 0U, 0U});
    } else {
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        DataCopyPad(gradUniqueGM[static_cast<uint64_t>(outOffset) * hiddenDim_ * sizeof(float)],
                    gradSumFp32.ReinterpretCast<uint8_t>(),
                    {1U, static_cast<uint16_t>(hiddenDim_ * sizeof(float)), 0U, 0U});
    }
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::InitFlagsAndDispls()
{
    if (aivId_ != 0) {
        return;
    }

    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagInit;
    flagInit.SetGlobalBuffer((__gm__ int32_t *)flagScratchGM_);
    DataCopy(flagInit, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    GlobalTensor<int32_t> sendCountsOutGM;
    sendCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
    GlobalTensor<int32_t> recvCountsOutGM;
    recvCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);

    int64_t sAccum = 0;
    int64_t rAccum = 0;
    for (uint32_t r = 0; r < numRanks_; r++) {
        LocalTensor<int64_t> displVal = tempBuf_.Get<int64_t>();
        displVal.SetValue(0, sAccum);
        EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> sdGM;
        sdGM.SetGlobalBuffer((__gm__ int64_t *)(sdisplsGM_ + r * UB_ALIGN));
        DataCopyParams displParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        DataCopyPad(sdGM, displVal, displParams);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        int32_t sCount = sendCountsOutGM.GetValue(r * SENDCOUNT_STRIDE_RATIO);
        sAccum += sCount;

        displVal.SetValue(0, rAccum);
        EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> rdGM;
        rdGM.SetGlobalBuffer((__gm__ int64_t *)(rdisplsGM_ + r * UB_ALIGN));
        DataCopyPad(rdGM, displVal, displParams);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        int32_t rCount = recvCountsOutGM.GetValue(r);
        rAccum += rCount;
    }
}

__aicore__ inline void EngramFetchGradArch35::ClearWinCounters()
{
    if (aivId_ != 0) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    uint32_t cleanElements = numRanks_ * sendersPerRank_ * STATE_OFFSET / sizeof(int32_t);
    uint32_t totalCleanElements = 2U * cleanElements;

    LocalTensor<int32_t> cleanLocal = pingBuf_.Get<int32_t>(totalCleanElements);
    Duplicate<int32_t>(cleanLocal, 0, totalCleanElements);
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();

    GlobalTensor<int32_t> countersGM;
    countersGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
    DataCopy(countersGM, cleanLocal, totalCleanElements);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::CrossRankBarrier()
{
    if (isSender_) {
        uint32_t dstRank = aivId_ % numRanks_;
        uint32_t senderIdx = aivId_ / numRanks_;
        if (senderIdx == 0U && dstRank != rankId_) {
            uint64_t handle = GetCommHandle(dstRank, 0U);
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, barrierFlagOffset_) + rankId_ * STATE_OFFSET;
            WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
        }
    }
    if (aivId_ == 0) {
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GM_ADDR localFlagAddr = localWinBase + barrierFlagOffset_ + rankId_ * STATE_OFFSET;
        LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
        Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        GlobalTensor<int32_t> flagGM;
        flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
        DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
    }
    SyncAll<true>();

    if (isFlagCore_) {
        GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
        GM_ADDR localFlagBase = localWinBase + barrierFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }
}

__aicore__ inline void EngramFetchGradArch35::Process()
{
    if ASCEND_IS_AIV {
        uint64_t cycleStartTotal = static_cast<uint64_t>(AscendC::GetSystemCycle());

        uint64_t cycleStartInit = static_cast<uint64_t>(AscendC::GetSystemCycle());
        InitFlagsAndDispls();
        uint64_t cycleEndInit = static_cast<uint64_t>(AscendC::GetSystemCycle());
        SyncAll<true>();

        uint64_t cycleStartClear = static_cast<uint64_t>(AscendC::GetSystemCycle());
        ClearWinCounters();
        uint64_t cycleEndClear = static_cast<uint64_t>(AscendC::GetSystemCycle());
        SyncAll<true>();

        uint64_t cycleStartBarrier = static_cast<uint64_t>(AscendC::GetSystemCycle());
        CrossRankBarrier();
        uint64_t cycleEndBarrier = static_cast<uint64_t>(AscendC::GetSystemCycle());

        uint64_t cycleStart1 = static_cast<uint64_t>(AscendC::GetSystemCycle());
        UnsortGrad();
        uint64_t cycleEnd1 = static_cast<uint64_t>(AscendC::GetSystemCycle());
        uint64_t cycleStart = static_cast<uint64_t>(AscendC::GetSystemCycle());
        ExchangeGrad();
        uint64_t cycleEnd = static_cast<uint64_t>(AscendC::GetSystemCycle());

        uint64_t cycleStartScatter = static_cast<uint64_t>(AscendC::GetSystemCycle());
        UniqueScatterAdd();
        uint64_t cycleEndScatter = static_cast<uint64_t>(AscendC::GetSystemCycle());

        uint64_t cycleEndTotal = static_cast<uint64_t>(AscendC::GetSystemCycle());
        uint64_t cycleDelta = cycleEnd - cycleStart;
        uint64_t sendBw = (exchangeSendCycles_ > 0) ? exchangeSendBytes_ / (exchangeSendCycles_ / 1000) / 536 / 2 : 0;
        uint64_t recvBw = (exchangeRecvCycles_ > 0) ? exchangeRecvBytes_ / (exchangeRecvCycles_ / 1000) / 536 / 2 : 0;
        uint64_t totalSendBytes = static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
        uint64_t totalCycles = cycleEndTotal - cycleStartTotal;
        uint64_t totalBw = (totalCycles > 1000) ? totalSendBytes / (totalCycles / 1000) / 536 / 2 : 0;
        printf("[EngramFetchGrad] rankId=%u, aivId=%u, total cycles=%llu (totalSendBytes=%llu, totalBW=%llu), "
               "InitFlagsAndDispls=%llu, ClearWinCounters=%llu, CrossRankBarrier=%llu, UnsortGrad=%llu, "
               "ExchangeGrad=%llu (sendCycles=%llu/sendBytes=%llu /sendBW=%llu, "
               "recvCycles=%llu/recvBytes=%llu /recvBW=%llu), UniqueScatterAdd=%llu\n",
               rankId_, aivId_,
               static_cast<unsigned long long>(totalCycles),
               static_cast<unsigned long long>(totalSendBytes),
               static_cast<unsigned long long>(totalBw),
               static_cast<unsigned long long>(cycleEndInit - cycleStartInit),
               static_cast<unsigned long long>(cycleEndClear - cycleStartClear),
               static_cast<unsigned long long>(cycleEndBarrier - cycleStartBarrier),
               static_cast<unsigned long long>(cycleEnd1 - cycleStart1),
               static_cast<unsigned long long>(cycleDelta),
               static_cast<unsigned long long>(exchangeSendCycles_),
               static_cast<unsigned long long>(exchangeSendBytes_),
               static_cast<unsigned long long>(sendBw),
               static_cast<unsigned long long>(exchangeRecvCycles_),
               static_cast<unsigned long long>(exchangeRecvBytes_),
               static_cast<unsigned long long>(recvBw),
               static_cast<unsigned long long>(cycleEndScatter - cycleStartScatter));
    }
}

} // namespace Mc2Kernel

#endif
