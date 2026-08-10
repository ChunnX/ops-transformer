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
#include "sort_with_index_int32.h"

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
    __aicore__ inline void SendGradLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void SendGradRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase);
    __aicore__ inline void RecvGradFromPeers();
    __aicore__ inline void UniqueScatterAdd();
    __aicore__ inline void InitCompanion(uint32_t numRecv);
    __aicore__ inline void RunSort(uint32_t numRecv);
    __aicore__ inline void WriteNumUniqueZero();
    __aicore__ inline void ZeroGradUnique(uint32_t numRecv);
    __aicore__ inline void CountUniquesParallel(uint32_t numRecv);
    __aicore__ inline void ScatterAddAtomicParallel(uint32_t numRecv);
    __aicore__ inline void CastGradUniqueToOutput();
    __aicore__ inline void InitFlagsAndDispls();
    __aicore__ inline void ClearWinCounters();
    __aicore__ inline void CrossRankBarrier();

    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value);
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

    uint64_t uniqueScatterAddStartCycle_{0};
    uint64_t uniqueScatterAddEndCycle_{0};
    uint64_t usaCompanionCycle_{0};
    uint64_t usaSortCycle_{0};
    uint64_t usaZeroCountCycle_{0};
    uint64_t usaScatterAddCycle_{0};
    uint64_t usaCastCycle_{0};
    uint64_t processStartCycle_{0};
    uint64_t processEndCycle_{0};

    uint64_t barrierFlagOffset_{0};
    uint64_t tokenFlagOffset_{0};
    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t tokenDataOffset_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxTokensPerSlot_{0};

    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
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
    GM_ADDR segCountGM_{0};
    GM_ADDR coreStartGM_{0};

    GM_ADDR sortTempValGM_{0};
    GM_ADDR sortTempIdxGM_{0};
    GM_ADDR sortCompanionGM_{0};
    GM_ADDR sortHistGm_{0};
    GM_ADDR sortPrefixGm_{0};
    GM_ADDR sortOffsetsGm_{0};
    GM_ADDR gradUniqueFp32GM_{0};

    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> hcommBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> indicesBuf_;
    TBuf<> gradSumBuf_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    SortWithIndexInt32::SortWithIndexKernel sorter_;
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

__aicore__ inline uint64_t EngramFetchGradArch35::GetCommHandle(uint32_t dstRank)
{
    uint32_t channelIdx = isSender_ ? SENDER_CHANNEL_IDX : RECEIVER_CHANNEL_IDX;
    return ctxPtr_->hcommHandle[dstRank * channelsPerRank_ + channelIdx];
}

__aicore__ inline int32_t EngramFetchGradArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                  uint32_t srcRank)
{
    GM_ADDR counterAddr = winBase + counterOffset + srcRank * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counterGM);
    return counterGM.GetValue(0);
}

__aicore__ inline void EngramFetchGradArch35::WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset,
                                                                 int32_t value)
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

    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) + rankId_ * STATE_OFFSET;
    WriteNbiChecked(handle, remoteCounterAddr, srcAddr, sizeof(int32_t));
    DrainChecked(handle);
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
    LocalTensor<uint8_t> tmp0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> tmp1 = pongBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);
    uint32_t tileLen = TILE_BYTES;
    uint64_t off = 0;
    uint32_t tileIdx = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? tileLen : (len - off);
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? tmp0 : tmp1;
        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }
        DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
        DataCopyPad(dstGm[off], tmp, mte3Params);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        off += thisLen;
        tileIdx++;
    }
    if (tileIdx >= 1U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
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

    winSize_ = tilingData->commBufferSize;

    uint64_t totalRanksOffset = numRanks_ * STATE_OFFSET;
    barrierFlagOffset_ = 0;
    tokenFlagOffset_ = totalRanksOffset;
    tokenWriteOffset_ = tokenFlagOffset_ + totalRanksOffset;
    tokenReadOffset_ = tokenWriteOffset_ + totalRanksOffset;
    tokenDataOffset_ = 4U * numRanks_ * STATE_OFFSET;

    uint64_t tokenAreaRaw = (winSize_ > tokenDataOffset_) ? (winSize_ - tokenDataOffset_) : 0U;
    uint64_t tokenArea = tokenAreaRaw;
    if (hiddenBytes_ > 0) {
        uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
        tokenArea = Ceil(tokenAreaRaw, hiddenBytes) * hiddenBytes;
        if (tokenArea > tokenAreaRaw) {
            tokenArea = (tokenAreaRaw / hiddenBytes) * hiddenBytes;
        }
    }
    tokenSlotSize_ = tokenArea / numRanks_ / NUM_SLOTS;
    maxTokensPerSlot_ = static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_));

    numSendCores_ = totalBlocks_ / 2U;
    if (numSendCores_ == 0U) {
        numSendCores_ = 1U;
    }
    numRecvCores_ = totalBlocks_ - numSendCores_;
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
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
    segCountGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t);
    coreStartGM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalBlocks_) * sizeof(int32_t);

    uint32_t maxSortCount = static_cast<uint32_t>(totalRecv_);
    uint32_t sortTileCountByCore = static_cast<uint32_t>(totalBlocks_);
    uint32_t sortTileCountByDefault = (maxSortCount + SortWithIndexInt32::DEFAULT_TILE_SIZE - 1U) / SortWithIndexInt32::DEFAULT_TILE_SIZE;
    uint32_t sortTileCount = (sortTileCountByCore > sortTileCountByDefault) ? sortTileCountByCore : sortTileCountByDefault;
    uint64_t sortTempSize = Ceil(static_cast<uint64_t>(maxSortCount) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    sortTempValGM_ = workspaceGM_ + wsOffset;
    wsOffset += sortTempSize;
    sortTempIdxGM_ = workspaceGM_ + wsOffset;
    wsOffset += sortTempSize;
    sortCompanionGM_ = workspaceGM_ + wsOffset;
    wsOffset += sortTempSize;
    sortHistGm_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(sortTileCount) * SortWithIndexInt32::HISTOGRAM_BINS * sizeof(int32_t);
    sortPrefixGm_ = workspaceGM_ + wsOffset;
    wsOffset += SortWithIndexInt32::HISTOGRAM_BINS * sizeof(int32_t);
    sortOffsetsGm_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(sortTileCount) * SortWithIndexInt32::HISTOGRAM_BINS * sizeof(int32_t);

    gradUniqueFp32GM_ = workspaceGM_ + wsOffset;
    wsOffset += static_cast<uint64_t>(totalRecv_) * static_cast<uint64_t>(hiddenDim_) * sizeof(float);

    sorter_.Init(maxSortCount, totalBlocks_, SortWithIndexInt32::DEFAULT_TILE_SIZE, *tpipe_);

    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(tempBuf_, statusBufSize);
    uint32_t gradSumBufSize = Ceil(static_cast<uint32_t>(hiddenDim_) * sizeof(float) * GRAD_SUB_BATCH, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gradSumBuf_, gradSumBufSize);
    uint32_t indicesBufSize = 4U * 1024U;
    if (indicesBufSize < statusBufSize) {
        indicesBufSize = statusBufSize;
    }
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
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
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    SendGradToPeers();
    RecvGradFromPeers();

    if (isFlagCore_) {
        GM_ADDR localFlagBase = localWinBase + tokenFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::SendGradToPeers()
{
    if (!isSender_) {
        return;
    }

    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];

    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
        GM_ADDR sSlot = sendCountsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t sendCount = sSlotGM.GetValue(0);

        GM_ADDR sDisplSlot = sdisplsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int64_t> sDisplSlotGM;
        sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
        int64_t sdispl = sDisplSlotGM.GetValue(0);

        if (dstRank == rankId_) {
            SendGradLocal(sendCount, sdispl, localWinBase);
            continue;
        }
        SendGradRemote(dstRank, sendCount, sdispl, localWinBase);
    }
}

__aicore__ inline void EngramFetchGradArch35::SendGradLocal(int32_t sendCount, int64_t sdispl, GM_ADDR localWinBase)
{
    if (sendCount > 0) {
        GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * UB_ALIGN;
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        LocalCopySlice(recvGradGM_ + rdispl * hiddenBytes_, gradSortedGM_ + sdispl * hiddenBytes_,
                       static_cast<uint64_t>(sendCount) * hiddenBytes_);
    }
    GM_ADDR localFlagAddr = localWinBase + tokenFlagOffset_ + rankId_ * STATE_OFFSET;
    LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    GlobalTensor<int32_t> flagGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
    DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::SendGradRemote(uint32_t dstRank, int32_t sendCount, int64_t sdispl,
                                                             GM_ADDR localWinBase)
{
    uint64_t handle = GetCommHandle(dstRank);
    uint32_t totalSent = 0;
    uint32_t localWriteCnt = 0;

    while (totalSent < static_cast<uint32_t>(sendCount)) {
        if (totalBlocks_ > 1U) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                TimeoutCheck(startTime);
            }
        }

        uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
        uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
        ascendc_assert(chunkLen != 0U, "ExchangeGrad chunkLen is 0");

        uint64_t slotOffset =
            tokenDataOffset_ + rankId_ * NUM_SLOTS * tokenSlotSize_ + (localWriteCnt % NUM_SLOTS) * tokenSlotSize_;
        GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
        GM_ADDR srcAddr = gradSortedGM_ + (sdispl + totalSent) * hiddenBytes_;
        uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

        GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, tokenWriteOffset_) + rankId_ * STATE_OFFSET;
        int32_t ret = hcomm_.WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                static_cast<uint64_t>(localWriteCnt + 1));
        ascendc_assert(ret == 0, "WriteWithNotifyNbi failed, ret=%d, tag=ExTok_data, rankId=%u, dstRank=%u", ret,
                       rankId_, dstRank);
        DrainChecked(handle);

        localWriteCnt++;
        totalSent += chunkLen;
    }

    GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, tokenFlagOffset_) + rankId_ * STATE_OFFSET;
    WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
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

        GM_ADDR rDisplSlot = rdisplsGM_ + srcRank * UB_ALIGN;
        GlobalTensor<int64_t> rDisplSlotGM;
        rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
        int64_t rdispl = rDisplSlotGM.GetValue(0);

        uint32_t totalReceived = 0;
        uint32_t localReadCnt = 0;

        while (totalReceived < static_cast<uint32_t>(recvCount)) {
            uint64_t startTime = static_cast<uint64_t>(AscendC::GetSystemCycle()) / ENGRAM_GRAD_CYCLES_PER_US;
            int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
            while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= localReadCnt) {
                remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
                TimeoutCheck(startTime);
            }

            uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
            uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;

            uint64_t slotOffset =
                tokenDataOffset_ + srcRank * NUM_SLOTS * tokenSlotSize_ + (localReadCnt % NUM_SLOTS) * tokenSlotSize_;
            GM_ADDR localSlotAddr = localWinBase + slotOffset;
            uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

            LocalCopySlice(recvGradGM_ + (rdispl + totalReceived) * hiddenBytes_, localSlotAddr, dataBytes);

            localReadCnt++;
            totalReceived += chunkLen;
            WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(localReadCnt));
        }
    }
}

__aicore__ inline void EngramFetchGradArch35::InitCompanion(uint32_t numRecv)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t start = aivId_ * chunk;
    uint32_t end = start + chunk;
    if (end > numRecv) {
        end = numRecv;
    }

    LocalTensor<int32_t> idxLocal = indicesBuf_.Get<int32_t>();
    GlobalTensor<int32_t> compGM;
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM_);

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = end - cur;
        if (batchLen > ENTRY_BATCH_CAP) {
            batchLen = ENTRY_BATCH_CAP;
        }
        for (uint32_t i = 0; i < batchLen; i++) {
            idxLocal.SetValue(i, static_cast<int32_t>(cur + i));
        }
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
        DataCopyParams cp = {1U, static_cast<uint16_t>(batchLen * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(compGM[cur], idxLocal, cp);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        cur += batchLen;
    }
}

__aicore__ inline void EngramFetchGradArch35::RunSort(uint32_t numRecv)
{
    sorter_.SetCount(numRecv);
    sorter_.SetMaxValue(static_cast<uint32_t>(numRanks_) * static_cast<uint32_t>(numEntriesPerRank_));

    GlobalTensor<int32_t> valGm, idxGm, outValGm, outIdxGm, tmpValGm, tmpIdxGm;
    GlobalTensor<int32_t> histGm, prefixGm, offsetsGm;
    valGm.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
    idxGm.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM_);
    outValGm.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
    outIdxGm.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM_);
    tmpValGm.SetGlobalBuffer((__gm__ int32_t *)sortTempValGM_);
    tmpIdxGm.SetGlobalBuffer((__gm__ int32_t *)sortTempIdxGM_);
    histGm.SetGlobalBuffer((__gm__ int32_t *)sortHistGm_);
    prefixGm.SetGlobalBuffer((__gm__ int32_t *)sortPrefixGm_);
    offsetsGm.SetGlobalBuffer((__gm__ int32_t *)sortOffsetsGm_);
    sorter_.Process(valGm, idxGm, outValGm, outIdxGm, tmpValGm, tmpIdxGm,
                    histGm, prefixGm, offsetsGm, *tpipe_);
}

__aicore__ inline void EngramFetchGradArch35::WriteNumUniqueZero()
{
    if (aivId_ != 0) {
        SyncAll<true>();
        return;
    }
    GlobalTensor<int32_t> numUniqueGM;
    numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM_);
    LocalTensor<int32_t> tmp = statusBuf_.Get<int32_t>();
    tmp.SetValue(0, 0);
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(numUniqueGM, tmp, params);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::UniqueScatterAdd()
{
    GlobalTensor<int32_t> numRecvGM;
    numRecvGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(numRecvGM);
    uint32_t numRecv = static_cast<uint32_t>(numRecvGM.GetValue(0));

    if (numRecv == 0) {
        WriteNumUniqueZero();
        return;
    }

    uint64_t t0 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    InitCompanion(numRecv);
    SyncAll<true>();
    uint64_t t1 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    RunSort(numRecv);
    SyncAll<true>();
    uint64_t t2 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    ZeroGradUnique(numRecv);
    CountUniquesParallel(numRecv);
    SyncAll<true>();
    uint64_t t3 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    ScatterAddAtomicParallel(numRecv);
    SyncAll<true>();
    uint64_t t4 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    CastGradUniqueToOutput();
    SyncAll<true>();
    uint64_t t5 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    usaCompanionCycle_ = t1 - t0;
    usaSortCycle_ = t2 - t1;
    usaZeroCountCycle_ = t3 - t2;
    usaScatterAddCycle_ = t4 - t3;
    usaCastCycle_ = t5 - t4;
}


__aicore__ inline void EngramFetchGradArch35::ZeroGradUnique(uint32_t numRecv)
{
    uint32_t zeroCount = numRecv;
    uint64_t totalFloats = static_cast<uint64_t>(zeroCount) * static_cast<uint64_t>(hiddenDim_);
    uint64_t chunk = (totalFloats + totalBlocks_ - 1U) / totalBlocks_;
    uint64_t start = static_cast<uint64_t>(aivId_) * chunk;
    uint64_t end = start + chunk;
    if (end > totalFloats) end = totalFloats;

    if (start < end) {
        LocalTensor<float> zeroBuf = pongBuf_.Get<float>();
        uint32_t tileFloats = TILE_BYTES / sizeof(float);
        Duplicate<float>(zeroBuf, 0.0f, tileFloats);
        EngramFetchGradSyncFunc<HardEvent::V_MTE3>();

        GM_ADDR dstBase = (outputDtype_ == ENGRAM_DT_FLOAT) ? gradUniqueOutGM_ : gradUniqueFp32GM_;
        GlobalTensor<float> dstGM;
        dstGM.SetGlobalBuffer((__gm__ float *)dstBase);
        for (uint64_t off = start; off < end; off += tileFloats) {
            uint64_t thisLen = end - off;
            if (thisLen > tileFloats) thisLen = tileFloats;
            DataCopyParams params{1U, static_cast<uint16_t>(thisLen * sizeof(float)), 0U, 0U};
            DataCopyPad(dstGM[off], zeroBuf, params);
        }
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void EngramFetchGradArch35::CountUniquesParallel(uint32_t numRecv)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t rawStart = aivId_ * chunk;
    uint32_t rawEnd = rawStart + chunk;
    if (rawEnd > numRecv) rawEnd = numRecv;

    GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);

    uint32_t start = rawStart;
    if (rawStart > 0 && rawStart < rawEnd) {
        LocalTensor<int32_t> probeUb = tempBuf_.Get<int32_t>();
        uint32_t probeLen = rawEnd - rawStart + 1U;
        if (probeLen > ENTRY_BATCH_CAP) probeLen = ENTRY_BATCH_CAP;
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(probeLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(probeUb, sortedEntryGM[rawStart - 1U], cpParams, cpPad);
        EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

        int32_t boundaryEntry = probeUb.GetValue(0);
        uint32_t p = 1U;
        while (p < probeLen && probeUb.GetValue(p) == boundaryEntry) {
            p++;
        }
        start = rawStart + p - 1U;
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        LocalTensor<int32_t> entryUb = tempBuf_.Get<int32_t>();
        int32_t prevEntry;
        bool isFirstElement = (start == 0);
        if (!isFirstElement) {
            DataCopyPadExtParams<int32_t> onePad{false, 0, 0, 0};
            DataCopyExtParams oneParams{1U, sizeof(int32_t), 0U, 0U, 0U};
            DataCopyPad(entryUb, sortedEntryGM[start - 1U], oneParams, onePad);
            EngramFetchGradSyncFunc<HardEvent::MTE2_S>();
            prevEntry = entryUb.GetValue(0);
        }

        uint32_t cur = start;
        while (cur < rawEnd) {
            uint32_t batchLen = rawEnd - cur;
            if (batchLen > ENTRY_BATCH_CAP) batchLen = ENTRY_BATCH_CAP;

            DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
            EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

            for (uint32_t i = 0; i < batchLen; i++) {
                int32_t entry = entryUb.GetValue(i);
                bool isNewUnique;
                if (isFirstElement) {
                    isNewUnique = true;
                    isFirstElement = false;
                } else {
                    isNewUnique = (entry != prevEntry);
                }
                if (isNewUnique) {
                    localUniqueCount++;
                }
                prevEntry = entry;
            }
            cur += batchLen;
        }
    }

    LocalTensor<int32_t> cntUb = statusBuf_.Get<int32_t>();
    cntUb.SetValue(0, static_cast<int32_t>(start));
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> coreStartGM;
    coreStartGM.SetGlobalBuffer((__gm__ int32_t *)coreStartGM_);
    DataCopyParams startParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(coreStartGM[aivId_], cntUb, startParams);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

    cntUb.SetValue(0, static_cast<int32_t>(localUniqueCount));
    EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
    GlobalTensor<int32_t> segCountGM;
    segCountGM.SetGlobalBuffer((__gm__ int32_t *)segCountGM_);
    DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(segCountGM[aivId_], cntUb, p);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::ScatterAddAtomicParallel(uint32_t numRecv)
{
    GlobalTensor<int32_t> coreStartGM;
    coreStartGM.SetGlobalBuffer((__gm__ int32_t *)coreStartGM_);
    LocalTensor<int32_t> startUb = tempBuf_.Get<int32_t>();
    uint32_t readCores = aivId_ + 2U;
    if (readCores > totalBlocks_) readCores = totalBlocks_;
    if (aivId_ == totalBlocks_ - 1U) readCores = totalBlocks_;
    DataCopyPadExtParams<int32_t> csPad{false, 0, 0, 0};
    DataCopyExtParams csParams{1U, static_cast<uint32_t>(readCores * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(startUb, coreStartGM, csParams, csPad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    uint32_t start = static_cast<uint32_t>(startUb.GetValue(aivId_));
    uint32_t end;
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        end = static_cast<uint32_t>(startUb.GetValue(aivId_ + 1U));
    }
    if (start >= end) return;

    int32_t preCoreOffset = 0;
    GlobalTensor<int32_t> segCountGM;
    segCountGM.SetGlobalBuffer((__gm__ int32_t *)segCountGM_);
    {
        DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
        DataCopyExtParams scParams{1U, static_cast<uint32_t>(aivId_ * sizeof(int32_t)), 0U, 0U, 0U};
        if (aivId_ > 0) {
            for (uint32_t core = 0; core < aivId_; core++) {
                DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(segCountGM[core]);
                preCoreOffset += segCountGM.GetValue(core);
            }
        }
    }

    GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
    GlobalTensor<int32_t> compGM;
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM_);
    GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM_);

    GM_ADDR atomicDstBase = (outputDtype_ == ENGRAM_DT_FLOAT) ? gradUniqueOutGM_ : gradUniqueFp32GM_;
    GlobalTensor<float> atomicDstGM;
    atomicDstGM.SetGlobalBuffer((__gm__ float *)atomicDstBase);

    int32_t prevEntry = 0;
    bool isFirstElement = true;

    LocalTensor<int32_t> entryUb = indicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> pingInt32 = pingBuf_.Get<int32_t>();
    LocalTensor<int32_t> compUb = pingInt32;
    LocalTensor<int32_t> uniqueUb = pingInt32[ENTRY_BATCH_CAP];
    LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * ENTRY_BATCH_CAP];
    LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * ENTRY_BATCH_CAP];
    LocalTensor<uint8_t> gradRaw = pongBuf_.Get<uint8_t>();
    LocalTensor<float> gradFp32 = gradSumBuf_.Get<float>();

    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;

    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = end - cur;
        if (batchLen > ENTRY_BATCH_CAP) batchLen = ENTRY_BATCH_CAP;

        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
        DataCopyPad(compUb, compGM[cur], cpParams, cpPad);
        EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

        int32_t inclusiveSum = 0;
        uint32_t tileUniqueCnt = 0;
        for (uint32_t i = 0; i < batchLen; i++) {
            int32_t entry = entryUb.GetValue(i);
            bool isNewUnique;
            if (isFirstElement) {
                isNewUnique = true;
                isFirstElement = false;
            } else {
                isNewUnique = (entry != prevEntry);
            }
            prevEntry = entry;
            if (isNewUnique) {
                inclusiveSum++;
                uniqueUb.SetValue(tileUniqueCnt, entry - static_cast<int32_t>(rankId_) * numEntriesPerRank_);
                tileUniqueCnt++;
            }
            compactIdxBuf.SetValue(i, runningOffset + inclusiveSum - 1);
            recvIdxBuf.SetValue(i, compUb.GetValue(i));
        }

        for (uint32_t subStart = 0; subStart < batchLen; subStart += GRAD_SUB_BATCH) {
            uint32_t subLen = batchLen - subStart;
            if (subLen > GRAD_SUB_BATCH) subLen = GRAD_SUB_BATCH;

            DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
            for (uint32_t j = 0; j < subLen; j++) {
                int32_t recvIdx = recvIdxBuf.GetValue(subStart + j);
                GM_ADDR gradAddr = recvGradGM_ + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
                DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
                GlobalTensor<uint8_t> gradSrcGM;
                gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
                DataCopyPad(gradRaw[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenBytes_)], gradSrcGM, gradParams, gradPad);
            }

            if (inputDtype_ == ENGRAM_DT_FLOAT) {
                EngramFetchGradSyncFunc<HardEvent::MTE2_MTE3>();
                LocalTensor<float> inT = gradRaw.ReinterpretCast<float>();
                for (uint32_t j = 0; j < subLen; j++) {
                    int32_t compactIdx = compactIdxBuf.GetValue(subStart + j);
                    SetAtomicAdd<float>();
                    DataCopy(atomicDstGM[static_cast<uint64_t>(compactIdx) * hiddenDim_],
                             inT[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenDim_)],
                             static_cast<uint32_t>(hiddenDim_));
                    SetAtomicNone();
                }
                EngramFetchGradSyncFunc<HardEvent::MTE3_MTE2>();
            } else {
                EngramFetchGradSyncFunc<HardEvent::MTE2_V>();
                uint32_t castCount = subLen * static_cast<uint32_t>(hiddenDim_);
                if (inputDtype_ == ENGRAM_DT_BFLOAT16) {
                    LocalTensor<bfloat16_t> inT = gradRaw.ReinterpretCast<bfloat16_t>();
                    Cast(gradFp32, inT, AscendC::RoundMode::CAST_NONE, castCount);
                } else {
                    LocalTensor<half> inT = gradRaw.ReinterpretCast<half>();
                    Cast(gradFp32, inT, AscendC::RoundMode::CAST_NONE, castCount);
                }
                PipeBarrier<PIPE_V>();
                EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
                for (uint32_t j = 0; j < subLen; j++) {
                    int32_t compactIdx = compactIdxBuf.GetValue(subStart + j);
                    SetAtomicAdd<float>();
                    DataCopy(atomicDstGM[static_cast<uint64_t>(compactIdx) * hiddenDim_],
                             gradFp32[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenDim_)],
                             static_cast<uint32_t>(hiddenDim_));
                    SetAtomicNone();
                }
                EngramFetchGradSyncFunc<HardEvent::MTE3_MTE2>();
            }
        }

        if (tileUniqueCnt > 0) {
            EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
            DataCopyParams ueParams{1U, static_cast<uint16_t>(tileUniqueCnt * sizeof(int32_t)), 0U, 0U};
            DataCopyPad(uniqueEntryOutGM[runningUniqueOffset], uniqueUb, ueParams);
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
            runningUniqueOffset += static_cast<int32_t>(tileUniqueCnt);
        }

        runningOffset += inclusiveSum;
        cur += batchLen;
    }
}

__aicore__ inline void EngramFetchGradArch35::CastGradUniqueToOutput()
{
    GlobalTensor<int32_t> segCountGM;
    segCountGM.SetGlobalBuffer((__gm__ int32_t *)segCountGM_);
    LocalTensor<int32_t> segCountUb = tempBuf_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
    DataCopyExtParams scParams{1U, static_cast<uint32_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(segCountUb, segCountGM, scParams, scPad);
    EngramFetchGradSyncFunc<HardEvent::MTE2_S>();

    int32_t numUnique = 0;
    for (uint32_t core = 0; core < totalBlocks_; core++) {
        numUnique += segCountUb.GetValue(core);
    }

    if (aivId_ == 0) {
        LocalTensor<int32_t> tmp = statusBuf_.Get<int32_t>();
        tmp.SetValue(0, numUnique);
        EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int32_t> numUniqueGM;
        numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM_);
        DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        DataCopyPad(numUniqueGM, tmp, p);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
    }

    if (numUnique == 0 || outputDtype_ == ENGRAM_DT_FLOAT) {
        SyncAll<true>();
        return;
    }

    uint32_t totalElements = static_cast<uint32_t>(numUnique) * static_cast<uint32_t>(hiddenDim_);
    uint32_t chunk = (totalElements + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t start = aivId_ * chunk;
    uint32_t end = start + chunk;
    if (end > totalElements) end = totalElements;
    if (start >= end) {
        SyncAll<true>();
        return;
    }

    GlobalTensor<float> gradUniqueFp32GM;
    gradUniqueFp32GM.SetGlobalBuffer((__gm__ float *)gradUniqueFp32GM_);
    GlobalTensor<uint8_t> gradUniqueOutGM;
    gradUniqueOutGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM_);

    LocalTensor<float> fp32Tile = pingBuf_.Get<float>();
    uint32_t tileLen = TILE_BYTES / sizeof(float);

    uint32_t off = start;
    while (off < end) {
        uint32_t thisLen = end - off;
        if (thisLen > tileLen) thisLen = tileLen;

        DataCopyExtParams inParams{1U, static_cast<uint32_t>(thisLen * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
        DataCopyPad(fp32Tile, gradUniqueFp32GM[off], inParams, inPad);
        EngramFetchGradSyncFunc<HardEvent::MTE2_V>();

        if (outputDtype_ == ENGRAM_DT_BFLOAT16) {
            LocalTensor<bfloat16_t> outTile = pongBuf_.Get<bfloat16_t>();
            Cast(outTile, fp32Tile, AscendC::RoundMode::CAST_RINT, thisLen);
            EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
            DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(bfloat16_t)), 0U, 0U};
            DataCopyPad(gradUniqueOutGM[static_cast<uint64_t>(off) * sizeof(bfloat16_t)],
                        outTile.ReinterpretCast<uint8_t>(), outParams);
        } else {
            LocalTensor<half> outTile = pongBuf_.Get<half>();
            Cast(outTile, fp32Tile, AscendC::RoundMode::CAST_RINT, thisLen);
            EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
            DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(half)), 0U, 0U};
            DataCopyPad(gradUniqueOutGM[static_cast<uint64_t>(off) * sizeof(half)],
                        outTile.ReinterpretCast<uint8_t>(), outParams);
        }
        EngramFetchGradSyncFunc<HardEvent::MTE3_MTE2>();
        off += thisLen;
    }
    SyncAll<true>();
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
    LocalTensor<int32_t> cleanLocal = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanLocal, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
    uint64_t cleanSize = numRanks_ * STATE_OFFSET / sizeof(int32_t);
    GlobalTensor<int32_t> flagGM;
    GlobalTensor<int32_t> writeGM;
    GlobalTensor<int32_t> readGM;
    flagGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenFlagOffset_));
    writeGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
    readGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenReadOffset_));
    DataCopy(flagGM, cleanLocal, cleanSize);
    DataCopy(writeGM, cleanLocal, cleanSize);
    DataCopy(readGM, cleanLocal, cleanSize);
    EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradArch35::CrossRankBarrier()
{
    if (isSender_) {
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            if (dstRank == rankId_)
                continue;
            uint64_t handle = GetCommHandle(dstRank);
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, barrierFlagOffset_) + rankId_ * STATE_OFFSET;
            WriteNbiChecked(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
            DrainChecked(handle);
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
    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradArch35::Process()
{
    if ASCEND_IS_AIV {
        processStartCycle_ = static_cast<uint64_t>(AscendC::GetSystemCycle());

        InitFlagsAndDispls();
        SyncAll<true>();

        ClearWinCounters();
        SyncAll<true>();

        CrossRankBarrier();

        UnsortGrad();
        ExchangeGrad();
        uniqueScatterAddStartCycle_ = static_cast<uint64_t>(AscendC::GetSystemCycle());
        UniqueScatterAdd();
        uniqueScatterAddEndCycle_ = static_cast<uint64_t>(AscendC::GetSystemCycle());

        processEndCycle_ = static_cast<uint64_t>(AscendC::GetSystemCycle());

       
            uint64_t totalUs = (processEndCycle_ - processStartCycle_) / 1000U;
            uint64_t usaUs = (uniqueScatterAddEndCycle_ - uniqueScatterAddStartCycle_) / 1000U;
            uint64_t preUs = (uniqueScatterAddStartCycle_ - processStartCycle_) / 1000U;
            AscendC::printf("ENGRAM total=%lluu pre=%lluu usa=%lluu | companion=%lluu sort=%lluu zeroCount=%lluu scatterAdd=%lluu cast=%lluu\n",
                            totalUs, preUs, usaUs,
                            usaCompanionCycle_ / 1000U, usaSortCycle_ / 1000U,
                            usaZeroCountCycle_ / 1000U, usaScatterAddCycle_ / 1000U, usaCastCycle_ / 1000U);
        
    }
}

} // namespace Mc2Kernel

#endif
