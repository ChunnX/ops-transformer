/*!
 * \file engram_fetch_grad_arch35.h
 * \brief engram_fetch_grad 算子 arch35 kernel 实现
 *
 * 整体流程（3 步）:
 *   Step 1: 反重排 — grad_sorted[i] = grad_fetched[perm[i]]（行 gather）
 *   Step 2: a2a 梯度交换 — quad-buffer 流控，counts 再次互换
 *   Step 3: owner unique + fp32 scatter-add
 */

#ifndef ENGRAM_FETCH_GRAD_ARCH35_H
#define ENGRAM_FETCH_GRAD_ARCH35_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_grad_tiling_data.h"
#include "../../../engram_fetch/op_kernel/engram_fetch_utils.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace Mc2Kernel {

using namespace AscendC;

template<AscendC::HardEvent event>
__aicore__ inline void EngramFetchGradSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class EngramFetchGradArch35 {
public:
    __aicore__ inline EngramFetchGradArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR gradFetched,
                                GM_ADDR permOut, GM_ADDR sendCountsOut,
                                GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut,
                                GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const EngramFetchGradTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void UnsortGrad();
    __aicore__ inline void ExchangeGrad();
    __aicore__ inline void UniqueScatterAdd();

    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank);
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value);
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);

    // ===== 基本资源 =====
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

    // ===== 运行时参数 =====
    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t hiddenDim_{0};
    int64_t totalRecv_{0};       // a2a 接收上界（来自 tilingData）
    int32_t inputDtype_{0};      // gradFetched dtype（ge::DataType）
    int32_t outputDtype_{0};     // gradUniqueOut dtype（ge::DataType）
    uint64_t winSize_{0};
    uint64_t ubSize_{0};

    // ===== Win 区偏移 =====
    uint64_t barrierFlagOffset_{0};
    uint64_t tokenFlagOffset_{0};
    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t tokenDataOffset_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxTokensPerSlot_{0};

    // ===== 核分配 =====
    uint32_t numSendCores_{0};
    uint32_t numRecvCores_{0};
    bool isSender_{false};
    bool isReceiver_{false};
    bool isFlagCore_{false};

    // ===== Workspace GM 地址 =====
    GM_ADDR gradSortedGM_{0};
    GM_ADDR recvGradGM_{0};
    GM_ADDR sendCountsGM_{0};
    GM_ADDR recvCountsGM_{0};
    GM_ADDR sdisplsGM_{0};
    GM_ADDR rdisplsGM_{0};
    GM_ADDR recvLocalEntryGM_{0};
    GM_ADDR counterScratchGM_{0};  // [aivNum×32B] WriteRemoteCounter per-core 32B 独占区
    GM_ADDR flagScratchGM_{0};

    // ===== UB Buffer =====
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
};

// ===== 实现 =====

__aicore__ inline GM_ADDR EngramFetchGradArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

__aicore__ inline uint64_t EngramFetchGradArch35::GetCommHandle(uint32_t dstRank)
{
    return ctxPtr_->hcommHandle[dstRank];
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
    if (dstRank == rankId_) return;

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
    hcomm_.WriteNbi(handle, remoteCounterAddr, srcAddr, sizeof(int32_t));
    hcomm_.Drain(handle);
}

__aicore__ inline void EngramFetchGradArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(expectCount);
    uint32_t pollCount = 0;
    while (sumOfFlag != compareFlag) {
        sumOfFlag = 0;
        for (uint32_t i = 0; i < numRanks_; i++) {
            GlobalTensor<int32_t> slotGM;
            slotGM.SetGlobalBuffer((__gm__ int32_t *)(statusWinBase + i * STATE_OFFSET));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(slotGM);
            int32_t flagVal = slotGM.GetValue(0);
            sumOfFlag += flagVal;
            if (pollCount <= 3U || pollCount % 100000U == 0) {
                printf("[engram_fetch_grad] WaitAllStatusFlags rankId=%u, poll=%u, srcRank=%u, flag=%d\n",
                       rankId_, pollCount, i, flagVal);
            }
        }
        pollCount++;
        if (pollCount <= 3U || pollCount % 100000U == 0) {
            printf("[engram_fetch_grad] WaitAllStatusFlags rankId=%u, poll=%u, sum=%d, expect=%d\n",
                   rankId_, pollCount, sumOfFlag, compareFlag);
        }
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

__aicore__ inline void EngramFetchGradArch35::Init(
    GM_ADDR commContext, GM_ADDR gradFetched,
    GM_ADDR permOut, GM_ADDR sendCountsOut,
    GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut,
    GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
    GM_ADDR workspaceGM, TPipe *pipe, const EngramFetchGradTilingData *tilingData)
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

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    hiddenDim_ = tilingData->hiddenDim;
    totalRecv_ = tilingData->totalRecv;
    inputDtype_ = tilingData->inputDtype;
    outputDtype_ = tilingData->outputDtype;
    ubSize_ = tilingData->ubSize;

    winSize_ = tilingData->commBufferSize;

    // Win 区布局: 4 个固定区域 + token 数据区
    // barrierFlagOffset_ 专门用于跨卡 barrier，与 tokenFlagOffset_ 隔离避免竞态
    barrierFlagOffset_ = 0;
    tokenFlagOffset_ = numRanks_ * STATE_OFFSET;
    tokenWriteOffset_ = 2U * numRanks_ * STATE_OFFSET;
    tokenReadOffset_ = 3U * numRanks_ * STATE_OFFSET;
    tokenDataOffset_ = 4U * numRanks_ * STATE_OFFSET;

    uint64_t tokenArea = (winSize_ > tokenDataOffset_) ? (winSize_ - tokenDataOffset_) : 0U;
    if (hiddenBytes_ > 0) {
        tokenArea = Ceil(tokenArea, static_cast<uint64_t>(hiddenBytes_)) * static_cast<uint64_t>(hiddenBytes_);
    }
    tokenSlotSize_ = tokenArea / numRanks_ / NUM_SLOTS;
    maxTokensPerSlot_ = static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_));

    // 核分配
    numSendCores_ = totalBlocks_ / 2U;
    if (numSendCores_ == 0U) { numSendCores_ = 1U; }
    numRecvCores_ = totalBlocks_ - numSendCores_;
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    // hcomm
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    // ping-pong
    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    // sendCounts/recvCounts/recvLocalEntry 直接用前向输出 tensor，不占 workspace
    // sendCountsOut 是 32B spaced (r * UB_ALIGN)，recvCountsOut 是紧凑 (r * sizeof(int32_t))
    sendCountsGM_ = sendCountsOutGM_;
    recvCountsGM_ = recvCountsOutGM_;
    recvLocalEntryGM_ = recvLocalEntryOutGM_;

    // Workspace 只保留 gradSorted/recvGrad/sdispls/rdispls/counterScratch/flagScratch
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

    // UB buffers
    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    // tempBuf_ 用于 sortedIdx（上界 = totalRecv）
    int64_t totalRecvUpperBound = totalRecv_;
    uint32_t tempBufSize = Ceil(static_cast<uint32_t>(totalRecvUpperBound) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    // gradSumBuf_ 用于 fp32 累加结果（hiddenDim 个 float）
    uint32_t gradSumBufSize = Ceil(static_cast<uint32_t>(hiddenDim_) * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gradSumBuf_, gradSumBufSize);
    uint32_t indicesBufSize = Ceil(static_cast<uint32_t>(numTokens_) * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    if (indicesBufSize < statusBufSize) { indicesBufSize = statusBufSize; }
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);

    printf("[engram_fetch_grad] Init: rankId=%u, numRanks=%u, aivId=%u, totalBlocks=%u, "
           "winSize=%lu, tokenSlot=%lu, maxTokPerSlot=%u, sendCores=%u, recvCores=%u\n",
           rankId_, numRanks_, aivId_, totalBlocks_, winSize_, tokenSlotSize_, maxTokensPerSlot_,
           numSendCores_, numRecvCores_);
}

// ===== Step 1: 反重排 =====
__aicore__ inline void EngramFetchGradArch35::UnsortGrad()
{
    printf("[engram_fetch_grad] UnsortGrad start: rankId=%u, aivId=%u, numTokens=%ld, hiddenBytes=%ld\n",
           rankId_, aivId_, numTokens_, hiddenBytes_);
    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer((__gm__ int32_t *)permOutGM_);

    for (int64_t i = static_cast<int64_t>(aivId_); i < numTokens_; i += static_cast<int64_t>(totalBlocks_)) {
        int32_t origIdx = permGM.GetValue(static_cast<uint32_t>(i));
        GM_ADDR src = gradFetchedGM_ + static_cast<uint64_t>(origIdx) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = gradSortedGM_ + static_cast<uint64_t>(i) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }

    SyncAll<true>();
    printf("[engram_fetch_grad] UnsortGrad done: rankId=%u\n", rankId_);
}

// ===== Step 2: a2a 梯度交换（quad-buffer 流控）=====
__aicore__ inline void EngramFetchGradArch35::ExchangeGrad()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    GlobalTensor<uint8_t> recvGradGM;
    recvGradGM.SetGlobalBuffer((__gm__ uint8_t *)recvGradGM_);

    // --- 发送核 ---
    if (isSender_) {
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            // 读 sendCounts[dstRank] 和 sdispls[dstRank]
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

            printf("[engram_fetch_grad] ExchangeGrad sender: rankId=%u, aivId=%u, dstRank=%u, sendCount=%d, sdispl=%ld\n",
                   rankId_, aivId_, dstRank, sendCount, sdispl);

            if (dstRank == rankId_) {
                // 本地：直接 DataCopy 到 recvGrad
                if (sendCount > 0) {
                    GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * UB_ALIGN;
                    GlobalTensor<int64_t> rDisplSlotGM;
                    rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
                    int64_t rdispl = rDisplSlotGM.GetValue(0);

                    LocalCopySlice(recvGradGM_ + rdispl * hiddenBytes_,
                                  gradSortedGM_ + sdispl * hiddenBytes_,
                                  static_cast<uint64_t>(sendCount) * hiddenBytes_);
                }
                // 本地 flag 预置 1
                GM_ADDR localFlagAddr = localWinBase + tokenFlagOffset_ + rankId_ * STATE_OFFSET;
                LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
                Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
                EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
                GlobalTensor<int32_t> flagGM;
                flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
                DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
                EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
                continue;
            }

            // 远端：quad-buffer 流控发送
            uint64_t handle = GetCommHandle(dstRank);
            uint32_t totalSent = 0;
            uint32_t localWriteCnt = 0;

            printf("[engram_fetch_grad] ExchangeGrad remote send start: rankId=%u, dstRank=%u, sendCount=%d, maxTokPerSlot=%u\n",
                   rankId_, dstRank, sendCount, maxTokensPerSlot_);

            while (totalSent < static_cast<uint32_t>(sendCount)) {
                if (totalBlocks_ > 1U) {
                    int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                    uint32_t pollCnt = 0;
                    while (localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                        remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                        pollCnt++;
                        if (pollCnt <= 3U || pollCnt % 100000U == 0) {
                            printf("[engram_fetch_grad] ExchangeGrad send wait: rankId=%u, dstRank=%u, writeCnt=%u, readCnt=%d\n",
                                   rankId_, dstRank, localWriteCnt, remoteReadCnt);
                        }
                    }
                }

                uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
                uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
                if (chunkLen == 0U) {
                    printf("[engram_fetch_grad][ERROR] ExchangeGrad send chunkLen=0: rankId=%u, dstRank=%u, sendCount=%d, totalSent=%u, maxTokPerSlot=%u\n",
                           rankId_, dstRank, sendCount, totalSent, maxTokensPerSlot_);
                    break;
                }

                uint64_t slotOffset = tokenDataOffset_ + rankId_ * tokenSlotSize_
                                    + (localWriteCnt % NUM_SLOTS) * tokenSlotSize_;
                GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
                GM_ADDR srcAddr = gradSortedGM_ + (sdispl + totalSent) * hiddenBytes_;
                uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

                printf("[engram_fetch_grad] ExchangeGrad send chunk: rankId=%u, dstRank=%u, chunk=%u, totalSent=%u, slot=%u\n",
                       rankId_, dstRank, chunkLen, totalSent, localWriteCnt % NUM_SLOTS);

                hcomm_.WriteNbi(handle, remoteSlotAddr, srcAddr, dataBytes);
                hcomm_.Drain(handle);

                localWriteCnt++;
                totalSent += chunkLen;
                WriteRemoteCounter(dstRank, tokenWriteOffset_, static_cast<int32_t>(localWriteCnt));
            }

            printf("[engram_fetch_grad] ExchangeGrad send done: rankId=%u, dstRank=%u, totalSent=%u\n",
                   rankId_, dstRank, totalSent);

            // 写完成 flag
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, tokenFlagOffset_) + rankId_ * STATE_OFFSET;
            hcomm_.WriteNbi(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
            hcomm_.Drain(handle);
        }
    }

    // --- 接收核 ---
    if (isReceiver_) {
        uint32_t recvIdx = aivId_ - numSendCores_;
        for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
            if (srcRank == rankId_) continue;

            // counts 互换: 接收用 recvCounts/rdispls
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

            printf("[engram_fetch_grad] ExchangeGrad receiver: rankId=%u, aivId=%u, srcRank=%u, recvCount=%d, rdispl=%ld\n",
                   rankId_, aivId_, srcRank, recvCount, rdispl);

            uint32_t totalReceived = 0;
            uint32_t localReadCnt = 0;

            while (totalReceived < static_cast<uint32_t>(recvCount)) {
                int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
                uint32_t pollCnt = 0;
                while (static_cast<uint32_t>(remoteWriteCnt) - localReadCnt == 0) {
                    remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
                    pollCnt++;
                    if (pollCnt <= 3U || pollCnt % 100000U == 0) {
                        printf("[engram_fetch_grad] ExchangeGrad recv wait: rankId=%u, srcRank=%u, writeCnt=%d, readCnt=%u\n",
                               rankId_, srcRank, remoteWriteCnt, localReadCnt);
                    }
                }

                uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
                uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;

                printf("[engram_fetch_grad] ExchangeGrad recv chunk: rankId=%u, srcRank=%u, chunk=%u, totalRecv=%u, slot=%u\n",
                       rankId_, srcRank, chunkLen, totalReceived, localReadCnt % NUM_SLOTS);

                uint64_t slotOffset = tokenDataOffset_ + srcRank * tokenSlotSize_
                                    + (localReadCnt % NUM_SLOTS) * tokenSlotSize_;
                GM_ADDR localSlotAddr = localWinBase + slotOffset;
                uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

                LocalCopySlice(recvGradGM_ + (rdispl + totalReceived) * hiddenBytes_,
                              localSlotAddr, dataBytes);

                localReadCnt++;
                totalReceived += chunkLen;
                WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(localReadCnt));
            }
            printf("[engram_fetch_grad] ExchangeGrad recv done: rankId=%u, srcRank=%u, totalRecv=%u\n",
                   rankId_, srcRank, totalReceived);
        }
    }

    // --- flag核 ---
    if (isFlagCore_) {
        printf("[engram_fetch_grad] ExchangeGrad flagCore wait: rankId=%u, aivId=%u\n", rankId_, aivId_);
        GM_ADDR localFlagBase = localWinBase + tokenFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        printf("[engram_fetch_grad] ExchangeGrad flagCore flags done: rankId=%u\n", rankId_);

        // write!=read 检查已注释（方向相反，会导致死锁）
        // for (uint32_t r = 0; r < numRanks_; r++) {
        //     if (r == rankId_) continue;
        //     int32_t writeCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, r);
        //     int32_t readCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, r);
        //     while (writeCnt != readCnt) {
        //         writeCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, r);
        //         readCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, r);
        //     }
        // }

        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
    printf("[engram_fetch_grad] ExchangeGrad done: rankId=%u\n", rankId_);
}

// ===== Step 3: owner unique + fp32 scatter-add =====
__aicore__ inline void EngramFetchGradArch35::UniqueScatterAdd()
{
    if (aivId_ == 0) {
        // 读 numRecv
        GlobalTensor<int32_t> numRecvGM;
        numRecvGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(numRecvGM);
        int32_t numRecv = numRecvGM.GetValue(0);

        printf("[engram_fetch_grad] UniqueScatterAdd: rankId=%u, numRecv=%d\n", rankId_, numRecv);

        if (numRecv == 0) {
            // 写 numUnique=0
            LocalTensor<int32_t> tmp = statusBuf_.Get<int32_t>();
            tmp.SetValue(0, 0);
            EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
            GlobalTensor<int32_t> numUniqueGM;
            numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM_);
            DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
            DataCopyPad(numUniqueGM, tmp, params);
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
            return;
        }

        // 读 recvLocalEntry 到 UB
        GlobalTensor<int32_t> recvLocalEntryGM;
        recvLocalEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM_);
        LocalTensor<int32_t> entryLocal = indicesBuf_.Get<int32_t>();

        // 转换为本地 entry 索引
        for (int32_t i = 0; i < numRecv; i++) {
            int32_t globalIdx = recvLocalEntryGM.GetValue(static_cast<uint32_t>(i));
            int32_t localIdx = globalIdx - static_cast<int32_t>(rankId_) * numEntriesPerRank_;
            entryLocal.SetValue(static_cast<uint32_t>(i), localIdx);
        }

        // 简单排序：冒泡排序（numRecv 通常不大，后续可优化为 vbitsort32）
        LocalTensor<int32_t> sortedIdx = tempBuf_.Get<int32_t>();
        for (int32_t i = 0; i < numRecv; i++) {
            sortedIdx.SetValue(static_cast<uint32_t>(i), i);
        }
        for (int32_t i = 0; i < numRecv - 1; i++) {
            for (int32_t j = 0; j < numRecv - 1 - i; j++) {
                int32_t idxJ = sortedIdx.GetValue(static_cast<uint32_t>(j));
                int32_t idxJ1 = sortedIdx.GetValue(static_cast<uint32_t>(j + 1));
                int32_t entryJ = entryLocal.GetValue(static_cast<uint32_t>(idxJ));
                int32_t entryJ1 = entryLocal.GetValue(static_cast<uint32_t>(idxJ1));
                if (entryJ > entryJ1) {
                    sortedIdx.SetValue(static_cast<uint32_t>(j), idxJ1);
                    sortedIdx.SetValue(static_cast<uint32_t>(j + 1), idxJ);
                }
            }
        }

        // 分段去重 + fp32 scatter-add
        GlobalTensor<uint8_t> recvGradGM;
        recvGradGM.SetGlobalBuffer((__gm__ uint8_t *)recvGradGM_);
        GlobalTensor<uint8_t> gradUniqueGM;
        gradUniqueGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM_);
        GlobalTensor<int32_t> uniqueLocalEntryGM;
        uniqueLocalEntryGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM_);

        LocalTensor<float> gradSum = gradSumBuf_.Get<float>();
        LocalTensor<float> gradTmp = pingBuf_.Get<float>();

        int32_t numUnique = 0;
        int32_t i = 0;
        while (i < numRecv) {
            int32_t origIdx = sortedIdx.GetValue(static_cast<uint32_t>(i));
            int32_t entry = entryLocal.GetValue(static_cast<uint32_t>(origIdx));

            uniqueLocalEntryGM.SetValue(static_cast<uint32_t>(numUnique), entry);

            // 清零 gradSum
            Duplicate<float>(gradSum, 0.0f, static_cast<uint32_t>(hiddenDim_));
            PipeBarrier<PIPE_V>();

            // 段内累加
            while (i < numRecv) {
                int32_t curIdx = sortedIdx.GetValue(static_cast<uint32_t>(i));
                int32_t curEntry = entryLocal.GetValue(static_cast<uint32_t>(curIdx));
                if (curEntry != entry) break;

                // 读 grad → cast fp32 → add
                GM_ADDR gradAddr = recvGradGM_ + static_cast<uint64_t>(curIdx) * hiddenBytes_;
                LocalTensor<uint8_t> gradRaw = pongBuf_.Get<uint8_t>();
                DataCopyExtParams cpParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
                DataCopyPadExtParams<uint8_t> cpPad{false, 0, 0, 0};
                GlobalTensor<uint8_t> gradSrcGM;
                gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
                DataCopyPad(gradRaw, gradSrcGM, cpParams, cpPad);
                EngramFetchGradSyncFunc<HardEvent::MTE2_V>();

                // 按 inputDtype cast → fp32
                if (inputDtype_ == 27) {  // DT_BF16
                    LocalTensor<bfloat16_t> inTensor = pongBuf_.Get<bfloat16_t>();
                    Cast(gradTmp, inTensor, AscendC::RoundMode::CAST_NONE,
                         static_cast<uint32_t>(hiddenDim_));
                    PipeBarrier<PIPE_V>();
                    Add(gradSum, gradSum, gradTmp, static_cast<uint32_t>(hiddenDim_));
                } else if (inputDtype_ == 1) {  // DT_FLOAT16
                    LocalTensor<half> inTensor = pongBuf_.Get<half>();
                    Cast(gradTmp, inTensor, AscendC::RoundMode::CAST_NONE,
                         static_cast<uint32_t>(hiddenDim_));
                    PipeBarrier<PIPE_V>();
                    Add(gradSum, gradSum, gradTmp, static_cast<uint32_t>(hiddenDim_));
                } else {  // DT_FLOAT(0): 直接用，无需 cast
                    LocalTensor<float> inTensor = pongBuf_.Get<float>();
                    Add(gradSum, gradSum, inTensor, static_cast<uint32_t>(hiddenDim_));
                }

                // 确保 Cast/Add (V) 读完 pongBuf_ 后，下一轮 DataCopyPad (MTE2) 才能覆盖
                EngramFetchGradSyncFunc<HardEvent::V_MTE2>();

                i++;
            }

            // 确保最后一个 Add (V) 完成后再做输出 Cast
            PipeBarrier<PIPE_V>();

            // 按 outputDtype cast fp32 → output，写结果
            if (outputDtype_ == 27) {  // DT_BF16
                LocalTensor<bfloat16_t> outTensor = pingBuf_.Get<bfloat16_t>();
                Cast(outTensor, gradSum, AscendC::RoundMode::CAST_ROUND,
                     static_cast<uint32_t>(hiddenDim_));
                EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
                DataCopyParams outParams = {1U, static_cast<uint16_t>(hiddenDim_ * sizeof(bfloat16_t)), 0U, 0U};
                DataCopyPad(gradUniqueGM[static_cast<uint64_t>(numUnique) * hiddenDim_ * sizeof(bfloat16_t)],
                            outTensor.ReinterpretCast<uint8_t>(), outParams);
            } else if (outputDtype_ == 1) {  // DT_FLOAT16
                LocalTensor<half> outTensor = pingBuf_.Get<half>();
                Cast(outTensor, gradSum, AscendC::RoundMode::CAST_ROUND,
                     static_cast<uint32_t>(hiddenDim_));
                EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
                DataCopyParams outParams = {1U, static_cast<uint16_t>(hiddenDim_ * sizeof(half)), 0U, 0U};
                DataCopyPad(gradUniqueGM[static_cast<uint64_t>(numUnique) * hiddenDim_ * sizeof(half)],
                            outTensor.ReinterpretCast<uint8_t>(), outParams);
            } else {  // DT_FLOAT(0): 直接写 fp32
                EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
                DataCopyParams outParams = {1U, static_cast<uint16_t>(hiddenDim_ * sizeof(float)), 0U, 0U};
                DataCopyPad(gradUniqueGM[static_cast<uint64_t>(numUnique) * hiddenDim_ * sizeof(float)],
                            gradSum.ReinterpretCast<uint8_t>(), outParams);
            }
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

            numUnique++;
        }

        // 写 numUnique
        LocalTensor<int32_t> tmp = statusBuf_.Get<int32_t>();
        tmp.SetValue(0, numUnique);
        EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int32_t> numUniqueOutGM;
        numUniqueOutGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM_);
        DataCopyParams numParams = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        DataCopyPad(numUniqueOutGM, tmp, numParams);
        EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

        printf("[engram_fetch_grad] UniqueScatterAdd done: rankId=%u, numUnique=%d\n", rankId_, numUnique);
    }

    SyncAll<true>();
}

// ===== Process =====
__aicore__ inline void EngramFetchGradArch35::Process()
{
    if ASCEND_IS_AIV {
        printf("[engram_fetch_grad] Process start: aivId=%u\n", aivId_);

        // 初始化 flagScratchGM_
        if (aivId_ == 0) {
            LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
            Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
            EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
            GlobalTensor<int32_t> flagInit;
            flagInit.SetGlobalBuffer((__gm__ int32_t *)flagScratchGM_);
            DataCopy(flagInit, flagLocal, STATE_OFFSET / sizeof(int32_t));
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();

            // 从前向输出计算 sdispls/rdispls 并写入 workspace
            // sendCounts/recvCounts 直接用前向输出 tensor，不需要拷贝
            GlobalTensor<int32_t> sendCountsOutGM;
            sendCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
            GlobalTensor<int32_t> recvCountsOutGM;
            recvCountsOutGM.SetGlobalBuffer((__gm__ int32_t *)recvCountsOutGM_);

            int64_t sAccum = 0;
            int64_t rAccum = 0;
            for (uint32_t r = 0; r < numRanks_; r++) {
                // 写 sdispls[r]
                LocalTensor<int64_t> displVal = tempBuf_.Get<int64_t>();
                displVal.SetValue(0, sAccum);
                EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
                GlobalTensor<int64_t> sdGM;
                sdGM.SetGlobalBuffer((__gm__ int64_t *)(sdisplsGM_ + r * UB_ALIGN));
                DataCopyParams displParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
                DataCopyPad(sdGM, displVal, displParams);
                EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
                int32_t sCount = sendCountsOutGM.GetValue(r * 8U);  // 32B spaced, 每 8 个取第一个
                sAccum += sCount;

                // 写 rdispls[r]
                displVal.SetValue(0, rAccum);
                EngramFetchGradSyncFunc<HardEvent::S_MTE3>();
                GlobalTensor<int64_t> rdGM;
                rdGM.SetGlobalBuffer((__gm__ int64_t *)(rdisplsGM_ + r * UB_ALIGN));
                DataCopyPad(rdGM, displVal, displParams);
                EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
                int32_t rCount = recvCountsOutGM.GetValue(r);  // 紧凑排列
                rAccum += rCount;

                printf("[engram_fetch_grad] displs: rankId=%u, r=%u, sendCount=%d, sdispl=%ld, recvCount=%d, rdispl=%ld\n",
                       rankId_, r, sCount, sAccum - sCount, rCount, rAccum - rCount);
            }
        }
        SyncAll<true>();

        // 清零 win 区的 3 个区域（tokenFlag/tokenWrite/tokenRead）
        // 放在 Process 开头，依赖 host 侧同步（torch.npu.synchronize）保证所有卡都到这里才开始 ExchangeGrad
        if (aivId_ == 0) {
            GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
            LocalTensor<int32_t> cleanLocal = statusBuf_.Get<int32_t>();
            Duplicate<int32_t>(cleanLocal, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
            EngramFetchGradSyncFunc<HardEvent::V_MTE3>();
            uint64_t cleanSize = numRanks_ * STATE_OFFSET / sizeof(int32_t);
            GlobalTensor<int32_t> flagGM, writeGM, readGM;
            flagGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenFlagOffset_));
            writeGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
            readGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenReadOffset_));
            DataCopy(flagGM, cleanLocal, cleanSize);
            DataCopy(writeGM, cleanLocal, cleanSize);
            DataCopy(readGM, cleanLocal, cleanSize);
            EngramFetchGradSyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        printf("[engram_fetch_grad] Process: win counters cleared\n");

        // 跨卡 barrier：用独立的 barrierFlag 区做握手，保证所有卡都清完 win 区才开始通信
        // 每个卡给所有对端发 flag=1，然后等所有 flag 到齐，再清零 barrierFlag 区
        if (isSender_) {
            for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
                if (dstRank == rankId_) continue;
                uint64_t handle = GetCommHandle(dstRank);
                GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, barrierFlagOffset_) + rankId_ * STATE_OFFSET;
                hcomm_.WriteNbi(handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET);
                hcomm_.Drain(handle);
            }
        }
        // 本地 flag 预置 1
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

        // flag 核等所有 flag 到齐，然后清零 barrierFlag 区
        if (isFlagCore_) {
            GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
            GM_ADDR localFlagBase = localWinBase + barrierFlagOffset_;
            WaitAllStatusFlags(localFlagBase, numRanks_);
            ClearStatusFlags(localFlagBase);
        }
        SyncAll<true>();
        printf("[engram_fetch_grad] Process: cross-rank barrier done\n");

        // Step 1: 反重排
        UnsortGrad();

        // Step 2: a2a 梯度交换
        ExchangeGrad();

        // Step 3: owner unique + fp32 scatter-add
        UniqueScatterAdd();

        printf("[engram_fetch_grad] Process done: rankId=%u, aivId=%u\n", rankId_, aivId_);
    }
}

} // namespace Mc2Kernel

#endif
