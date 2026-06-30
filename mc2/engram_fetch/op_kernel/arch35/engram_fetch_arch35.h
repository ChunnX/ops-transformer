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
 * \file engram_fetch_arch35.h
 * \brief engram_fetch算子arch35 kernel实现
 */


#ifndef ENGRAM_FETCH_ARCH35_H
#define ENGRAM_FETCH_ARCH35_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_tiling_data.h"
#include "../engram_fetch.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace Mc2Kernel {

using namespace AscendC;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class EngramFetchArch35 {
public:
    __aicore__ inline EngramFetchArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
                                TPipe *pipe, const EngramFetchTilingData *tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);
    __aicore__ inline void CopyContextToUb();
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);
    __aicore__ inline void ScatterByRank(const LocalTensor<int32_t> &indicesLocal,
                                         const LocalTensor<uint32_t> &rankCounts,
                                         const LocalTensor<uint32_t> &rankOffsets,
                                         const LocalTensor<uint32_t> &tokenIdxInRank,
                                         uint32_t batchLen);

    TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    uint64_t ubSize_{0};
    uint32_t indicesBatchSize_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};

    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};
    TBuf<> indicesBuf_;
    TBuf<> hcommBuf_;
    TBuf<> rankCountsBuf_;
    TBuf<> rankOffsetsBuf_;
    TBuf<> tokenIdxInRankBuf_;
    TBuf<> commBufferBuf_;
    TBuf<> hcommHandleBuf_;
    TBuf<> rankIDsBuf_;
    TBuf<> positionsBuf_;
    TBuf<> divisorBuf_;
    TBuf<> maskBuf_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchArch35::Init(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR workspaceGM,
    TPipe *pipe, const EngramFetchTilingData *tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    aivId_ = GetBlockIdx();
    (void)workspaceGM;

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;

    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    ubSize_ = tilingData->ubSize;

    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    uint32_t countsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);
    uint32_t offsetsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankOffsetsBuf_, offsetsBufSize);
    uint32_t commBufferBufSize = Ceil(numRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(commBufferBuf_, commBufferBufSize);
    uint32_t hcommHandleBufSize = Ceil(numRanks_ * sizeof(uint64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(hcommHandleBuf_, hcommHandleBufSize);

    constexpr uint64_t ubReserved = 8U * 1024U;
    // ScatterByRank 需要: indices + tokenIdxInRank + rankIDs + positions + divisor (各4字节) + mask(每元素1字节近似)
    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U;
    uint64_t usedUb = HCOMM_INIT_SIZE + TILE_BYTES * 2U + countsBufSize + offsetsBufSize
                    + commBufferBufSize + hcommHandleBufSize;
    uint64_t availableUb = (ubSize_ > usedUb + ubReserved) ? (ubSize_ - usedUb - ubReserved) : 0U;
    uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / bytesPerIndice);
    uint32_t numTokens = static_cast<uint32_t>(numTokens_);
    indicesBatchSize_ = (numTokens <= maxBatchSize) ? numTokens : maxBatchSize;
    if (indicesBatchSize_ == 0U) {
        indicesBatchSize_ = 1U;
    }

    uint32_t indicesBufSize = Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    uint32_t tokenIdxInRankBufSize = Ceil(indicesBatchSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenIdxInRankBuf_, tokenIdxInRankBufSize);

    uint32_t compareCntMax = Ceil(indicesBatchSize_ * sizeof(int32_t), ALIGNED_LEN_256) *
                             ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t int32BufSize = compareCntMax * sizeof(int32_t);
    tpipe_->InitBuffer(rankIDsBuf_, int32BufSize);
    tpipe_->InitBuffer(positionsBuf_, int32BufSize);
    tpipe_->InitBuffer(divisorBuf_, int32BufSize);
    uint32_t maskBufSize = Ceil(Ceil(compareCntMax, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(maskBuf_, maskBufSize);
}

__aicore__ inline void EngramFetchArch35::CopyContextToUb()
{
    LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
    GlobalTensor<uint64_t> commBufferGm;
    commBufferGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->commBuffer[0]);
    DataCopyExtParams cpComm{1U, numRanks_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint64_t> padComm{false, 0, 0, 0};
    DataCopyPad(commBufferLocal, commBufferGm, cpComm, padComm);

    LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
    GlobalTensor<uint64_t> hcommHandleGm;
    hcommHandleGm.SetGlobalBuffer((__gm__ uint64_t *)&ctxPtr_->hcommHandle_[0]);
    DataCopyExtParams cpHandle{1U, numRanks_ * static_cast<uint32_t>(sizeof(uint64_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint64_t> padHandle{false, 0, 0, 0};
    DataCopyPad(hcommHandleLocal, hcommHandleGm, cpHandle, padHandle);
}

__aicore__ inline void EngramFetchArch35::CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
{
    GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

__aicore__ inline void EngramFetchArch35::ScatterByRank(
    const LocalTensor<int32_t> &indicesLocal,
    const LocalTensor<uint32_t> &rankCounts,
    const LocalTensor<uint32_t> &rankOffsets,
    const LocalTensor<uint32_t> &tokenIdxInRank,
    uint32_t batchLen)
{
    uint32_t compareCntMax = Ceil(batchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

    LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
    LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
    LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();

    ArithProgression<int32_t>(positions, 0, 1, batchLen);
    Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax);

    Duplicate<int32_t>(rankIDs, static_cast<int32_t>(numRanks_), compareCntMax);
    PipeBarrier<PIPE_V>();

    Div<int32_t>(rankIDs, indicesLocal, divisor, batchLen);
    PipeBarrier<PIPE_V>();
    SyncFunc<HardEvent::V_S>();

    uint32_t runningOffset = 0;
    uint32_t totalBlocks = GetBlockNum();
    for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks) {
        CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ, compareCntMax);
        PipeBarrier<PIPE_V>();

        LocalTensor<int32_t> dstRegion = tokenIdxInRank[runningOffset].ReinterpretCast<int32_t>();
        uint64_t rsvdCnt = 0;
        GatherMask(dstRegion, positions, mask.ReinterpretCast<uint32_t>(), true,
                   batchLen, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        uint32_t count = static_cast<uint32_t>(rsvdCnt);
        rankOffsets.SetValue(ownerRank, runningOffset);
        rankCounts.SetValue(ownerRank, count);
        runningOffset += count;
    }

    SyncFunc<HardEvent::V_S>();
}

__aicore__ inline void EngramFetchArch35::Process()
{
    if ASCEND_IS_AIV {
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            return;
        }

        uint32_t totalBlocks = GetBlockNum();

        CopyContextToUb();

        LocalTensor<uint64_t> commBufferLocal = commBufferBuf_.Get<uint64_t>();
        LocalTensor<uint64_t> hcommHandleLocal = hcommHandleBuf_.Get<uint64_t>();
        LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
        LocalTensor<uint32_t> rankCounts = rankCountsBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> rankOffsets = rankOffsetsBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

        uint64_t hiddenBytes = static_cast<uint64_t>(hiddenBytes_);
        uint32_t numTokens = static_cast<uint32_t>(numTokens_);
        uint32_t numEntriesPerRank = static_cast<uint32_t>(numEntriesPerRank_);

        uint32_t indicesBatchStart = 0;
        while (indicesBatchStart < numTokens) {
            uint32_t indicesBatchLen = indicesBatchSize_;
            if (indicesBatchStart + indicesBatchLen > numTokens) {
                indicesBatchLen = numTokens - indicesBatchStart;
            }

            CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
            SyncFunc<HardEvent::MTE2_S>();

            ScatterByRank(indicesLocal, rankCounts, rankOffsets, tokenIdxInRank, indicesBatchLen);

            for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks) {
                uint32_t idxStart = ownerRank * numEntriesPerRank;
                uint32_t rankStart = rankOffsets(ownerRank);
                uint32_t cnt = rankCounts(ownerRank);

                uint32_t channelIdx = ownerRank;
                uint32_t pendingReadCount = 0;
                for (uint32_t b = 0; b < cnt; b++) {
                    uint32_t i = tokenIdxInRank(rankStart + b);
                    int32_t globalIdx = indicesLocal(i);
                    uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - idxStart;
                    uint64_t globalTokenIdx = indicesBatchStart + i;
                    GM_ADDR dst = fetchedGM_ + globalTokenIdx * hiddenBytes;

                    if (ownerRank == rankId_) {
                        GM_ADDR src = (GM_ADDR)commBufferLocal(rankId_)
                                      + static_cast<uint64_t>(localEntryIdx) * hiddenBytes;
                        LocalCopySlice(dst, src, hiddenBytes);
                    } else {
                        GM_ADDR remoteSrcAddr = (GM_ADDR)commBufferLocal(ownerRank)
                                                + static_cast<uint64_t>(localEntryIdx) * hiddenBytes;
                        hcomm_.ReadNbi<false>(hcommHandleLocal(channelIdx), dst, remoteSrcAddr, hiddenBytes);
                        pendingReadCount++;
                        if (pendingReadCount >= READ_COMMIT_THRESHOLD) {
                            hcomm_.Commit(hcommHandleLocal(channelIdx));
                            pendingReadCount = 0;
                        }
                    }
                }

                if (pendingReadCount > 0) {
                    hcomm_.Commit(hcommHandleLocal(channelIdx));
                }
            }

            indicesBatchStart += indicesBatchLen;
        }
    }
}

__aicore__ inline void EngramFetchArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
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

        DataCopy(tmp, srcGm[off], thisLen);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        DataCopy(dstGm[off], tmp, thisLen);
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
} // namespace Mc2Kernel

#endif