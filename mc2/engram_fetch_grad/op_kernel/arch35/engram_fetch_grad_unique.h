/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_UNIQUE_H
#define ENGRAM_FETCH_GRAD_UNIQUE_H

#include "kernel_operator.h"
#include "adv_api/math/cumsum.h"
#include "../engram_fetch_grad_utils.h"

namespace EngramFetchGradUnique {

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc(AscendC::TPipe &pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

static constexpr AscendC::CumSumConfig cumsumCfg{true, false, false,
                                                   AscendC::CumSumAlgorithm::CUMSUM_ALGORITHM_LINEBYLINE};

static __aicore__ inline uint32_t GetDtypeSize(int32_t dtype)
{
    if (dtype == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        return sizeof(bfloat16_t);
    }
    if (dtype == Mc2Kernel::ENGRAM_DT_FLOAT16) {
        return sizeof(half);
    }
    return sizeof(float);
}

class EngramFetchGradUnique {
public:
    __aicore__ inline EngramFetchGradUnique() = default;

    __aicore__ inline void Init(uint32_t aivId, uint32_t totalBlocks, uint32_t rankId, uint32_t numRanks,
                                int32_t numEntriesPerRank, int64_t hiddenDim, int64_t hiddenBytes, int32_t inputDtype,
                                int32_t outputDtype, AscendC::TPipe *pipe, AscendC::TBuf<> &entryBuf,
                                AscendC::TBuf<> &gradBuf, AscendC::TBuf<> &indicesBuf, AscendC::TBuf<> &tempBuf,
                                AscendC::TBuf<> &statusBuf, AscendC::TBuf<> &castBuf, AscendC::TBuf<> &accumBuf);

    __aicore__ inline void ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM);

    __aicore__ inline void RunCount(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR coreStartGM,
                                    GM_ADDR segCountGM);
    __aicore__ inline void RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                          GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
                                          GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM, GM_ADDR coreStartGM,
                                          GM_ADDR segCountGM, GM_ADDR sortedSrcIdxGM);

    __aicore__ inline void WriteNumUniqueZero(GM_ADDR numUniqueOutGM);

    __aicore__ inline void SetCastBuf(AscendC::TBuf<> &castBuf) { castBuf_ = &castBuf; }
    __aicore__ inline void SetAccumBuf(AscendC::TBuf<> &accumBuf) { accumBuf_ = &accumBuf; }
    __aicore__ inline void SetGradSubBatch(uint32_t batch) { gradSubBatch_ = batch; }

private:
    __aicore__ inline void ScatterAccumulateParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                     GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                     GM_ADDR recvGradGM, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                     GM_ADDR sortedSrcIdxGM);
    __aicore__ inline void WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM);
    __aicore__ inline void LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                         uint32_t &start, uint32_t &end, int32_t &globalUnique);
    __aicore__ inline uint32_t ProcessScatterBatch(uint32_t cur, uint32_t end, int32_t &runningCompactIdx,
                                                    int32_t &runningWriteOffset, int32_t &prevEntry,
                                                    bool &isFirstElement, GM_ADDR recvLocalEntryOutGM,
                                                    GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                    GM_ADDR recvGradGM, GM_ADDR sortedSrcIdxGM);
    __aicore__ inline uint32_t ProcessBatchIndex(uint32_t cur, uint32_t batchLen, int32_t runningCompactIdx,
                                                   int32_t &prevEntry, bool &isFirstElement, int32_t &inclusiveSum,
                                                   GM_ADDR recvLocalEntryOutGM, GM_ADDR sortedSrcIdxGM);
    __aicore__ inline void WriteBatchUnique(uint32_t batchUniqueCnt, int32_t &runningWriteOffset,
                                             GM_ADDR uniqueLocalEntryOutGM);

    __aicore__ inline void FlushAccum(GM_ADDR gradUniqueOutGM);
    __aicore__ inline void DirectCopyGrad(AscendC::LocalTensor<uint8_t> &sortedGradBuf, uint32_t j,
                                          int32_t compactIdx, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateSubBatch(AscendC::LocalTensor<uint8_t> &sortedGradBuf, uint32_t subStart,
                                               uint32_t subLen, uint32_t batchLen, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void CastInputToFp32(AscendC::LocalTensor<float> &dst,
                                           AscendC::LocalTensor<uint8_t> &src, uint32_t j);

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t hiddenDim_{0};
    int64_t hiddenBytes_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    AscendC::TPipe *pipe_{nullptr};
    AscendC::TBuf<> *entryBuf_{nullptr};
    AscendC::TBuf<> *gradBuf_{nullptr};
    AscendC::TBuf<> *indicesBuf_{nullptr};
    AscendC::TBuf<> *tempBuf_{nullptr};
    AscendC::TBuf<> *statusBuf_{nullptr};
    AscendC::TBuf<> *castBuf_{nullptr};
    AscendC::TBuf<> *accumBuf_{nullptr};
    uint32_t gradSubBatch_{Mc2Kernel::GRAD_SUB_BATCH};

    int32_t compactIdx_{-1};
    bool inAccum_{false};
    int32_t flushEvtVMte3_{0};
    int32_t flushEvtMte3V_{0};
};

__aicore__ inline void EngramFetchGradUnique::Init(uint32_t aivId, uint32_t totalBlocks, uint32_t rankId,
                                                    uint32_t numRanks, int32_t numEntriesPerRank, int64_t hiddenDim,
                                                    int64_t hiddenBytes, int32_t inputDtype, int32_t outputDtype,
                                                    AscendC::TPipe *pipe, AscendC::TBuf<> &entryBuf,
                                                    AscendC::TBuf<> &gradBuf, AscendC::TBuf<> &indicesBuf,
                                                    AscendC::TBuf<> &tempBuf, AscendC::TBuf<> &statusBuf,
                                                    AscendC::TBuf<> &castBuf, AscendC::TBuf<> &accumBuf)
{
    aivId_ = aivId;
    totalBlocks_ = totalBlocks;
    rankId_ = rankId;
    numRanks_ = numRanks;
    numEntriesPerRank_ = numEntriesPerRank;
    hiddenDim_ = hiddenDim;
    hiddenBytes_ = hiddenBytes;
    inputDtype_ = inputDtype;
    outputDtype_ = outputDtype;
    pipe_ = pipe;
    entryBuf_ = &entryBuf;
    gradBuf_ = &gradBuf;
    indicesBuf_ = &indicesBuf;
    tempBuf_ = &tempBuf;
    statusBuf_ = &statusBuf;
    castBuf_ = &castBuf;
    accumBuf_ = &accumBuf;
}

__aicore__ inline void EngramFetchGradUnique::RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                             GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
                                                             GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                             GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                             GM_ADDR sortedSrcIdxGM)
{
    AscendC::SyncAll<true>();

    flushEvtVMte3_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
    flushEvtMte3V_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_V));

    ScatterAccumulateParallel(numRecv, recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                               coreStartGM, segCountGM, sortedSrcIdxGM);

    if (inAccum_) {
        FlushAccum(gradUniqueOutGM);
        inAccum_ = false;
    }

    WriteNumUnique(segCountGM, numUniqueOutGM);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);

    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(flushEvtVMte3_);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(flushEvtMte3V_);
}

__aicore__ inline void EngramFetchGradUnique::WriteNumUniqueZero(GM_ADDR numUniqueOutGM)
{
    if (aivId_ != 0) {
        AscendC::SyncAll<true>();
        return;
    }
    AscendC::GlobalTensor<int32_t> numUniqueGM;
    numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM);
    AscendC::LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
    tmp.SetValue(0, 0);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(numUniqueGM, tmp, params);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchGradUnique::ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM)
{
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t totalBytes = static_cast<uint64_t>(numRecv) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    uint64_t chunk = (totalBytes + totalBlocks_ - 1U) / totalBlocks_;
    uint64_t start = static_cast<uint64_t>(aivId_) * chunk;
    uint64_t end = start + chunk;
    if (end > totalBytes) {
        end = totalBytes;
    }

    if (start < end) {
        AscendC::LocalTensor<float> zeroBuf = gradBuf_->Get<float>();
        uint32_t tileFloats = Mc2Kernel::TILE_BYTES / sizeof(float);
        AscendC::Duplicate<float>(zeroBuf, 0.0f, tileFloats);
        SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);

        AscendC::GlobalTensor<uint8_t> dstGM;
        dstGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM);
        for (uint64_t off = start; off < end; off += Mc2Kernel::TILE_BYTES) {
            uint64_t thisLen = end - off;
            if (thisLen > Mc2Kernel::TILE_BYTES) {
                thisLen = Mc2Kernel::TILE_BYTES;
            }
            AscendC::DataCopyParams params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
            AscendC::DataCopyPad(dstGM[off], zeroBuf.ReinterpretCast<uint8_t>(), params);
        }
    }
}

__aicore__ inline void EngramFetchGradUnique::RunCount(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                       GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t rawStart = aivId_ * chunk;
    uint32_t rawEnd = rawStart + chunk;
    if (rawEnd > numRecv) {
        rawEnd = numRecv;
    }

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);

    int32_t prevEntry = 0;
    uint32_t start = rawStart;
    if (rawStart > 0 && rawStart < rawEnd) {
        AscendC::LocalTensor<int32_t> probeUb = tempBuf_->Get<int32_t>();
        uint32_t probeLen = rawEnd - rawStart + 1U;
        if (probeLen > Mc2Kernel::ENTRY_BATCH_CAP) {
            probeLen = Mc2Kernel::ENTRY_BATCH_CAP;
        }
        AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(probeLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(probeUb, sortedEntryGM[rawStart - 1U], cpParams, cpPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

        int32_t boundaryEntry = probeUb.GetValue(0);
        uint32_t p = 1U;
        while (p < probeLen && probeUb.GetValue(p) == boundaryEntry) {
            p++;
        }
        start = rawStart + p - 1U;
        prevEntry = boundaryEntry;
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        AscendC::LocalTensor<int32_t> entryUb = tempBuf_->Get<int32_t>();
        AscendC::LocalTensor<int32_t> diffUb = indicesBuf_->Get<int32_t>();
        AscendC::LocalTensor<int32_t> sumUb = statusBuf_->Get<int32_t>();
        bool isFirstElement = (start == 0);

        uint32_t cur = start;
        while (cur < rawEnd) {
            uint32_t batchLen = rawEnd - cur;
            if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
                batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
            }
            AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
            SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);

            if (isFirstElement && batchLen == 1) {
                SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
                localUniqueCount++;
                prevEntry = entryUb.GetValue(0);
                isFirstElement = false;
            } else if (batchLen >= 2U) {
                uint32_t cmpLen = batchLen - 1U;
                AscendC::Sub<int32_t>(diffUb, entryUb[1], entryUb, cmpLen);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mins<int32_t>(diffUb, diffUb, 1, cmpLen);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::WholeReduceSum<int32_t>(sumUb, diffUb, cmpLen, 1U, 1U, 1U, 8U);
                SyncFunc<AscendC::HardEvent::V_S>(*pipe_);
                int32_t batchNewUnique = sumUb.GetValue(0);

                if (isFirstElement) {
                    localUniqueCount += batchNewUnique + 1;
                    isFirstElement = false;
                } else {
                    int32_t firstEntry = entryUb.GetValue(0);
                    if (firstEntry != prevEntry) {
                        localUniqueCount += batchNewUnique + 1;
                    } else {
                        localUniqueCount += batchNewUnique;
                    }
                }
                prevEntry = entryUb.GetValue(batchLen - 1U);
            }
            cur += batchLen;
        }
    }

    AscendC::LocalTensor<int32_t> cntUb = statusBuf_->Get<int32_t>();
    cntUb.SetValue(0, static_cast<int32_t>(start));
    cntUb.SetValue(1, static_cast<int32_t>(localUniqueCount));

    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::DataCopyParams oneParam{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(coreStartGMT[aivId_], cntUb, oneParam);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::DataCopyPad(segCountGMT[aivId_], cntUb[1], oneParam);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
}

__aicore__ inline void EngramFetchGradUnique::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM,
                                                            GM_ADDR segCountGM, uint32_t &start, uint32_t &end,
                                                            int32_t &globalUnique)
{
    AscendC::GlobalTensor<int32_t> coreStartGM_;
    coreStartGM_.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGM_;
    segCountGM_.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::LocalTensor<int32_t> startUb = tempBuf_->Get<int32_t>();
    uint32_t arrayBytes = totalBlocks_ * sizeof(int32_t);
    AscendC::DataCopyPadExtParams<int32_t> csPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams csParams{1U, static_cast<uint32_t>(arrayBytes), 0U, 0U, 0U};
    AscendC::DataCopyPad(startUb, coreStartGM_, csParams, csPad);
    AscendC::DataCopyPad(startUb[totalBlocks_], segCountGM_, csParams, csPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    AscendC::LocalTensor<int32_t> segCountUb = startUb[totalBlocks_];
    globalUnique = 0;
    for (uint32_t core = 0; core < aivId_; core++) {
        globalUnique += segCountUb.GetValue(core);
    }

    if (segCountUb.GetValue(aivId_) == 0) {
        start = 0;
        end = 0;
        return;
    }
    start = static_cast<uint32_t>(startUb.GetValue(aivId_));
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        uint32_t nextCore = aivId_ + 1U;
        while (nextCore < totalBlocks_ - 1U && segCountUb.GetValue(nextCore) == 0) {
            nextCore++;
        }
        end = static_cast<uint32_t>(startUb.GetValue(nextCore));
    }
}

__aicore__ inline void EngramFetchGradUnique::CastInputToFp32(AscendC::LocalTensor<float> &dst,
                                                              AscendC::LocalTensor<uint8_t> &src, uint32_t j)
{
    uint32_t hd = static_cast<uint32_t>(hiddenDim_);
    uint64_t offset = static_cast<uint64_t>(j) * static_cast<uint32_t>(hiddenBytes_);
    if (inputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::LocalTensor<bfloat16_t> inT = src[offset].ReinterpretCast<bfloat16_t>();
        AscendC::Cast(dst, inT, AscendC::RoundMode::CAST_NONE, hd);
    } else if (inputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
        AscendC::LocalTensor<half> inT = src[offset].ReinterpretCast<half>();
        AscendC::Cast(dst, inT, AscendC::RoundMode::CAST_NONE, hd);
    } else {
        AscendC::LocalTensor<float> inT = src[offset].ReinterpretCast<float>();
        AscendC::DataCopy(dst, inT, hd);
    }
}

__aicore__ inline void EngramFetchGradUnique::FlushAccum(GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accum = accumBuf_->Get<float>();
    AscendC::LocalTensor<uint8_t> castOut = castBuf_->Get<uint8_t>();
    uint32_t hd = static_cast<uint32_t>(hiddenDim_);
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);

    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::LocalTensor<bfloat16_t> outT = castOut.ReinterpretCast<bfloat16_t>();
        AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, hd);
    } else if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
        AscendC::LocalTensor<half> outT = castOut.ReinterpretCast<half>();
        AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, hd);
    } else {
        AscendC::LocalTensor<float> outT = castOut.ReinterpretCast<float>();
        AscendC::DataCopy(outT, accum, hd);
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3_);

    constexpr uint32_t MAX_BLOCK_BYTES = 65535U;
    uint32_t totalBytes = hd * outDtypeSize;
    uint32_t numBlocks = (totalBytes + MAX_BLOCK_BYTES - 1U) / MAX_BLOCK_BYTES;
    uint64_t gmOffset = static_cast<uint64_t>(compactIdx_) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    AscendC::GlobalTensor<uint8_t> dstGM;
    dstGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM);
    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t byteOffset = b * MAX_BLOCK_BYTES;
        uint32_t remaining = totalBytes - byteOffset;
        uint16_t blkBytes = static_cast<uint16_t>(remaining > MAX_BLOCK_BYTES ? MAX_BLOCK_BYTES : remaining);
        AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
        AscendC::DataCopyPad(dstGM[gmOffset + byteOffset], castOut[byteOffset], params);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3V_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3V_);
}

__aicore__ inline void EngramFetchGradUnique::DirectCopyGrad(AscendC::LocalTensor<uint8_t> &sortedGradBuf,
                                                              uint32_t j, int32_t compactIdx,
                                                              GM_ADDR gradUniqueOutGM)
{
    uint64_t inOffset = static_cast<uint64_t>(j) * static_cast<uint32_t>(hiddenBytes_);
    uint64_t gmOffset = static_cast<uint64_t>(compactIdx) * static_cast<uint64_t>(hiddenBytes_);
    AscendC::GlobalTensor<uint8_t> dstGM;
    dstGM.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(flushEvtVMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(flushEvtVMte3_);
    constexpr uint32_t MAX_BLOCK_BYTES = 65535U;
    uint32_t totalBytes = static_cast<uint32_t>(hiddenBytes_);
    uint32_t numBlocks = (totalBytes + MAX_BLOCK_BYTES - 1U) / MAX_BLOCK_BYTES;
    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t byteOffset = b * MAX_BLOCK_BYTES;
        uint32_t remaining = totalBytes - byteOffset;
        uint16_t blkBytes = static_cast<uint16_t>(remaining > MAX_BLOCK_BYTES ? MAX_BLOCK_BYTES : remaining);
        AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
        AscendC::DataCopyPad(dstGM[gmOffset + byteOffset], sortedGradBuf[inOffset + byteOffset], params);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(flushEvtMte3V_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(flushEvtMte3V_);
}

__aicore__ inline void EngramFetchGradUnique::AccumulateSubBatch(AscendC::LocalTensor<uint8_t> &sortedGradBuf,
                                                                  uint32_t subStart, uint32_t subLen,
                                                                  uint32_t batchLen, GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accum = accumBuf_->Get<float>();
    AscendC::LocalTensor<float> tempFp32 = castBuf_->Get<float>();

    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];

    uint32_t hd = static_cast<uint32_t>(hiddenDim_);

    for (uint32_t j = 0; j < subLen; j++) {
        uint32_t idx = subStart + j;
        int32_t compactIdx = compactIdxBuf.GetValue(idx);

        if (compactIdx == compactIdx_) {
            CastInputToFp32(tempFp32, sortedGradBuf, j);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add<float>(accum, accum, tempFp32, hd);
            AscendC::PipeBarrier<PIPE_V>();
            inAccum_ = true;
        } else {
            if (inAccum_) {
                FlushAccum(gradUniqueOutGM);
            }
            bool isLastInBatch = (idx + 1U >= batchLen);
            bool isDirectCopy = !isLastInBatch &&
                                (compactIdxBuf.GetValue(idx + 1U) != compactIdx) &&
                                (compactIdx_ != compactIdx);
            if (isDirectCopy) {
                DirectCopyGrad(sortedGradBuf, j, compactIdx, gradUniqueOutGM);
                inAccum_ = false;
                compactIdx_ = -1;
            } else {
                AscendC::Duplicate<float>(accum, 0.0f, hd);
                AscendC::PipeBarrier<PIPE_V>();
                CastInputToFp32(accum, sortedGradBuf, j);
                AscendC::PipeBarrier<PIPE_V>();
                compactIdx_ = compactIdx;
                inAccum_ = true;
            }
        }
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchIndex(uint32_t cur, uint32_t batchLen,
                                                                      int32_t runningCompactIdx, int32_t &prevEntry,
                                                                      bool &isFirstElement, int32_t &inclusiveSum,
                                                                      GM_ADDR recvLocalEntryOutGM,
                                                                      GM_ADDR sortedSrcIdxGM)
{
    AscendC::LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> srcIdxUb = pingInt32;
    AscendC::LocalTensor<int32_t> uniqueEntryUb = pingInt32[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> sortedSrcIdxBuf = pingInt32[2 * Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    AscendC::GlobalTensor<int32_t> srcIdxGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
    srcIdxGM.SetGlobalBuffer((__gm__ int32_t *)sortedSrcIdxGM);
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    AscendC::DataCopyPad(srcIdxUb, srcIdxGM[cur], cpParams, cpPad);
    SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);

    sortedSrcIdxBuf = srcIdxUb;
    AscendC::DataCopy(sortedSrcIdxBuf, srcIdxUb, batchLen);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<int32_t> maskBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<float> floatBuf = tempBuf_->Get<float>();
    AscendC::LocalTensor<int32_t> inclBuf = tempBuf_->Get<int32_t>();

    uint32_t cmpLen = batchLen - 1U;
    AscendC::Sub<int32_t>(maskBuf, entryUb[1], entryUb, cmpLen);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins<int32_t>(maskBuf, maskBuf, 1, cmpLen);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Cast<float, int32_t>(floatBuf, maskBuf, AscendC::RoundMode::CAST_RINT, cmpLen);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::CumSumInfo cumsumInfo{1U, cmpLen};
    AscendC::CumSum<float, cumsumCfg>(floatBuf, floatBuf, floatBuf, cumsumInfo);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast<int32_t, float>(inclBuf, floatBuf, AscendC::RoundMode::CAST_RINT, cmpLen);
    AscendC::PipeBarrier<PIPE_V>();

    if (isFirstElement) {
        inclusiveSum = 1;
    } else {
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
        int32_t firstEntry = entryUb.GetValue(0);
        inclusiveSum = (firstEntry != prevEntry) ? 1 : 0;
    }

    if (batchLen >= 2U) {
        AscendC::Adds<int32_t>(compactIdxBuf[1], inclBuf, static_cast<int32_t>(runningCompactIdx + inclusiveSum - 1),
                               cmpLen);
        AscendC::PipeBarrier<PIPE_V>();
    }
    compactIdxBuf.SetValue(0, runningCompactIdx + inclusiveSum - 1);

    int32_t bias = static_cast<int32_t>(rankId_) * numEntriesPerRank_;
    AscendC::Adds<int32_t>(entryUb, entryUb, -bias, batchLen);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::GatherMaskParams gatherParams{1U, 1U, 8U, 8U};
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask<int32_t>(uniqueEntryUb, entryUb, static_cast<uint8_t>(0), true, batchLen, gatherParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t batchUniqueCnt = static_cast<uint32_t>(rsvdCnt);
    inclusiveSum += static_cast<int32_t>(batchUniqueCnt) - 1;
    prevEntry = entryUb.GetValue(batchLen - 1U) + bias;
    SyncFunc<AscendC::HardEvent::V_S>(*pipe_);
    isFirstElement = false;

    SyncFunc<AscendC::HardEvent::V_S>(*pipe_);
    return batchUniqueCnt;
}

__aicore__ inline void EngramFetchGradUnique::WriteBatchUnique(uint32_t batchUniqueCnt, int32_t &runningWriteOffset,
                                                                GM_ADDR uniqueLocalEntryOutGM)
{
    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> uniqueEntryUb = pingInt32[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM);
    SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);
    AscendC::DataCopyParams ueParams{1U, static_cast<uint16_t>(batchUniqueCnt * sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(uniqueEntryOutGM[runningWriteOffset], uniqueEntryUb, ueParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    runningWriteOffset += static_cast<int32_t>(batchUniqueCnt);
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessScatterBatch(
    uint32_t cur, uint32_t end, int32_t &runningCompactIdx, int32_t &runningWriteOffset, int32_t &prevEntry,
    bool &isFirstElement, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR recvGradGM, GM_ADDR sortedSrcIdxGM)
{
    uint32_t batchLen = end - cur;
    if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
        batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
    }

    int32_t inclusiveSum = 0;
    uint32_t batchUniqueCnt = ProcessBatchIndex(cur, batchLen, runningCompactIdx, prevEntry, isFirstElement, inclusiveSum,
                                                 recvLocalEntryOutGM, sortedSrcIdxGM);

    uint32_t maxGradPerBatch = Mc2Kernel::TILE_BYTES / static_cast<uint32_t>(hiddenBytes_);
    if (maxGradPerBatch < 1U) {
        maxGradPerBatch = 1U;
    }
    if (maxGradPerBatch > gradSubBatch_) {
        maxGradPerBatch = gradSubBatch_;
    }
    uint32_t gradBufHalf = 32U * 1024U;

    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> sortedSrcIdxBuf = pingInt32[2 * Mc2Kernel::ENTRY_BATCH_CAP];

    AscendC::LocalTensor<uint8_t> gradPing = gradBuf_->Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> gradPong = gradPing[gradBufHalf];

    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};

    event_t evtMte2V_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtMte2V_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte2_0 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));
    event_t evtVMte2_1 = static_cast<event_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE2));

    uint32_t tileIdx = 0;
    for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxGradPerBatch) {
            subLen = maxGradPerBatch;
        }
        uint32_t bufIdx = tileIdx % 2U;
        AscendC::LocalTensor<uint8_t> sortedGradBuf = (bufIdx == 0U) ? gradPing : gradPong;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(bufIdx == 0U ? evtVMte2_0 : evtVMte2_1);
        }

        for (uint32_t j = 0; j < subLen; j++) {
            int32_t srcIdx = sortedSrcIdxBuf.GetValue(subStart + j);
            GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(srcIdx) * hiddenBytes_;
            AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
            AscendC::GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
            AscendC::DataCopyPad(sortedGradBuf[static_cast<uint64_t>(j) * static_cast<uint32_t>(hiddenBytes_)], gradSrcGM,
                                  gradParams, gradPad);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(bufIdx == 0U ? evtMte2V_0 : evtMte2V_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(bufIdx == 0U ? evtMte2V_0 : evtMte2V_1);

        AccumulateSubBatch(sortedGradBuf, subStart, subLen, batchLen, gradUniqueOutGM);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(bufIdx == 0U ? evtVMte2_0 : evtVMte2_1);
        tileIdx++;
    }

    if (tileIdx >= 1U) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>((tileIdx - 1U) % 2U == 0U ? evtVMte2_0 : evtVMte2_1);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tileIdx % 2U == 0U ? evtVMte2_0 : evtVMte2_1);
    }
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V_0);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V_1);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2_0);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2_1);

    if (batchUniqueCnt > 0) {
        WriteBatchUnique(batchUniqueCnt, runningWriteOffset, uniqueLocalEntryOutGM);
    }
    runningCompactIdx += inclusiveSum;
    return cur + batchLen;
}

__aicore__ inline void EngramFetchGradUnique::ScatterAccumulateParallel(
    uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR recvGradGM, GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortedSrcIdxGM)
{
    int32_t globalUnique;
    uint32_t start;
    uint32_t end;
    LoadCoreRange(numRecv, coreStartGM, segCountGM, start, end, globalUnique);
    if (start >= end) {
        return;
    }
    int32_t runningCompactIdx = globalUnique;
    int32_t runningWriteOffset = globalUnique;
    int32_t prevEntry = 0;
    bool isFirstElement = true;

    compactIdx_ = -1;
    inAccum_ = false;

    uint32_t cur = start;
    while (cur < end) {
        cur = ProcessScatterBatch(cur, end, runningCompactIdx, runningWriteOffset, prevEntry, isFirstElement,
                                   recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                                   sortedSrcIdxGM);
    }
}

__aicore__ inline void EngramFetchGradUnique::WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM)
{
    if (aivId_ == 0) {
        AscendC::GlobalTensor<int32_t> segCountGMT;
        segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
        AscendC::LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams scParams{1U, static_cast<uint32_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(segCountUb, segCountGMT, scParams, scPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

        int32_t numUnique = 0;
        for (uint32_t core = 0; core < totalBlocks_; core++) {
            numUnique += segCountUb.GetValue(core);
        }

        AscendC::LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
        tmp.SetValue(0, numUnique);
        SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
        AscendC::GlobalTensor<int32_t> numUniqueGM;
        numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM);
        AscendC::DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        AscendC::DataCopyPad(numUniqueGM, tmp, p);
        SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    }
}


} // namespace EngramFetchGradUnique

#endif
