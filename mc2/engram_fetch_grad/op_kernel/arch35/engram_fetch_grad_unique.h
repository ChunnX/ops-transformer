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
#include "../engram_fetch_grad_utils.h"

namespace EngramFetchGradUnique {

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc(AscendC::TPipe &pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

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
                                          GM_ADDR segCountGM, GM_ADDR sortCompanionGM);

    __aicore__ inline void WriteNumUniqueZero(GM_ADDR numUniqueOutGM);

    __aicore__ inline void SetCastBuf(AscendC::TBuf<> &castBuf) { castBuf_ = &castBuf; }
    __aicore__ inline void SetAccumBuf(AscendC::TBuf<> &accumBuf) { accumBuf_ = &accumBuf; }
    __aicore__ inline void SetGradSubBatch(uint32_t batch) { gradSubBatch_ = batch; }

    uint64_t usaZeroCycle_{0};
    uint64_t usaCountCycle_{0};
    uint64_t usaScatterAddCycle_{0};
    uint64_t usaCastCycle_{0};

private:
    __aicore__ inline void CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR coreStartGM,
                                                 GM_ADDR segCountGM);
    __aicore__ inline void FixCoreStartForSkippedCores(GM_ADDR coreStartGM, GM_ADDR segCountGM);
    __aicore__ inline void ScatterAccumulateParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                     GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                     GM_ADDR recvGradGM, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                     GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM);
    __aicore__ inline void LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, uint32_t &start, uint32_t &end);
    __aicore__ inline int32_t ComputePreCoreOffset(GM_ADDR segCountGM);
    __aicore__ inline uint32_t ProcessScatterBatch(uint32_t cur, uint32_t end, int32_t &runningOffset,
                                                    int32_t &runningUniqueOffset, int32_t &prevEntry,
                                                    bool &isFirstElement, GM_ADDR recvLocalEntryOutGM,
                                                    GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                    GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline uint32_t ProcessBatchUnique(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                   int32_t &prevEntry, bool &isFirstElement, int32_t &inclusiveSum,
                                                   GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                             GM_ADDR uniqueLocalEntryOutGM);

    template <typename T>
    __aicore__ inline void FlushAccumTyped(AscendC::GlobalTensor<T> &dstGM);
    template <typename T>
    __aicore__ inline void AccumulateSubBatchTyped(AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t subStart,
                                                    uint32_t subLen, GM_ADDR gradUniqueOutGM);
    template <typename OutT>
    __aicore__ inline void CastToOutputType(AscendC::LocalTensor<OutT> &outT,
                                            AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t count);

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

    int32_t myPreCoreOffset_{0};
    int32_t mySegCount_{0};

    int32_t accumCompactIdx_{-1};
    bool accumDirty_{false};
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

__aicore__ inline void EngramFetchGradUnique::RunCount(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                       GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    CountUniquesParallel(numRecv, recvLocalEntryOutGM, coreStartGM, segCountGM);
}

__aicore__ inline void EngramFetchGradUnique::RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                             GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
                                                             GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                             GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                             GM_ADDR sortCompanionGM)
{
    FixCoreStartForSkippedCores(coreStartGM, segCountGM);
    AscendC::SyncAll<true>();

    flushEvtVMte3_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::V_MTE3));
    flushEvtMte3V_ = static_cast<int32_t>(pipe_->FetchEventID(AscendC::HardEvent::MTE3_V));

    ScatterAccumulateParallel(numRecv, recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                               coreStartGM, segCountGM, sortCompanionGM);

    if (accumDirty_) {
        uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
        uint64_t gmByteOffset = static_cast<uint64_t>(accumCompactIdx_) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
        if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::GlobalTensor<bfloat16_t> dstGM;
            dstGM.SetGlobalBuffer((__gm__ bfloat16_t *)(gradUniqueOutGM + gmByteOffset));
            FlushAccumTyped<bfloat16_t>(dstGM);
        } else if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
            AscendC::GlobalTensor<half> dstGM;
            dstGM.SetGlobalBuffer((__gm__ half *)(gradUniqueOutGM + gmByteOffset));
            FlushAccumTyped<half>(dstGM);
        } else {
            AscendC::GlobalTensor<float> dstGM;
            dstGM.SetGlobalBuffer((__gm__ float *)(gradUniqueOutGM + gmByteOffset));
            FlushAccumTyped<float>(dstGM);
        }
        accumDirty_ = false;
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

__aicore__ inline void EngramFetchGradUnique::CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
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
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        AscendC::LocalTensor<int32_t> entryUb = tempBuf_->Get<int32_t>();
        int32_t prevEntry = 0;
        bool isFirstElement = (start == 0);
        if (!isFirstElement) {
            AscendC::DataCopyPadExtParams<int32_t> onePad{false, 0, 0, 0};
            AscendC::DataCopyExtParams oneParams{1U, sizeof(int32_t), 0U, 0U, 0U};
            AscendC::DataCopyPad(entryUb, sortedEntryGM[start - 1U], oneParams, onePad);
            SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
            prevEntry = entryUb.GetValue(0);
        }

        uint32_t cur = start;
        while (cur < rawEnd) {
            uint32_t batchLen = rawEnd - cur;
            if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
                batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
            }
            AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
            SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
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

    AscendC::LocalTensor<int32_t> cntUb = statusBuf_->Get<int32_t>();
    cntUb.SetValue(0, static_cast<int32_t>(start));
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::DataCopyParams startParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(coreStartGMT[aivId_], cntUb, startParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);

    cntUb.SetValue(0, static_cast<int32_t>(localUniqueCount));
    mySegCount_ = static_cast<int32_t>(localUniqueCount);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    AscendC::DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(segCountGMT[aivId_], cntUb, p);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
}

__aicore__ inline void EngramFetchGradUnique::FixCoreStartForSkippedCores(GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    if (aivId_ != 0) {
        return;
    }

    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);

    int32_t nextValidStart = -1;
    for (int32_t core = static_cast<int32_t>(totalBlocks_) - 1; core >= 0; core--) {
        int32_t segCnt = segCountGMT.GetValue(core);
        if (segCnt == 0) {
            if (nextValidStart >= 0) {
                AscendC::LocalTensor<int32_t> fixUb = statusBuf_->Get<int32_t>();
                fixUb.SetValue(0, nextValidStart);
                SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
                AscendC::DataCopyParams fixParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
                AscendC::DataCopyPad(coreStartGMT[core], fixUb, fixParams);
                SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
            }
        } else {
            nextValidStart = coreStartGMT.GetValue(core);
        }
    }
}

__aicore__ inline void EngramFetchGradUnique::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, uint32_t &start,
                                                            uint32_t &end)
{
    AscendC::GlobalTensor<int32_t> coreStartGM_;
    coreStartGM_.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    AscendC::LocalTensor<int32_t> startUb = tempBuf_->Get<int32_t>();
    uint32_t readCores = aivId_ + 2U;
    if (readCores > totalBlocks_) {
        readCores = totalBlocks_;
    }
    if (aivId_ == totalBlocks_ - 1U) {
        readCores = totalBlocks_;
    }
    AscendC::DataCopyPadExtParams<int32_t> csPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams csParams{1U, static_cast<uint32_t>(readCores * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(startUb, coreStartGM_, csParams, csPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    start = static_cast<uint32_t>(startUb.GetValue(aivId_));
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        end = static_cast<uint32_t>(startUb.GetValue(aivId_ + 1U));
    }
}

__aicore__ inline int32_t EngramFetchGradUnique::ComputePreCoreOffset(GM_ADDR segCountGM)
{
    int32_t preCoreOffset = 0;
    if (aivId_ > 0) {
        AscendC::GlobalTensor<int32_t> segCountGMT;
        segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
        AscendC::LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams scParams{1U, static_cast<uint32_t>(aivId_ * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(segCountUb, segCountGMT, scParams, scPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
        for (uint32_t core = 0; core < aivId_; core++) {
            preCoreOffset += segCountUb.GetValue(core);
        }
    }
    return preCoreOffset;
}

template <typename T>
__aicore__ inline void EngramFetchGradUnique::FlushAccumTyped(AscendC::GlobalTensor<T> &dstGM)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3_);

    AscendC::LocalTensor<T> accum = accumBuf_->Get<T>();
    constexpr uint32_t MAX_BLOCK_BYTES = 65535U;
    uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(T);
    uint32_t numBlocks = (totalBytes + MAX_BLOCK_BYTES - 1U) / MAX_BLOCK_BYTES;
    uint64_t elemBase = static_cast<uint64_t>(accumCompactIdx_) * static_cast<uint64_t>(hiddenDim_);
    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t byteOffset = b * MAX_BLOCK_BYTES;
        uint32_t elemOffset = byteOffset / sizeof(T);
        uint32_t remaining = totalBytes - byteOffset;
        uint16_t blkBytes = static_cast<uint16_t>(remaining > MAX_BLOCK_BYTES ? MAX_BLOCK_BYTES : remaining);
        AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
        AscendC::DataCopyPad(dstGM[elemBase + elemOffset], accum[elemOffset], params);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3V_);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3V_);
}

template <typename OutT>
__aicore__ inline void EngramFetchGradUnique::CastToOutputType(AscendC::LocalTensor<OutT> &outT,
                                                                AscendC::LocalTensor<uint8_t> &gradRaw, uint32_t count)
{
    if (inputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::LocalTensor<bfloat16_t> inT = gradRaw.ReinterpretCast<bfloat16_t>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_NONE, count);
    } else if (inputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
        AscendC::LocalTensor<half> inT = gradRaw.ReinterpretCast<half>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_NONE, count);
    } else {
        AscendC::LocalTensor<float> inT = gradRaw.ReinterpretCast<float>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_RINT, count);
    }
}

template <typename T>
__aicore__ inline void EngramFetchGradUnique::AccumulateSubBatchTyped(AscendC::LocalTensor<uint8_t> &gradRaw,
                                                                       uint32_t subStart, uint32_t subLen,
                                                                       GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<T> gradTyped = gradRaw.ReinterpretCast<T>();
    AscendC::LocalTensor<T> accum = accumBuf_->Get<T>();
    AscendC::GlobalTensor<T> dstGM;
    dstGM.SetGlobalBuffer((__gm__ T *)gradUniqueOutGM);

    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];

    for (uint32_t j = 0; j < subLen; j++) {
        int32_t compactIdx = compactIdxBuf.GetValue(subStart + j);
        if (compactIdx != accumCompactIdx_) {
            if (accumDirty_) {
                FlushAccumTyped<T>(dstGM);
            }
            AscendC::Duplicate<T>(accum, static_cast<T>(0), static_cast<uint32_t>(hiddenDim_));
            AscendC::PipeBarrier<PIPE_V>();
            accumCompactIdx_ = compactIdx;
        }
        AscendC::Add<T>(accum, accum, gradTyped[j * static_cast<uint32_t>(hiddenDim_)],
                        static_cast<uint32_t>(hiddenDim_));
        AscendC::PipeBarrier<PIPE_V>();
        accumDirty_ = true;
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchUnique(uint32_t cur, uint32_t batchLen,
                                                                      int32_t runningOffset, int32_t &prevEntry,
                                                                      bool &isFirstElement, int32_t &inclusiveSum,
                                                                      GM_ADDR recvLocalEntryOutGM,
                                                                      GM_ADDR sortCompanionGM)
{
    AscendC::LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> compUb = pingInt32;
    AscendC::LocalTensor<int32_t> uniqueUb = pingInt32[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * Mc2Kernel::ENTRY_BATCH_CAP];

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    AscendC::GlobalTensor<int32_t> compGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM);
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    AscendC::DataCopyPad(compUb, compGM[cur], cpParams, cpPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    inclusiveSum = 0;
    uint32_t tileUniqueCnt = 0;
    for (uint32_t i = 0; i < batchLen; i++) {
        int32_t entry = entryUb.GetValue(i);
        bool isNewUnique = isFirstElement || (entry != prevEntry);
        isFirstElement = false;
        prevEntry = entry;
        if (isNewUnique) {
            inclusiveSum++;
            uniqueUb.SetValue(tileUniqueCnt, entry - static_cast<int32_t>(rankId_) * numEntriesPerRank_);
            tileUniqueCnt++;
        }
        compactIdxBuf.SetValue(i, runningOffset + inclusiveSum - 1);
        recvIdxBuf.SetValue(i, compUb.GetValue(i));
    }
    return tileUniqueCnt;
}

__aicore__ inline void EngramFetchGradUnique::WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                                                GM_ADDR uniqueLocalEntryOutGM)
{
    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> uniqueUb = pingInt32[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams ueParams{1U, static_cast<uint16_t>(tileUniqueCnt * sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(uniqueEntryOutGM[runningUniqueOffset], uniqueUb, ueParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    runningUniqueOffset += static_cast<int32_t>(tileUniqueCnt);
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessScatterBatch(
    uint32_t cur, uint32_t end, int32_t &runningOffset, int32_t &runningUniqueOffset, int32_t &prevEntry,
    bool &isFirstElement, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR recvGradGM, GM_ADDR sortCompanionGM)
{
    uint32_t batchLen = end - cur;
    if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
        batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
    }

    int32_t inclusiveSum = 0;
    uint32_t tileUniqueCnt = ProcessBatchUnique(cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
                                                 recvLocalEntryOutGM, sortCompanionGM);

    uint32_t maxGradPerBatch = Mc2Kernel::TILE_BYTES / static_cast<uint32_t>(hiddenBytes_);
    if (maxGradPerBatch < 1U) {
        maxGradPerBatch = 1U;
    }
    if (maxGradPerBatch > gradSubBatch_) {
        maxGradPerBatch = gradSubBatch_;
    }
    uint32_t gradBufHalf = 32U * 1024U;

    AscendC::LocalTensor<int32_t> pingInt32 = entryBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * Mc2Kernel::ENTRY_BATCH_CAP];

    AscendC::LocalTensor<uint8_t> gradPing = gradBuf_->Get<uint8_t>();
    AscendC::LocalTensor<uint8_t> gradPong = gradPing[gradBufHalf];

    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};

    bool needCast = (inputDtype_ != outputDtype_);
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t castHalfFloats = 0;
    if (needCast) {
        uint32_t castHalfBytes = (static_cast<uint32_t>(hiddenDim_) * outDtypeSize * maxGradPerBatch +
                                  Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        castHalfFloats = castHalfBytes / sizeof(float);
    }

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
        AscendC::LocalTensor<uint8_t> gradRaw = (bufIdx == 0U) ? gradPing : gradPong;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(bufIdx == 0U ? evtVMte2_0 : evtVMte2_1);
        }

        for (uint32_t j = 0; j < subLen; j++) {
            int32_t recvIdx = recvIdxBuf.GetValue(subStart + j);
            GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
            AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
            AscendC::GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
            AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * static_cast<uint32_t>(hiddenBytes_)], gradSrcGM,
                                  gradParams, gradPad);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(bufIdx == 0U ? evtMte2V_0 : evtMte2V_1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(bufIdx == 0U ? evtMte2V_0 : evtMte2V_1);

        if (needCast) {
            AscendC::LocalTensor<float> castPingF = castBuf_->Get<float>();
            AscendC::LocalTensor<float> castPongF = castPingF[castHalfFloats];
            AscendC::LocalTensor<uint8_t> castRaw = (bufIdx == 0U) ?
                castPingF.ReinterpretCast<uint8_t>() : castPongF.ReinterpretCast<uint8_t>();
            uint32_t castCount = subLen * static_cast<uint32_t>(hiddenDim_);
            if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
                AscendC::LocalTensor<bfloat16_t> outT = castRaw.ReinterpretCast<bfloat16_t>();
                CastToOutputType<bfloat16_t>(outT, gradRaw, castCount);
            } else if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
                AscendC::LocalTensor<half> outT = castRaw.ReinterpretCast<half>();
                CastToOutputType<half>(outT, gradRaw, castCount);
            } else {
                AscendC::LocalTensor<float> outT = castRaw.ReinterpretCast<float>();
                CastToOutputType<float>(outT, gradRaw, castCount);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
                AccumulateSubBatchTyped<bfloat16_t>(castRaw, subStart, subLen, gradUniqueOutGM);
            } else if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
                AccumulateSubBatchTyped<half>(castRaw, subStart, subLen, gradUniqueOutGM);
            } else {
                AccumulateSubBatchTyped<float>(castRaw, subStart, subLen, gradUniqueOutGM);
            }
        } else {
            if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
                AccumulateSubBatchTyped<bfloat16_t>(gradRaw, subStart, subLen, gradUniqueOutGM);
            } else if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT16) {
                AccumulateSubBatchTyped<half>(gradRaw, subStart, subLen, gradUniqueOutGM);
            } else {
                AccumulateSubBatchTyped<float>(gradRaw, subStart, subLen, gradUniqueOutGM);
            }
        }

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

    if (tileUniqueCnt > 0) {
        WriteBatchUnique(tileUniqueCnt, runningUniqueOffset, uniqueLocalEntryOutGM);
    }
    runningOffset += inclusiveSum;
    return cur + batchLen;
}

__aicore__ inline void EngramFetchGradUnique::ScatterAccumulateParallel(
    uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR recvGradGM, GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM)
{
    int32_t preCoreOffset = ComputePreCoreOffset(segCountGM);
    myPreCoreOffset_ = preCoreOffset;

    uint32_t start;
    uint32_t end;
    LoadCoreRange(numRecv, coreStartGM, start, end);
    if (start >= end) {
        return;
    }
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;

    accumCompactIdx_ = -1;
    accumDirty_ = false;

    uint32_t cur = start;
    while (cur < end) {
        cur = ProcessScatterBatch(cur, end, runningOffset, runningUniqueOffset, prevEntry, isFirstElement,
                                   recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                                   sortCompanionGM);
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
