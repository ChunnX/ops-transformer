#ifndef ENGRAM_FETCH_GRAD_UNIQUE_H
#define ENGRAM_FETCH_GRAD_UNIQUE_H

#include "kernel_operator.h"
#include "../engram_fetch_grad_utils.h"

namespace EngramFetchGradUnique {

using namespace AscendC;
using namespace Mc2Kernel;

template <HardEvent event>
__aicore__ inline void SyncFunc(TPipe &pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

class UniqueScatterKernel {
public:
    __aicore__ inline UniqueScatterKernel() = default;

    __aicore__ inline void Init(
        uint32_t aivId, uint32_t totalBlocks, uint32_t rankId, uint32_t numRanks,
        int32_t numEntriesPerRank, int64_t hiddenDim, int64_t hiddenBytes,
        int32_t inputDtype, int32_t outputDtype, TPipe *pipe,
        TBuf<> &pingBuf, TBuf<> &pongBuf, TBuf<> &indicesBuf,
        TBuf<> &tempBuf, TBuf<> &statusBuf, TBuf<> &gradSumBuf)
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
        pingBuf_ = &pingBuf;
        pongBuf_ = &pongBuf;
        indicesBuf_ = &indicesBuf;
        tempBuf_ = &tempBuf;
        statusBuf_ = &statusBuf;
        gradSumBuf_ = &gradSumBuf;
    }

    __aicore__ inline void Run(
        uint32_t numRecv,
        GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
        GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
        GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM);

    __aicore__ inline void WriteNumUniqueZero(GM_ADDR numUniqueOutGM);

    uint64_t usaZeroCountCycle_{0};
    uint64_t usaScatterAddCycle_{0};
    uint64_t usaCastCycle_{0};

private:
    __aicore__ inline void ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM);
    __aicore__ inline void CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                 GM_ADDR coreStartGM, GM_ADDR segCountGM);
    __aicore__ inline void ScatterAddAtomicParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                    GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                    GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
                                                    GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void CastGradUniqueToOutput(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM,
                                                  GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM);
    __aicore__ inline void LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, uint32_t &start, uint32_t &end);
    __aicore__ inline int32_t ComputePreCoreOffset(GM_ADDR segCountGM);
    __aicore__ inline uint32_t ProcessScatterBatch(
        uint32_t cur, uint32_t end, int32_t &runningOffset, int32_t &runningUniqueOffset,
        int32_t &prevEntry, bool &isFirstElement,
        GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
        GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void ScatterGradSubBatch(uint32_t subStart, uint32_t subLen,
                                               GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM);
    __aicore__ inline void AtomicAddGrad(GlobalTensor<float> &atomicDstGM, LocalTensor<float> &srcT,
                                         uint32_t subStart, uint32_t subLen);
    __aicore__ inline uint32_t ProcessBatchUnique(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                   int32_t &prevEntry, bool &isFirstElement,
                                                   int32_t &inclusiveSum,
                                                   GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                            GM_ADDR uniqueLocalEntryOutGM);

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t hiddenDim_{0};
    int64_t hiddenBytes_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    TPipe *pipe_{nullptr};
    TBuf<> *pingBuf_{nullptr};
    TBuf<> *pongBuf_{nullptr};
    TBuf<> *indicesBuf_{nullptr};
    TBuf<> *tempBuf_{nullptr};
    TBuf<> *statusBuf_{nullptr};
    TBuf<> *gradSumBuf_{nullptr};
};

__aicore__ inline void UniqueScatterKernel::Run(
    uint32_t numRecv,
    GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
    GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
    GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM)
{
    uint64_t t0 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    ZeroGradUnique(numRecv, gradUniqueOutGM, gradUniqueFp32GM);
    CountUniquesParallel(numRecv, recvLocalEntryOutGM, coreStartGM, segCountGM);
    SyncAll<true>();
    uint64_t t1 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    ScatterAddAtomicParallel(numRecv, recvLocalEntryOutGM, uniqueLocalEntryOutGM,
                             gradUniqueOutGM, gradUniqueFp32GM, recvGradGM,
                             coreStartGM, segCountGM, sortCompanionGM);
    SyncAll<true>();
    uint64_t t2 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    CastGradUniqueToOutput(segCountGM, numUniqueOutGM, gradUniqueOutGM, gradUniqueFp32GM);
    SyncAll<true>();
    uint64_t t3 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    usaZeroCountCycle_ = t1 - t0;
    usaScatterAddCycle_ = t2 - t1;
    usaCastCycle_ = t3 - t2;
}

__aicore__ inline void UniqueScatterKernel::WriteNumUniqueZero(GM_ADDR numUniqueOutGM)
{
    if (aivId_ != 0) {
        SyncAll<true>();
        return;
    }
    GlobalTensor<int32_t> numUniqueGM;
    numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM);
    LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
    tmp.SetValue(0, 0);
    SyncFunc<HardEvent::S_MTE3>(*pipe_);
    DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(numUniqueGM, tmp, params);
    SyncFunc<HardEvent::MTE3_S>(*pipe_);
    SyncAll<true>();
}

__aicore__ inline void UniqueScatterKernel::ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM)
{
    uint32_t zeroCount = numRecv;
    uint64_t totalFloats = static_cast<uint64_t>(zeroCount) * static_cast<uint64_t>(hiddenDim_);
    uint64_t chunk = (totalFloats + totalBlocks_ - 1U) / totalBlocks_;
    uint64_t start = static_cast<uint64_t>(aivId_) * chunk;
    uint64_t end = start + chunk;
    if (end > totalFloats) end = totalFloats;

    if (start < end) {
        LocalTensor<float> zeroBuf = pongBuf_->Get<float>();
        uint32_t tileFloats = TILE_BYTES / sizeof(float);
        Duplicate<float>(zeroBuf, 0.0f, tileFloats);
        SyncFunc<HardEvent::V_MTE3>(*pipe_);

        GM_ADDR dstBase = (outputDtype_ == ENGRAM_DT_FLOAT) ? gradUniqueOutGM : gradUniqueFp32GM;
        GlobalTensor<float> dstGM;
        dstGM.SetGlobalBuffer((__gm__ float *)dstBase);
        for (uint64_t off = start; off < end; off += tileFloats) {
            uint64_t thisLen = end - off;
            if (thisLen > tileFloats) thisLen = tileFloats;
            DataCopyParams params{1U, static_cast<uint16_t>(thisLen * sizeof(float)), 0U, 0U};
            DataCopyPad(dstGM[off], zeroBuf, params);
        }
        SyncFunc<HardEvent::MTE3_S>(*pipe_);
    }
}

__aicore__ inline void UniqueScatterKernel::CountUniquesParallel(
    uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t rawStart = aivId_ * chunk;
    uint32_t rawEnd = rawStart + chunk;
    if (rawEnd > numRecv) rawEnd = numRecv;

    GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);

    uint32_t start = rawStart;
    if (rawStart > 0 && rawStart < rawEnd) {
        LocalTensor<int32_t> probeUb = tempBuf_->Get<int32_t>();
        uint32_t probeLen = rawEnd - rawStart + 1U;
        if (probeLen > ENTRY_BATCH_CAP) probeLen = ENTRY_BATCH_CAP;
        DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
        DataCopyExtParams cpParams{1U, static_cast<uint32_t>(probeLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(probeUb, sortedEntryGM[rawStart - 1U], cpParams, cpPad);
        SyncFunc<HardEvent::MTE2_S>(*pipe_);

        int32_t boundaryEntry = probeUb.GetValue(0);
        uint32_t p = 1U;
        while (p < probeLen && probeUb.GetValue(p) == boundaryEntry) {
            p++;
        }
        start = rawStart + p - 1U;
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        LocalTensor<int32_t> entryUb = tempBuf_->Get<int32_t>();
        int32_t prevEntry;
        bool isFirstElement = (start == 0);
        if (!isFirstElement) {
            DataCopyPadExtParams<int32_t> onePad{false, 0, 0, 0};
            DataCopyExtParams oneParams{1U, sizeof(int32_t), 0U, 0U, 0U};
            DataCopyPad(entryUb, sortedEntryGM[start - 1U], oneParams, onePad);
            SyncFunc<HardEvent::MTE2_S>(*pipe_);
            prevEntry = entryUb.GetValue(0);
        }

        uint32_t cur = start;
        while (cur < rawEnd) {
            uint32_t batchLen = rawEnd - cur;
            if (batchLen > ENTRY_BATCH_CAP) batchLen = ENTRY_BATCH_CAP;
            DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
            SyncFunc<HardEvent::MTE2_S>(*pipe_);
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

    LocalTensor<int32_t> cntUb = statusBuf_->Get<int32_t>();
    cntUb.SetValue(0, static_cast<int32_t>(start));
    SyncFunc<HardEvent::S_MTE3>(*pipe_);
    GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    DataCopyParams startParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(coreStartGMT[aivId_], cntUb, startParams);
    SyncFunc<HardEvent::MTE3_S>(*pipe_);

    cntUb.SetValue(0, static_cast<int32_t>(localUniqueCount));
    SyncFunc<HardEvent::S_MTE3>(*pipe_);
    GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPad(segCountGMT[aivId_], cntUb, p);
    SyncFunc<HardEvent::MTE3_S>(*pipe_);
}

__aicore__ inline void UniqueScatterKernel::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, uint32_t &start, uint32_t &end)
{
    GlobalTensor<int32_t> coreStartGM_;
    coreStartGM_.SetGlobalBuffer((__gm__ int32_t *)coreStartGM);
    LocalTensor<int32_t> startUb = tempBuf_->Get<int32_t>();
    uint32_t readCores = aivId_ + 2U;
    if (readCores > totalBlocks_) readCores = totalBlocks_;
    if (aivId_ == totalBlocks_ - 1U) readCores = totalBlocks_;
    DataCopyPadExtParams<int32_t> csPad{false, 0, 0, 0};
    DataCopyExtParams csParams{1U, static_cast<uint32_t>(readCores * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(startUb, coreStartGM_, csParams, csPad);
    SyncFunc<HardEvent::MTE2_S>(*pipe_);

    start = static_cast<uint32_t>(startUb.GetValue(aivId_));
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        end = static_cast<uint32_t>(startUb.GetValue(aivId_ + 1U));
    }
}

__aicore__ inline int32_t UniqueScatterKernel::ComputePreCoreOffset(GM_ADDR segCountGM)
{
    int32_t preCoreOffset = 0;
    GlobalTensor<int32_t> segCountGM_;
    segCountGM_.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    if (aivId_ > 0) {
        for (uint32_t core = 0; core < aivId_; core++) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(segCountGM_[core]);
            preCoreOffset += segCountGM_.GetValue(core);
        }
    }
    return preCoreOffset;
}

__aicore__ inline void UniqueScatterKernel::AtomicAddGrad(
    GlobalTensor<float> &atomicDstGM, LocalTensor<float> &srcT, uint32_t subStart, uint32_t subLen)
{
    LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * ENTRY_BATCH_CAP];
    for (uint32_t j = 0; j < subLen; j++) {
        int32_t compactIdx = compactIdxBuf.GetValue(subStart + j);
        SetAtomicAdd<float>();
        DataCopy(atomicDstGM[static_cast<uint64_t>(compactIdx) * hiddenDim_],
                 srcT[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenDim_)],
                 static_cast<uint32_t>(hiddenDim_));
        SetAtomicNone();
    }
}

__aicore__ inline void UniqueScatterKernel::ScatterGradSubBatch(
    uint32_t subStart, uint32_t subLen, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM)
{
    LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * ENTRY_BATCH_CAP];
    LocalTensor<uint8_t> gradRaw = pongBuf_->Get<uint8_t>();
    LocalTensor<float> gradFp32 = gradSumBuf_->Get<float>();
    GM_ADDR atomicDstBase = (outputDtype_ == ENGRAM_DT_FLOAT) ? gradUniqueOutGM : gradUniqueFp32GM;
    GlobalTensor<float> atomicDstGM;
    atomicDstGM.SetGlobalBuffer((__gm__ float *)atomicDstBase);

    DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
    for (uint32_t j = 0; j < subLen; j++) {
        int32_t recvIdx = recvIdxBuf.GetValue(subStart + j);
        GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
        DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
        GlobalTensor<uint8_t> gradSrcGM;
        gradSrcGM.SetGlobalBuffer((__gm__ uint8_t *)gradAddr);
        DataCopyPad(gradRaw[static_cast<uint32_t>(j) * static_cast<uint32_t>(hiddenBytes_)], gradSrcGM, gradParams, gradPad);
    }

    if (inputDtype_ == ENGRAM_DT_FLOAT) {
        SyncFunc<HardEvent::MTE2_MTE3>(*pipe_);
        LocalTensor<float> inT = gradRaw.ReinterpretCast<float>();
        AtomicAddGrad(atomicDstGM, inT, subStart, subLen);
        SyncFunc<HardEvent::MTE3_MTE2>(*pipe_);
    } else {
        SyncFunc<HardEvent::MTE2_V>(*pipe_);
        uint32_t castCount = subLen * static_cast<uint32_t>(hiddenDim_);
        if (inputDtype_ == ENGRAM_DT_BFLOAT16) {
            LocalTensor<bfloat16_t> inT = gradRaw.ReinterpretCast<bfloat16_t>();
            Cast(gradFp32, inT, AscendC::RoundMode::CAST_NONE, castCount);
        } else {
            LocalTensor<half> inT = gradRaw.ReinterpretCast<half>();
            Cast(gradFp32, inT, AscendC::RoundMode::CAST_NONE, castCount);
        }
        PipeBarrier<PIPE_V>();
        SyncFunc<HardEvent::V_MTE3>(*pipe_);
        AtomicAddGrad(atomicDstGM, gradFp32, subStart, subLen);
        SyncFunc<HardEvent::MTE3_MTE2>(*pipe_);
    }
}

__aicore__ inline uint32_t UniqueScatterKernel::ProcessBatchUnique(
    uint32_t cur, uint32_t batchLen, int32_t runningOffset,
    int32_t &prevEntry, bool &isFirstElement, int32_t &inclusiveSum,
    GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM)
{
    LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();
    LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    LocalTensor<int32_t> compUb = pingInt32;
    LocalTensor<int32_t> uniqueUb = pingInt32[ENTRY_BATCH_CAP];
    LocalTensor<int32_t> recvIdxBuf = pingInt32[2 * ENTRY_BATCH_CAP];
    LocalTensor<int32_t> compactIdxBuf = pingInt32[3 * ENTRY_BATCH_CAP];

    GlobalTensor<int32_t> sortedEntryGM, compGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t *)recvLocalEntryOutGM);
    compGM.SetGlobalBuffer((__gm__ int32_t *)sortCompanionGM);
    DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    DataCopyPad(compUb, compGM[cur], cpParams, cpPad);
    SyncFunc<HardEvent::MTE2_S>(*pipe_);

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

__aicore__ inline void UniqueScatterKernel::WriteBatchUnique(uint32_t tileUniqueCnt, int32_t &runningUniqueOffset,
                                                            GM_ADDR uniqueLocalEntryOutGM)
{
    LocalTensor<int32_t> pingInt32 = pingBuf_->Get<int32_t>();
    LocalTensor<int32_t> uniqueUb = pingInt32[ENTRY_BATCH_CAP];
    GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t *)uniqueLocalEntryOutGM);
    SyncFunc<HardEvent::S_MTE3>(*pipe_);
    DataCopyParams ueParams{1U, static_cast<uint16_t>(tileUniqueCnt * sizeof(int32_t)), 0U, 0U};
    DataCopyPad(uniqueEntryOutGM[runningUniqueOffset], uniqueUb, ueParams);
    SyncFunc<HardEvent::MTE3_S>(*pipe_);
    runningUniqueOffset += static_cast<int32_t>(tileUniqueCnt);
}

__aicore__ inline uint32_t UniqueScatterKernel::ProcessScatterBatch(
    uint32_t cur, uint32_t end, int32_t &runningOffset, int32_t &runningUniqueOffset,
    int32_t &prevEntry, bool &isFirstElement,
    GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM, GM_ADDR sortCompanionGM)
{
    uint32_t batchLen = end - cur;
    if (batchLen > ENTRY_BATCH_CAP) batchLen = ENTRY_BATCH_CAP;

    int32_t inclusiveSum = 0;
    uint32_t tileUniqueCnt = ProcessBatchUnique(
        cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
        recvLocalEntryOutGM, sortCompanionGM);

    for (uint32_t subStart = 0; subStart < batchLen; subStart += GRAD_SUB_BATCH) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > GRAD_SUB_BATCH) subLen = GRAD_SUB_BATCH;
        ScatterGradSubBatch(subStart, subLen, gradUniqueOutGM, gradUniqueFp32GM, recvGradGM);
    }

    if (tileUniqueCnt > 0) {
        WriteBatchUnique(tileUniqueCnt, runningUniqueOffset, uniqueLocalEntryOutGM);
    }
    runningOffset += inclusiveSum;
    return cur + batchLen;
}

__aicore__ inline void UniqueScatterKernel::ScatterAddAtomicParallel(
    uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
    GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM, GM_ADDR recvGradGM,
    GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM)
{
    uint32_t start, end;
    LoadCoreRange(numRecv, coreStartGM, start, end);
    if (start >= end) return;

    int32_t preCoreOffset = ComputePreCoreOffset(segCountGM);
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;

    uint32_t cur = start;
    while (cur < end) {
        cur = ProcessScatterBatch(cur, end, runningOffset, runningUniqueOffset, prevEntry, isFirstElement,
                                  recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM,
                                  gradUniqueFp32GM, recvGradGM, sortCompanionGM);
    }
}

__aicore__ inline void UniqueScatterKernel::CastGradUniqueToOutput(
    GM_ADDR segCountGM, GM_ADDR numUniqueOutGM, GM_ADDR gradUniqueOutGM, GM_ADDR gradUniqueFp32GM)
{
    GlobalTensor<int32_t> segCountGM_;
    segCountGM_.SetGlobalBuffer((__gm__ int32_t *)segCountGM);
    LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
    DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
    DataCopyExtParams scParams{1U, static_cast<uint32_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(segCountUb, segCountGM_, scParams, scPad);
    SyncFunc<HardEvent::MTE2_S>(*pipe_);

    int32_t numUnique = 0;
    for (uint32_t core = 0; core < totalBlocks_; core++) {
        numUnique += segCountUb.GetValue(core);
    }

    if (aivId_ == 0) {
        LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
        tmp.SetValue(0, numUnique);
        SyncFunc<HardEvent::S_MTE3>(*pipe_);
        GlobalTensor<int32_t> numUniqueGM;
        numUniqueGM.SetGlobalBuffer((__gm__ int32_t *)numUniqueOutGM);
        DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        DataCopyPad(numUniqueGM, tmp, p);
        SyncFunc<HardEvent::MTE3_S>(*pipe_);
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

    GlobalTensor<float> gradUniqueFp32GMT;
    gradUniqueFp32GMT.SetGlobalBuffer((__gm__ float *)gradUniqueFp32GM);
    GlobalTensor<uint8_t> gradUniqueOutGMT;
    gradUniqueOutGMT.SetGlobalBuffer((__gm__ uint8_t *)gradUniqueOutGM);

    LocalTensor<float> fp32Tile = pingBuf_->Get<float>();
    uint32_t tileLen = TILE_BYTES / sizeof(float);

    uint32_t off = start;
    while (off < end) {
        uint32_t thisLen = end - off;
        if (thisLen > tileLen) thisLen = tileLen;

        DataCopyExtParams inParams{1U, static_cast<uint32_t>(thisLen * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
        DataCopyPad(fp32Tile, gradUniqueFp32GMT[off], inParams, inPad);
        SyncFunc<HardEvent::MTE2_V>(*pipe_);

        if (outputDtype_ == ENGRAM_DT_BFLOAT16) {
            LocalTensor<bfloat16_t> outTile = pongBuf_->Get<bfloat16_t>();
            Cast(outTile, fp32Tile, AscendC::RoundMode::CAST_RINT, thisLen);
            SyncFunc<HardEvent::V_MTE3>(*pipe_);
            DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(bfloat16_t)), 0U, 0U};
            DataCopyPad(gradUniqueOutGMT[static_cast<uint64_t>(off) * sizeof(bfloat16_t)],
                        outTile.ReinterpretCast<uint8_t>(), outParams);
        } else {
            LocalTensor<half> outTile = pongBuf_->Get<half>();
            Cast(outTile, fp32Tile, AscendC::RoundMode::CAST_RINT, thisLen);
            SyncFunc<HardEvent::V_MTE3>(*pipe_);
            DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(half)), 0U, 0U};
            DataCopyPad(gradUniqueOutGMT[static_cast<uint64_t>(off) * sizeof(half)],
                        outTile.ReinterpretCast<uint8_t>(), outParams);
        }
        SyncFunc<HardEvent::MTE3_MTE2>(*pipe_);
        off += thisLen;
    }
    SyncAll<true>();
}

} // namespace EngramFetchGradUnique

#endif
