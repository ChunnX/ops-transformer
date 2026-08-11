/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_SORT_INT32_H
#define ENGRAM_FETCH_GRAD_SORT_INT32_H
/*
 * Minimal 1D LSD radix sort with index — positive int32 values.
 */

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace EngramFetchGradSortInt32Int32 {

using namespace AscendC;

enum : uint32_t {
    HISTOGRAM_BINS = 256,
    DEFAULT_TILE_SIZE = 4096,
    BITS_PER_BYTE = 8,
    BYTES_PER_INT32 = 4,
    VECTOR_LEN_INT32 = 64,
    VECTOR_LEN_INT16 = 128,
    VECTOR_LEN_INT8 = 256,
};

constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
__aicore__ inline uint32_t AlignUpU32(uint32_t bytes)
{
    return (bytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
}

template <HardEvent evt>
__aicore__ inline void WaitPipe(TPipe &pipe)
{
    event_t e = static_cast<event_t>(pipe.FetchEventID(evt));
    SetFlag<evt>(e);
    WaitFlag<evt>(e);
}

__aicore__ inline void ComputeTileHistogram(
    __local_mem__ uint32_t *valueUb, uint32_t elementCount, uint32_t byteRound,
    __local_mem__ uint16_t *histogramUb, __local_mem__ uint16_t *cumulativeUb,
    __local_mem__ uint8_t *keyUb, __local_mem__ uint32_t *exclusiveUb);

class EngramFetchGradSortInt32 {
public:
    __aicore__ inline EngramFetchGradSortInt32() = default;

    __aicore__ inline void Init(uint32_t totalElements, uint32_t numCores, uint32_t tileSize,
                                TPipe &pipe);

    __aicore__ inline void SetCount(uint32_t n);

    __aicore__ inline void SetMaxValue(uint32_t maxVal);

    __aicore__ inline void Process(
        GlobalTensor<int32_t> &valueGm, GlobalTensor<int32_t> &indexGm,
        GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
        GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
        GlobalTensor<int32_t> &histGm, GlobalTensor<int32_t> &prefixGm,
        GlobalTensor<int32_t> &tileOffsetsGm, TPipe &pipe);

    __aicore__ inline void Process(
        GM_ADDR valueGm, GM_ADDR indexGm,
        GM_ADDR tempValueGm, GM_ADDR tempIndexGm,
        GM_ADDR histGm, GM_ADDR prefixGm, GM_ADDR tileOffsetsGm, TPipe &pipe);

private:
    __aicore__ inline void ComputePrefixAndOffsets(
        uint32_t tc, GlobalTensor<int32_t> &prefixGm, GlobalTensor<int32_t> &tileOffsetsGm,
        GlobalTensor<int32_t> &histGm, TPipe &pipe);

    __aicore__ inline GlobalTensor<int32_t> *GetSrcValueGm(
        uint32_t byteRound, GlobalTensor<int32_t> &valueGm,
        GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &outputValueGm);

    __aicore__ inline void GetDstBuffers(
        uint32_t byteRound, GlobalTensor<int32_t> &indexGm,
        GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
        GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
        GlobalTensor<int32_t> *&srcIdx, GlobalTensor<int32_t> *&dstValue,
        GlobalTensor<int32_t> *&dstIndex);

    __aicore__ inline void ProcessHistStage(
        uint32_t byteRound, uint32_t batchCount,
        GlobalTensor<int32_t> *srcValueGm, GlobalTensor<int32_t> &histGm, TPipe &pipe);

    __aicore__ inline void ScatterOneBin(
        uint32_t binStart, uint32_t binCount, int32_t gmPos,
        LocalTensor<int32_t> &gatheredBuf, LocalTensor<int32_t> &valueLocal,
        LocalTensor<int32_t> &indexLocal, LocalTensor<uint32_t> &sortIndices,
        GlobalTensor<int32_t> *dstValue, GlobalTensor<int32_t> *dstIndex, TPipe &pipe);

    __aicore__ inline void ProcessScatterStage(
        uint32_t byteRound, uint32_t batchCount,
        GlobalTensor<int32_t> *srcValueGm, GlobalTensor<int32_t> &indexGm,
        GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
        GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
        GlobalTensor<int32_t> &histGm, GlobalTensor<int32_t> &prefixGm,
        GlobalTensor<int32_t> &tileOffsetsGm, TPipe &pipe);

    __aicore__ inline void CopyTempToOutput(
        GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
        GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
        TPipe &pipe);

    __aicore__ inline void ProcessOneByteRound(
        uint32_t byteRound, GlobalTensor<int32_t> &valueGm, GlobalTensor<int32_t> &indexGm,
        GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
        GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
        GlobalTensor<int32_t> &histGm, GlobalTensor<int32_t> &prefixGm,
        GlobalTensor<int32_t> &tileOffsetsGm, TPipe &pipe);

    static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

    TBuf<TPosition::VECCALC> valueBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> keyBuffer_;
    TBuf<TPosition::VECCALC> histogramBuffer_;
    TBuf<TPosition::VECCALC> cumulativeBuffer_;
    TBuf<TPosition::VECCALC> exclusiveBuffer_;
    TBuf<TPosition::VECCALC> prefixBuffer_;
    TBuf<TPosition::VECCALC> sortTempBuffer_;
    TBuf<TPosition::VECCALC> sortedKeyBuffer_;
    TBuf<TPosition::VECCALC> histWideBuffer_;
    TQue<QuePosition::VECCALC, 1> prefixQueue_;

    uint32_t elementCount_{0};
    uint32_t tileElements_{0};
    uint32_t tileCount_{0};
    uint32_t coreCount_{0};
    uint32_t byteRounds_{BYTES_PER_INT32};
};

// ==== implementations ====

__aicore__ inline void ComputeTileHistogram(
    __local_mem__ uint32_t *valueUb, uint32_t elementCount, uint32_t byteRound,
    __local_mem__ uint16_t *histogramUb, __local_mem__ uint16_t *cumulativeUb,
    __local_mem__ uint8_t *keyUb, __local_mem__ uint32_t *exclusiveUb)
{
    uint32_t shiftBits = byteRound * BITS_PER_BYTE;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint32_t> input0, input1, input2, input3;
        Reg::RegTensor<uint16_t> hist0, hist1, cumu0, cumu1;
        Reg::MaskReg mask32 = Reg::CreateMask<uint32_t>();
        Reg::MaskReg mask16 = Reg::CreateMask<uint16_t>();
        Reg::Duplicate(hist0, (uint16_t)0, mask16);
        Reg::Duplicate(hist1, (uint16_t)0, mask16);
        Reg::Duplicate(cumu0, (uint16_t)0, mask16);
        Reg::Duplicate(cumu1, (uint16_t)0, mask16);
        for (uint16_t i = 0; i < CeilDiv(elementCount, VECTOR_LEN_INT8); i++) {
            Reg::MaskReg elemMask = Reg::UpdateMask<uint8_t>(elementCount);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input0, valueUb, VECTOR_LEN_INT32);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input1, valueUb, VECTOR_LEN_INT32);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input2, valueUb, VECTOR_LEN_INT32);
            Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(input3, valueUb, VECTOR_LEN_INT32);
            Reg::RegTensor<uint32_t> s0, s1, s2, s3;
            Reg::ShiftRights<uint32_t, int16_t>(s0, input0, shiftBits, mask32);
            Reg::ShiftRights<uint32_t, int16_t>(s1, input1, shiftBits, mask32);
            Reg::ShiftRights<uint32_t, int16_t>(s2, input2, shiftBits, mask32);
            Reg::ShiftRights<uint32_t, int16_t>(s3, input3, shiftBits, mask32);
            Reg::RegTensor<uint16_t> d0, d1, d2, d3;
            Reg::DeInterleave(d0, d1, (Reg::RegTensor<uint16_t> &)s0, (Reg::RegTensor<uint16_t> &)s1);
            Reg::DeInterleave(d2, d3, (Reg::RegTensor<uint16_t> &)s2, (Reg::RegTensor<uint16_t> &)s3);
            Reg::RegTensor<uint8_t> b0, b1;
            Reg::DeInterleave(b0, b1, (Reg::RegTensor<uint8_t> &)d0, (Reg::RegTensor<uint8_t> &)d2);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(keyUb, b0, VECTOR_LEN_INT8, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0,
                            Reg::HistogramsType::FREQUENCY>(hist0, b0, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1,
                            Reg::HistogramsType::FREQUENCY>(hist1, b0, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0,
                            Reg::HistogramsType::ACCUMULATE>(cumu0, b0, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1,
                            Reg::HistogramsType::ACCUMULATE>(cumu1, b0, elemMask);
        }
        Reg::RegTensor<uint16_t> excl0, excl1, zero;
        Reg::Sub(excl0, cumu0, hist0, mask16);
        Reg::Sub(excl1, cumu1, hist1, mask16);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramUb, hist0, VECTOR_LEN_INT16, mask16);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(histogramUb, hist1, VECTOR_LEN_INT16, mask16);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(cumulativeUb, excl0, VECTOR_LEN_INT16, mask16);
        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(cumulativeUb, excl1, VECTOR_LEN_INT16, mask16);
        Reg::Duplicate(zero, (uint16_t)0, mask16);
        Reg::RegTensor<uint32_t> sum0, sum1, sum2, sum3, acc0, acc1, acc2, acc3;
        Reg::Interleave((Reg::RegTensor<uint16_t> &)sum0, (Reg::RegTensor<uint16_t> &)sum1, excl0, zero);
        Reg::Interleave((Reg::RegTensor<uint16_t> &)sum2, (Reg::RegTensor<uint16_t> &)sum3, excl1, zero);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(acc0, exclusiveUb, VECTOR_LEN_INT32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(acc1, exclusiveUb, VECTOR_LEN_INT32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(acc2, exclusiveUb, VECTOR_LEN_INT32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(acc3, exclusiveUb, VECTOR_LEN_INT32);
        Reg::Add(acc0, acc0, sum0, mask32);
        Reg::Add(acc1, acc1, sum1, mask32);
        Reg::Add(acc2, acc2, sum2, mask32);
        Reg::Add(acc3, acc3, sum3, mask32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc0, VECTOR_LEN_INT32, mask32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc1, VECTOR_LEN_INT32, mask32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc2, VECTOR_LEN_INT32, mask32);
        Reg::DataCopy<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc3, VECTOR_LEN_INT32, mask32);
    }
}

__aicore__ inline void EngramFetchGradSortInt32::Init(uint32_t totalElements, uint32_t numCores,
                                                      uint32_t tileSize, TPipe &pipe)
{
    elementCount_ = totalElements;
    coreCount_ = numCores;
    tileElements_ = tileSize > 0 ? tileSize : DEFAULT_TILE_SIZE;
    tileCount_ = CeilDiv(elementCount_, tileElements_);

    pipe.InitBuffer(valueBuffer_, static_cast<uint32_t>(AlignUpU32(tileElements_ * sizeof(int32_t))));
    pipe.InitBuffer(indexBuffer_, static_cast<uint32_t>(AlignUpU32(tileElements_ * sizeof(int32_t))));
    pipe.InitBuffer(keyBuffer_, tileElements_);
    pipe.InitBuffer(histogramBuffer_, HISTOGRAM_BINS * sizeof(int32_t));
    pipe.InitBuffer(cumulativeBuffer_, HISTOGRAM_BINS * sizeof(uint16_t));
    pipe.InitBuffer(exclusiveBuffer_, static_cast<uint32_t>(AlignUpU32(tileElements_ * sizeof(uint32_t))));
    pipe.InitBuffer(prefixBuffer_, HISTOGRAM_BINS * sizeof(int32_t));
    pipe.InitBuffer(sortTempBuffer_, static_cast<uint32_t>(AlignUpU32(tileElements_ * sizeof(int32_t))));
    pipe.InitBuffer(sortedKeyBuffer_, tileElements_);
    pipe.InitBuffer(histWideBuffer_, HISTOGRAM_BINS * sizeof(int32_t));
    pipe.InitBuffer(prefixQueue_, 1, HISTOGRAM_BINS * sizeof(int32_t));
}

__aicore__ inline void EngramFetchGradSortInt32::SetCount(uint32_t n)
{
    elementCount_ = n;
    uint32_t dynTile = CeilDiv(n, coreCount_);
    if (dynTile > 0 && dynTile < tileElements_) {
        tileElements_ = dynTile;
    }
    tileCount_ = CeilDiv(n, tileElements_);
}

__aicore__ inline void EngramFetchGradSortInt32::SetMaxValue(uint32_t maxVal)
{
    if (maxVal <= 0xFFU) {
        byteRounds_ = 1U;
    } else if (maxVal <= 0xFFFFU) {
        byteRounds_ = 2U;
    } else if (maxVal <= 0xFFFFFFU) {
        byteRounds_ = 3U;
    } else {
        byteRounds_ = BYTES_PER_INT32;
    }
}

__aicore__ inline void EngramFetchGradSortInt32::ComputePrefixAndOffsets(
    uint32_t tc, GlobalTensor<int32_t> &prefixGm, GlobalTensor<int32_t> &tileOffsetsGm,
    GlobalTensor<int32_t> &histGm, TPipe &pipe)
{
    LocalTensor<int32_t> histogram = prefixQueue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> prefixUb = histWideBuffer_.Get<int32_t>();
    DataCopyExtParams param{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> padParam{false, 0, 0, 0};
    DataCopyParams offsetParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};

    Duplicate(prefixUb, (int32_t)0, HISTOGRAM_BINS);

    for (uint32_t t = 0; t < tc; t++) {
        WaitPipe<HardEvent::V_MTE3>(pipe);
        DataCopyPad(tileOffsetsGm[t * HISTOGRAM_BINS], prefixUb, offsetParam);
        WaitPipe<HardEvent::MTE3_MTE2>(pipe);
        DataCopyPad(histogram, histGm[t * HISTOGRAM_BINS], param, padParam);
        WaitPipe<HardEvent::MTE2_V>(pipe);
        Add(prefixUb, prefixUb, histogram, HISTOGRAM_BINS);
    }

    WaitPipe<HardEvent::V_S>(pipe);
    int32_t running = 0;
    for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
        int32_t count = prefixUb.GetValue(b);
        prefixUb.SetValue(b, running);
        running += count;
    }

    WaitPipe<HardEvent::S_MTE3>(pipe);
    DataCopyPad(prefixGm, prefixUb, offsetParam);
    WaitPipe<HardEvent::MTE3_S>(pipe);
    prefixQueue_.FreeTensor(histogram);
}

__aicore__ inline GlobalTensor<int32_t> *EngramFetchGradSortInt32::GetSrcValueGm(
    uint32_t byteRound, GlobalTensor<int32_t> &valueGm,
    GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &outputValueGm)
{
    if (byteRound == 0) {
        return &valueGm;
    }
    if (byteRound % 2 == 1) {
        return &tempValueGm;
    }
    return &outputValueGm;
}

__aicore__ inline void EngramFetchGradSortInt32::GetDstBuffers(
    uint32_t byteRound, GlobalTensor<int32_t> &indexGm,
    GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
    GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
    GlobalTensor<int32_t> *&srcIdx, GlobalTensor<int32_t> *&dstValue,
    GlobalTensor<int32_t> *&dstIndex)
{
    if (byteRound == 0) {
        srcIdx = &indexGm;
        dstValue = &tempValueGm;
        dstIndex = &tempIndexGm;
    } else if (byteRound % 2 == 1) {
        srcIdx = &tempIndexGm;
        dstValue = &outputValueGm;
        dstIndex = &outputIndexGm;
    } else {
        srcIdx = &outputIndexGm;
        dstValue = &tempValueGm;
        dstIndex = &tempIndexGm;
    }
}

__aicore__ inline void EngramFetchGradSortInt32::ProcessHistStage(
    uint32_t byteRound, uint32_t batchCount,
    GlobalTensor<int32_t> *srcValueGm, GlobalTensor<int32_t> &histGm, TPipe &pipe)
{
    DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    event_t evtVS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
    event_t evtSMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));

    for (uint32_t batch = 0; batch < batchCount; batch++) {
        uint32_t firstTile = batch * coreCount_;
        uint32_t batchCores = MinU32(tileCount_ - firstTile, coreCount_);
        uint32_t coreId = GetBlockIdx();
        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();
            LocalTensor<uint16_t> histLocal = histogramBuffer_.Get<uint16_t>();
            LocalTensor<uint16_t> cumsumLocal = cumulativeBuffer_.Get<uint16_t>();
            LocalTensor<uint32_t> exclusiveLocal = exclusiveBuffer_.Get<uint32_t>();
            LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();

            uint32_t valBytes = tileLen * sizeof(int32_t);
            DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
            DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
            SetFlag<HardEvent::MTE2_V>(evtMte2V);
            WaitFlag<HardEvent::MTE2_V>(evtMte2V);

            Duplicate(exclusiveLocal, (uint32_t)0, HISTOGRAM_BINS);

            ComputeTileHistogram(
                (__local_mem__ uint32_t *)valueLocal.GetPhyAddr(), tileLen, byteRound,
                (__local_mem__ uint16_t *)histLocal.GetPhyAddr(),
                (__local_mem__ uint16_t *)cumsumLocal.GetPhyAddr(),
                (__local_mem__ uint8_t *)keyLocal.GetPhyAddr(),
                (__local_mem__ uint32_t *)exclusiveLocal.GetPhyAddr());

            SetFlag<HardEvent::V_S>(evtVS);
            WaitFlag<HardEvent::V_S>(evtVS);

            LocalTensor<int32_t> histWide = histWideBuffer_.Get<int32_t>();
            for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
                histWide.SetValue(b, static_cast<int32_t>(histLocal.GetValue(b)));
            }

            SetFlag<HardEvent::S_MTE3>(evtSMte3);
            WaitFlag<HardEvent::S_MTE3>(evtSMte3);

            DataCopyParams histOutParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};
            DataCopyPad(histGm[tileId * HISTOGRAM_BINS], histWide, histOutParam);
        }
        WaitPipe<HardEvent::MTE3_S>(pipe);
        SyncAll<true>();
    }
}

__aicore__ inline void EngramFetchGradSortInt32::ScatterOneBin(
    uint32_t binStart, uint32_t binCount, int32_t gmPos,
    LocalTensor<int32_t> &gatheredBuf, LocalTensor<int32_t> &valueLocal,
    LocalTensor<int32_t> &indexLocal, LocalTensor<uint32_t> &sortIndices,
    GlobalTensor<int32_t> *dstValue, GlobalTensor<int32_t> *dstIndex, TPipe &pipe)
{
    DataCopyParams dcp{1U, static_cast<uint16_t>(binCount * sizeof(int32_t)), 0U, 0U};

    for (uint32_t j = 0; j < binCount; j++) {
        gatheredBuf.SetValue(j, valueLocal.GetValue(sortIndices.GetValue(binStart + j)));
    }
    WaitPipe<HardEvent::S_MTE3>(pipe);
    DataCopyPad((*dstValue)[gmPos], gatheredBuf, dcp);
    WaitPipe<HardEvent::MTE3_S>(pipe);

    for (uint32_t j = 0; j < binCount; j++) {
        gatheredBuf.SetValue(j, indexLocal.GetValue(sortIndices.GetValue(binStart + j)));
    }
    WaitPipe<HardEvent::S_MTE3>(pipe);
    DataCopyPad((*dstIndex)[gmPos], gatheredBuf, dcp);
    WaitPipe<HardEvent::MTE3_S>(pipe);
}

__aicore__ inline void EngramFetchGradSortInt32::ProcessScatterStage(
    uint32_t byteRound, uint32_t batchCount,
    GlobalTensor<int32_t> *srcValueGm, GlobalTensor<int32_t> &indexGm,
    GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
    GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
    GlobalTensor<int32_t> &histGm, GlobalTensor<int32_t> &prefixGm,
    GlobalTensor<int32_t> &tileOffsetsGm, TPipe &pipe)
{
    DataCopyExtParams histParam{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2S = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
    event_t evtSV = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    event_t evtVS2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));

    for (uint32_t batch = 0; batch < batchCount; batch++) {
        uint32_t firstTile = batch * coreCount_;
        uint32_t batchCores = MinU32(tileCount_ - firstTile, coreCount_);
        uint32_t coreId = GetBlockIdx();
        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            GlobalTensor<int32_t> *srcIdx;
            GlobalTensor<int32_t> *dstValue;
            GlobalTensor<int32_t> *dstIndex;
            GetDstBuffers(byteRound, indexGm, outputValueGm, outputIndexGm,
                          tempValueGm, tempIndexGm, srcIdx, dstValue, dstIndex);

            LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();
            LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
            LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();
            LocalTensor<uint8_t> sortedKey = sortedKeyBuffer_.Get<uint8_t>();
            LocalTensor<uint32_t> sortIndices = exclusiveBuffer_.Get<uint32_t>();
            LocalTensor<int32_t> prefixLocal = prefixBuffer_.Get<int32_t>();
            LocalTensor<int32_t> offsetLocal = histogramBuffer_.Get<int32_t>();

            uint32_t valBytes = tileLen * sizeof(int32_t);
            DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
            DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
            DataCopyPad(indexLocal, (*srcIdx)[offset], valParam, valPad);
            DataCopyPad(prefixLocal, prefixGm, histParam, valPad);
            DataCopyPad(offsetLocal, tileOffsetsGm[tileId * HISTOGRAM_BINS], histParam, valPad);

            SetFlag<HardEvent::MTE2_S>(evtMte2S);
            WaitFlag<HardEvent::MTE2_S>(evtMte2S);

            for (uint32_t i = 0; i < tileLen; i++) {
                keyLocal.SetValue(i, (uint8_t)((valueLocal.GetValue(i) >> (byteRound * BITS_PER_BYTE)) & 0xFF));
            }

            SetFlag<HardEvent::S_V>(evtSV);
            WaitFlag<HardEvent::S_V>(evtSV);

            LocalTensor<uint8_t> sortTemp = sortTempBuffer_.Get<uint8_t>();
            Sort<uint8_t, false, sortConfig>(sortedKey, sortIndices, keyLocal, sortTemp, tileLen);

            SetFlag<HardEvent::V_S>(evtVS2);
            WaitFlag<HardEvent::V_S>(evtVS2);

            LocalTensor<int32_t> gatheredBuf = sortTempBuffer_.Get<int32_t>();

            uint32_t binStart = 0;
            uint8_t curBin = sortedKey.GetValue(0);
            for (uint32_t i = 1; i <= tileLen; i++) {
                if (i == tileLen || sortedKey.GetValue(i) != curBin) {
                    uint32_t binCount = i - binStart;
                    int32_t gmPos = prefixLocal.GetValue(curBin) + offsetLocal.GetValue(curBin);
                    ScatterOneBin(binStart, binCount, gmPos, gatheredBuf, valueLocal,
                                  indexLocal, sortIndices, dstValue, dstIndex, pipe);
                    if (i < tileLen) {
                        curBin = sortedKey.GetValue(i);
                        binStart = i;
                    }
                }
            }
        }
        WaitPipe<HardEvent::S_MTE3>(pipe);
        SyncAll<true>();
    }
}

__aicore__ inline void EngramFetchGradSortInt32::CopyTempToOutput(
    GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
    GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
    TPipe &pipe)
{
    uint32_t chunk = CeilDiv(elementCount_, coreCount_);
    uint32_t start = GetBlockIdx() * chunk;
    uint32_t end = start + chunk;
    if (end > elementCount_) {
        end = elementCount_;
    }
    if (start >= end) {
        SyncAll<true>();
        return;
    }

    DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    LocalTensor<int32_t> tmpBuf = valueBuffer_.Get<int32_t>();
    uint32_t remaining = end - start;
    uint32_t off = 0;
    while (off < remaining) {
        uint32_t thisLen = remaining - off;
        if (thisLen > tileElements_) {
            thisLen = tileElements_;
        }
        DataCopyExtParams thisParams{1U, static_cast<uint32_t>(thisLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(tmpBuf, tempValueGm[start + off], thisParams, cpPad);
        WaitPipe<HardEvent::MTE2_S>(pipe);
        DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(outputValueGm[start + off], tmpBuf, outParams);
        WaitPipe<HardEvent::MTE3_S>(pipe);

        DataCopyPad(tmpBuf, tempIndexGm[start + off], thisParams, cpPad);
        WaitPipe<HardEvent::MTE2_S>(pipe);
        DataCopyPad(outputIndexGm[start + off], tmpBuf, outParams);
        WaitPipe<HardEvent::MTE3_S>(pipe);
        off += thisLen;
    }
    SyncAll<true>();
}

__aicore__ inline void EngramFetchGradSortInt32::ProcessOneByteRound(
    uint32_t byteRound, GlobalTensor<int32_t> &valueGm, GlobalTensor<int32_t> &indexGm,
    GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
    GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
    GlobalTensor<int32_t> &histGm, GlobalTensor<int32_t> &prefixGm,
    GlobalTensor<int32_t> &tileOffsetsGm, TPipe &pipe)
{
    uint32_t batchCount = CeilDiv(tileCount_, coreCount_);
    GlobalTensor<int32_t> *srcValueGm = GetSrcValueGm(byteRound, valueGm, tempValueGm, outputValueGm);

    ProcessHistStage(byteRound, batchCount, srcValueGm, histGm, pipe);

    if (GetBlockIdx() == 0) {
        ComputePrefixAndOffsets(tileCount_, prefixGm, tileOffsetsGm, histGm, pipe);
    }
    SyncAll<true>();

    ProcessScatterStage(byteRound, batchCount, srcValueGm, indexGm, outputValueGm,
                        outputIndexGm, tempValueGm, tempIndexGm, histGm, prefixGm,
                        tileOffsetsGm, pipe);
}

__aicore__ inline void EngramFetchGradSortInt32::Process(
    GlobalTensor<int32_t> &valueGm, GlobalTensor<int32_t> &indexGm,
    GlobalTensor<int32_t> &outputValueGm, GlobalTensor<int32_t> &outputIndexGm,
    GlobalTensor<int32_t> &tempValueGm, GlobalTensor<int32_t> &tempIndexGm,
    GlobalTensor<int32_t> &histGm, GlobalTensor<int32_t> &prefixGm,
    GlobalTensor<int32_t> &tileOffsetsGm, TPipe &pipe)
{
    for (uint32_t byteRound = 0; byteRound < byteRounds_; byteRound++) {
        ProcessOneByteRound(byteRound, valueGm, indexGm, outputValueGm, outputIndexGm,
                            tempValueGm, tempIndexGm, histGm, prefixGm, tileOffsetsGm, pipe);
    }

    if (byteRounds_ % 2 == 1) {
        CopyTempToOutput(tempValueGm, tempIndexGm, outputValueGm, outputIndexGm, pipe);
    }
}

__aicore__ inline void EngramFetchGradSortInt32::Process(
    GM_ADDR valueGm, GM_ADDR indexGm,
    GM_ADDR tempValueGm, GM_ADDR tempIndexGm,
    GM_ADDR histGm, GM_ADDR prefixGm, GM_ADDR tileOffsetsGm, TPipe &pipe)
{
    GlobalTensor<int32_t> valT, idxT, outValT, outIdxT, tmpValT, tmpIdxT, histT, prefixT, offsetsT;
    valT.SetGlobalBuffer((__gm__ int32_t *)valueGm);
    idxT.SetGlobalBuffer((__gm__ int32_t *)indexGm);
    outValT.SetGlobalBuffer((__gm__ int32_t *)valueGm);
    outIdxT.SetGlobalBuffer((__gm__ int32_t *)indexGm);
    tmpValT.SetGlobalBuffer((__gm__ int32_t *)tempValueGm);
    tmpIdxT.SetGlobalBuffer((__gm__ int32_t *)tempIndexGm);
    histT.SetGlobalBuffer((__gm__ int32_t *)histGm);
    prefixT.SetGlobalBuffer((__gm__ int32_t *)prefixGm);
    offsetsT.SetGlobalBuffer((__gm__ int32_t *)tileOffsetsGm);
    Process(valT, idxT, outValT, outIdxT, tmpValT, tmpIdxT, histT, prefixT, offsetsT, pipe);
}

} // namespace EngramFetchGradSortInt32Int32
#endif
