/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_SORT_H
#define ENGRAM_FETCH_GRAD_SORT_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace EngramFetchGradSort {

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

constexpr static AscendC::MicroAPI::CastTrait castTraitU162U32Even = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitU162U32Odd = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
__aicore__ inline uint32_t AlignUpU32(uint32_t bytes)
{
    return (bytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
}

template <AscendC::HardEvent evt>
__aicore__ inline void WaitPipe(AscendC::TPipe &pipe)
{
    event_t e = static_cast<event_t>(pipe.FetchEventID(evt));
    AscendC::SetFlag<evt>(e);
    AscendC::WaitFlag<evt>(e);
}

__aicore__ inline void ComputeTileHistogram(
    __local_mem__ uint32_t *valueUb, uint32_t elementCount, uint32_t shiftBits,
    __local_mem__ uint16_t *histogramUb, __local_mem__ uint16_t *cumulativeUb,
    __local_mem__ uint8_t *keyUb, __local_mem__ uint32_t *exclusiveUb,
    __local_mem__ uint32_t *histWideUb);

class EngramFetchGradSort {
public:
    __aicore__ inline EngramFetchGradSort() = default;

    __aicore__ inline static uint64_t GetWorkspaceSize(uint32_t totalElements, uint32_t numCores);

    __aicore__ inline void Init(uint32_t totalElements, uint32_t numCores,
                                GM_ADDR valueGm, GM_ADDR indexGm, GM_ADDR workspaceGm, AscendC::TPipe &pipe);

    __aicore__ inline void SetMaxValue(uint32_t maxVal);

    __aicore__ inline void SetValueOffset(int32_t offset);

    __aicore__ inline void Process(uint32_t actualCount, AscendC::TPipe &pipe);

private:
    __aicore__ inline void ApplyValueOffset(AscendC::TPipe &pipe);

    __aicore__ inline void RestoreValueOffset(AscendC::TPipe &pipe);

    __aicore__ inline void ComputePrefixAndOffsets(
        uint32_t tc, AscendC::GlobalTensor<int32_t> &prefixGm, AscendC::GlobalTensor<int32_t> &tileOffsetsGm,
        AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe);

    __aicore__ inline AscendC::GlobalTensor<int32_t> *GetSrcValueGm(
        uint32_t byteRound, AscendC::GlobalTensor<int32_t> &valueGm,
        AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &outputValueGm);

    __aicore__ inline void GetDstBuffers(
        uint32_t byteRound, AscendC::GlobalTensor<int32_t> &indexGm,
        AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
        AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
        AscendC::GlobalTensor<int32_t> *&srcIdx, AscendC::GlobalTensor<int32_t> *&dstValue,
        AscendC::GlobalTensor<int32_t> *&dstIndex);

    __aicore__ inline void ProcessHist(
        uint32_t shiftBits, uint32_t batchCount,
        AscendC::GlobalTensor<int32_t> *srcValueGm, AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe);

    __aicore__ inline void ScatterOneKey(
        uint32_t keyStart, uint32_t keyCount, int32_t gmPos,
        AscendC::LocalTensor<int32_t> &gatheredBuf, AscendC::LocalTensor<int32_t> &valueLocal,
        AscendC::LocalTensor<int32_t> &indexLocal, AscendC::LocalTensor<uint32_t> &sortIndices,
        AscendC::GlobalTensor<int32_t> *dstValue, AscendC::GlobalTensor<int32_t> *dstIndex, AscendC::TPipe &pipe);

    __aicore__ inline void ProcessScatter(
        uint32_t execRound, uint32_t shiftBits, uint32_t batchCount,
        AscendC::GlobalTensor<int32_t> *srcValueGm, AscendC::GlobalTensor<int32_t> &indexGm,
        AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
        AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
        AscendC::GlobalTensor<int32_t> &histGm, AscendC::GlobalTensor<int32_t> &prefixGm,
        AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe);

    __aicore__ inline void CopyTempToOutput(
        AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
        AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
        AscendC::TPipe &pipe);

    __aicore__ inline void ProcessOneByteRound(
        uint32_t execRound, uint32_t shiftByte,
        AscendC::GlobalTensor<int32_t> &valueGm, AscendC::GlobalTensor<int32_t> &indexGm,
        AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
        AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
        AscendC::GlobalTensor<int32_t> &histGm, AscendC::GlobalTensor<int32_t> &prefixGm,
        AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe);

    __aicore__ inline void ComputeCoreRange(
        AscendC::GlobalTensor<int32_t> &prefixGm,
        uint32_t &myBucketStart, uint32_t &myBucketEnd,
        AscendC::TPipe &pipe);

    __aicore__ inline void ProcessLocalLsdRound(
        uint32_t execRound, uint32_t shiftByte, uint32_t coreStart, uint32_t coreEnd,
        AscendC::TPipe &pipe);

    static constexpr AscendC::SortConfig sortConfig{AscendC::SortType::RADIX_SORT, false};

    AscendC::TBuf<AscendC::TPosition::VECCALC> valueBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> keyBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> histogramBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumulativeBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> exclusiveBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortTempBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortedKeyBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> histWideBuffer_;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> prefixQueue_;

    uint32_t elementCount_{0};
    uint32_t tileElements_{0};
    uint32_t tileCount_{0};
    uint32_t coreCount_{0};
    uint32_t byteRounds_{BYTES_PER_INT32};
    int32_t valueOffset_{0};

    AscendC::GlobalTensor<int32_t> valueGm_;
    AscendC::GlobalTensor<int32_t> indexGm_;
    AscendC::GlobalTensor<int32_t> tempValueGm_;
    AscendC::GlobalTensor<int32_t> tempIndexGm_;
    AscendC::GlobalTensor<int32_t> histGm_;
    AscendC::GlobalTensor<int32_t> prefixGm_;
    AscendC::GlobalTensor<int32_t> tileOffsetsGm_;
};

// ==== implementations ====

__aicore__ inline void ComputeTileHistogram(
    __local_mem__ uint32_t *valueUb, uint32_t elementCount, uint32_t shiftBits,
    __local_mem__ uint16_t *histogramUb, __local_mem__ uint16_t *cumulativeUb,
    __local_mem__ uint8_t *keyUb, __local_mem__ uint32_t *exclusiveUb,
    __local_mem__ uint32_t *histWideUb)
{
    __VEC_SCOPE__
    {
        using namespace AscendC::Reg;
        RegTensor<uint32_t> input0, input1, input2, input3;
        RegTensor<uint16_t> hist0, hist1, cumu0, cumu1;
        MaskReg mask32 = CreateMask<uint32_t>();
        MaskReg mask16 = CreateMask<uint16_t>();
        Duplicate(hist0, (uint16_t)0, mask16);
        Duplicate(hist1, (uint16_t)0, mask16);
        Duplicate(cumu0, (uint16_t)0, mask16);
        Duplicate(cumu1, (uint16_t)0, mask16);
        for (uint16_t i = 0; i < CeilDiv(elementCount, VECTOR_LEN_INT8); i++) {
            MaskReg elemMask = UpdateMask<uint8_t>(elementCount);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input0, valueUb, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input1, valueUb, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input2, valueUb, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input3, valueUb, VECTOR_LEN_INT32);
            RegTensor<uint32_t> s0, s1, s2, s3;
            ShiftRights<uint32_t, int16_t>(s0, input0, shiftBits, mask32);
            ShiftRights<uint32_t, int16_t>(s1, input1, shiftBits, mask32);
            ShiftRights<uint32_t, int16_t>(s2, input2, shiftBits, mask32);
            ShiftRights<uint32_t, int16_t>(s3, input3, shiftBits, mask32);
            RegTensor<uint16_t> d0, d1, d2, d3;
            DeInterleave(d0, d1, (RegTensor<uint16_t> &)s0, (RegTensor<uint16_t> &)s1);
            DeInterleave(d2, d3, (RegTensor<uint16_t> &)s2, (RegTensor<uint16_t> &)s3);
            RegTensor<uint8_t> b0, b1;
            DeInterleave(b0, b1, (RegTensor<uint8_t> &)d0, (RegTensor<uint8_t> &)d2);
            DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(keyUb, b0, VECTOR_LEN_INT8, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0,
                       HistogramsType::FREQUENCY>(hist0, b0, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1,
                       HistogramsType::FREQUENCY>(hist1, b0, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0,
                       HistogramsType::ACCUMULATE>(cumu0, b0, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1,
                       HistogramsType::ACCUMULATE>(cumu1, b0, elemMask);
        }
        RegTensor<uint16_t> excl0, excl1, zero;
        Sub(excl0, cumu0, hist0, mask16);
        Sub(excl1, cumu1, hist1, mask16);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(histogramUb, hist0, VECTOR_LEN_INT16, mask16);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(histogramUb, hist1, VECTOR_LEN_INT16, mask16);
        RegTensor<uint32_t> hw0Even, hw0Odd, hw1Even, hw1Odd, hw0, hw1;
        Cast<uint32_t, uint16_t, castTraitU162U32Even>(hw0Even, hist0, mask16);
        Cast<uint32_t, uint16_t, castTraitU162U32Odd>(hw0Odd, hist0, mask16);
        Interleave<uint32_t>(hw0, hw1, hw0Even, hw0Odd);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(histWideUb, hw0, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(histWideUb, hw1, VECTOR_LEN_INT32, mask32);
        Cast<uint32_t, uint16_t, castTraitU162U32Even>(hw1Even, hist1, mask16);
        Cast<uint32_t, uint16_t, castTraitU162U32Odd>(hw1Odd, hist1, mask16);
        Interleave<uint32_t>(hw0, hw1, hw1Even, hw1Odd);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(histWideUb, hw0, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(histWideUb, hw1, VECTOR_LEN_INT32, mask32);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(cumulativeUb, excl0, VECTOR_LEN_INT16, mask16);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE>(cumulativeUb, excl1, VECTOR_LEN_INT16, mask16);
        Duplicate(zero, (uint16_t)0, mask16);
        RegTensor<uint32_t> sum0, sum1, sum2, sum3, acc0, acc1, acc2, acc3;
        Interleave((RegTensor<uint16_t> &)sum0, (RegTensor<uint16_t> &)sum1, excl0, zero);
        Interleave((RegTensor<uint16_t> &)sum2, (RegTensor<uint16_t> &)sum3, excl1, zero);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(acc0, exclusiveUb, VECTOR_LEN_INT32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(acc1, exclusiveUb, VECTOR_LEN_INT32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(acc2, exclusiveUb, VECTOR_LEN_INT32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(acc3, exclusiveUb, VECTOR_LEN_INT32);
        Add(acc0, acc0, sum0, mask32);
        Add(acc1, acc1, sum1, mask32);
        Add(acc2, acc2, sum2, mask32);
        Add(acc3, acc3, sum3, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc0, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc1, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc2, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(exclusiveUb, acc3, VECTOR_LEN_INT32, mask32);
    }
}

__aicore__ inline uint64_t EngramFetchGradSort::GetWorkspaceSize(uint32_t totalElements,
                                                                 uint32_t numCores)
{
    uint32_t tileElements = DEFAULT_TILE_SIZE;
    uint32_t dynTile = CeilDiv(totalElements, numCores);
    if (dynTile > 0 && dynTile < tileElements) {
        tileElements = dynTile;
    }
    uint32_t tileCount = CeilDiv(totalElements, tileElements);
    uint32_t sortTileCountByCore = numCores;
    uint32_t sortTileCountByDefault = CeilDiv(totalElements, DEFAULT_TILE_SIZE);
    uint32_t sortTileCount = (sortTileCountByCore > sortTileCountByDefault) ?
                                 sortTileCountByCore :
                                 sortTileCountByDefault;
    uint64_t sortTempBytes = static_cast<uint64_t>(totalElements) * sizeof(int32_t);
    uint64_t sortTempSize = (sortTempBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
    return 2 * sortTempSize +
           2 * static_cast<uint64_t>(sortTileCount) * HISTOGRAM_BINS * sizeof(int32_t) +
           HISTOGRAM_BINS * sizeof(int32_t);
}

__aicore__ inline void EngramFetchGradSort::Init(uint32_t totalElements, uint32_t numCores,
                                                 GM_ADDR valueGm, GM_ADDR indexGm,
                                                 GM_ADDR workspaceGm, AscendC::TPipe &pipe)
{
    elementCount_ = totalElements;
    coreCount_ = numCores;
    tileElements_ = DEFAULT_TILE_SIZE;
    tileCount_ = CeilDiv(elementCount_, tileElements_);

    uint32_t sortTileCountByCore = numCores;
    uint32_t sortTileCountByDefault = CeilDiv(elementCount_, DEFAULT_TILE_SIZE);
    uint32_t sortTileCount = (sortTileCountByCore > sortTileCountByDefault) ?
                                 sortTileCountByCore :
                                 sortTileCountByDefault;
    uint64_t sortTempBytes = static_cast<uint64_t>(elementCount_) * sizeof(int32_t);
    uint64_t sortTempSize = (sortTempBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
    uint64_t histSize = static_cast<uint64_t>(sortTileCount) * HISTOGRAM_BINS * sizeof(int32_t);
    uint64_t offset = 0;
    valueGm_.SetGlobalBuffer((__gm__ int32_t *)valueGm);
    indexGm_.SetGlobalBuffer((__gm__ int32_t *)indexGm);
    tempValueGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));
    offset += sortTempSize;
    tempIndexGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));
    offset += sortTempSize;
    histGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));
    offset += histSize;
    prefixGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));
    offset += HISTOGRAM_BINS * sizeof(int32_t);
    tileOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));

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

__aicore__ inline void EngramFetchGradSort::SetMaxValue(uint32_t maxVal)
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

__aicore__ inline void EngramFetchGradSort::SetValueOffset(int32_t offset)
{
    valueOffset_ = offset;
}

__aicore__ inline void EngramFetchGradSort::ApplyValueOffset(AscendC::TPipe &pipe)
{
    if (valueOffset_ == 0) {
        return;
    }

    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));

    uint32_t chunk = CeilDiv(elementCount_, coreCount_);
    uint32_t start = AscendC::GetBlockIdx() * chunk;
    uint32_t end = MinU32(start + chunk, elementCount_);

    for (uint32_t off = start; off < end; off += tileElements_) {
        uint32_t tileLen = MinU32(end - off, tileElements_);
        AscendC::LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();

        uint32_t valBytes = tileLen * sizeof(int32_t);
        AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(valueLocal, valueGm_[off], valParam, valPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);

        AscendC::Adds<int32_t>(valueLocal, valueLocal, static_cast<int32_t>(-valueOffset_),
                               static_cast<int32_t>(tileLen));

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);

        AscendC::DataCopyParams outParams{1U, static_cast<uint16_t>(valBytes), 0U, 0U};
        AscendC::DataCopyPad(valueGm_[off], valueLocal, outParams);
        WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
    }

    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchGradSort::RestoreValueOffset(AscendC::TPipe &pipe)
{
    if (valueOffset_ == 0) {
        return;
    }

    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));

    uint32_t chunk = CeilDiv(elementCount_, coreCount_);
    uint32_t start = AscendC::GetBlockIdx() * chunk;
    uint32_t end = MinU32(start + chunk, elementCount_);

    for (uint32_t off = start; off < end; off += tileElements_) {
        uint32_t tileLen = MinU32(end - off, tileElements_);
        AscendC::LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();

        uint32_t valBytes = tileLen * sizeof(int32_t);
        AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(valueLocal, valueGm_[off], valParam, valPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);

        AscendC::Adds<int32_t>(valueLocal, valueLocal, valueOffset_, static_cast<int32_t>(tileLen));

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);

        AscendC::DataCopyParams outParams{1U, static_cast<uint16_t>(valBytes), 0U, 0U};
        AscendC::DataCopyPad(valueGm_[off], valueLocal, outParams);
        WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
    }

    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchGradSort::ComputePrefixAndOffsets(
    uint32_t tc, AscendC::GlobalTensor<int32_t> &prefixGm, AscendC::GlobalTensor<int32_t> &tileOffsetsGm,
    AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe)
{
    AscendC::LocalTensor<int32_t> histogram = prefixQueue_.AllocTensor<int32_t>();
    AscendC::LocalTensor<int32_t> prefixUb = histWideBuffer_.Get<int32_t>();
    AscendC::DataCopyExtParams param{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> padParam{false, 0, 0, 0};
    AscendC::DataCopyParams offsetParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};

    AscendC::Duplicate(prefixUb, (int32_t)0, HISTOGRAM_BINS);

    for (uint32_t t = 0; t < tc; t++) {
        WaitPipe<AscendC::HardEvent::V_MTE3>(pipe);
        AscendC::DataCopyPad(tileOffsetsGm[t * HISTOGRAM_BINS], prefixUb, offsetParam);
        WaitPipe<AscendC::HardEvent::MTE3_MTE2>(pipe);
        AscendC::DataCopyPad(histogram, histGm[t * HISTOGRAM_BINS], param, padParam);
        WaitPipe<AscendC::HardEvent::MTE2_V>(pipe);
        AscendC::Add(prefixUb, prefixUb, histogram, HISTOGRAM_BINS);
    }

    WaitPipe<AscendC::HardEvent::V_S>(pipe);
    int32_t running = 0;
    for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
        int32_t count = prefixUb.GetValue(b);
        prefixUb.SetValue(b, running);
        running += count;
    }

    WaitPipe<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::DataCopyPad(prefixGm, prefixUb, offsetParam);
    WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
    prefixQueue_.FreeTensor(histogram);
}

__aicore__ inline AscendC::GlobalTensor<int32_t> *EngramFetchGradSort::GetSrcValueGm(
    uint32_t byteRound, AscendC::GlobalTensor<int32_t> &valueGm,
    AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &outputValueGm)
{
    if (byteRound == 0) {
        return &valueGm;
    }
    if (byteRound % 2 == 1) {
        return &tempValueGm;
    }
    return &outputValueGm;
}

__aicore__ inline void EngramFetchGradSort::GetDstBuffers(
    uint32_t byteRound, AscendC::GlobalTensor<int32_t> &indexGm,
    AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
    AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
    AscendC::GlobalTensor<int32_t> *&srcIdx, AscendC::GlobalTensor<int32_t> *&dstValue,
    AscendC::GlobalTensor<int32_t> *&dstIndex)
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

__aicore__ inline void EngramFetchGradSort::ProcessHist(
    uint32_t shiftBits, uint32_t batchCount,
    AscendC::GlobalTensor<int32_t> *srcValueGm, AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe)
{
    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));

    for (uint32_t batch = 0; batch < batchCount; batch++) {
        uint32_t firstTile = batch * coreCount_;
        uint32_t batchCores = MinU32(tileCount_ - firstTile, coreCount_);
        uint32_t coreId = AscendC::GetBlockIdx();
        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            AscendC::LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();
            AscendC::LocalTensor<uint16_t> histLocal = histogramBuffer_.Get<uint16_t>();
            AscendC::LocalTensor<uint16_t> cumsumLocal = cumulativeBuffer_.Get<uint16_t>();
            AscendC::LocalTensor<uint32_t> exclusiveLocal = exclusiveBuffer_.Get<uint32_t>();
            AscendC::LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();

            uint32_t valBytes = tileLen * sizeof(int32_t);
            AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);

            AscendC::Duplicate(exclusiveLocal, (uint32_t)0, HISTOGRAM_BINS);

            AscendC::LocalTensor<int32_t> histWide = histWideBuffer_.Get<int32_t>();

            ComputeTileHistogram(
                (__local_mem__ uint32_t *)valueLocal.GetPhyAddr(), tileLen, shiftBits,
                (__local_mem__ uint16_t *)histLocal.GetPhyAddr(),
                (__local_mem__ uint16_t *)cumsumLocal.GetPhyAddr(),
                (__local_mem__ uint8_t *)keyLocal.GetPhyAddr(),
                (__local_mem__ uint32_t *)exclusiveLocal.GetPhyAddr(),
                (__local_mem__ uint32_t *)histWide.GetPhyAddr());

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);

            AscendC::DataCopyParams histOutParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};
            AscendC::DataCopyPad(histGm[tileId * HISTOGRAM_BINS], histWide, histOutParam);
        }
        WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
        AscendC::SyncAll<true>();
    }
}

__aicore__ inline void EngramFetchGradSort::ScatterOneKey(
    uint32_t keyStart, uint32_t keyCount, int32_t gmPos,
    AscendC::LocalTensor<int32_t> &gatheredBuf, AscendC::LocalTensor<int32_t> &valueLocal,
    AscendC::LocalTensor<int32_t> &indexLocal, AscendC::LocalTensor<uint32_t> &sortIndices,
    AscendC::GlobalTensor<int32_t> *dstValue, AscendC::GlobalTensor<int32_t> *dstIndex, AscendC::TPipe &pipe)
{
    AscendC::DataCopyParams dcp{1U, static_cast<uint16_t>(keyCount * sizeof(int32_t)), 0U, 0U};

    for (uint32_t j = 0; j < keyCount; j++) {
        gatheredBuf.SetValue(j, valueLocal.GetValue(sortIndices.GetValue(keyStart + j)));
    }
    WaitPipe<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::DataCopyPad((*dstValue)[gmPos], gatheredBuf, dcp);
    WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);

    for (uint32_t j = 0; j < keyCount; j++) {
        gatheredBuf.SetValue(j, indexLocal.GetValue(sortIndices.GetValue(keyStart + j)));
    }
    WaitPipe<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::DataCopyPad((*dstIndex)[gmPos], gatheredBuf, dcp);
    WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
}

__aicore__ inline void EngramFetchGradSort::ProcessScatter(
    uint32_t execRound, uint32_t shiftBits, uint32_t batchCount,
    AscendC::GlobalTensor<int32_t> *srcValueGm, AscendC::GlobalTensor<int32_t> &indexGm,
    AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
    AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
    AscendC::GlobalTensor<int32_t> &histGm, AscendC::GlobalTensor<int32_t> &prefixGm,
    AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe)
{
    AscendC::DataCopyExtParams histParam{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtMte2S = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_S));
    event_t evtVS2 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_S));

    for (uint32_t batch = 0; batch < batchCount; batch++) {
        uint32_t firstTile = batch * coreCount_;
        uint32_t batchCores = MinU32(tileCount_ - firstTile, coreCount_);
        uint32_t coreId = AscendC::GetBlockIdx();
        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            AscendC::GlobalTensor<int32_t> *srcIdx;
            AscendC::GlobalTensor<int32_t> *dstValue;
            AscendC::GlobalTensor<int32_t> *dstIndex;
            GetDstBuffers(execRound, indexGm, outputValueGm, outputIndexGm,
                          tempValueGm, tempIndexGm, srcIdx, dstValue, dstIndex);

            AscendC::LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();
            AscendC::LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
            AscendC::LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();
            AscendC::LocalTensor<uint8_t> sortedKey = sortedKeyBuffer_.Get<uint8_t>();
            AscendC::LocalTensor<uint32_t> sortIndices = exclusiveBuffer_.Get<uint32_t>();
            AscendC::LocalTensor<int32_t> prefixLocal = prefixBuffer_.Get<int32_t>();
            AscendC::LocalTensor<int32_t> offsetLocal = histogramBuffer_.Get<int32_t>();

            uint32_t valBytes = tileLen * sizeof(int32_t);
            AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
            AscendC::DataCopyPad(indexLocal, (*srcIdx)[offset], valParam, valPad);
            AscendC::DataCopyPad(prefixLocal, prefixGm, histParam, valPad);
            AscendC::DataCopyPad(offsetLocal, tileOffsetsGm[tileId * HISTOGRAM_BINS], histParam, valPad);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);

            uint32_t shiftBitsLocal = shiftBits;
            __VEC_SCOPE__
            {
                using namespace AscendC::Reg;
                __local_mem__ uint32_t *valuePtr = (__local_mem__ uint32_t *)valueLocal.GetPhyAddr();
                __local_mem__ uint8_t *keyPtr = (__local_mem__ uint8_t *)keyLocal.GetPhyAddr();
                MaskReg mask32 = CreateMask<uint32_t>();
                for (uint16_t i = 0; i < CeilDiv(tileLen, VECTOR_LEN_INT8); i++) {
                    MaskReg elemMask = UpdateMask<uint8_t>(tileLen);
                    RegTensor<uint32_t> input0, input1, input2, input3;
                    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input0, valuePtr, VECTOR_LEN_INT32);
                    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input1, valuePtr, VECTOR_LEN_INT32);
                    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input2, valuePtr, VECTOR_LEN_INT32);
                    DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input3, valuePtr, VECTOR_LEN_INT32);
                    RegTensor<uint32_t> s0, s1, s2, s3;
                    ShiftRights<uint32_t, int16_t>(s0, input0, shiftBitsLocal, mask32);
                    ShiftRights<uint32_t, int16_t>(s1, input1, shiftBitsLocal, mask32);
                    ShiftRights<uint32_t, int16_t>(s2, input2, shiftBitsLocal, mask32);
                    ShiftRights<uint32_t, int16_t>(s3, input3, shiftBitsLocal, mask32);
                    RegTensor<uint16_t> d0, d1, d2, d3;
                    DeInterleave(d0, d1, (RegTensor<uint16_t> &)s0, (RegTensor<uint16_t> &)s1);
                    DeInterleave(d2, d3, (RegTensor<uint16_t> &)s2, (RegTensor<uint16_t> &)s3);
                    RegTensor<uint8_t> b0, b1;
                    DeInterleave(b0, b1, (RegTensor<uint8_t> &)d0, (RegTensor<uint8_t> &)d2);
                    DataCopy<uint8_t, PostLiteral::POST_MODE_UPDATE>(keyPtr, b0, VECTOR_LEN_INT8, elemMask);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<uint8_t> sortTemp = sortTempBuffer_.Get<uint8_t>();
            AscendC::Sort<uint8_t, false, sortConfig>(sortedKey, sortIndices, keyLocal, sortTemp, tileLen);

            AscendC::SetFlag<AscendC::HardEvent::V_S>(evtVS2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtVS2);

            AscendC::LocalTensor<int32_t> gatheredBuf = sortTempBuffer_.Get<int32_t>();

            uint32_t keyStart = 0;
            uint8_t curKey = sortedKey.GetValue(0);
            for (uint32_t i = 1; i <= tileLen; i++) {
                if (i == tileLen || sortedKey.GetValue(i) != curKey) {
                    uint32_t keyCount = i - keyStart;
                    int32_t gmPos = prefixLocal.GetValue(curKey) + offsetLocal.GetValue(curKey);
                    ScatterOneKey(keyStart, keyCount, gmPos, gatheredBuf, valueLocal,
                                  indexLocal, sortIndices, dstValue, dstIndex, pipe);
                    if (i < tileLen) {
                        curKey = sortedKey.GetValue(i);
                        keyStart = i;
                    }
                }
            }
        }
        WaitPipe<AscendC::HardEvent::S_MTE3>(pipe);
        AscendC::SyncAll<true>();
    }
}

__aicore__ inline void EngramFetchGradSort::CopyTempToOutput(
    AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
    AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
    AscendC::TPipe &pipe)
{
    uint32_t chunk = CeilDiv(elementCount_, coreCount_);
    uint32_t start = AscendC::GetBlockIdx() * chunk;
    uint32_t end = start + chunk;
    if (end > elementCount_) {
        end = elementCount_;
    }
    if (start >= end) {
        AscendC::SyncAll<true>();
        return;
    }

    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::LocalTensor<int32_t> tmpBuf = valueBuffer_.Get<int32_t>();
    uint32_t remaining = end - start;
    uint32_t off = 0;
    while (off < remaining) {
        uint32_t thisLen = remaining - off;
        if (thisLen > tileElements_) {
            thisLen = tileElements_;
        }
        AscendC::DataCopyExtParams thisParams{1U, static_cast<uint32_t>(thisLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(tmpBuf, tempValueGm[start + off], thisParams, cpPad);
        WaitPipe<AscendC::HardEvent::MTE2_S>(pipe);
        AscendC::DataCopyParams outParams{1U, static_cast<uint16_t>(thisLen * sizeof(int32_t)), 0U, 0U};
        AscendC::DataCopyPad(outputValueGm[start + off], tmpBuf, outParams);
        WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);

        AscendC::DataCopyPad(tmpBuf, tempIndexGm[start + off], thisParams, cpPad);
        WaitPipe<AscendC::HardEvent::MTE2_S>(pipe);
        AscendC::DataCopyPad(outputIndexGm[start + off], tmpBuf, outParams);
        WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
        off += thisLen;
    }
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchGradSort::ProcessOneByteRound(
    uint32_t execRound, uint32_t shiftByte,
    AscendC::GlobalTensor<int32_t> &valueGm, AscendC::GlobalTensor<int32_t> &indexGm,
    AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
    AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
    AscendC::GlobalTensor<int32_t> &histGm, AscendC::GlobalTensor<int32_t> &prefixGm,
    AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe)
{
    uint32_t batchCount = CeilDiv(tileCount_, coreCount_);
    AscendC::GlobalTensor<int32_t> *srcValueGm = GetSrcValueGm(execRound, valueGm, tempValueGm, outputValueGm);

    ProcessHist(shiftByte * BITS_PER_BYTE, batchCount, srcValueGm, histGm, pipe);

    if (AscendC::GetBlockIdx() == 0) {
        ComputePrefixAndOffsets(tileCount_, prefixGm, tileOffsetsGm, histGm, pipe);
    }
    AscendC::SyncAll<true>();

    ProcessScatter(execRound, shiftByte * BITS_PER_BYTE, batchCount, srcValueGm, indexGm, outputValueGm,
                   outputIndexGm, tempValueGm, tempIndexGm, histGm, prefixGm,
                   tileOffsetsGm, pipe);
}

__aicore__ inline void EngramFetchGradSort::ComputeCoreRange(
    AscendC::GlobalTensor<int32_t> &prefixGm,
    uint32_t &myBucketStart, uint32_t &myBucketEnd,
    AscendC::TPipe &pipe)
{
    AscendC::LocalTensor<int32_t> prefixLocal = prefixBuffer_.Get<int32_t>();
    AscendC::DataCopyExtParams param{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> padParam{false, 0, 0, 0};
    AscendC::DataCopyPad(prefixLocal, prefixGm, param, padParam);
    WaitPipe<AscendC::HardEvent::MTE2_S>(pipe);

    uint32_t coreId = AscendC::GetBlockIdx();

    // uint32_t nonEmpty = 0;
    // for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
    //     uint32_t prev = (b == 0) ? 0 : static_cast<uint32_t>(prefixLocal.GetValue(b));
    //     uint32_t next = (b + 1 == HISTOGRAM_BINS) ? elementCount_ : static_cast<uint32_t>(prefixLocal.GetValue(b + 1));
    //     uint32_t cnt = next - prev;
    //     if (cnt > 0) {
    //         AscendC::printf("SORT core=%uu bucket[%u]=%uu\n", coreId, b, cnt);
    //         nonEmpty++;
    //     }
    // }
    // AscendC::printf("SORT core=%uu nonEmptyBuckets=%uu total=%uu\n", coreId, nonEmpty, elementCount_);

    uint32_t bucketsPerCore = CeilDiv(HISTOGRAM_BINS, coreCount_);
    myBucketStart = coreId * bucketsPerCore;
    myBucketEnd = myBucketStart + bucketsPerCore;
    if (myBucketEnd > HISTOGRAM_BINS) {
        myBucketEnd = HISTOGRAM_BINS;
    }
}

__aicore__ inline void EngramFetchGradSort::ProcessLocalLsdRound(
    uint32_t execRound, uint32_t shiftByte, uint32_t coreStart, uint32_t coreEnd,
    AscendC::TPipe &pipe)
{
    if (coreStart >= coreEnd) {
        return;
    }

    AscendC::GlobalTensor<int32_t> *srcValueGm = GetSrcValueGm(execRound, valueGm_, tempValueGm_, valueGm_);
    AscendC::GlobalTensor<int32_t> *srcIdx;
    AscendC::GlobalTensor<int32_t> *dstValue;
    AscendC::GlobalTensor<int32_t> *dstIndex;
    GetDstBuffers(execRound, indexGm_, valueGm_, indexGm_,
                  tempValueGm_, tempIndexGm_, srcIdx, dstValue, dstIndex);

    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVS = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_S));
    event_t evtMte2S = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_S));
    event_t evtSV = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::S_V));

    AscendC::LocalTensor<int32_t> localHist = histWideBuffer_.Get<int32_t>();
    AscendC::Duplicate(localHist, (int32_t)0, HISTOGRAM_BINS);

    uint32_t shiftBits = shiftByte * BITS_PER_BYTE;
    for (uint32_t off = coreStart; off < coreEnd; off += tileElements_) {
        uint32_t tileLen = MinU32(coreEnd - off, tileElements_);
        AscendC::LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint16_t> histLocal = histogramBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<uint16_t> cumsumLocal = cumulativeBuffer_.Get<uint16_t>();
        AscendC::LocalTensor<uint32_t> exclusiveLocal = exclusiveBuffer_.Get<uint32_t>();
        AscendC::LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<int32_t> histWide = prefixBuffer_.Get<int32_t>();

        uint32_t valBytes = tileLen * sizeof(int32_t);
        AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(valueLocal, (*srcValueGm)[off], valParam, valPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);

        AscendC::Duplicate(exclusiveLocal, (uint32_t)0, HISTOGRAM_BINS);
        ComputeTileHistogram(
            (__local_mem__ uint32_t *)valueLocal.GetPhyAddr(), tileLen, shiftBits,
            (__local_mem__ uint16_t *)histLocal.GetPhyAddr(),
            (__local_mem__ uint16_t *)cumsumLocal.GetPhyAddr(),
            (__local_mem__ uint8_t *)keyLocal.GetPhyAddr(),
            (__local_mem__ uint32_t *)exclusiveLocal.GetPhyAddr(),
            (__local_mem__ uint32_t *)histWide.GetPhyAddr());

        AscendC::SetFlag<AscendC::HardEvent::V_S>(evtVS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtVS);

        for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
            localHist.SetValue(b, localHist.GetValue(b) + static_cast<int32_t>(histLocal.GetValue(b)));
        }
    }

    WaitPipe<AscendC::HardEvent::V_S>(pipe);
    AscendC::LocalTensor<int32_t> localPrefix = prefixBuffer_.Get<int32_t>();
    int32_t running = 0;
    for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
        localPrefix.SetValue(b, running);
        running += localHist.GetValue(b);
    }

    AscendC::LocalTensor<int32_t> runningOffset = histogramBuffer_.Get<int32_t>();
    AscendC::Duplicate(runningOffset, (int32_t)0, HISTOGRAM_BINS);

    for (uint32_t off = coreStart; off < coreEnd; off += tileElements_) {
        uint32_t tileLen = MinU32(coreEnd - off, tileElements_);
        AscendC::LocalTensor<int32_t> valueLocal = valueBuffer_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> indexLocal = indexBuffer_.Get<int32_t>();
        AscendC::LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> sortedKey = sortedKeyBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint32_t> sortIndices = exclusiveBuffer_.Get<uint32_t>();

        uint32_t valBytes = tileLen * sizeof(int32_t);
        AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(valueLocal, (*srcValueGm)[off], valParam, valPad);
        AscendC::DataCopyPad(indexLocal, (*srcIdx)[off], valParam, valPad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);

        for (uint32_t i = 0; i < tileLen; i++) {
            keyLocal.SetValue(i, (uint8_t)((valueLocal.GetValue(i) >> shiftBits) & 0xFF));
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(evtSV);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtSV);

        AscendC::LocalTensor<uint8_t> sortTemp = sortTempBuffer_.Get<uint8_t>();
        AscendC::Sort<uint8_t, false, sortConfig>(sortedKey, sortIndices, keyLocal, sortTemp, tileLen);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(evtVS);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtVS);

        AscendC::LocalTensor<int32_t> gatheredBuf = sortTempBuffer_.Get<int32_t>();
        uint32_t keyStart = 0;
        uint8_t curKey = sortedKey.GetValue(0);
        for (uint32_t i = 1; i <= tileLen; i++) {
            if (i == tileLen || sortedKey.GetValue(i) != curKey) {
                uint32_t keyCount = i - keyStart;
                int32_t gmPos = static_cast<int32_t>(coreStart) +
                                localPrefix.GetValue(curKey) +
                                runningOffset.GetValue(curKey);
                ScatterOneKey(keyStart, keyCount, gmPos, gatheredBuf, valueLocal,
                              indexLocal, sortIndices, dstValue, dstIndex, pipe);
                runningOffset.SetValue(curKey, runningOffset.GetValue(curKey) + static_cast<int32_t>(keyCount));
                if (i < tileLen) {
                    curKey = sortedKey.GetValue(i);
                    keyStart = i;
                }
            }
        }
    }
}

__aicore__ inline void EngramFetchGradSort::Process(uint32_t actualCount, AscendC::TPipe &pipe)
{
    elementCount_ = actualCount;
    uint32_t dynTile = CeilDiv(actualCount, coreCount_);
    if (dynTile > 0 && dynTile < tileElements_) {
        tileElements_ = dynTile;
    }
    tileCount_ = CeilDiv(elementCount_, tileElements_);

    // AscendC::printf("SORT byteRounds=%uu msbByte=%uu\n", byteRounds_, byteRounds_ - 1U);
    ApplyValueOffset(pipe);

    ProcessOneByteRound(0, byteRounds_ - 1U, valueGm_, indexGm_, valueGm_, indexGm_,
                        tempValueGm_, tempIndexGm_, histGm_, prefixGm_, tileOffsetsGm_, pipe);

    if (byteRounds_ <= 1U) {
        CopyTempToOutput(tempValueGm_, tempIndexGm_, valueGm_, indexGm_, pipe);
        RestoreValueOffset(pipe);
        return;
    }

    uint32_t myBucketStart = 0;
    uint32_t myBucketEnd = 0;
    ComputeCoreRange(prefixGm_, myBucketStart, myBucketEnd, pipe);

    AscendC::LocalTensor<int32_t> prefixLocal = prefixBuffer_.Get<int32_t>();
    uint32_t savedBoundaries[HISTOGRAM_BINS + 1];
    savedBoundaries[0] = 0;
    for (uint32_t b = 1; b <= HISTOGRAM_BINS; b++) {
        savedBoundaries[b] = (b == HISTOGRAM_BINS) ? elementCount_
            : static_cast<uint32_t>(prefixLocal.GetValue(b));
    }

    for (uint32_t execRound = 1U; execRound < byteRounds_; execRound++) {
        uint32_t shiftByte = execRound - 1U;
        for (uint32_t b = myBucketStart; b < myBucketEnd; b++) {
            uint32_t bucketStart = savedBoundaries[b];
            uint32_t bucketEnd = savedBoundaries[b + 1];
            if (bucketEnd <= bucketStart) {
                continue;
            }
            ProcessLocalLsdRound(execRound, shiftByte, bucketStart, bucketEnd, pipe);
        }
        AscendC::SyncAll<true>();
    }

    if (byteRounds_ % 2 == 1) {
        CopyTempToOutput(tempValueGm_, tempIndexGm_, valueGm_, indexGm_, pipe);
    }

    RestoreValueOffset(pipe);
}

} // namespace EngramFetchGradSort
#endif
