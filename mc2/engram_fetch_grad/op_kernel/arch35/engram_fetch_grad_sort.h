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
#include "adv_api/math/cumsum.h"
#include "simt_api/asc_simt.h"

constexpr uint32_t SIMT_THREAD_NUM = 1024;

__simt_vf__ LAUNCH_BOUND(SIMT_THREAD_NUM)
__aicore__ inline void ScatterWithPosSimt(uint32_t tileLen, __ubuf__ uint8_t *sortedKey,
                                          __ubuf__ uint32_t *sortedIndex, __ubuf__ uint32_t *binStart,
                                          __ubuf__ uint32_t *basePos, __ubuf__ int32_t *value,
                                          __ubuf__ int32_t *index, __gm__ int32_t *dstValueGm,
                                          __gm__ int32_t *dstIndexGm)
{
    for (uint32_t i = threadIdx.x; i < tileLen; i += SIMT_THREAD_NUM) {
        uint8_t k = sortedKey[i];
        uint32_t gmPos = basePos[k] + i - binStart[k];
        uint32_t srcIdx = sortedIndex[i];
        dstValueGm[gmPos] = value[srcIdx];
        dstIndexGm[gmPos] = index[srcIdx];
    }
}

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

__aicore__ inline void ComputeTileHistogram(__local_mem__ uint32_t *valueUb, uint32_t elementCount, uint32_t byteRound,
                                            int32_t valueOffset, __local_mem__ uint16_t *histogramUb,
                                            __local_mem__ uint16_t *cumulativeUb, __local_mem__ uint8_t *keyUb,
                                            __local_mem__ uint32_t *exclusiveUb, __local_mem__ uint32_t *histWideUb);

class EngramFetchGradSort {
public:
    __aicore__ inline EngramFetchGradSort() = default;

    __aicore__ inline static uint64_t GetWorkspaceSize(uint32_t totalElements, uint32_t numCores);

    __aicore__ inline static uint32_t GetUbSize(uint32_t totalElements, uint32_t numCores);

    __aicore__ inline void Init(uint32_t totalElements, uint32_t numCores, GM_ADDR valueGm, GM_ADDR indexGm,
                                GM_ADDR workspaceGm, AscendC::TPipe &pipe,
                                AscendC::TBufPool<AscendC::TPosition::VECCALC, 16> &pool);

    __aicore__ inline void SetMaxValue(uint32_t maxVal);

    __aicore__ inline void SetValueOffset(int32_t offset);

    __aicore__ inline void Process(uint32_t actualCount, AscendC::TPipe &pipe);

    uint64_t sortCopyTempCycle_{0};
    uint64_t sortRoundHistCycle_[4]{0};
    uint64_t sortRoundPrefixCycle_[4]{0};
    uint64_t sortRoundScatterCycle_[4]{0};
    uint64_t sortHistMte2Cycle_[4]{0};
    uint64_t sortHistComputeCycle_[4]{0};
    uint64_t sortHistMte3Cycle_[4]{0};
    uint64_t sortScatterMte2Cycle_[4]{0};
    uint64_t sortScatterKeyCycle_[4]{0};
    uint64_t sortScatterWriteCycle_[4]{0};

private:
    __aicore__ inline void ComputePrefixAndOffsets(uint32_t tc, AscendC::GlobalTensor<int32_t> &prefixGm,
                                                   AscendC::GlobalTensor<int32_t> &tileOffsetsGm,
                                                   AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe);

    __aicore__ inline AscendC::GlobalTensor<int32_t> *GetSrcValueGm(uint32_t byteRound,
                                                                    AscendC::GlobalTensor<int32_t> &valueGm,
                                                                    AscendC::GlobalTensor<int32_t> &tempValueGm,
                                                                    AscendC::GlobalTensor<int32_t> &outputValueGm);

    __aicore__ inline void GetDstBuffers(
        uint32_t byteRound, AscendC::GlobalTensor<int32_t> &indexGm, AscendC::GlobalTensor<int32_t> &outputValueGm,
        AscendC::GlobalTensor<int32_t> &outputIndexGm, AscendC::GlobalTensor<int32_t> &tempValueGm,
        AscendC::GlobalTensor<int32_t> &tempIndexGm, AscendC::GlobalTensor<int32_t> *&srcIdx,
        AscendC::GlobalTensor<int32_t> *&dstValue, AscendC::GlobalTensor<int32_t> *&dstIndex);

    __aicore__ inline void ProcessHist(uint32_t byteRound, uint32_t batchCount,
                                       AscendC::GlobalTensor<int32_t> *srcValueGm,
                                       AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe);

    __aicore__ inline void ProcessScatter(
        uint32_t byteRound, uint32_t batchCount, AscendC::GlobalTensor<int32_t> *srcValueGm,
        AscendC::GlobalTensor<int32_t> &indexGm, AscendC::GlobalTensor<int32_t> &outputValueGm,
        AscendC::GlobalTensor<int32_t> &outputIndexGm, AscendC::GlobalTensor<int32_t> &tempValueGm,
        AscendC::GlobalTensor<int32_t> &tempIndexGm, AscendC::GlobalTensor<int32_t> &histGm,
        AscendC::GlobalTensor<int32_t> &prefixGm, AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe);

    __aicore__ inline void CopyTempToOutput(AscendC::GlobalTensor<int32_t> &tempValueGm,
                                            AscendC::GlobalTensor<int32_t> &tempIndexGm,
                                            AscendC::GlobalTensor<int32_t> &outputValueGm,
                                            AscendC::GlobalTensor<int32_t> &outputIndexGm, AscendC::TPipe &pipe);

    __aicore__ inline void ProcessOneByteRound(
        uint32_t byteRound, AscendC::GlobalTensor<int32_t> &valueGm, AscendC::GlobalTensor<int32_t> &indexGm,
        AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
        AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
        AscendC::GlobalTensor<int32_t> &histGm, AscendC::GlobalTensor<int32_t> &prefixGm,
        AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe);

    static constexpr AscendC::SortConfig sortConfig{AscendC::SortType::RADIX_SORT, false};
    static constexpr AscendC::CumSumConfig prefixScanCfg{true, false, false,
                                                         AscendC::CumSumAlgorithm::CUMSUM_ALGORITHM_LINEBYLINE};

    AscendC::TBuf<AscendC::TPosition::VECCALC> sortWorkspace_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> keyBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> histogramBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cumulativeBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prefixBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sortedKeyBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> histWideBuffer_;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> prefixQueue_;

    uint32_t elementCount_{0};
    uint32_t tileElements_{0};
    uint32_t tileAlignBytes_{0};
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
    AscendC::GlobalTensor<int32_t> coreSumsGm_;
    AscendC::GlobalTensor<int32_t> tileOffsetsGm_;
    GM_ADDR keyLocalGm_{nullptr};
};

// ==== implementations ====

__aicore__ inline void ComputeTileHistogram(__local_mem__ uint32_t *valueUb, uint32_t elementCount, uint32_t byteRound,
                                            int32_t valueOffset, __local_mem__ uint16_t *histogramUb,
                                            __local_mem__ uint16_t *cumulativeUb, __local_mem__ uint8_t *keyUb,
                                            __local_mem__ uint32_t *exclusiveUb, __local_mem__ uint32_t *histWideUb)
{
    uint32_t shiftBits = byteRound * BITS_PER_BYTE;
    __VEC_SCOPE__
    {
        using namespace AscendC::Reg;
        RegTensor<uint32_t> input0, input1, input2, input3;
        RegTensor<uint16_t> hist0, hist1, cumu0, cumu1;
        RegTensor<uint32_t> offsetReg;
        MaskReg mask32 = CreateMask<uint32_t>();
        MaskReg mask16 = CreateMask<uint16_t>();
        Duplicate(hist0, (uint16_t)0, mask16);
        Duplicate(hist1, (uint16_t)0, mask16);
        Duplicate(cumu0, (uint16_t)0, mask16);
        Duplicate(cumu1, (uint16_t)0, mask16);
        Duplicate(offsetReg, static_cast<uint32_t>(valueOffset), mask32);
        for (uint16_t i = 0; i < CeilDiv(elementCount, VECTOR_LEN_INT8); i++) {
            MaskReg elemMask = UpdateMask<uint8_t>(elementCount);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input0, valueUb, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input1, valueUb, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input2, valueUb, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(input3, valueUb, VECTOR_LEN_INT32);
            Sub(input0, input0, offsetReg, mask32);
            Sub(input1, input1, offsetReg, mask32);
            Sub(input2, input2, offsetReg, mask32);
            Sub(input3, input3, offsetReg, mask32);
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
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::FREQUENCY>(hist0, b0, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::FREQUENCY>(hist1, b0, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN0, HistogramsType::ACCUMULATE>(cumu0, b0, elemMask);
            Histograms<uint8_t, uint16_t, HistogramsBinType::BIN1, HistogramsType::ACCUMULATE>(cumu1, b0, elemMask);
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

__aicore__ inline uint64_t EngramFetchGradSort::GetWorkspaceSize(uint32_t totalElements, uint32_t numCores)
{
    uint32_t tileElements = DEFAULT_TILE_SIZE;
    uint32_t dynTile = CeilDiv(totalElements, numCores);
    if (dynTile > 0 && dynTile < tileElements) {
        tileElements = dynTile;
    }
    uint32_t tileCount = CeilDiv(totalElements, tileElements);
    uint32_t sortTileCountByCore = numCores;
    uint32_t sortTileCountByDefault = CeilDiv(totalElements, DEFAULT_TILE_SIZE);
    uint32_t sortTileCount =
        (sortTileCountByCore > sortTileCountByDefault) ? sortTileCountByCore : sortTileCountByDefault;
    uint32_t tileAlignBytes = static_cast<uint32_t>(AlignUpU32(tileElements * sizeof(int32_t)));
    uint64_t sortTempBytes = static_cast<uint64_t>(totalElements) * sizeof(int32_t);
    uint64_t sortTempSize = (sortTempBytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES;
    uint64_t keyLocalSize = static_cast<uint64_t>(tileCount) * static_cast<uint64_t>(tileAlignBytes);
    return 2 * sortTempSize + 2 * static_cast<uint64_t>(sortTileCount) * HISTOGRAM_BINS * sizeof(int32_t) +
           HISTOGRAM_BINS * sizeof(int32_t) + static_cast<uint64_t>(numCores) * HISTOGRAM_BINS * sizeof(int32_t) +
           keyLocalSize;
}

__aicore__ inline uint32_t EngramFetchGradSort::GetUbSize(uint32_t totalElements, uint32_t numCores)
{
    uint32_t tileElements = DEFAULT_TILE_SIZE;
    uint32_t tileAlignBytes = static_cast<uint32_t>(AlignUpU32(tileElements * sizeof(int32_t)));
    uint32_t phaseHistSize = 2U * tileAlignBytes;
    uint32_t phasePrefixSize = numCores * HISTOGRAM_BINS * sizeof(int32_t) + tileAlignBytes;
    uint32_t phaseScatterSize = 5U * tileAlignBytes;
    uint32_t sortWorkspaceSize = phaseHistSize;
    if (phasePrefixSize > sortWorkspaceSize) {
        sortWorkspaceSize = phasePrefixSize;
    }
    if (phaseScatterSize > sortWorkspaceSize) {
        sortWorkspaceSize = phaseScatterSize;
    }
    uint32_t smallBufs = tileElements + HISTOGRAM_BINS * sizeof(int32_t) + HISTOGRAM_BINS * sizeof(uint16_t) +
                         HISTOGRAM_BINS * sizeof(int32_t) + tileElements + HISTOGRAM_BINS * sizeof(int32_t) +
                         HISTOGRAM_BINS * sizeof(int32_t);
    return sortWorkspaceSize + smallBufs;
}

__aicore__ inline void EngramFetchGradSort::Init(uint32_t totalElements, uint32_t numCores, GM_ADDR valueGm,
                                                 GM_ADDR indexGm, GM_ADDR workspaceGm, AscendC::TPipe &pipe,
                                                 AscendC::TBufPool<AscendC::TPosition::VECCALC, 16> &pool)
{
    elementCount_ = totalElements;
    coreCount_ = numCores;
    tileElements_ = DEFAULT_TILE_SIZE;
    tileCount_ = CeilDiv(elementCount_, tileElements_);

    uint32_t sortTileCountByCore = numCores;
    uint32_t sortTileCountByDefault = CeilDiv(elementCount_, DEFAULT_TILE_SIZE);
    uint32_t sortTileCount =
        (sortTileCountByCore > sortTileCountByDefault) ? sortTileCountByCore : sortTileCountByDefault;
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
    coreSumsGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));
    offset += static_cast<uint64_t>(numCores) * HISTOGRAM_BINS * sizeof(int32_t);
    tileOffsetsGm_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGm + offset));
    offset += static_cast<uint64_t>(sortTileCount) * HISTOGRAM_BINS * sizeof(int32_t);

    uint32_t tileAlignBytes = static_cast<uint32_t>(AlignUpU32(tileElements_ * sizeof(int32_t)));
    keyLocalGm_ = workspaceGm + offset;
    tileAlignBytes_ = tileAlignBytes;
    uint32_t phaseHistSize = 2U * tileAlignBytes;
    uint32_t phasePrefixSize = static_cast<uint32_t>(numCores) * HISTOGRAM_BINS * sizeof(int32_t) + tileAlignBytes;
    uint32_t phaseScatterSize = 5U * tileAlignBytes;
    uint32_t sortWorkspaceSize = phaseHistSize;
    if (phasePrefixSize > sortWorkspaceSize) {
        sortWorkspaceSize = phasePrefixSize;
    }
    if (phaseScatterSize > sortWorkspaceSize) {
        sortWorkspaceSize = phaseScatterSize;
    }
    pool.InitBuffer(sortWorkspace_, sortWorkspaceSize);
    pool.InitBuffer(keyBuffer_, tileElements_);
    pool.InitBuffer(histogramBuffer_, HISTOGRAM_BINS * sizeof(int32_t));
    pool.InitBuffer(cumulativeBuffer_, HISTOGRAM_BINS * sizeof(uint16_t));
    pool.InitBuffer(prefixBuffer_, HISTOGRAM_BINS * sizeof(int32_t));
    pool.InitBuffer(sortedKeyBuffer_, tileElements_);
    pool.InitBuffer(histWideBuffer_, HISTOGRAM_BINS * sizeof(int32_t));
    pool.InitBuffer(prefixQueue_, 1, HISTOGRAM_BINS * sizeof(int32_t));
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

__aicore__ inline void EngramFetchGradSort::SetValueOffset(int32_t offset) { valueOffset_ = offset; }

__aicore__ inline void EngramFetchGradSort::ComputePrefixAndOffsets(uint32_t tc,
                                                                    AscendC::GlobalTensor<int32_t> &prefixGm,
                                                                    AscendC::GlobalTensor<int32_t> &tileOffsetsGm,
                                                                    AscendC::GlobalTensor<int32_t> &histGm,
                                                                    AscendC::TPipe &pipe)
{
    AscendC::LocalTensor<int32_t> histogram = prefixQueue_.AllocTensor<int32_t>();
    AscendC::DataCopyExtParams param{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> padParam{false, 0, 0, 0};
    AscendC::DataCopyParams offsetParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};

    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte2 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));
    event_t evtVMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
    event_t evtMte3Mte2 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    event_t evtMte3S = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE3_S));

    uint32_t coreId = AscendC::GetBlockIdx();
    uint32_t tilesPerCore = CeilDiv(tc, coreCount_);
    uint32_t coreStart = coreId * tilesPerCore;
    uint32_t coreEnd = MinU32(coreStart + tilesPerCore, tc);

    AscendC::LocalTensor<int32_t> localSum = histWideBuffer_.Get<int32_t>();
    AscendC::Duplicate(localSum, (int32_t)0, HISTOGRAM_BINS);

    for (uint32_t t = coreStart; t < coreEnd; t++) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2);
        AscendC::DataCopyPad(histogram, histGm[t * HISTOGRAM_BINS], param, padParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::Add(localSum, localSum, histogram, HISTOGRAM_BINS);
    }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            // SIMT writes dst GM via the V pipe; next byte-round's ProcessHist reads it via MTE2.
            // Make V stores visible to MTE2 explicitly (don't rely on SyncAll flushing V->GM).
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2);
    AscendC::DataCopyPad(coreSumsGm_[coreId * HISTOGRAM_BINS], localSum, offsetParam);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);

    AscendC::SyncAll<true>();

    AscendC::LocalTensor<int32_t> coreSumsBatch = sortWorkspace_.GetWithOffset<int32_t>(coreCount_ * HISTOGRAM_BINS, 0);
    AscendC::DataCopyExtParams batchParam{1U, static_cast<uint32_t>(coreCount_ * HISTOGRAM_BINS * sizeof(int32_t)), 0U,
                                          0U, 0U};
    AscendC::DataCopyPad(coreSumsBatch, coreSumsGm_, batchParam, padParam);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);

    AscendC::LocalTensor<int32_t> prefixUb = histWideBuffer_.Get<int32_t>();
    AscendC::LocalTensor<int32_t> totalBuf = prefixBuffer_.Get<int32_t>();
    __local_mem__ uint32_t *sumsPtr = (__local_mem__ uint32_t *)coreSumsBatch.GetPhyAddr();

    __VEC_SCOPE__
    {
        using namespace AscendC::Reg;
        MaskReg mask32 = CreateMask<uint32_t>();

        RegTensor<uint32_t> acc0, acc1, acc2, acc3;
        Duplicate(acc0, (uint32_t)0, mask32);
        Duplicate(acc1, (uint32_t)0, mask32);
        Duplicate(acc2, (uint32_t)0, mask32);
        Duplicate(acc3, (uint32_t)0, mask32);

        for (uint16_t c = 0; c < coreId; c++) {
            __local_mem__ uint32_t *ptr = sumsPtr + c * HISTOGRAM_BINS;
            RegTensor<uint32_t> r0, r1, r2, r3;
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r0, ptr, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r1, ptr, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r2, ptr, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r3, ptr, VECTOR_LEN_INT32);
            Add(acc0, acc0, r0, mask32);
            Add(acc1, acc1, r1, mask32);
            Add(acc2, acc2, r2, mask32);
            Add(acc3, acc3, r3, mask32);
        }

        __local_mem__ uint32_t *offsetPtr = (__local_mem__ uint32_t *)prefixUb.GetPhyAddr();
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(offsetPtr, acc0, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(offsetPtr, acc1, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(offsetPtr, acc2, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(offsetPtr, acc3, VECTOR_LEN_INT32, mask32);

        for (uint16_t c = coreId; c < coreCount_; c++) {
            __local_mem__ uint32_t *ptr = sumsPtr + c * HISTOGRAM_BINS;
            RegTensor<uint32_t> r0, r1, r2, r3;
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r0, ptr, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r1, ptr, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r2, ptr, VECTOR_LEN_INT32);
            DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(r3, ptr, VECTOR_LEN_INT32);
            Add(acc0, acc0, r0, mask32);
            Add(acc1, acc1, r1, mask32);
            Add(acc2, acc2, r2, mask32);
            Add(acc3, acc3, r3, mask32);
        }

        __local_mem__ uint32_t *totalPtr = (__local_mem__ uint32_t *)totalBuf.GetPhyAddr();
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(totalPtr, acc0, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(totalPtr, acc1, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(totalPtr, acc2, VECTOR_LEN_INT32, mask32);
        DataCopy<uint32_t, PostLiteral::POST_MODE_UPDATE>(totalPtr, acc3, VECTOR_LEN_INT32, mask32);
    }

    AscendC::PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<float> floatBuf = histogramBuffer_.Get<float>();
    AscendC::LocalTensor<int32_t> inclBuf =
        sortWorkspace_.GetWithOffset<int32_t>(HISTOGRAM_BINS, coreCount_ * HISTOGRAM_BINS * sizeof(int32_t));

    AscendC::Cast<float, int32_t>(floatBuf, totalBuf, AscendC::RoundMode::CAST_RINT, HISTOGRAM_BINS);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::CumSumInfo prefixScanInfo{1U, HISTOGRAM_BINS};
    AscendC::CumSum<float, prefixScanCfg>(floatBuf, floatBuf, floatBuf, prefixScanInfo);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Cast<int32_t, float>(inclBuf, floatBuf, AscendC::RoundMode::CAST_RINT, HISTOGRAM_BINS);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(totalBuf, inclBuf, totalBuf, HISTOGRAM_BINS);

    for (uint32_t t = coreStart; t < coreEnd; t++) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
        AscendC::DataCopyPad(tileOffsetsGm[t * HISTOGRAM_BINS], prefixUb, offsetParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2);
        AscendC::DataCopyPad(histogram, histGm[t * HISTOGRAM_BINS], param, padParam);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
        AscendC::Add(prefixUb, prefixUb, histogram, HISTOGRAM_BINS);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(evtMte3S);
    prefixQueue_.FreeTensor(histogram);
}

__aicore__ inline AscendC::GlobalTensor<int32_t> *EngramFetchGradSort::GetSrcValueGm(
    uint32_t byteRound, AscendC::GlobalTensor<int32_t> &valueGm, AscendC::GlobalTensor<int32_t> &tempValueGm,
    AscendC::GlobalTensor<int32_t> &outputValueGm)
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
    uint32_t byteRound, AscendC::GlobalTensor<int32_t> &indexGm, AscendC::GlobalTensor<int32_t> &outputValueGm,
    AscendC::GlobalTensor<int32_t> &outputIndexGm, AscendC::GlobalTensor<int32_t> &tempValueGm,
    AscendC::GlobalTensor<int32_t> &tempIndexGm, AscendC::GlobalTensor<int32_t> *&srcIdx,
    AscendC::GlobalTensor<int32_t> *&dstValue, AscendC::GlobalTensor<int32_t> *&dstIndex)
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

__aicore__ inline void EngramFetchGradSort::ProcessHist(uint32_t byteRound, uint32_t batchCount,
                                                        AscendC::GlobalTensor<int32_t> *srcValueGm,
                                                        AscendC::GlobalTensor<int32_t> &histGm, AscendC::TPipe &pipe)
{
    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtVMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));

    uint64_t mte2Total = 0;
    uint64_t computeTotal = 0;
    uint64_t mte3Total = 0;

    for (uint32_t batch = 0; batch < batchCount; batch++) {
        uint32_t firstTile = batch * coreCount_;
        uint32_t batchCores = MinU32(tileCount_ - firstTile, coreCount_);
        uint32_t coreId = AscendC::GetBlockIdx();
        // uint64_t t2 = 0;
        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            AscendC::LocalTensor<int32_t> valueLocal = sortWorkspace_.GetWithOffset<int32_t>(tileElements_, 0);
            AscendC::LocalTensor<uint16_t> histLocal = histogramBuffer_.Get<uint16_t>();
            AscendC::LocalTensor<uint16_t> cumsumLocal = cumulativeBuffer_.Get<uint16_t>();
            AscendC::LocalTensor<uint32_t> exclusiveLocal =
                sortWorkspace_.GetWithOffset<uint32_t>(tileElements_, tileAlignBytes_);
            AscendC::LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();

            uint32_t valBytes = tileLen * sizeof(int32_t);
            AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
            // uint64_t t0 = static_cast<uint64_t>(AscendC::GetSystemCycle());
            AscendC::DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            // uint64_t t1 = static_cast<uint64_t>(AscendC::GetSystemCycle());
            // mte2Total += t1 - t0;

            AscendC::Duplicate(exclusiveLocal, (uint32_t)0, HISTOGRAM_BINS);

            AscendC::LocalTensor<int32_t> histWide = histWideBuffer_.Get<int32_t>();

            ComputeTileHistogram(
                (__local_mem__ uint32_t *)valueLocal.GetPhyAddr(), tileLen, byteRound, valueOffset_,
                (__local_mem__ uint16_t *)histLocal.GetPhyAddr(), (__local_mem__ uint16_t *)cumsumLocal.GetPhyAddr(),
                (__local_mem__ uint8_t *)keyLocal.GetPhyAddr(), (__local_mem__ uint32_t *)exclusiveLocal.GetPhyAddr(),
                (__local_mem__ uint32_t *)histWide.GetPhyAddr());

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            // t2 = static_cast<uint64_t>(AscendC::GetSystemCycle());
            // computeTotal += t2 - t1;

            AscendC::DataCopyParams histOutParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};
            AscendC::DataCopyPad(histGm[tileId * HISTOGRAM_BINS], histWide, histOutParam);

            AscendC::GlobalTensor<uint8_t> keyLocalGmT;
            keyLocalGmT.SetGlobalBuffer((__gm__ uint8_t *)(keyLocalGm_ + static_cast<uint64_t>(tileId) * tileAlignBytes_));
            AscendC::DataCopyParams keyOutParam{1U, static_cast<uint16_t>(tileLen), 0U, 0U};
            AscendC::DataCopyPad(keyLocalGmT, keyLocal, keyOutParam);
        }
        WaitPipe<AscendC::HardEvent::MTE3_S>(pipe);
        // uint64_t t3 = static_cast<uint64_t>(AscendC::GetSystemCycle());
        // mte3Total += t3 - t2;
    }
    AscendC::SyncAll<true>();
    // sortHistMte2Cycle_[byteRound] += mte2Total;
    // sortHistComputeCycle_[byteRound] += computeTotal;
    // sortHistMte3Cycle_[byteRound] += mte3Total;
}

__aicore__ inline void EngramFetchGradSort::ProcessScatter(
    uint32_t byteRound, uint32_t batchCount, AscendC::GlobalTensor<int32_t> *srcValueGm,
    AscendC::GlobalTensor<int32_t> &indexGm, AscendC::GlobalTensor<int32_t> &outputValueGm,
    AscendC::GlobalTensor<int32_t> &outputIndexGm, AscendC::GlobalTensor<int32_t> &tempValueGm,
    AscendC::GlobalTensor<int32_t> &tempIndexGm, AscendC::GlobalTensor<int32_t> &histGm,
    AscendC::GlobalTensor<int32_t> &prefixGm, AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe)
{
    AscendC::DataCopyExtParams histParam{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};
    event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_V));
    event_t evtMte2S = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::MTE2_S));
    event_t evtVS2 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_S));
    event_t evtVMte3 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE3));
    event_t evtSV = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::S_V));
    event_t evtVMte2 = static_cast<event_t>(pipe.FetchEventID(AscendC::HardEvent::V_MTE2));

    // uint64_t mte2Total = 0;
    // uint64_t keyTotal = 0;
    // uint64_t writeTotal = 0;

    for (uint32_t batch = 0; batch < batchCount; batch++) {
        uint32_t firstTile = batch * coreCount_;
        uint32_t batchCores = MinU32(tileCount_ - firstTile, coreCount_);
        uint32_t coreId = AscendC::GetBlockIdx();

        AscendC::LocalTensor<int32_t> valueLocal = sortWorkspace_.GetWithOffset<int32_t>(tileElements_, 0);
        AscendC::LocalTensor<int32_t> indexLocal =
            sortWorkspace_.GetWithOffset<int32_t>(tileElements_, tileAlignBytes_);
        AscendC::LocalTensor<uint8_t> keyLocal = keyBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint8_t> sortedKey = sortedKeyBuffer_.Get<uint8_t>();
        AscendC::LocalTensor<uint32_t> sortIndices =
            sortWorkspace_.GetWithOffset<uint32_t>(tileElements_, 2U * tileAlignBytes_);
        AscendC::LocalTensor<int32_t> prefixLocal = prefixBuffer_.Get<int32_t>();
        AscendC::LocalTensor<int32_t> offsetLocal = histogramBuffer_.Get<int32_t>();

        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            AscendC::GlobalTensor<int32_t> *srcIdx;
            AscendC::GlobalTensor<int32_t> *dstValue;
            AscendC::GlobalTensor<int32_t> *dstIndex;
            GetDstBuffers(byteRound, indexGm, outputValueGm, outputIndexGm, tempValueGm, tempIndexGm, srcIdx, dstValue,
                          dstIndex);

            uint32_t valBytes = tileLen * sizeof(int32_t);
            AscendC::DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
            AscendC::DataCopyPad(indexLocal, (*srcIdx)[offset], valParam, valPad);
        }

        if (batch == 0) {
            AscendC::SyncAll<true>();
        }

        if (coreId < batchCores) {
            uint32_t tileId = firstTile + coreId;
            uint32_t offset = tileId * tileElements_;
            uint32_t tileLen = MinU32(elementCount_ - offset, tileElements_);

            AscendC::GlobalTensor<int32_t> *srcIdx;
            AscendC::GlobalTensor<int32_t> *dstValue;
            AscendC::GlobalTensor<int32_t> *dstIndex;
            GetDstBuffers(byteRound, indexGm, outputValueGm, outputIndexGm, tempValueGm, tempIndexGm, srcIdx, dstValue,
                          dstIndex);

            AscendC::DataCopyPad(offsetLocal, tileOffsetsGm[tileId * HISTOGRAM_BINS], histParam, valPad);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evtMte2S);

            AscendC::LocalTensor<uint8_t> sortTemp =
                sortWorkspace_.GetWithOffset<uint8_t>(tileAlignBytes_, 3U * tileAlignBytes_);
            AscendC::GlobalTensor<uint8_t> keyLocalGmT;
            keyLocalGmT.SetGlobalBuffer((__gm__ uint8_t *)(keyLocalGm_ + static_cast<uint64_t>(tileId) * tileAlignBytes_));
            AscendC::DataCopyParams keyLoadParam{1U, static_cast<uint16_t>(tileLen), 0U, 0U};
            AscendC::DataCopyPad(keyLocal, keyLocalGmT, keyLoadParam);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V);
            AscendC::Sort<uint8_t, false, sortConfig>(sortedKey, sortIndices, keyLocal, sortTemp, tileLen);

            AscendC::SetFlag<AscendC::HardEvent::V_S>(evtVS2);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(evtVS2);

            // Recompute binStart (within-tile exclusive prefix per bin) as uint32 (k*4 is 4B
            // aligned; uint16 at k*2 triggered VEC UB alignment err 340) by scanning sorted keys.
            AscendC::LocalTensor<uint32_t> binStartUbT =
                sortWorkspace_.GetWithOffset<uint32_t>(HISTOGRAM_BINS, 4U * tileAlignBytes_);
            if (tileLen > 0U) {
                uint8_t prev = sortedKey.GetValue(0);
                binStartUbT.SetValue(prev, (uint32_t)0);
                for (uint32_t i = 1U; i < tileLen; i++) {
                    uint8_t k = sortedKey.GetValue(i);
                    if (k != prev) {
                        binStartUbT.SetValue(k, (uint32_t)i);
                        prev = k;
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(evtSV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(evtSV);

            AscendC::PipeBarrier<PIPE_V>();

            AscendC::LocalTensor<int32_t> basePosUb = histWideBuffer_.Get<int32_t>();
            AscendC::Add(basePosUb, prefixLocal, offsetLocal, HISTOGRAM_BINS);

            AscendC::PipeBarrier<PIPE_V>();

            __gm__ int32_t *dstValPtr = (__gm__ int32_t *)(*dstValue).GetPhyAddr();
            __gm__ int32_t *dstIdxPtr = (__gm__ int32_t *)(*dstIndex).GetPhyAddr();
            __ubuf__ uint8_t *sortedKeyUb = (__ubuf__ uint8_t *)sortedKey.GetPhyAddr();
            __ubuf__ uint32_t *sortedIdxUb = (__ubuf__ uint32_t *)sortIndices.GetPhyAddr();
            __ubuf__ uint32_t *binStartUb = (__ubuf__ uint32_t *)binStartUbT.GetPhyAddr();
            __ubuf__ uint32_t *basePosPtr = (__ubuf__ uint32_t *)basePosUb.GetPhyAddr();
            __ubuf__ int32_t *valUb = (__ubuf__ int32_t *)valueLocal.GetPhyAddr();
            __ubuf__ int32_t *idxUb = (__ubuf__ int32_t *)indexLocal.GetPhyAddr();

            asc_vf_call<ScatterWithPosSimt>(dim3(SIMT_THREAD_NUM), tileLen, sortedKeyUb, sortedIdxUb, binStartUb,
                                            basePosPtr, valUb, idxUb, dstValPtr, dstIdxPtr);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtVMte3);
        }
        AscendC::SyncAll<true>();
    }
    // sortScatterMte2Cycle_[byteRound] += mte2Total;
    // sortScatterKeyCycle_[byteRound] += keyTotal;
    // sortScatterWriteCycle_[byteRound] += writeTotal;
}

__aicore__ inline void EngramFetchGradSort::CopyTempToOutput(AscendC::GlobalTensor<int32_t> &tempValueGm,
                                                             AscendC::GlobalTensor<int32_t> &tempIndexGm,
                                                             AscendC::GlobalTensor<int32_t> &outputValueGm,
                                                             AscendC::GlobalTensor<int32_t> &outputIndexGm,
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
    AscendC::LocalTensor<int32_t> tmpBuf = sortWorkspace_.GetWithOffset<int32_t>(tileElements_, 0);
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
    uint32_t byteRound, AscendC::GlobalTensor<int32_t> &valueGm, AscendC::GlobalTensor<int32_t> &indexGm,
    AscendC::GlobalTensor<int32_t> &outputValueGm, AscendC::GlobalTensor<int32_t> &outputIndexGm,
    AscendC::GlobalTensor<int32_t> &tempValueGm, AscendC::GlobalTensor<int32_t> &tempIndexGm,
    AscendC::GlobalTensor<int32_t> &histGm, AscendC::GlobalTensor<int32_t> &prefixGm,
    AscendC::GlobalTensor<int32_t> &tileOffsetsGm, AscendC::TPipe &pipe)
{
    uint32_t batchCount = CeilDiv(tileCount_, coreCount_);
    AscendC::GlobalTensor<int32_t> *srcValueGm = GetSrcValueGm(byteRound, valueGm, tempValueGm, outputValueGm);

    // uint64_t h0 = static_cast<uint64_t>(AscendC::GetSystemCycle());
    ProcessHist(byteRound, batchCount, srcValueGm, histGm, pipe);
    // uint64_t h1 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    ComputePrefixAndOffsets(tileCount_, prefixGm, tileOffsetsGm, histGm, pipe);
    // uint64_t p1 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    ProcessScatter(byteRound, batchCount, srcValueGm, indexGm, outputValueGm, outputIndexGm, tempValueGm, tempIndexGm,
                   histGm, prefixGm, tileOffsetsGm, pipe);
    // uint64_t s1 = static_cast<uint64_t>(AscendC::GetSystemCycle());

    // sortRoundHistCycle_[byteRound] += h1 - h0;
    // sortRoundPrefixCycle_[byteRound] += p1 - h1;
    // sortRoundScatterCycle_[byteRound] += s1 - p1;
}

__aicore__ inline void EngramFetchGradSort::Process(uint32_t actualCount, AscendC::TPipe &pipe)
{
    elementCount_ = actualCount;
    uint32_t dynTile = CeilDiv(actualCount, coreCount_);
    if (dynTile > 0 && dynTile < tileElements_) {
        tileElements_ = dynTile;
    }
    tileCount_ = CeilDiv(elementCount_, tileElements_);

    for (uint32_t byteRound = 0; byteRound < byteRounds_; byteRound++) {
        ProcessOneByteRound(byteRound, valueGm_, indexGm_, valueGm_, indexGm_, tempValueGm_, tempIndexGm_, histGm_,
                            prefixGm_, tileOffsetsGm_, pipe);
    }

    if (byteRounds_ % 2 == 1) {
        CopyTempToOutput(tempValueGm_, tempIndexGm_, valueGm_, indexGm_, pipe);
    }
}

} // namespace EngramFetchGradSort
#endif
