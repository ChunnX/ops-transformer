#ifndef SORT_WITH_INDEX_INT32_H
#define SORT_WITH_INDEX_INT32_H
/*
 * Minimal 1D LSD radix sort with index — positive int32 values.
 */

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

namespace SortWithIndexInt32 {

using namespace AscendC;

enum : uint32_t {
    HISTOGRAM_BINS     = 256,
    DEFAULT_TILE_SIZE  = 4096,
    BITS_PER_BYTE      = 8,
    BYTES_PER_INT32    = 4,
    VECTOR_LEN_INT32   = 64,
    VECTOR_LEN_INT16   = 128,
    VECTOR_LEN_INT8    = 256,
};

constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)  { return a < b ? a : b; }
__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1U) / b; }
__aicore__ inline uint32_t AlignUpU32(uint32_t bytes) { return (bytes + UB_BLOCK_BYTES - 1U) / UB_BLOCK_BYTES * UB_BLOCK_BYTES; }

template <HardEvent evt>
__aicore__ inline void WaitPipe(TPipe& pipe)
{
    event_t e = static_cast<event_t>(pipe.FetchEventID(evt));
    SetFlag<evt>(e);
    WaitFlag<evt>(e);
}

// ---- SIMD histogram: count byte-key frequencies for one tile ----
__aicore__ inline void ComputeTileHistogram(
    __local_mem__ uint32_t* valueUb, uint32_t elementCount, uint32_t byteRound,
    __local_mem__ uint16_t* histogramUb, __local_mem__ uint16_t* cumulativeUb,
    __local_mem__ uint8_t* keyUb, __local_mem__ uint32_t* exclusiveUb)
{
    uint32_t shiftBits = byteRound * BITS_PER_BYTE;
    __VEC_SCOPE__ {
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
            Reg::DeInterleave(d0, d1, (Reg::RegTensor<uint16_t>&)s0, (Reg::RegTensor<uint16_t>&)s1);
            Reg::DeInterleave(d2, d3, (Reg::RegTensor<uint16_t>&)s2, (Reg::RegTensor<uint16_t>&)s3);
            Reg::RegTensor<uint8_t> b0, b1;
            Reg::DeInterleave(b0, b1, (Reg::RegTensor<uint8_t>&)d0, (Reg::RegTensor<uint8_t>&)d2);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(keyUb, b0, VECTOR_LEN_INT8, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::FREQUENCY>(hist0, b0, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::FREQUENCY>(hist1, b0, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN0, Reg::HistogramsType::ACCUMULATE>(cumu0, b0, elemMask);
            Reg::Histograms<uint8_t, uint16_t, Reg::HistogramsBinType::BIN1, Reg::HistogramsType::ACCUMULATE>(cumu1, b0, elemMask);
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
        Reg::Interleave((Reg::RegTensor<uint16_t>&)sum0, (Reg::RegTensor<uint16_t>&)sum1, excl0, zero);
        Reg::Interleave((Reg::RegTensor<uint16_t>&)sum2, (Reg::RegTensor<uint16_t>&)sum3, excl1, zero);
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

// ===================================================================
// SORT KERNEL
// ===================================================================
struct SortWithIndexKernel {
    TBuf<TPosition::VECCALC> valueBuffer;
    TBuf<TPosition::VECCALC> indexBuffer;
    TBuf<TPosition::VECCALC> keyBuffer;
    TBuf<TPosition::VECCALC> histogramBuffer;
    TBuf<TPosition::VECCALC> cumulativeBuffer;
    TBuf<TPosition::VECCALC> exclusiveBuffer;
    TBuf<TPosition::VECCALC> prefixBuffer;
    TBuf<TPosition::VECCALC> sortTempBuffer;
    TBuf<TPosition::VECCALC> sortedKeyBuffer;
    TBuf<TPosition::VECCALC> histWideBuffer;
    TQue<QuePosition::VECCALC, 1> prefixQueue;

    uint32_t elementCount;
    uint32_t tileElements;
    uint32_t tileCount;
    uint32_t coreCount;

    static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

    __aicore__ inline void Init(uint32_t totalElements, uint32_t numCores, uint32_t tileSize,
                                TPipe& pipe)
    {
        elementCount = totalElements;
        coreCount    = numCores;
        tileElements = tileSize > 0 ? tileSize : DEFAULT_TILE_SIZE;
        tileCount    = CeilDiv(elementCount, tileElements);

        pipe.InitBuffer(valueBuffer,      static_cast<uint32_t>(AlignUpU32(tileElements * sizeof(int32_t))));
        pipe.InitBuffer(indexBuffer,      static_cast<uint32_t>(AlignUpU32(tileElements * sizeof(int32_t))));
        pipe.InitBuffer(keyBuffer,        tileElements);
        pipe.InitBuffer(histogramBuffer,  HISTOGRAM_BINS * sizeof(int32_t));
        pipe.InitBuffer(cumulativeBuffer, HISTOGRAM_BINS * sizeof(uint16_t));
        pipe.InitBuffer(exclusiveBuffer,  static_cast<uint32_t>(AlignUpU32(tileElements * sizeof(uint32_t))));
        pipe.InitBuffer(prefixBuffer,     HISTOGRAM_BINS * sizeof(int32_t));
        pipe.InitBuffer(sortTempBuffer,   static_cast<uint32_t>(AlignUpU32(tileElements * 4)));
        pipe.InitBuffer(sortedKeyBuffer,  tileElements);
        pipe.InitBuffer(histWideBuffer,   HISTOGRAM_BINS * sizeof(int32_t));
        pipe.InitBuffer(prefixQueue, 1, HISTOGRAM_BINS * sizeof(int32_t));
    }

    __aicore__ inline void SetCount(uint32_t n) { elementCount = n; tileCount = CeilDiv(n, tileElements); }

    __aicore__ inline void ComputePrefixAndOffsets(
        uint32_t tc, GlobalTensor<int32_t>& prefixGm, GlobalTensor<int32_t>& tileOffsetsGm,
        GlobalTensor<int32_t>& histGm, TPipe& pipe)
    {
        AscendC::printf("Prefix START: tileCount=%u\n", tc);
        LocalTensor<int32_t> histogram = prefixQueue.AllocTensor<int32_t>();
        DataCopyExtParams param{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> padParam{false, 0, 0, 0};

        int32_t prefix[256];
        for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) prefix[b] = 0;

        event_t evtMte2S = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
        for (uint32_t t = 0; t < tc; t++) {
            DataCopyPad(histogram, histGm[t * HISTOGRAM_BINS], param, padParam);
            SetFlag<HardEvent::MTE2_S>(evtMte2S);
            WaitFlag<HardEvent::MTE2_S>(evtMte2S);
            for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
                tileOffsetsGm.SetValue(t * HISTOGRAM_BINS + b, prefix[b]);
                prefix[b] += histogram.GetValue(b);
            }
        }

        int32_t running = 0;
        for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) {
            int32_t count = prefix[b];
            prefix[b] = running;
            running += count;
        }

        for (uint32_t b = 0; b < HISTOGRAM_BINS; b++) histogram.SetValue(b, prefix[b]);
        WaitPipe<HardEvent::S_MTE3>(pipe);
        DataCopyParams outParam{1U, static_cast<uint16_t>(HISTOGRAM_BINS * sizeof(int32_t)), 0U, 0U};
        DataCopyPad(prefixGm, histogram, outParam);
        WaitPipe<HardEvent::MTE3_S>(pipe);
        prefixQueue.FreeTensor(histogram);
        AscendC::printf("Prefix DONE: total=%d\n", running);
    }

    __aicore__ inline void ProcessOneByteRound(
        uint32_t byteRound,
        GlobalTensor<int32_t>& valueGm, GlobalTensor<int32_t>& indexGm,
        GlobalTensor<int32_t>& outputValueGm, GlobalTensor<int32_t>& outputIndexGm,
        GlobalTensor<int32_t>& tempValueGm, GlobalTensor<int32_t>& tempIndexGm,
        GlobalTensor<int32_t>& histGm, GlobalTensor<int32_t>& prefixGm,
        GlobalTensor<int32_t>& tileOffsetsGm, TPipe& pipe)
    {
        uint32_t batchCount = CeilDiv(tileCount, coreCount);
        GlobalTensor<int32_t>* srcValueGm;
        if (byteRound == 0) {
            srcValueGm = &valueGm;
        } else if (byteRound % 2 == 1) {
            srcValueGm = &tempValueGm;
        } else {
            srcValueGm = &outputValueGm;
        }

        DataCopyExtParams histParam{1U, HISTOGRAM_BINS * sizeof(int32_t), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> valPad{false, 0, 0, 0};

        event_t evtMte2V = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        event_t evtVS = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        event_t evtSMte3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_MTE3));

        for (uint32_t batch = 0; batch < batchCount; batch++) {
            uint32_t firstTile  = batch * coreCount;
            uint32_t batchCores = MinU32(tileCount - firstTile, coreCount);
            uint32_t coreId     = GetBlockIdx();

            if (coreId < batchCores) {
                uint32_t tileId  = firstTile + coreId;
                uint32_t offset  = tileId * tileElements;
                uint32_t tileLen = MinU32(elementCount - offset, tileElements);

                LocalTensor<int32_t>  valueLocal     = valueBuffer.Get<int32_t>();
                LocalTensor<uint16_t> histLocal      = histogramBuffer.Get<uint16_t>();
                LocalTensor<uint16_t> cumsumLocal    = cumulativeBuffer.Get<uint16_t>();
                LocalTensor<uint32_t> exclusiveLocal = exclusiveBuffer.Get<uint32_t>();
                LocalTensor<uint8_t>  keyLocal       = keyBuffer.Get<uint8_t>();

                uint32_t valBytes = tileLen * sizeof(int32_t);
                DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
                DataCopyPad(valueLocal, (*srcValueGm)[offset], valParam, valPad);
                SetFlag<HardEvent::MTE2_V>(evtMte2V);
                WaitFlag<HardEvent::MTE2_V>(evtMte2V);

                Duplicate(exclusiveLocal, (uint32_t)0, HISTOGRAM_BINS);

                ComputeTileHistogram(
                    (__local_mem__ uint32_t*)valueLocal.GetPhyAddr(), tileLen, byteRound,
                    (__local_mem__ uint16_t*)histLocal.GetPhyAddr(),
                    (__local_mem__ uint16_t*)cumsumLocal.GetPhyAddr(),
                    (__local_mem__ uint8_t*)keyLocal.GetPhyAddr(),
                    (__local_mem__ uint32_t*)exclusiveLocal.GetPhyAddr());

                SetFlag<HardEvent::V_S>(evtVS);
                WaitFlag<HardEvent::V_S>(evtVS);

                LocalTensor<int32_t> histWide = histWideBuffer.Get<int32_t>();
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

        AscendC::printf("Sort round=%u HIST done, block=%u\n", byteRound, GetBlockIdx());

        if (GetBlockIdx() == 0) {
            ComputePrefixAndOffsets(tileCount, prefixGm, tileOffsetsGm, histGm, pipe);
        }
        SyncAll<true>();

        AscendC::printf("Sort round=%u PREFIX done, block=%u\n", byteRound, GetBlockIdx());

        event_t evtMte2S = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_S));
        event_t evtSV = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
        event_t evtVS2 = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));

        for (uint32_t batch = 0; batch < batchCount; batch++) {
            uint32_t firstTile  = batch * coreCount;
            uint32_t batchCores = MinU32(tileCount - firstTile, coreCount);
            uint32_t coreId     = GetBlockIdx();

            if (coreId < batchCores) {
                uint32_t tileId  = firstTile + coreId;
                uint32_t offset  = tileId * tileElements;
                uint32_t tileLen = MinU32(elementCount - offset, tileElements);

                GlobalTensor<int32_t>* srcIdx;
                GlobalTensor<int32_t>* dstValue;
                GlobalTensor<int32_t>* dstIndex;
                if (byteRound == 0) {
                    srcIdx   = &indexGm;
                    dstValue = &tempValueGm;
                    dstIndex = &tempIndexGm;
                } else if (byteRound % 2 == 1) {
                    srcIdx   = &tempIndexGm;
                    dstValue = &outputValueGm;
                    dstIndex = &outputIndexGm;
                } else {
                    srcIdx   = &outputIndexGm;
                    dstValue = &tempValueGm;
                    dstIndex = &tempIndexGm;
                }

                LocalTensor<int32_t>  valueLocal  = valueBuffer.Get<int32_t>();
                LocalTensor<int32_t>  indexLocal  = indexBuffer.Get<int32_t>();
                LocalTensor<uint8_t>  keyLocal    = keyBuffer.Get<uint8_t>();
                LocalTensor<uint8_t>  sortedKey   = sortedKeyBuffer.Get<uint8_t>();
                LocalTensor<uint32_t> sortIndices = exclusiveBuffer.Get<uint32_t>();
                LocalTensor<int32_t>  prefixLocal = prefixBuffer.Get<int32_t>();
                LocalTensor<int32_t>  offsetLocal = histogramBuffer.Get<int32_t>();

                uint32_t valBytes = tileLen * sizeof(int32_t);
                DataCopyExtParams valParam{1U, valBytes, 0U, 0U, 0U};
                DataCopyPad(valueLocal,  (*srcValueGm)[offset], valParam, valPad);
                DataCopyPad(indexLocal,  (*srcIdx)[offset], valParam, valPad);
                DataCopyPad(prefixLocal, prefixGm, histParam, valPad);
                DataCopyPad(offsetLocal, tileOffsetsGm[tileId * HISTOGRAM_BINS], histParam, valPad);

                SetFlag<HardEvent::MTE2_S>(evtMte2S);
                WaitFlag<HardEvent::MTE2_S>(evtMte2S);

                for (uint32_t i = 0; i < tileLen; i++)
                    keyLocal.SetValue(i, (uint8_t)((valueLocal.GetValue(i) >> (byteRound * BITS_PER_BYTE)) & 0xFF));

                SetFlag<HardEvent::S_V>(evtSV);
                WaitFlag<HardEvent::S_V>(evtSV);

                LocalTensor<uint8_t> sortTemp = sortTempBuffer.Get<uint8_t>();
                AscendC::printf("Sort round=%u SCATTER before Sort: block=%u, tileId=%u, tileLen=%u\n",
                                byteRound, GetBlockIdx(), tileId, tileLen);
                Sort<uint8_t, false, sortConfig>(sortedKey, sortIndices, keyLocal, sortTemp, tileLen);

                SetFlag<HardEvent::V_S>(evtVS2);
                WaitFlag<HardEvent::V_S>(evtVS2);

                AscendC::printf("Sort round=%u SCATTER after Sort: block=%u, tileId=%u\n",
                                byteRound, GetBlockIdx(), tileId);
                for (uint32_t i = 0; i < tileLen; i++) {
                    uint32_t sortedIdx = sortIndices.GetValue(i);
                    uint8_t  key       = keyLocal.GetValue(sortedIdx);
                    int32_t  binOffset = offsetLocal.GetValue(key);
                    int32_t  position  = prefixLocal.GetValue(key) + binOffset;
                    (*dstValue).SetValue(position, valueLocal.GetValue(sortedIdx));
                    (*dstIndex).SetValue(position, indexLocal.GetValue(sortedIdx));
                    offsetLocal.SetValue(key, binOffset + 1);
                }
            }
            WaitPipe<HardEvent::S_MTE3>(pipe);
            SyncAll<true>();
        }
    }

    __aicore__ inline void Process(
        GlobalTensor<int32_t>& valueGm, GlobalTensor<int32_t>& indexGm,
        GlobalTensor<int32_t>& outputValueGm, GlobalTensor<int32_t>& outputIndexGm,
        GlobalTensor<int32_t>& tempValueGm, GlobalTensor<int32_t>& tempIndexGm,
        GlobalTensor<int32_t>& histGm, GlobalTensor<int32_t>& prefixGm,
        GlobalTensor<int32_t>& tileOffsetsGm, TPipe& pipe)
    {
        for (uint32_t byteRound = 0; byteRound < BYTES_PER_INT32; byteRound++) {
            AscendC::printf("Sort ROUND %u START: block=%u, tileCount=%u, batchCount=%u\n",
                            byteRound, GetBlockIdx(), tileCount,
                            CeilDiv(tileCount, coreCount));
            ProcessOneByteRound(byteRound, valueGm, indexGm, outputValueGm, outputIndexGm,
                                tempValueGm, tempIndexGm, histGm, prefixGm, tileOffsetsGm, pipe);
            AscendC::printf("Sort ROUND %u DONE: block=%u\n", byteRound, GetBlockIdx());
        }
        AscendC::printf("Sort ALL ROUNDS DONE: block=%u\n", GetBlockIdx());
    }
};

} // namespace SortWithIndexInt32
#endif
