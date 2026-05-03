/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef SLIDING_TILE_ATTENTION_H
#define SLIDING_TILE_ATTENTION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sliding_tile_attention_tiling_data.h"

namespace SlidingTileAttentionKernel {
using namespace AscendC;

constexpr float NEG_INF = -3.4028234663852886e+38F;
constexpr float EXP_LOWER_BOUND = -80.0F;
constexpr float EXP_UPPER_BOUND = 80.0F;
constexpr float LN2 = 0.6931471805599453F;
constexpr float INV_LN2 = 1.4426950408889634F;
constexpr int64_t VECTOR_HEAD_DIM_LIMIT = 128;
constexpr int64_t VECTOR_ALIGN_ELEMS = 16;
constexpr int64_t BUFFER_NUM = 2;

template <bool IS_BF16>
struct DataTypeSelector {
    using Type = half;
};

template <>
struct DataTypeSelector<true> {
    using Type = bfloat16_t;
};

__aicore__ inline int64_t MaxInt64(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

__aicore__ inline int64_t MinInt64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

__aicore__ inline int64_t AlignUpInt64(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline int64_t ClampStart(int64_t center, int64_t window, int64_t limit)
{
    if (limit <= window) {
        return 0;
    }
    int64_t start = center - (window >> 1);
    start = MaxInt64(start, 0);
    start = MinInt64(start, limit - window);
    return start;
}

__aicore__ inline float Pow2Int(int64_t exponent)
{
    float result = 1.0F;
    if (exponent >= 0) {
        for (int64_t i = 0; i < exponent; ++i) {
            result *= 2.0F;
        }
    } else {
        for (int64_t i = 0; i < -exponent; ++i) {
            result *= 0.5F;
        }
    }
    return result;
}

__aicore__ inline float FloorFloat(float value)
{
    int64_t intValue = static_cast<int64_t>(value);
    if (static_cast<float>(intValue) > value) {
        --intValue;
    }
    return static_cast<float>(intValue);
}

__aicore__ inline float ExpApprox(float value)
{
    if (value <= EXP_LOWER_BOUND) {
        return 0.0F;
    }
    if (value >= EXP_UPPER_BOUND) {
        value = EXP_UPPER_BOUND;
    }

    const float exponentFloat = FloorFloat(value * INV_LN2);
    const int64_t exponent = static_cast<int64_t>(exponentFloat);
    const float reduced = value - exponentFloat * LN2;
    const float reduced2 = reduced * reduced;
    const float reduced3 = reduced2 * reduced;
    const float reduced4 = reduced3 * reduced;
    const float reduced5 = reduced4 * reduced;
    const float reduced6 = reduced5 * reduced;
    const float expReduced = 1.0F + reduced + 0.5F * reduced2 + 0.1666666667F * reduced3 +
                             0.0416666667F * reduced4 + 0.0083333333F * reduced5 +
                             0.0013888889F * reduced6;
    return Pow2Int(exponent) * expReduced;
}

template <bool IS_BF16>
class SlidingTileAttention {
public:
    using DataType = typename DataTypeSelector<IS_BF16>::Type;

    __aicore__ inline SlidingTileAttention() = default;
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                                const SlidingTileAttentionTilingData *tilingData)
    {
        tiling_ = tilingData;
        const int64_t total = tiling_->batchSize * tiling_->headNum * tiling_->seqLen * tiling_->headDim;
        qDataGm_.SetGlobalBuffer((__gm__ DataType *)q, total);
        kDataGm_.SetGlobalBuffer((__gm__ DataType *)k, total);
        vDataGm_.SetGlobalBuffer((__gm__ DataType *)v, total);
        yDataGm_.SetGlobalBuffer((__gm__ DataType *)y, total);
        scale_ = tiling_->scale;

        pipe_.InitBuffer(qDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(kDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(vDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(yDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(qFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(kFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(vFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(tmpFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(accFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceWorkBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t blockNum = AscendC::GetBlockNum();
        const int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t loopIdx = 0;
        for (int64_t queryLinear = blockIdx; queryLinear < tiling_->totalQueryNum; queryLinear += blockNum) {
            ProcessQuery(queryLinear, static_cast<uint32_t>(loopIdx % BUFFER_NUM));
            ++loopIdx;
        }
    }

private:
    __aicore__ inline uint32_t AlignedHeadDim() const
    {
        return static_cast<uint32_t>(AlignUpInt64(tiling_->headDim, VECTOR_ALIGN_ELEMS));
    }

    __aicore__ inline int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim) const
    {
        return (((batch * tiling_->headNum + head) * tiling_->seqLen + seq) * tiling_->headDim + dim);
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> GetBufferSlice(TBuf<QuePosition::VECCALC> &buf, uint32_t bufferIdx)
    {
        LocalTensor<T> local = buf.Get<T>();
        return local[bufferIdx * VECTOR_HEAD_DIM_LIMIT];
    }

    __aicore__ inline void CopyDataRowToLocal(GlobalTensor<DataType> &gm, LocalTensor<DataType> &local,
                                              int64_t offset, uint32_t bufferIdx)
    {
        const uint32_t headDim = static_cast<uint32_t>(tiling_->headDim);
        const uint32_t alignedHeadDim = AlignedHeadDim();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headDim * sizeof(DataType)), 0, 0, 0};
        DataCopyPadExtParams<DataType> padParams{true, 0, static_cast<uint8_t>(alignedHeadDim - headDim),
                                                 DataType(0)};
        DataCopyPad(local, gm[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
    }

    __aicore__ inline void CopyDataRowToFloat(GlobalTensor<DataType> &gm, TBuf<QuePosition::VECCALC> &dataBuf,
                                             LocalTensor<float> &floatLocal, int64_t offset, uint32_t bufferIdx)
    {
        LocalTensor<DataType> dataLocal = GetBufferSlice<DataType>(dataBuf, bufferIdx);
        CopyDataRowToLocal(gm, dataLocal, offset, bufferIdx);
        Cast(floatLocal, dataLocal, RoundMode::CAST_NONE, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreFloatRow(int64_t offset, LocalTensor<float> &floatLocal, uint32_t bufferIdx)
    {
        LocalTensor<DataType> dataLocal = GetBufferSlice<DataType>(yDataBuf_, bufferIdx);
        if constexpr (IS_BF16) {
            Cast(dataLocal, floatLocal, RoundMode::CAST_RINT, AlignedHeadDim());
        } else {
            Cast(dataLocal, floatLocal, RoundMode::CAST_ROUND, AlignedHeadDim());
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling_->headDim * sizeof(DataType)), 0, 0, 0};
        DataCopyPad(yDataGm_[offset], dataLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
    }

    __aicore__ inline void StoreZeroRow(int64_t offset, uint32_t bufferIdx)
    {
        LocalTensor<float> accLocal = GetBufferSlice<float>(accFloatBuf_, bufferIdx);
        Duplicate(accLocal, 0.0F, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
        StoreFloatRow(offset, accLocal, bufferIdx);
    }

    __aicore__ inline float DotScoreVector(LocalTensor<float> &qLocal, int64_t kvBase, uint32_t bufferIdx)
    {
        LocalTensor<float> kLocal = GetBufferSlice<float>(kFloatBuf_, bufferIdx);
        CopyDataRowToFloat(kDataGm_, kDataBuf_, kLocal, kvBase, bufferIdx);

        LocalTensor<float> tmpLocal = GetBufferSlice<float>(tmpFloatBuf_, bufferIdx);
        Mul(tmpLocal, qLocal, kLocal, AlignedHeadDim());
        PipeBarrier<PIPE_V>();

        LocalTensor<float> reduceLocal = GetBufferSlice<float>(reduceFloatBuf_, bufferIdx);
        LocalTensor<float> reduceWorkLocal = GetBufferSlice<float>(reduceWorkBuf_, bufferIdx);
        ReduceSum(reduceLocal, tmpLocal, reduceWorkLocal, AlignedHeadDim());
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        return reduceLocal.GetValue(0) * scale_;
    }

    __aicore__ inline void DecodeTileCoord(int64_t tileId, int64_t &tileT, int64_t &tileH, int64_t &tileW) const
    {
        const int64_t tilesHW = tiling_->tileNumH * tiling_->tileNumW;
        tileT = tileId / tilesHW;
        const int64_t rem = tileId - tileT * tilesHW;
        tileH = rem / tiling_->tileNumW;
        tileW = rem - tileH * tiling_->tileNumW;
    }

    __aicore__ inline void GetImageWindowRange(int64_t head, int64_t querySeq, int64_t &tBegin, int64_t &tEnd,
                                               int64_t &hBegin, int64_t &hEnd, int64_t &wBegin, int64_t &wEnd) const
    {
        const int64_t tileId = querySeq / tiling_->tileTokenNum;
        int64_t qTileT = 0;
        int64_t qTileH = 0;
        int64_t qTileW = 0;
        DecodeTileCoord(tileId, qTileT, qTileH, qTileW);
        tBegin = ClampStart(qTileT, tiling_->windowT[head], tiling_->tileNumT);
        hBegin = ClampStart(qTileH, tiling_->windowH[head], tiling_->tileNumH);
        wBegin = ClampStart(qTileW, tiling_->windowW[head], tiling_->tileNumW);
        tEnd = MinInt64(tBegin + tiling_->windowT[head], tiling_->tileNumT);
        hEnd = MinInt64(hBegin + tiling_->windowH[head], tiling_->tileNumH);
        wEnd = MinInt64(wBegin + tiling_->windowW[head], tiling_->tileNumW);
    }

    __aicore__ inline void AccumulateOnlineValueVector(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                       int64_t kvBase, float &maxScore, float &denom,
                                                       uint32_t bufferIdx)
    {
        const float score = DotScoreVector(qLocal, kvBase, bufferIdx);
        if (score > maxScore) {
            const float oldScale = ExpApprox(maxScore - score);
            Muls(accLocal, accLocal, oldScale, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
            denom *= oldScale;
            maxScore = score;
        }

        const float weight = ExpApprox(score - maxScore);
        LocalTensor<float> vLocal = GetBufferSlice<float>(vFloatBuf_, bufferIdx);
        CopyDataRowToFloat(vDataGm_, vDataBuf_, vLocal, kvBase, bufferIdx);

        LocalTensor<float> tmpLocal = GetBufferSlice<float>(tmpFloatBuf_, bufferIdx);
        Muls(tmpLocal, vLocal, weight, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
        Add(accLocal, accLocal, tmpLocal, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
        denom += weight;
    }

    __aicore__ inline void AccumulateImageWindowOnlineVector(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                            int64_t batch, int64_t head, int64_t seq, float &maxScore,
                                                            float &denom, uint32_t bufferIdx)
    {
        int64_t tBegin = 0;
        int64_t tEnd = 0;
        int64_t hBegin = 0;
        int64_t hEnd = 0;
        int64_t wBegin = 0;
        int64_t wEnd = 0;
        GetImageWindowRange(head, seq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
        for (int64_t tt = tBegin; tt < tEnd; ++tt) {
            for (int64_t hh = hBegin; hh < hEnd; ++hh) {
                for (int64_t ww = wBegin; ww < wEnd; ++ww) {
                    const int64_t baseTile = ((tt * tiling_->tileNumH + hh) * tiling_->tileNumW + ww) *
                                             tiling_->tileTokenNum;
                    for (int64_t offset = 0; offset < tiling_->tileTokenNum; ++offset) {
                        const int64_t kvSeq = baseTile + offset;
                        if (kvSeq < tiling_->imageSeqLen) {
                            AccumulateOnlineValueVector(qLocal, accLocal, TensorOffset(batch, head, kvSeq, 0),
                                                        maxScore, denom, bufferIdx);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline float ComputeOnlineValueVector(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                     int64_t batch, int64_t head, int64_t seq, bool isTextQuery,
                                                     uint32_t bufferIdx)
    {
        // Online softmax avoids separate max, denom, and value passes over the same KV window.
        float maxScore = NEG_INF;
        float denom = 0.0F;
        Duplicate(accLocal, 0.0F, AlignedHeadDim());
        PipeBarrier<PIPE_V>();

        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                AccumulateOnlineValueVector(qLocal, accLocal, TensorOffset(batch, head, kvSeq, 0), maxScore, denom,
                                            bufferIdx);
            }
        } else {
            AccumulateImageWindowOnlineVector(qLocal, accLocal, batch, head, seq, maxScore, denom, bufferIdx);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    const int64_t kvSeq = tiling_->imageSeqLen + i;
                    AccumulateOnlineValueVector(qLocal, accLocal, TensorOffset(batch, head, kvSeq, 0), maxScore,
                                                denom, bufferIdx);
                }
            }
        }
        return denom;
    }

    __aicore__ inline void ProcessQueryVector(int64_t queryLinear, uint32_t bufferIdx)
    {
        const int64_t seq = queryLinear % tiling_->seqLen;
        const int64_t tmp0 = queryLinear / tiling_->seqLen;
        const int64_t head = tmp0 % tiling_->headNum;
        const int64_t batch = tmp0 / tiling_->headNum;
        const bool isTextQuery = (tiling_->hasText != 0) && (seq >= tiling_->imageSeqLen) &&
                                 (seq < tiling_->imageSeqLen + tiling_->textLength);
        const bool isIgnoredPad = seq >= tiling_->imageSeqLen + tiling_->textLength;
        const int64_t qBase = TensorOffset(batch, head, seq, 0);

        if (isIgnoredPad) {
            StoreZeroRow(qBase, bufferIdx);
            return;
        }

        LocalTensor<float> qLocal = GetBufferSlice<float>(qFloatBuf_, bufferIdx);
        CopyDataRowToFloat(qDataGm_, qDataBuf_, qLocal, qBase, bufferIdx);

        LocalTensor<float> accLocal = GetBufferSlice<float>(accFloatBuf_, bufferIdx);
        const float denom = ComputeOnlineValueVector(qLocal, accLocal, batch, head, seq, isTextQuery, bufferIdx);
        if (denom > 0.0F) {
            Muls(accLocal, accLocal, 1.0F / denom, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
        } else {
            Duplicate(accLocal, 0.0F, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
        }
        StoreFloatRow(qBase, accLocal, bufferIdx);
    }

    __aicore__ inline void ProcessQuery(int64_t queryLinear, uint32_t bufferIdx)
    {
        ProcessQueryVector(queryLinear, bufferIdx);
    }

private:
    GlobalTensor<DataType> qDataGm_;
    GlobalTensor<DataType> kDataGm_;
    GlobalTensor<DataType> vDataGm_;
    GlobalTensor<DataType> yDataGm_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> qDataBuf_;
    TBuf<QuePosition::VECCALC> kDataBuf_;
    TBuf<QuePosition::VECCALC> vDataBuf_;
    TBuf<QuePosition::VECCALC> yDataBuf_;
    TBuf<QuePosition::VECCALC> qFloatBuf_;
    TBuf<QuePosition::VECCALC> kFloatBuf_;
    TBuf<QuePosition::VECCALC> vFloatBuf_;
    TBuf<QuePosition::VECCALC> tmpFloatBuf_;
    TBuf<QuePosition::VECCALC> accFloatBuf_;
    TBuf<QuePosition::VECCALC> reduceFloatBuf_;
    TBuf<QuePosition::VECCALC> reduceWorkBuf_;
    const SlidingTileAttentionTilingData *tiling_ = nullptr;
    float scale_ = 0.0F;
};
} // namespace SlidingTileAttentionKernel

#endif // SLIDING_TILE_ATTENTION_H
