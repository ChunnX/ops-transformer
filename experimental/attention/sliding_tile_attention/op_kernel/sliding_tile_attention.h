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

__aicore__ inline float SqrtApprox(float value)
{
    if (value <= 0.0F) {
        return 0.0F;
    }

    float result = value > 1.0F ? value : 1.0F;
    constexpr int64_t iterationNum = 8;
    for (int64_t i = 0; i < iterationNum; ++i) {
        result = 0.5F * (result + value / result);
    }
    return result;
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

__aicore__ inline float InvSqrtApprox(float value)
{
    const float sqrtValue = SqrtApprox(value);
    return sqrtValue > 0.0F ? 1.0F / sqrtValue : 0.0F;
}

__aicore__ inline float Bf16BitsToFloat(uint16_t value)
{
    const int64_t sign = (value >> 15) & 1;
    const int64_t exponent = (value >> 7) & 0xFF;
    const int64_t mantissa = value & 0x7F;
    if (exponent == 0 && mantissa == 0) {
        return sign == 0 ? 0.0F : -0.0F;
    }
    if (exponent == 0xFF) {
        return sign == 0 ? 3.4028234663852886e+38F : -3.4028234663852886e+38F;
    }

    float magnitude = 0.0F;
    if (exponent == 0) {
        magnitude = (static_cast<float>(mantissa) / 128.0F) * Pow2Int(-126);
    } else {
        magnitude = (1.0F + static_cast<float>(mantissa) / 128.0F) * Pow2Int(exponent - 127);
    }
    return sign == 0 ? magnitude : -magnitude;
}

__aicore__ inline uint16_t FloatToBf16Bits(float value)
{
    if (value == 0.0F) {
        return 0;
    }

    uint16_t sign = 0;
    float magnitude = value;
    if (magnitude < 0.0F) {
        sign = 0x8000;
        magnitude = -magnitude;
    }
    if (magnitude >= 3.3895313892515355e+38F) {
        return sign | 0x7F7F;
    }

    int64_t exponent = 0;
    while (magnitude >= 2.0F) {
        magnitude *= 0.5F;
        ++exponent;
    }
    while (magnitude < 1.0F && exponent > -126) {
        magnitude *= 2.0F;
        --exponent;
    }

    if (exponent <= -126 && magnitude < 1.0F) {
        const int64_t subnormalMantissa = static_cast<int64_t>(magnitude * 128.0F + 0.5F);
        return sign | static_cast<uint16_t>(subnormalMantissa > 0x7F ? 0x7F : subnormalMantissa);
    }

    int64_t exponentBits = exponent + 127;
    int64_t mantissa = static_cast<int64_t>((magnitude - 1.0F) * 128.0F + 0.5F);
    if (mantissa >= 128) {
        mantissa = 0;
        ++exponentBits;
    }
    if (exponentBits >= 0xFF) {
        exponentBits = 0xFE;
        mantissa = 0x7F;
    }
    return sign | static_cast<uint16_t>((exponentBits << 7) | mantissa);
}

__aicore__ inline float HalfBitsToFloat(uint16_t value)
{
    const int64_t sign = (value >> 15) & 1;
    const int64_t exponent = (value >> 10) & 0x1F;
    const int64_t mantissa = value & 0x3FF;
    if (exponent == 0 && mantissa == 0) {
        return sign == 0 ? 0.0F : -0.0F;
    }
    if (exponent == 0x1F) {
        return sign == 0 ? 65504.0F : -65504.0F;
    }

    float magnitude = 0.0F;
    if (exponent == 0) {
        magnitude = (static_cast<float>(mantissa) / 1024.0F) * Pow2Int(-14);
    } else {
        magnitude = (1.0F + static_cast<float>(mantissa) / 1024.0F) * Pow2Int(exponent - 15);
    }
    return sign == 0 ? magnitude : -magnitude;
}

__aicore__ inline uint16_t FloatToHalfBits(float value)
{
    if (value == 0.0F) {
        return 0;
    }

    uint16_t sign = 0;
    float magnitude = value;
    if (magnitude < 0.0F) {
        sign = 0x8000;
        magnitude = -magnitude;
    }
    if (magnitude >= 65504.0F) {
        return sign | 0x7BFF;
    }

    int64_t exponent = 0;
    while (magnitude >= 2.0F) {
        magnitude *= 0.5F;
        ++exponent;
    }
    while (magnitude < 1.0F && exponent > -14) {
        magnitude *= 2.0F;
        --exponent;
    }

    if (exponent <= -14 && magnitude < 1.0F) {
        const int64_t subnormalMantissa = static_cast<int64_t>(magnitude * 1024.0F + 0.5F);
        return sign | static_cast<uint16_t>(subnormalMantissa > 0x3FF ? 0x3FF : subnormalMantissa);
    }

    int64_t exponentBits = exponent + 15;
    int64_t mantissa = static_cast<int64_t>((magnitude - 1.0F) * 1024.0F + 0.5F);
    if (mantissa >= 1024) {
        mantissa = 0;
        ++exponentBits;
    }
    if (exponentBits >= 0x1F) {
        exponentBits = 0x1E;
        mantissa = 0x3FF;
    }
    return sign | static_cast<uint16_t>((exponentBits << 10) | mantissa);
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
        qGm_.SetGlobalBuffer((__gm__ uint16_t *)q, total);
        kGm_.SetGlobalBuffer((__gm__ uint16_t *)k, total);
        vGm_.SetGlobalBuffer((__gm__ uint16_t *)v, total);
        yGm_.SetGlobalBuffer((__gm__ uint16_t *)y, total);
        qDataGm_.SetGlobalBuffer((__gm__ DataType *)q, total);
        kDataGm_.SetGlobalBuffer((__gm__ DataType *)k, total);
        vDataGm_.SetGlobalBuffer((__gm__ DataType *)v, total);
        yDataGm_.SetGlobalBuffer((__gm__ DataType *)y, total);
        scale_ = InvSqrtApprox(static_cast<float>(tiling_->headDim));

        pipe_.InitBuffer(qDataBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(kDataBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(vDataBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(yDataBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(qFloatBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(kFloatBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(vFloatBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(tmpFloatBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(accFloatBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceFloatBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceWorkBuf_, VECTOR_HEAD_DIM_LIMIT * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t blockNum = AscendC::GetBlockNum();
        const int64_t blockIdx = AscendC::GetBlockIdx();
        for (int64_t queryLinear = blockIdx; queryLinear < tiling_->totalQueryNum; queryLinear += blockNum) {
            ProcessQuery(queryLinear);
        }
    }

private:
    __aicore__ inline float LoadGmValueAsFloat(GlobalTensor<uint16_t> &gm, int64_t offset)
    {
        if constexpr (IS_BF16) {
            return Bf16BitsToFloat(gm.GetValue(offset));
        } else {
            return HalfBitsToFloat(gm.GetValue(offset));
        }
    }

    __aicore__ inline void SetOutputValue(int64_t offset, float value)
    {
        if constexpr (IS_BF16) {
            yGm_.SetValue(offset, FloatToBf16Bits(value));
        } else {
            yGm_.SetValue(offset, FloatToHalfBits(value));
        }
    }

    __aicore__ inline bool CanUseVectorPath() const
    {
        return tiling_->headDim > 0 && tiling_->headDim <= VECTOR_HEAD_DIM_LIMIT;
    }

    __aicore__ inline uint32_t AlignedHeadDim() const
    {
        return static_cast<uint32_t>(AlignUpInt64(tiling_->headDim, VECTOR_ALIGN_ELEMS));
    }

    __aicore__ inline int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim) const
    {
        return (((batch * tiling_->headNum + head) * tiling_->seqLen + seq) * tiling_->headDim + dim);
    }

    __aicore__ inline float DotScore(int64_t qBase, int64_t kvBase)
    {
        float score = 0.0F;
        for (int64_t dim = 0; dim < tiling_->headDim; ++dim) {
            score += LoadGmValueAsFloat(qGm_, qBase + dim) * LoadGmValueAsFloat(kGm_, kvBase + dim);
        }
        return score * scale_;
    }

    __aicore__ inline void CopyDataRowToLocal(GlobalTensor<DataType> &gm, LocalTensor<DataType> &local,
                                              int64_t offset)
    {
        const uint32_t headDim = static_cast<uint32_t>(tiling_->headDim);
        const uint32_t alignedHeadDim = AlignedHeadDim();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headDim * sizeof(DataType)), 0, 0, 0};
        DataCopyPadExtParams<DataType> padParams{true, 0, static_cast<uint8_t>(alignedHeadDim - headDim),
                                                 DataType(0)};
        DataCopyPad(local, gm[offset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void CopyDataRowToFloat(GlobalTensor<DataType> &gm, TBuf<QuePosition::VECCALC> &dataBuf,
                                             LocalTensor<float> &floatLocal, int64_t offset)
    {
        LocalTensor<DataType> dataLocal = dataBuf.Get<DataType>();
        CopyDataRowToLocal(gm, dataLocal, offset);
        Cast(floatLocal, dataLocal, RoundMode::CAST_NONE, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreFloatRow(int64_t offset, LocalTensor<float> &floatLocal)
    {
        LocalTensor<DataType> dataLocal = yDataBuf_.Get<DataType>();
        if constexpr (IS_BF16) {
            Cast(dataLocal, floatLocal, RoundMode::CAST_RINT, AlignedHeadDim());
        } else {
            Cast(dataLocal, floatLocal, RoundMode::CAST_ROUND, AlignedHeadDim());
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling_->headDim * sizeof(DataType)), 0, 0, 0};
        DataCopyPad(yDataGm_[offset], dataLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    __aicore__ inline void StoreZeroRow(int64_t offset)
    {
        LocalTensor<float> accLocal = accFloatBuf_.Get<float>();
        Duplicate(accLocal, 0.0F, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
        StoreFloatRow(offset, accLocal);
    }

    __aicore__ inline float DotScoreVector(LocalTensor<float> &qLocal, int64_t kvBase)
    {
        LocalTensor<float> kLocal = kFloatBuf_.Get<float>();
        CopyDataRowToFloat(kDataGm_, kDataBuf_, kLocal, kvBase);

        LocalTensor<float> tmpLocal = tmpFloatBuf_.Get<float>();
        Mul(tmpLocal, qLocal, kLocal, AlignedHeadDim());
        PipeBarrier<PIPE_V>();

        LocalTensor<float> reduceLocal = reduceFloatBuf_.Get<float>();
        LocalTensor<float> reduceWorkLocal = reduceWorkBuf_.Get<float>();
        ReduceSum(reduceLocal, tmpLocal, reduceWorkLocal, AlignedHeadDim());
        SetFlag<HardEvent::V_S>(EVENT_ID1);
        WaitFlag<HardEvent::V_S>(EVENT_ID1);
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

    __aicore__ inline float ReduceImageWindowMax(int64_t qBase, int64_t batch, int64_t head, int64_t seq)
    {
        float maxScore = NEG_INF;
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
                            const float score = DotScore(qBase, TensorOffset(batch, head, kvSeq, 0));
                            maxScore = score > maxScore ? score : maxScore;
                        }
                    }
                }
            }
        }
        return maxScore;
    }

    __aicore__ inline float ReduceImageWindowDenom(int64_t qBase, int64_t batch, int64_t head, int64_t seq,
                                                  float maxScore)
    {
        float denom = 0.0F;
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
                            denom += ExpApprox(DotScore(qBase, TensorOffset(batch, head, kvSeq, 0)) - maxScore);
                        }
                    }
                }
            }
        }
        return denom;
    }

    __aicore__ inline float ReduceImageWindowValue(int64_t qBase, int64_t batch, int64_t head, int64_t seq, int64_t dim,
                                                  float maxScore)
    {
        float acc = 0.0F;
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
                            const int64_t kvBase = TensorOffset(batch, head, kvSeq, 0);
                            const float weight = ExpApprox(DotScore(qBase, kvBase) - maxScore);
                            acc += weight * LoadGmValueAsFloat(vGm_, kvBase + dim);
                        }
                    }
                }
            }
        }
        return acc;
    }

    __aicore__ inline void AccumulateOnlineValueVector(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                       int64_t kvBase, float &maxScore, float &denom)
    {
        const float score = DotScoreVector(qLocal, kvBase);
        if (score > maxScore) {
            const float oldScale = ExpApprox(maxScore - score);
            Muls(accLocal, accLocal, oldScale, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
            denom *= oldScale;
            maxScore = score;
        }

        const float weight = ExpApprox(score - maxScore);
        LocalTensor<float> vLocal = vFloatBuf_.Get<float>();
        CopyDataRowToFloat(vDataGm_, vDataBuf_, vLocal, kvBase);

        LocalTensor<float> tmpLocal = tmpFloatBuf_.Get<float>();
        Muls(tmpLocal, vLocal, weight, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
        Add(accLocal, accLocal, tmpLocal, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
        denom += weight;
    }

    __aicore__ inline void AccumulateImageWindowOnlineVector(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                            int64_t batch, int64_t head, int64_t seq, float &maxScore,
                                                            float &denom)
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
                                                        maxScore, denom);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline float ReduceMax(int64_t qBase, int64_t batch, int64_t head, int64_t seq, bool isTextQuery)
    {
        float maxScore = NEG_INF;
        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                const float score = DotScore(qBase, TensorOffset(batch, head, kvSeq, 0));
                maxScore = score > maxScore ? score : maxScore;
            }
        } else {
            maxScore = ReduceImageWindowMax(qBase, batch, head, seq);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    const int64_t kvSeq = tiling_->imageSeqLen + i;
                    const float score = DotScore(qBase, TensorOffset(batch, head, kvSeq, 0));
                    maxScore = score > maxScore ? score : maxScore;
                }
            }
        }
        return maxScore;
    }

    __aicore__ inline float ReduceDenom(int64_t qBase, int64_t batch, int64_t head, int64_t seq, bool isTextQuery,
                                        float maxScore)
    {
        float denom = 0.0F;
        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                denom += ExpApprox(DotScore(qBase, TensorOffset(batch, head, kvSeq, 0)) - maxScore);
            }
        } else {
            denom = ReduceImageWindowDenom(qBase, batch, head, seq, maxScore);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    const int64_t kvSeq = tiling_->imageSeqLen + i;
                    denom += ExpApprox(DotScore(qBase, TensorOffset(batch, head, kvSeq, 0)) - maxScore);
                }
            }
        }
        return denom;
    }

    __aicore__ inline float ReduceValue(int64_t qBase, int64_t batch, int64_t head, int64_t seq, int64_t dim,
                                        bool isTextQuery, float maxScore)
    {
        float acc = 0.0F;
        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                const int64_t kvBase = TensorOffset(batch, head, kvSeq, 0);
                const float weight = ExpApprox(DotScore(qBase, kvBase) - maxScore);
                acc += weight * LoadGmValueAsFloat(vGm_, kvBase + dim);
            }
        } else {
            acc = ReduceImageWindowValue(qBase, batch, head, seq, dim, maxScore);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    const int64_t kvSeq = tiling_->imageSeqLen + i;
                    const int64_t kvBase = TensorOffset(batch, head, kvSeq, 0);
                    const float weight = ExpApprox(DotScore(qBase, kvBase) - maxScore);
                    acc += weight * LoadGmValueAsFloat(vGm_, kvBase + dim);
                }
            }
        }
        return acc;
    }

    __aicore__ inline float ComputeOnlineValueVector(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                     int64_t batch, int64_t head, int64_t seq, bool isTextQuery)
    {
        // Online softmax avoids separate max, denom, and value passes over the same KV window.
        float maxScore = NEG_INF;
        float denom = 0.0F;
        Duplicate(accLocal, 0.0F, AlignedHeadDim());
        PipeBarrier<PIPE_V>();

        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                AccumulateOnlineValueVector(qLocal, accLocal, TensorOffset(batch, head, kvSeq, 0), maxScore, denom);
            }
        } else {
            AccumulateImageWindowOnlineVector(qLocal, accLocal, batch, head, seq, maxScore, denom);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    const int64_t kvSeq = tiling_->imageSeqLen + i;
                    AccumulateOnlineValueVector(qLocal, accLocal, TensorOffset(batch, head, kvSeq, 0), maxScore,
                                                denom);
                }
            }
        }
        return denom;
    }

    __aicore__ inline void ProcessQueryVector(int64_t queryLinear)
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
            StoreZeroRow(qBase);
            return;
        }

        LocalTensor<float> qLocal = qFloatBuf_.Get<float>();
        CopyDataRowToFloat(qDataGm_, qDataBuf_, qLocal, qBase);

        LocalTensor<float> accLocal = accFloatBuf_.Get<float>();
        const float denom = ComputeOnlineValueVector(qLocal, accLocal, batch, head, seq, isTextQuery);
        if (denom > 0.0F) {
            Muls(accLocal, accLocal, 1.0F / denom, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
        } else {
            Duplicate(accLocal, 0.0F, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
        }
        StoreFloatRow(qBase, accLocal);
    }

    __aicore__ inline void ProcessQuery(int64_t queryLinear)
    {
        if (CanUseVectorPath()) {
            ProcessQueryVector(queryLinear);
            return;
        }

        const int64_t seq = queryLinear % tiling_->seqLen;
        const int64_t tmp0 = queryLinear / tiling_->seqLen;
        const int64_t head = tmp0 % tiling_->headNum;
        const int64_t batch = tmp0 / tiling_->headNum;
        const bool isTextQuery = (tiling_->hasText != 0) && (seq >= tiling_->imageSeqLen) &&
                                 (seq < tiling_->imageSeqLen + tiling_->textLength);
        const bool isIgnoredPad = seq >= tiling_->imageSeqLen + tiling_->textLength;
        const int64_t qBase = TensorOffset(batch, head, seq, 0);

        if (isIgnoredPad) {
            for (int64_t dim = 0; dim < tiling_->headDim; ++dim) {
                SetOutputValue(qBase + dim, 0.0F);
            }
            return;
        }

        const float maxScore = ReduceMax(qBase, batch, head, seq, isTextQuery);
        const float denom = ReduceDenom(qBase, batch, head, seq, isTextQuery, maxScore);
        for (int64_t dim = 0; dim < tiling_->headDim; ++dim) {
            const float acc = ReduceValue(qBase, batch, head, seq, dim, isTextQuery, maxScore);
            const float value = denom > 0.0F ? acc / denom : 0.0F;
            SetOutputValue(qBase + dim, value);
        }
    }

private:
    GlobalTensor<uint16_t> qGm_;
    GlobalTensor<uint16_t> kGm_;
    GlobalTensor<uint16_t> vGm_;
    GlobalTensor<uint16_t> yGm_;
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
