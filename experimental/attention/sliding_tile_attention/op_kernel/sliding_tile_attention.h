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
constexpr int64_t SCALAR_CAST_LEN = 16;

__aicore__ inline int64_t MaxInt64(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

__aicore__ inline int64_t MinInt64(int64_t a, int64_t b)
{
    return a < b ? a : b;
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

template <typename T>
class SlidingTileAttention {
public:
    __aicore__ inline SlidingTileAttention() = default;
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                                const SlidingTileAttentionTilingData *tilingData)
    {
        tiling_ = tilingData;
        const int64_t total = tiling_->batchSize * tiling_->headNum * tiling_->seqLen * tiling_->headDim;
        qGm_.SetGlobalBuffer((__gm__ T *)q, total);
        kGm_.SetGlobalBuffer((__gm__ T *)k, total);
        vGm_.SetGlobalBuffer((__gm__ T *)v, total);
        yGm_.SetGlobalBuffer((__gm__ T *)y, total);
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            qRawGm_.SetGlobalBuffer((__gm__ uint16_t *)q, total);
            kRawGm_.SetGlobalBuffer((__gm__ uint16_t *)k, total);
            vRawGm_.SetGlobalBuffer((__gm__ uint16_t *)v, total);
            yRawGm_.SetGlobalBuffer((__gm__ uint16_t *)y, total);
        }
        scale_ = InvSqrtApprox(static_cast<float>(tiling_->headDim));
        pipe_.InitBuffer(outFloatBuf_, SCALAR_CAST_LEN * sizeof(float));
        pipe_.InitBuffer(outCastBuf_, SCALAR_CAST_LEN * sizeof(T));
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
    __aicore__ inline float LoadGmValueAsFloat(GlobalTensor<T> &gm, GlobalTensor<uint16_t> &rawGm, int64_t offset)
    {
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            return Bf16BitsToFloat(rawGm.GetValue(offset));
        } else {
            return AscendC::ToFloat(gm.GetValue(offset));
        }
    }

    __aicore__ inline void SetOutputValue(int64_t offset, float value)
    {
        if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
            yRawGm_.SetValue(offset, FloatToBf16Bits(value));
            return;
        }
        LocalTensor<float> outFloat = outFloatBuf_.Get<float>();
        LocalTensor<T> outCast = outCastBuf_.Get<T>();
        Duplicate(outFloat, value, SCALAR_CAST_LEN);
        Cast(outCast, outFloat, RoundMode::CAST_NONE, SCALAR_CAST_LEN);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[offset], outCast, copyParams);
    }

    __aicore__ inline int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim) const
    {
        return (((batch * tiling_->headNum + head) * tiling_->seqLen + seq) * tiling_->headDim + dim);
    }

    __aicore__ inline float DotScore(int64_t qBase, int64_t kvBase)
    {
        float score = 0.0F;
        for (int64_t dim = 0; dim < tiling_->headDim; ++dim) {
            score += LoadGmValueAsFloat(qGm_, qRawGm_, qBase + dim) *
                     LoadGmValueAsFloat(kGm_, kRawGm_, kvBase + dim);
        }
        return score * scale_;
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
                            acc += weight * LoadGmValueAsFloat(vGm_, vRawGm_, kvBase + dim);
                        }
                    }
                }
            }
        }
        return acc;
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
                acc += weight * LoadGmValueAsFloat(vGm_, vRawGm_, kvBase + dim);
            }
        } else {
            acc = ReduceImageWindowValue(qBase, batch, head, seq, dim, maxScore);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    const int64_t kvSeq = tiling_->imageSeqLen + i;
                    const int64_t kvBase = TensorOffset(batch, head, kvSeq, 0);
                    const float weight = ExpApprox(DotScore(qBase, kvBase) - maxScore);
                    acc += weight * LoadGmValueAsFloat(vGm_, vRawGm_, kvBase + dim);
                }
            }
        }
        return acc;
    }

    __aicore__ inline void ProcessQuery(int64_t queryLinear)
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
    TPipe pipe_;
    TBuf<TPosition::VECCALC> outFloatBuf_;
    TBuf<TPosition::VECCALC> outCastBuf_;
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<uint16_t> qRawGm_;
    GlobalTensor<uint16_t> kRawGm_;
    GlobalTensor<uint16_t> vRawGm_;
    GlobalTensor<uint16_t> yRawGm_;
    const SlidingTileAttentionTilingData *tiling_ = nullptr;
    float scale_ = 0.0F;
};
} // namespace SlidingTileAttentionKernel

#endif // SLIDING_TILE_ATTENTION_H
