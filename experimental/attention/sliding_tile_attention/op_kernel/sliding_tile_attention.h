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
    __aicore__ inline float ToFloat(T value) const
    {
        return static_cast<float>(value);
    }

    __aicore__ inline int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim) const
    {
        return (((batch * tiling_->headNum + head) * tiling_->seqLen + seq) * tiling_->headDim + dim);
    }

    __aicore__ inline float DotScore(int64_t qBase, int64_t kvBase) const
    {
        float score = 0.0F;
        for (int64_t dim = 0; dim < tiling_->headDim; ++dim) {
            score += ToFloat(qGm_.GetValue(qBase + dim)) * ToFloat(kGm_.GetValue(kvBase + dim));
        }
        return score / sqrtf(static_cast<float>(tiling_->headDim));
    }

    __aicore__ inline void DecodeTileCoord(int64_t tileId, int64_t &tileT, int64_t &tileH, int64_t &tileW) const
    {
        const int64_t tilesHW = tiling_->tileNumH * tiling_->tileNumW;
        tileT = tileId / tilesHW;
        const int64_t rem = tileId - tileT * tilesHW;
        tileH = rem / tiling_->tileNumW;
        tileW = rem - tileH * tiling_->tileNumW;
    }

    template <typename Func>
    __aicore__ inline void VisitImageWindow(int64_t head, int64_t querySeq, Func func) const
    {
        const int64_t tileId = querySeq / tiling_->tileTokenNum;
        int64_t qTileT = 0;
        int64_t qTileH = 0;
        int64_t qTileW = 0;
        DecodeTileCoord(tileId, qTileT, qTileH, qTileW);
        const int64_t tBegin = ClampStart(qTileT, tiling_->windowT[head], tiling_->tileNumT);
        const int64_t hBegin = ClampStart(qTileH, tiling_->windowH[head], tiling_->tileNumH);
        const int64_t wBegin = ClampStart(qTileW, tiling_->windowW[head], tiling_->tileNumW);
        const int64_t tEnd = MinInt64(tBegin + tiling_->windowT[head], tiling_->tileNumT);
        const int64_t hEnd = MinInt64(hBegin + tiling_->windowH[head], tiling_->tileNumH);
        const int64_t wEnd = MinInt64(wBegin + tiling_->windowW[head], tiling_->tileNumW);
        for (int64_t tt = tBegin; tt < tEnd; ++tt) {
            for (int64_t hh = hBegin; hh < hEnd; ++hh) {
                for (int64_t ww = wBegin; ww < wEnd; ++ww) {
                    const int64_t baseTile = ((tt * tiling_->tileNumH + hh) * tiling_->tileNumW + ww) *
                                             tiling_->tileTokenNum;
                    for (int64_t offset = 0; offset < tiling_->tileTokenNum; ++offset) {
                        const int64_t kvSeq = baseTile + offset;
                        if (kvSeq < tiling_->imageSeqLen) {
                            func(kvSeq);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline float ReduceMax(int64_t qBase, int64_t batch, int64_t head, int64_t seq, bool isTextQuery) const
    {
        float maxScore = NEG_INF;
        auto update = [&](int64_t kvSeq) {
            const float score = DotScore(qBase, TensorOffset(batch, head, kvSeq, 0));
            maxScore = score > maxScore ? score : maxScore;
        };
        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                update(kvSeq);
            }
        } else {
            VisitImageWindow(head, seq, update);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    update(tiling_->imageSeqLen + i);
                }
            }
        }
        return maxScore;
    }

    __aicore__ inline float ReduceDenom(int64_t qBase, int64_t batch, int64_t head, int64_t seq, bool isTextQuery,
                                        float maxScore) const
    {
        float denom = 0.0F;
        auto update = [&](int64_t kvSeq) {
            denom += expf(DotScore(qBase, TensorOffset(batch, head, kvSeq, 0)) - maxScore);
        };
        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                update(kvSeq);
            }
        } else {
            VisitImageWindow(head, seq, update);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    update(tiling_->imageSeqLen + i);
                }
            }
        }
        return denom;
    }

    __aicore__ inline float ReduceValue(int64_t qBase, int64_t batch, int64_t head, int64_t seq, int64_t dim,
                                        bool isTextQuery, float maxScore) const
    {
        float acc = 0.0F;
        auto update = [&](int64_t kvSeq) {
            const int64_t kvBase = TensorOffset(batch, head, kvSeq, 0);
            const float weight = expf(DotScore(qBase, kvBase) - maxScore);
            acc += weight * ToFloat(vGm_.GetValue(kvBase + dim));
        };
        if (isTextQuery) {
            for (int64_t kvSeq = 0; kvSeq < tiling_->imageSeqLen + tiling_->textLength; ++kvSeq) {
                update(kvSeq);
            }
        } else {
            VisitImageWindow(head, seq, update);
            if (tiling_->hasText != 0) {
                for (int64_t i = 0; i < tiling_->textLength; ++i) {
                    update(tiling_->imageSeqLen + i);
                }
            }
        }
        return acc;
    }

    __aicore__ inline void ProcessQuery(int64_t queryLinear) const
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
                yGm_.SetValue(qBase + dim, static_cast<T>(0.0F));
            }
            return;
        }

        const float maxScore = ReduceMax(qBase, batch, head, seq, isTextQuery);
        const float denom = ReduceDenom(qBase, batch, head, seq, isTextQuery, maxScore);
        for (int64_t dim = 0; dim < tiling_->headDim; ++dim) {
            const float acc = ReduceValue(qBase, batch, head, seq, dim, isTextQuery, maxScore);
            const float value = denom > 0.0F ? acc / denom : 0.0F;
            yGm_.SetValue(qBase + dim, static_cast<T>(value));
        }
    }

private:
    GlobalTensor<T> qGm_;
    GlobalTensor<T> kGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> yGm_;
    const SlidingTileAttentionTilingData *tiling_ = nullptr;
};
} // namespace SlidingTileAttentionKernel

#endif // SLIDING_TILE_ATTENTION_H
