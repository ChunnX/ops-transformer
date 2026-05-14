/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#ifndef SLIDING_TILE_ATTENTION_H
#define SLIDING_TILE_ATTENTION_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sliding_tile_attention_tiling_data.h"

#ifndef STA_USE_VECIN_SCORE_HANDOFF
#define STA_USE_VECIN_SCORE_HANDOFF 1
#endif

#ifndef STA_USE_VECIN_PROB_OUT_HANDOFF
#define STA_USE_VECIN_PROB_OUT_HANDOFF 0
#endif

namespace SlidingTileAttentionKernel {
using namespace AscendC;
using namespace matmul;

constexpr float NEG_INF = -3.4028234663852886e+38F;
constexpr float EXP_LOWER_BOUND = -80.0F;
constexpr float EXP_UPPER_BOUND = 80.0F;
constexpr float LN2 = 0.6931471805599453F;
constexpr float INV_LN2 = 1.4426950408889634F;
constexpr int64_t VECTOR_HEAD_DIM_LIMIT = 128;
constexpr int64_t VECTOR_ALIGN_ELEMS = 16;
constexpr int64_t VECTOR_FP32_BLOCK_ELEMS = 8;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t KV_BLOCK_NUM = 128;
constexpr int64_t SCORE_HANDOFF_UB_OFFSET = 128 * 1024;
constexpr uint64_t SYNC_AIV_TO_AIC_0 = 3;
constexpr uint64_t SYNC_AIV_TO_AIC_1 = 7;
constexpr uint64_t SYNC_AIC_TO_AIV_0 = 5;
constexpr uint64_t SYNC_AIC_TO_AIV_1 = 9;
constexpr uint64_t SYNC_AIV_PROB_TO_AIC_0 = 11;
constexpr uint64_t SYNC_AIV_PROB_TO_AIC_1 = 13;
constexpr uint64_t SYNC_AIC_OUT_TO_AIV_0 = 15;
constexpr uint64_t SYNC_AIC_OUT_TO_AIV_1 = 17;

__aicore__ inline uint64_t AivToAicSyncFlag(uint32_t bufferIdx)
{
    return bufferIdx == 0 ? SYNC_AIV_TO_AIC_0 : SYNC_AIV_TO_AIC_1;
}

__aicore__ inline uint64_t AicToAivSyncFlag(uint32_t bufferIdx)
{
    return bufferIdx == 0 ? SYNC_AIC_TO_AIV_0 : SYNC_AIC_TO_AIV_1;
}

__aicore__ inline uint64_t AivProbToAicSyncFlag(uint32_t bufferIdx)
{
    return bufferIdx == 0 ? SYNC_AIV_PROB_TO_AIC_0 : SYNC_AIV_PROB_TO_AIC_1;
}

__aicore__ inline uint64_t AicOutToAivSyncFlag(uint32_t bufferIdx)
{
    return bufferIdx == 0 ? SYNC_AIC_OUT_TO_AIV_0 : SYNC_AIC_OUT_TO_AIV_1;
}

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

__aicore__ inline void DuplicateTailFp32(LocalTensor<float> &tensor, int64_t begin, uint32_t end, float value)
{
    if (begin >= static_cast<int64_t>(end)) {
        return;
    }

    int64_t offset = begin;
    const int64_t lane = offset % VECTOR_FP32_BLOCK_ELEMS;
    if (lane != 0) {
        const int64_t alignedOffset = offset - lane;
        const uint32_t firstCount =
            static_cast<uint32_t>(MinInt64(VECTOR_FP32_BLOCK_ELEMS - lane, static_cast<int64_t>(end) - offset));
        uint64_t mask[2] = {((1ULL << firstCount) - 1ULL) << static_cast<uint32_t>(lane), 0};
        Duplicate(tensor[alignedOffset], value, mask, 1, 1, VECTOR_FP32_BLOCK_ELEMS);
        offset += firstCount;
    }

    if (offset < static_cast<int64_t>(end)) {
        Duplicate(tensor[offset], value, end - static_cast<uint32_t>(offset));
    }
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
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const SlidingTileAttentionTilingData *tilingData)
    {
        tiling_ = tilingData;
        const int64_t total = tiling_->batchSize * tiling_->headNum * tiling_->seqLen * tiling_->headDim;
        qDataGm_.SetGlobalBuffer((__gm__ DataType *)q, total);
        kDataGm_.SetGlobalBuffer((__gm__ DataType *)k, total);
        vDataGm_.SetGlobalBuffer((__gm__ DataType *)v, total);
        yDataGm_.SetGlobalBuffer((__gm__ DataType *)y, total);
        scale_ = tiling_->scale;

        const int64_t cubeCoreIdx = GetCubeCoreIdx();
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        __gm__ uint8_t *base = (__gm__ uint8_t *)workspace + cubeCoreIdx * tiling_->cubeWorkspacePerCore;
        scoreGm_.SetGlobalBuffer((__gm__ float *)base, scoreElems * BUFFER_NUM);
        probGm_.SetGlobalBuffer((__gm__ DataType *)(base + scoreElems * BUFFER_NUM * sizeof(float)),
                                scoreElems * BUFFER_NUM);
        outGm_.SetGlobalBuffer((__gm__ float *)(base + scoreElems * BUFFER_NUM * 2 * sizeof(float)),
                               outElems * BUFFER_NUM);

        pipe_.InitBuffer(qDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(kDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(vDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(yDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(qFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(kFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(vFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(tmpFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(accFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(accTileFloatBuf_, tiling_->qBlock * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceWorkBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(scoreFloatBuf_, BUFFER_NUM * KV_BLOCK_NUM * sizeof(float));
        pipe_.InitBuffer(weightFloatBuf_, BUFFER_NUM * KV_BLOCK_NUM * sizeof(float));
        pipe_.InitBuffer(maxFloatBuf_, tiling_->qBlock * sizeof(float));
        pipe_.InitBuffer(denomFloatBuf_, tiling_->qBlock * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t cubeCoreIdx = GetCubeCoreIdx();
        const int64_t cubeCoreNum = tiling_->aicCoreNum > 0 ? tiling_->aicCoreNum : GetBlockNum();
        for (int64_t taskIdx = cubeCoreIdx; taskIdx < tiling_->cubeTaskNum; taskIdx += cubeCoreNum) {
            ProcessScoreTileTask(taskIdx);
        }
    }

private:
    __aicore__ inline int64_t GetTaskRatioValue() const
    {
        const int64_t ratio = static_cast<int64_t>(GetTaskRation());
        return ratio > 0 ? ratio : 1;
    }

    __aicore__ inline int64_t GetCubeCoreIdx() const
    {
        return static_cast<int64_t>(GetBlockIdx()) / GetTaskRatioValue();
    }

    __aicore__ inline int64_t GetSubBlockIdxValue() const
    {
        return static_cast<int64_t>(GetSubBlockIdx());
    }

    __aicore__ inline uint32_t AlignedHeadDim() const
    {
        return static_cast<uint32_t>(AlignUpInt64(tiling_->headDim, VECTOR_ALIGN_ELEMS));
    }

    __aicore__ inline int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim) const
    {
        return (((batch * tiling_->headNum + head) * tiling_->seqLen + seq) * tiling_->headDim + dim);
    }

    __aicore__ inline void DecodeQueryLinear(int64_t queryLinear, int64_t &batch, int64_t &head, int64_t &seq) const
    {
        seq = queryLinear % tiling_->seqLen;
        const int64_t tmp = queryLinear / tiling_->seqLen;
        head = tmp % tiling_->headNum;
        batch = tmp / tiling_->headNum;
    }

    __aicore__ inline void DecodeCubeTask(int64_t taskIdx, int64_t &batch, int64_t &head, int64_t &seq) const
    {
        const int64_t seqBlockNum = (tiling_->seqLen + tiling_->qBlock - 1) / tiling_->qBlock;
        const int64_t seqBlock = taskIdx % seqBlockNum;
        const int64_t tmp = taskIdx / seqBlockNum;
        head = tmp % tiling_->headNum;
        batch = tmp / tiling_->headNum;
        seq = seqBlock * tiling_->qBlock;
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> GetBufferSlice(TBuf<QuePosition::VECCALC> &buf, uint32_t bufferIdx)
    {
        LocalTensor<T> local = buf.Get<T>();
        return local[bufferIdx * VECTOR_HEAD_DIM_LIMIT];
    }

    template <typename T>
    __aicore__ inline LocalTensor<T> GetKvBlockSlice(TBuf<QuePosition::VECCALC> &buf, uint32_t bufferIdx)
    {
        LocalTensor<T> local = buf.Get<T>();
        return local[bufferIdx * KV_BLOCK_NUM];
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

    __aicore__ inline int64_t TileLinearId(int64_t tileT, int64_t tileH, int64_t tileW) const
    {
        return (tileT * tiling_->tileNumH + tileH) * tiling_->tileNumW + tileW;
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

    __aicore__ inline bool IsKvVisible(int64_t head, int64_t querySeq, int64_t kvSeq, bool isTextQuery) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        if (kvSeq < 0 || kvSeq >= validSeqLen) {
            return false;
        }
        if (isTextQuery) {
            return true;
        }
        if (kvSeq >= tiling_->imageSeqLen) {
            return tiling_->hasText != 0;
        }

        int64_t tBegin = 0;
        int64_t tEnd = 0;
        int64_t hBegin = 0;
        int64_t hEnd = 0;
        int64_t wBegin = 0;
        int64_t wEnd = 0;
        GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
        const int64_t kvTileId = kvSeq / tiling_->tileTokenNum;
        int64_t kvTileT = 0;
        int64_t kvTileH = 0;
        int64_t kvTileW = 0;
        DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);
        return kvTileT >= tBegin && kvTileT < tEnd && kvTileH >= hBegin && kvTileH < hEnd &&
               kvTileW >= wBegin && kvTileW < wEnd;
    }

    __aicore__ inline bool IsImageTileVisibleByCoord(int64_t head, int64_t querySeq, int64_t kvTileT,
                                                     int64_t kvTileH, int64_t kvTileW) const
    {
        int64_t tBegin = 0;
        int64_t tEnd = 0;
        int64_t hBegin = 0;
        int64_t hEnd = 0;
        int64_t wBegin = 0;
        int64_t wEnd = 0;
        GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
        return kvTileT >= tBegin && kvTileT < tEnd && kvTileH >= hBegin && kvTileH < hEnd &&
               kvTileW >= wBegin && kvTileW < wEnd;
    }

    __aicore__ inline bool IsKvBlockRelevant(int64_t head, int64_t querySeqBegin, int64_t qCount,
                                             int64_t kvSeqBegin, int64_t kvCount) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        const int64_t kvSeqEnd = MinInt64(kvSeqBegin + kvCount, validSeqLen);
        if (qCount <= 0 || kvSeqBegin >= kvSeqEnd) {
            return false;
        }

        for (int64_t row = 0; row < qCount; ++row) {
            const int64_t querySeq = querySeqBegin + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            const bool isTextQuery = (tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen);
            if (isTextQuery) {
                return true;
            }
            if (tiling_->hasText != 0 && kvSeqEnd > tiling_->imageSeqLen) {
                return true;
            }

            const int64_t kvImageBegin = kvSeqBegin;
            const int64_t kvImageEnd = MinInt64(kvSeqEnd, tiling_->imageSeqLen);
            if (kvImageBegin >= kvImageEnd) {
                continue;
            }

            int64_t tBegin = 0;
            int64_t tEnd = 0;
            int64_t hBegin = 0;
            int64_t hEnd = 0;
            int64_t wBegin = 0;
            int64_t wEnd = 0;
            GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);

            const int64_t kvTileBegin = kvImageBegin / tiling_->tileTokenNum;
            const int64_t kvTileEnd = (kvImageEnd - 1) / tiling_->tileTokenNum;
            for (int64_t kvTileId = kvTileBegin; kvTileId <= kvTileEnd; ++kvTileId) {
                int64_t kvTileT = 0;
                int64_t kvTileH = 0;
                int64_t kvTileW = 0;
                DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);
                if (kvTileT >= tBegin && kvTileT < tEnd && kvTileH >= hBegin && kvTileH < hEnd &&
                    kvTileW >= wBegin && kvTileW < wEnd) {
                    return true;
                }
            }
        }
        return false;
    }

    __aicore__ inline bool IsImageKvRangeInWindow(int64_t head, int64_t querySeq, int64_t kvImageBegin,
                                                  int64_t kvImageEnd) const
    {
        if (kvImageBegin >= kvImageEnd) {
            return true;
        }

        int64_t tBegin = 0;
        int64_t tEnd = 0;
        int64_t hBegin = 0;
        int64_t hEnd = 0;
        int64_t wBegin = 0;
        int64_t wEnd = 0;
        GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);

        const int64_t kvTileBegin = kvImageBegin / tiling_->tileTokenNum;
        const int64_t kvTileEnd = (kvImageEnd - 1) / tiling_->tileTokenNum;
        for (int64_t kvTileId = kvTileBegin; kvTileId <= kvTileEnd; ++kvTileId) {
            int64_t kvTileT = 0;
            int64_t kvTileH = 0;
            int64_t kvTileW = 0;
            DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);
            if (kvTileT < tBegin || kvTileT >= tEnd || kvTileH < hBegin || kvTileH >= hEnd ||
                kvTileW < wBegin || kvTileW >= wEnd) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool IsKvBlockFullyVisible(int64_t head, int64_t querySeq, int64_t kvSeqBegin,
                                                 int64_t kvCount, bool isTextQuery) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        const int64_t kvSeqEnd = MinInt64(kvSeqBegin + kvCount, validSeqLen);
        if (kvSeqBegin >= kvSeqEnd) {
            return false;
        }
        if (isTextQuery) {
            return true;
        }
        if (kvSeqBegin >= tiling_->imageSeqLen) {
            return tiling_->hasText != 0;
        }
        if (kvSeqEnd > tiling_->imageSeqLen && tiling_->hasText == 0) {
            return false;
        }
        return IsImageKvRangeInWindow(head, querySeq, kvSeqBegin, MinInt64(kvSeqEnd, tiling_->imageSeqLen));
    }

    __aicore__ inline bool IsImageTileRelevant(int64_t head, int64_t querySeqBegin, int64_t qCount,
                                               int64_t kvTileId) const
    {
        int64_t kvTileT = 0;
        int64_t kvTileH = 0;
        int64_t kvTileW = 0;
        DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);

        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        for (int64_t row = 0; row < qCount; ++row) {
            const int64_t querySeq = querySeqBegin + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            if ((tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen)) {
                return true;
            }

            int64_t tBegin = 0;
            int64_t tEnd = 0;
            int64_t hBegin = 0;
            int64_t hEnd = 0;
            int64_t wBegin = 0;
            int64_t wEnd = 0;
            GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
            if (kvTileT >= tBegin && kvTileT < tEnd && kvTileH >= hBegin && kvTileH < hEnd &&
                kvTileW >= wBegin && kvTileW < wEnd) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void GetImageTileScanRange(int64_t head, int64_t querySeqBegin, int64_t qCount,
                                                 int64_t &tileBegin, int64_t &tileEnd) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        const int64_t imageTileNum = (tiling_->imageSeqLen + tiling_->tileTokenNum - 1) / tiling_->tileTokenNum;
        tileBegin = imageTileNum;
        tileEnd = 0;

        for (int64_t row = 0; row < qCount; ++row) {
            const int64_t querySeq = querySeqBegin + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            if ((tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen)) {
                tileBegin = 0;
                tileEnd = imageTileNum;
                return;
            }

            int64_t tBegin = 0;
            int64_t tEnd = 0;
            int64_t hBegin = 0;
            int64_t hEnd = 0;
            int64_t wBegin = 0;
            int64_t wEnd = 0;
            GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
            tileBegin = MinInt64(tileBegin, TileLinearId(tBegin, hBegin, wBegin));
            tileEnd = MaxInt64(tileEnd, TileLinearId(tEnd - 1, hEnd - 1, wEnd - 1) + 1);
        }

        if (tileBegin >= tileEnd) {
            tileBegin = 0;
            tileEnd = 0;
        }
    }

    __aicore__ inline void GetTextScanRange(int64_t &textBegin, int64_t &textEnd) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        if (tiling_->hasText != 0 && tiling_->textLength > 0) {
            textBegin = tiling_->imageSeqLen;
            textEnd = validSeqLen;
        } else {
            textBegin = validSeqLen;
            textEnd = validSeqLen;
        }
    }

    __aicore__ inline void CopyScoreRowToLocal(LocalTensor<float> &scoreLocal, uint32_t rowInBlock,
                                               int64_t kvCount, uint32_t bufferIdx)
    {
        const int64_t scoreOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->kvBlock +
                                    static_cast<int64_t>(rowInBlock) * tiling_->kvBlock;
        const uint32_t copyCount =
            static_cast<uint32_t>(AlignUpInt64(tiling_->kvBlock, VECTOR_FP32_BLOCK_ELEMS));
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0F};
        DataCopyPad(scoreLocal, scoreGm_[scoreOffset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
    }

    __aicore__ inline LocalTensor<float> GetScoreHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + static_cast<int64_t>(bufferIdx) * scoreElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, scoreElems);
    }

    __aicore__ inline LocalTensor<DataType> GetProbHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t probElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + BUFFER_NUM * scoreElems * sizeof(float) +
                               static_cast<int64_t>(bufferIdx) * probElems * sizeof(DataType);
        return LocalTensor<DataType>(TPosition::VECIN, offset, probElems);
    }

    __aicore__ inline LocalTensor<float> GetOutHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t probElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + BUFFER_NUM * scoreElems * sizeof(float) +
                               BUFFER_NUM * probElems * sizeof(DataType) +
                               static_cast<int64_t>(bufferIdx) * outElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, outElems);
    }

    __aicore__ inline LocalTensor<float> GetScoreHandoffRow(uint32_t rowInBlock, int64_t kvCount,
                                                            uint32_t bufferIdx) const
    {
        LocalTensor<float> scoreTensor = GetScoreHandoffTensor(bufferIdx);
        const int64_t rowStride = AlignUpInt64(kvCount, VECTOR_FP32_BLOCK_ELEMS);
        return scoreTensor[static_cast<int64_t>(rowInBlock) * rowStride];
    }

    __aicore__ inline LocalTensor<DataType> GetProbHandoffRow(uint32_t rowInBlock, uint32_t bufferIdx) const
    {
        LocalTensor<DataType> probTensor = GetProbHandoffTensor(bufferIdx);
        return probTensor[static_cast<int64_t>(rowInBlock) * tiling_->kvBlock];
    }

    __aicore__ inline LocalTensor<float> GetOutHandoffRow(uint32_t rowInBlock, uint32_t bufferIdx) const
    {
        LocalTensor<float> outTensor = GetOutHandoffTensor(bufferIdx);
        return outTensor[static_cast<int64_t>(rowInBlock) * tiling_->headDimAlign];
    }

    __aicore__ inline void CopyScoreHandoffRowToLocal(LocalTensor<float> &scoreLocal, uint32_t rowInBlock,
                                                      int64_t kvCount, uint32_t bufferIdx)
    {
        const uint32_t copyCount =
            static_cast<uint32_t>(AlignUpInt64(tiling_->kvBlock, VECTOR_FP32_BLOCK_ELEMS));
        LocalTensor<float> scoreHandoff = GetScoreHandoffRow(rowInBlock, kvCount, bufferIdx);
        DataCopy(scoreLocal, scoreHandoff, copyCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
    }

    __aicore__ inline void StoreProbRow(uint32_t rowInBlock, LocalTensor<float> &weightLocal, uint32_t bufferIdx)
    {
        LocalTensor<DataType> probLocal = GetKvBlockSlice<DataType>(kDataBuf_, bufferIdx);
        if constexpr (IS_BF16) {
            Cast(probLocal, weightLocal, RoundMode::CAST_RINT, static_cast<uint32_t>(tiling_->kvBlock));
        } else {
            Cast(probLocal, weightLocal, RoundMode::CAST_ROUND, static_cast<uint32_t>(tiling_->kvBlock));
        }
        PipeBarrier<PIPE_V>();
#if STA_USE_VECIN_PROB_OUT_HANDOFF
        LocalTensor<DataType> probHandoff = GetProbHandoffRow(rowInBlock, bufferIdx);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        DataCopy(probHandoff, probLocal, static_cast<uint32_t>(tiling_->kvBlock));
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
#else
        const int64_t probOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->kvBlock +
                                   static_cast<int64_t>(rowInBlock) * tiling_->kvBlock;
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(tiling_->kvBlock * sizeof(DataType)), 0, 0, 0};
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        DataCopyPad(probGm_[probOffset], probLocal, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
#endif
    }

    __aicore__ inline void AddPvOutRowToAcc(LocalTensor<float> &accLocal, uint32_t rowInBlock, uint32_t bufferIdx)
    {
        LocalTensor<float> outLocal = GetBufferSlice<float>(tmpFloatBuf_, bufferIdx);
#if STA_USE_VECIN_PROB_OUT_HANDOFF
        LocalTensor<float> outHandoff = GetOutHandoffRow(rowInBlock, bufferIdx);
        DataCopy(outLocal, outHandoff, AlignedHeadDim());
#else
        const int64_t outOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->headDimAlign +
                                  static_cast<int64_t>(rowInBlock) * tiling_->headDim;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tiling_->headDim * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{
            true, 0, static_cast<uint8_t>(AlignedHeadDim() - static_cast<uint32_t>(tiling_->headDim)), 0.0F};
        DataCopyPad(outLocal, outGm_[outOffset], copyParams, padParams);
#endif
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        Add(accLocal, accLocal, outLocal, AlignedHeadDim());
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<float> GetAccTileRow(uint32_t rowInBlock)
    {
        LocalTensor<float> accTile = accTileFloatBuf_.Get<float>();
        return accTile[static_cast<int64_t>(rowInBlock) * VECTOR_HEAD_DIM_LIMIT];
    }

    __aicore__ inline void AccumulateAicScoreBlock(LocalTensor<float> &accLocal, int64_t batch, int64_t head,
                                                   int64_t querySeq, int64_t kvSeqStart, int64_t kvCount,
                                                   uint32_t rowInBlock, float &maxScore, float &denom,
                                                   bool isTextQuery, bool isFullVisible, bool isSingleImageKvTile,
                                                   uint32_t bufferIdx)
    {
        const uint32_t calcCount =
            static_cast<uint32_t>(AlignUpInt64(tiling_->kvBlock, VECTOR_FP32_BLOCK_ELEMS));
        LocalTensor<float> scoreLocal = GetKvBlockSlice<float>(scoreFloatBuf_, bufferIdx);
#if STA_USE_VECIN_SCORE_HANDOFF
        CopyScoreHandoffRowToLocal(scoreLocal, rowInBlock, kvCount, bufferIdx);
#else
        CopyScoreRowToLocal(scoreLocal, rowInBlock, kvCount, bufferIdx);
#endif
        Muls(scoreLocal, scoreLocal, scale_, calcCount);
        PipeBarrier<PIPE_V>();
        DuplicateTailFp32(scoreLocal, kvCount, calcCount, NEG_INF);

        if (!isFullVisible) {
            if (isSingleImageKvTile) {
                Duplicate(scoreLocal, NEG_INF, static_cast<uint32_t>(kvCount));
            } else {
                for (int64_t i = 0; i < kvCount; ++i) {
                    if (!IsKvVisible(head, querySeq, kvSeqStart + i, isTextQuery)) {
                        scoreLocal.SetValue(i, NEG_INF);
                    }
                }
            }
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<float> reduceLocal = GetBufferSlice<float>(reduceFloatBuf_, bufferIdx);
        LocalTensor<float> reduceWorkLocal = GetBufferSlice<float>(reduceWorkBuf_, bufferIdx);
        ReduceMax(reduceLocal, scoreLocal, reduceWorkLocal, calcCount, false);
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        const float blockMaxScore = reduceLocal.GetValue(0);
        if (blockMaxScore == NEG_INF) {
            LocalTensor<float> weightLocal = GetKvBlockSlice<float>(weightFloatBuf_, bufferIdx);
            Duplicate(weightLocal, 0.0F, calcCount);
            PipeBarrier<PIPE_V>();
            StoreProbRow(rowInBlock, weightLocal, bufferIdx);
            return;
        }

        if (blockMaxScore > maxScore) {
            const float oldScale = ExpApprox(maxScore - blockMaxScore);
            Muls(accLocal, accLocal, oldScale, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
            denom *= oldScale;
            maxScore = blockMaxScore;
        }

        LocalTensor<float> weightLocal = GetKvBlockSlice<float>(weightFloatBuf_, bufferIdx);
        Adds(weightLocal, scoreLocal, -maxScore, calcCount);
        PipeBarrier<PIPE_V>();
        Exp(weightLocal, weightLocal, calcCount);
        PipeBarrier<PIPE_V>();
        StoreProbRow(rowInBlock, weightLocal, bufferIdx);

        ReduceSum(reduceLocal, weightLocal, reduceWorkLocal, calcCount);
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);

        denom += reduceLocal.GetValue(0);
    }

    __aicore__ inline void ProcessKvRange(int64_t batch, int64_t head, int64_t seq, int64_t qCount,
                                          int64_t kvBegin, int64_t kvEnd, int64_t &tileLoop,
                                          LocalTensor<float> &maxLocal, LocalTensor<float> &denomLocal)
    {
        for (int64_t kvSeq = kvBegin; kvSeq < kvEnd; kvSeq += tiling_->kvBlock) {
            const int64_t kvCount = MinInt64(tiling_->kvBlock, kvEnd - kvSeq);
            if (!IsKvBlockRelevant(head, seq, qCount, kvSeq, kvCount)) {
                continue;
            }
            const uint32_t bufferIdx = static_cast<uint32_t>(tileLoop % BUFFER_NUM);
            CrossCoreWaitFlag(AicToAivSyncFlag(bufferIdx));
            for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
                const int64_t querySeq = seq + row;
                if (querySeq >= tiling_->imageSeqLen + tiling_->textLength) {
                    continue;
                }
                const bool isTextQuery = (tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen) &&
                                         (querySeq < tiling_->imageSeqLen + tiling_->textLength);
                const bool isFullVisible = IsKvBlockFullyVisible(head, querySeq, kvSeq, kvCount, isTextQuery);
                LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
                float maxScore = maxLocal.GetValue(row);
                float denom = denomLocal.GetValue(row);
                AccumulateAicScoreBlock(accRow, batch, head, querySeq, kvSeq, kvCount, static_cast<uint32_t>(row),
                                        maxScore, denom, isTextQuery, isFullVisible, false, bufferIdx);
                maxLocal.SetValue(row, maxScore);
                denomLocal.SetValue(row, denom);
            }
            CrossCoreSetFlag<2, PIPE_MTE3>(AivProbToAicSyncFlag(bufferIdx));
            CrossCoreWaitFlag(AicOutToAivSyncFlag(bufferIdx));
            for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
                const int64_t querySeq = seq + row;
                if (querySeq >= tiling_->imageSeqLen + tiling_->textLength) {
                    continue;
                }
                LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
                AddPvOutRowToAcc(accRow, static_cast<uint32_t>(row), bufferIdx);
            }
            CrossCoreSetFlag<2, PIPE_MTE2>(AivToAicSyncFlag(bufferIdx));
            ++tileLoop;
        }
    }

    __aicore__ inline void ProcessVisibleKvRange(int64_t batch, int64_t head, int64_t seq, int64_t qCount,
                                                 int64_t kvBegin, int64_t kvEnd, int64_t &tileLoop,
                                                 LocalTensor<float> &maxLocal, LocalTensor<float> &denomLocal)
    {
        const bool isSingleImageQueryTile =
            seq < tiling_->imageSeqLen && seq + qCount <= tiling_->imageSeqLen &&
            (seq / tiling_->tileTokenNum == (seq + qCount - 1) / tiling_->tileTokenNum);
        for (int64_t kvSeq = kvBegin; kvSeq < kvEnd; kvSeq += tiling_->kvBlock) {
            const int64_t kvCount = MinInt64(tiling_->kvBlock, kvEnd - kvSeq);
            const int64_t kvTileId = kvSeq / tiling_->tileTokenNum;
            int64_t kvTileT = 0;
            int64_t kvTileH = 0;
            int64_t kvTileW = 0;
            DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);
            const bool blockImageTileVisible =
                isSingleImageQueryTile && IsImageTileVisibleByCoord(head, seq, kvTileT, kvTileH, kvTileW);
            const uint32_t bufferIdx = static_cast<uint32_t>(tileLoop % BUFFER_NUM);
            CrossCoreWaitFlag(AicToAivSyncFlag(bufferIdx));
            for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
                const int64_t querySeq = seq + row;
                if (querySeq >= tiling_->imageSeqLen + tiling_->textLength) {
                    continue;
                }
                const bool isTextQuery = (tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen) &&
                                         (querySeq < tiling_->imageSeqLen + tiling_->textLength);
                const bool isFullVisible =
                    isTextQuery ||
                    (isSingleImageQueryTile
                         ? blockImageTileVisible
                         : IsImageTileVisibleByCoord(head, querySeq, kvTileT, kvTileH, kvTileW));
                LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
                float maxScore = maxLocal.GetValue(row);
                float denom = denomLocal.GetValue(row);
                AccumulateAicScoreBlock(accRow, batch, head, querySeq, kvSeq, kvCount, static_cast<uint32_t>(row),
                                        maxScore, denom, isTextQuery, isFullVisible, !isTextQuery, bufferIdx);
                maxLocal.SetValue(row, maxScore);
                denomLocal.SetValue(row, denom);
            }
            CrossCoreSetFlag<2, PIPE_MTE3>(AivProbToAicSyncFlag(bufferIdx));
            CrossCoreWaitFlag(AicOutToAivSyncFlag(bufferIdx));
            for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
                const int64_t querySeq = seq + row;
                if (querySeq >= tiling_->imageSeqLen + tiling_->textLength) {
                    continue;
                }
                LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
                AddPvOutRowToAcc(accRow, static_cast<uint32_t>(row), bufferIdx);
            }
            CrossCoreSetFlag<2, PIPE_MTE2>(AivToAicSyncFlag(bufferIdx));
            ++tileLoop;
        }
    }

    __aicore__ inline void ProcessImageKvTiles(int64_t batch, int64_t head, int64_t seq, int64_t qCount,
                                               int64_t &tileLoop, LocalTensor<float> &maxLocal,
                                               LocalTensor<float> &denomLocal)
    {
        int64_t tileBegin = 0;
        int64_t tileEnd = 0;
        GetImageTileScanRange(head, seq, qCount, tileBegin, tileEnd);
        for (int64_t kvTileId = tileBegin; kvTileId < tileEnd; ++kvTileId) {
            if (!IsImageTileRelevant(head, seq, qCount, kvTileId)) {
                continue;
            }
            const int64_t kvBegin = kvTileId * tiling_->tileTokenNum;
            const int64_t kvEnd = MinInt64(kvBegin + tiling_->tileTokenNum, tiling_->imageSeqLen);
            ProcessVisibleKvRange(batch, head, seq, qCount, kvBegin, kvEnd, tileLoop, maxLocal, denomLocal);
        }
    }

    __aicore__ inline void ProcessScoreTileTask(int64_t taskIdx)
    {
        int64_t batch = 0;
        int64_t head = 0;
        int64_t seq = 0;
        DecodeCubeTask(taskIdx, batch, head, seq);
        const int64_t qCount = MinInt64(tiling_->qBlock, tiling_->seqLen - seq);
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        if (qCount <= 0 || validSeqLen <= 0) {
            return;
        }

        LocalTensor<float> maxLocal = maxFloatBuf_.Get<float>();
        LocalTensor<float> denomLocal = denomFloatBuf_.Get<float>();
        for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
            const int64_t querySeq = seq + row;
            const int64_t qBase = TensorOffset(batch, head, querySeq, 0);
            LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
            if (querySeq >= validSeqLen) {
                StoreZeroRow(qBase, static_cast<uint32_t>(row % BUFFER_NUM));
                continue;
            }
            Duplicate(accRow, 0.0F, AlignedHeadDim());
            maxLocal.SetValue(row, NEG_INF);
            denomLocal.SetValue(row, 0.0F);
        }
        PipeBarrier<PIPE_V>();

        int64_t textBegin = 0;
        int64_t textEnd = 0;
        GetTextScanRange(textBegin, textEnd);

        int64_t tileLoop = 0;
        ProcessImageKvTiles(batch, head, seq, qCount, tileLoop, maxLocal, denomLocal);
        ProcessKvRange(batch, head, seq, qCount, textBegin, textEnd, tileLoop, maxLocal, denomLocal);

        for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
            const int64_t querySeq = seq + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            const int64_t qBase = TensorOffset(batch, head, querySeq, 0);
            LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
            const float denom = denomLocal.GetValue(row);
            if (denom > 0.0F) {
                Muls(accRow, accRow, 1.0F / denom, AlignedHeadDim());
                PipeBarrier<PIPE_V>();
            } else {
                Duplicate(accRow, 0.0F, AlignedHeadDim());
                PipeBarrier<PIPE_V>();
            }
            StoreFloatRow(qBase, accRow, static_cast<uint32_t>(row % BUFFER_NUM));
        }
    }

    __aicore__ inline void AccumulateOnlineValueBlock(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                      int64_t batch, int64_t head, int64_t kvSeqStart,
                                                      int64_t kvCount, float &maxScore, float &denom,
                                                      uint32_t bufferIdx)
    {
        const uint32_t kvCountU32 = static_cast<uint32_t>(kvCount);
        LocalTensor<float> scoreLocal = GetKvBlockSlice<float>(scoreFloatBuf_, bufferIdx);
        float blockMaxScore = NEG_INF;
        for (int64_t i = 0; i < kvCount; ++i) {
            const int64_t kvBase = TensorOffset(batch, head, kvSeqStart + i, 0);
            const float score = DotScoreVector(qLocal, kvBase, bufferIdx);
            scoreLocal.SetValue(i, score);
            blockMaxScore = score > blockMaxScore ? score : blockMaxScore;
        }

        if (blockMaxScore > maxScore) {
            const float oldScale = ExpApprox(maxScore - blockMaxScore);
            Muls(accLocal, accLocal, oldScale, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
            denom *= oldScale;
            maxScore = blockMaxScore;
        }

        LocalTensor<float> weightLocal = GetKvBlockSlice<float>(weightFloatBuf_, bufferIdx);
        Adds(weightLocal, scoreLocal, -maxScore, kvCountU32);
        PipeBarrier<PIPE_V>();
        Exp(weightLocal, weightLocal, kvCountU32);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> reduceLocal = GetBufferSlice<float>(reduceFloatBuf_, bufferIdx);
        LocalTensor<float> reduceWorkLocal = GetBufferSlice<float>(reduceWorkBuf_, bufferIdx);
        ReduceSum(reduceLocal, weightLocal, reduceWorkLocal, kvCountU32);
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);

        for (int64_t i = 0; i < kvCount; ++i) {
            const int64_t kvBase = TensorOffset(batch, head, kvSeqStart + i, 0);
            const float weight = weightLocal.GetValue(i);
            LocalTensor<float> vLocal = GetBufferSlice<float>(vFloatBuf_, bufferIdx);
            CopyDataRowToFloat(vDataGm_, vDataBuf_, vLocal, kvBase, bufferIdx);

            LocalTensor<float> tmpLocal = GetBufferSlice<float>(tmpFloatBuf_, bufferIdx);
            Muls(tmpLocal, vLocal, weight, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
            Add(accLocal, accLocal, tmpLocal, AlignedHeadDim());
            PipeBarrier<PIPE_V>();
        }
        denom += reduceLocal.GetValue(0);
    }

    __aicore__ inline void AccumulateKvSeqRangeOnlineBlock(LocalTensor<float> &qLocal, LocalTensor<float> &accLocal,
                                                           int64_t batch, int64_t head, int64_t kvSeqBegin,
                                                           int64_t kvSeqEnd, float &maxScore, float &denom,
                                                           uint32_t bufferIdx)
    {
        for (int64_t kvSeq = kvSeqBegin; kvSeq < kvSeqEnd; kvSeq += KV_BLOCK_NUM) {
            const int64_t kvCount = MinInt64(KV_BLOCK_NUM, kvSeqEnd - kvSeq);
            AccumulateOnlineValueBlock(qLocal, accLocal, batch, head, kvSeq, kvCount, maxScore, denom, bufferIdx);
        }
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
                    const int64_t kvSeqEnd = MinInt64(baseTile + tiling_->tileTokenNum, tiling_->imageSeqLen);
                    AccumulateKvSeqRangeOnlineBlock(qLocal, accLocal, batch, head, baseTile, kvSeqEnd, maxScore, denom,
                                                    bufferIdx);
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
            AccumulateKvSeqRangeOnlineBlock(qLocal, accLocal, batch, head, 0,
                                            tiling_->imageSeqLen + tiling_->textLength, maxScore, denom, bufferIdx);
        } else {
            AccumulateImageWindowOnlineVector(qLocal, accLocal, batch, head, seq, maxScore, denom, bufferIdx);
            if (tiling_->hasText != 0) {
                AccumulateKvSeqRangeOnlineBlock(qLocal, accLocal, batch, head, tiling_->imageSeqLen,
                                                tiling_->imageSeqLen + tiling_->textLength, maxScore, denom,
                                                bufferIdx);
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
    GlobalTensor<float> scoreGm_;
    GlobalTensor<DataType> probGm_;
    GlobalTensor<float> outGm_;
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
    TBuf<QuePosition::VECCALC> accTileFloatBuf_;
    TBuf<QuePosition::VECCALC> reduceFloatBuf_;
    TBuf<QuePosition::VECCALC> reduceWorkBuf_;
    TBuf<QuePosition::VECCALC> scoreFloatBuf_;
    TBuf<QuePosition::VECCALC> weightFloatBuf_;
    TBuf<QuePosition::VECCALC> maxFloatBuf_;
    TBuf<QuePosition::VECCALC> denomFloatBuf_;
    const SlidingTileAttentionTilingData *tiling_ = nullptr;
    float scale_ = 0.0F;
};

template <bool IS_BF16>
class SlidingTileAttentionCube {
public:
    using DataType = typename DataTypeSelector<IS_BF16>::Type;
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, DataType>;
    using BTypeTrans = MatmulType<TPosition::GM, CubeFormat::ND, DataType, true>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, DataType>;
#if STA_USE_VECIN_SCORE_HANDOFF
    using ScoreCType = MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, float>;
#else
    using ScoreCType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
#endif
#if STA_USE_VECIN_PROB_OUT_HANDOFF
    using ProbAType = MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, DataType>;
    using OutCType = MatmulType<TPosition::VECIN, CubeFormat::ND_ALIGN, float>;
#else
    using ProbAType = MatmulType<TPosition::GM, CubeFormat::ND, DataType>;
    using OutCType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
#endif
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    static constexpr MatmulConfig MM_CFG = GetNormalConfig(true);

    __aicore__ inline SlidingTileAttentionCube() = default;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const SlidingTileAttentionTilingData *tilingData)
    {
        (void)y;
        tiling_ = tilingData;
        const int64_t total = tiling_->batchSize * tiling_->headNum * tiling_->seqLen * tiling_->headDim;
        qDataGm_.SetGlobalBuffer((__gm__ DataType *)q, total);
        kDataGm_.SetGlobalBuffer((__gm__ DataType *)k, total);
        vDataGm_.SetGlobalBuffer((__gm__ DataType *)v, total);

        const int64_t coreOffset = static_cast<int64_t>(GetBlockIdx()) * tiling_->cubeWorkspacePerCore;
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        __gm__ uint8_t *base = (__gm__ uint8_t *)workspace + coreOffset;
        scoreGm_.SetGlobalBuffer((__gm__ float *)base, scoreElems * BUFFER_NUM);
        probGm_.SetGlobalBuffer((__gm__ DataType *)(base + scoreElems * BUFFER_NUM * sizeof(float)),
                                scoreElems * BUFFER_NUM);
        outGm_.SetGlobalBuffer((__gm__ float *)(base + scoreElems * BUFFER_NUM * 2 * sizeof(float)),
                               outElems * BUFFER_NUM);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockNum = GetBlockNum();
        const int64_t blockIdx = GetBlockIdx();
        for (int64_t taskIdx = blockIdx; taskIdx < tiling_->cubeTaskNum; taskIdx += blockNum) {
            int64_t batch = 0;
            int64_t head = 0;
            int64_t seq = 0;
            DecodeCubeTask(taskIdx, batch, head, seq);
            const int64_t qCount = MinInt64(tiling_->qBlock, tiling_->seqLen - seq);
            int64_t textBegin = 0;
            int64_t textEnd = 0;
            GetTextScanRange(textBegin, textEnd);
            int64_t tileLoop = 0;
            ProcessImageKvTiles(taskIdx, head, seq, qCount, tileLoop);
            ProcessKvRange(taskIdx, head, seq, qCount, textBegin, textEnd, tileLoop);
            const int64_t drainBegin = tileLoop > BUFFER_NUM ? tileLoop - BUFFER_NUM : 0;
            for (int64_t drainLoop = drainBegin; drainLoop < tileLoop; ++drainLoop) {
                CrossCoreWaitFlag(AivToAicSyncFlag(static_cast<uint32_t>(drainLoop % BUFFER_NUM)));
            }
        }
    }

private:
    __aicore__ inline int64_t TensorOffset(int64_t batch, int64_t head, int64_t seq, int64_t dim) const
    {
        return (((batch * tiling_->headNum + head) * tiling_->seqLen + seq) * tiling_->headDim + dim);
    }

    __aicore__ inline void DecodeQueryLinear(int64_t queryLinear, int64_t &batch, int64_t &head, int64_t &seq) const
    {
        seq = queryLinear % tiling_->seqLen;
        const int64_t tmp = queryLinear / tiling_->seqLen;
        head = tmp % tiling_->headNum;
        batch = tmp / tiling_->headNum;
    }

    __aicore__ inline void DecodeCubeTask(int64_t taskIdx, int64_t &batch, int64_t &head, int64_t &seq) const
    {
        const int64_t seqBlockNum = (tiling_->seqLen + tiling_->qBlock - 1) / tiling_->qBlock;
        const int64_t seqBlock = taskIdx % seqBlockNum;
        const int64_t tmp = taskIdx / seqBlockNum;
        head = tmp % tiling_->headNum;
        batch = tmp / tiling_->headNum;
        seq = seqBlock * tiling_->qBlock;
    }

    __aicore__ inline void DecodeTileCoord(int64_t tileId, int64_t &tileT, int64_t &tileH, int64_t &tileW) const
    {
        const int64_t tilesHW = tiling_->tileNumH * tiling_->tileNumW;
        tileT = tileId / tilesHW;
        const int64_t rem = tileId - tileT * tilesHW;
        tileH = rem / tiling_->tileNumW;
        tileW = rem - tileH * tiling_->tileNumW;
    }

    __aicore__ inline int64_t TileLinearId(int64_t tileT, int64_t tileH, int64_t tileW) const
    {
        return (tileT * tiling_->tileNumH + tileH) * tiling_->tileNumW + tileW;
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

    __aicore__ inline bool IsKvBlockRelevant(int64_t head, int64_t querySeqBegin, int64_t qCount,
                                             int64_t kvSeqBegin, int64_t kvCount) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        const int64_t kvSeqEnd = MinInt64(kvSeqBegin + kvCount, validSeqLen);
        if (qCount <= 0 || kvSeqBegin >= kvSeqEnd) {
            return false;
        }

        for (int64_t row = 0; row < qCount; ++row) {
            const int64_t querySeq = querySeqBegin + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            const bool isTextQuery = (tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen);
            if (isTextQuery) {
                return true;
            }
            if (tiling_->hasText != 0 && kvSeqEnd > tiling_->imageSeqLen) {
                return true;
            }

            const int64_t kvImageBegin = kvSeqBegin;
            const int64_t kvImageEnd = MinInt64(kvSeqEnd, tiling_->imageSeqLen);
            if (kvImageBegin >= kvImageEnd) {
                continue;
            }

            int64_t tBegin = 0;
            int64_t tEnd = 0;
            int64_t hBegin = 0;
            int64_t hEnd = 0;
            int64_t wBegin = 0;
            int64_t wEnd = 0;
            GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);

            const int64_t kvTileBegin = kvImageBegin / tiling_->tileTokenNum;
            const int64_t kvTileEnd = (kvImageEnd - 1) / tiling_->tileTokenNum;
            for (int64_t kvTileId = kvTileBegin; kvTileId <= kvTileEnd; ++kvTileId) {
                int64_t kvTileT = 0;
                int64_t kvTileH = 0;
                int64_t kvTileW = 0;
                DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);
                if (kvTileT >= tBegin && kvTileT < tEnd && kvTileH >= hBegin && kvTileH < hEnd &&
                    kvTileW >= wBegin && kvTileW < wEnd) {
                    return true;
                }
            }
        }
        return false;
    }

    __aicore__ inline bool IsImageTileRelevant(int64_t head, int64_t querySeqBegin, int64_t qCount,
                                               int64_t kvTileId) const
    {
        int64_t kvTileT = 0;
        int64_t kvTileH = 0;
        int64_t kvTileW = 0;
        DecodeTileCoord(kvTileId, kvTileT, kvTileH, kvTileW);

        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        for (int64_t row = 0; row < qCount; ++row) {
            const int64_t querySeq = querySeqBegin + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            if ((tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen)) {
                return true;
            }

            int64_t tBegin = 0;
            int64_t tEnd = 0;
            int64_t hBegin = 0;
            int64_t hEnd = 0;
            int64_t wBegin = 0;
            int64_t wEnd = 0;
            GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
            if (kvTileT >= tBegin && kvTileT < tEnd && kvTileH >= hBegin && kvTileH < hEnd &&
                kvTileW >= wBegin && kvTileW < wEnd) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void GetImageTileScanRange(int64_t head, int64_t querySeqBegin, int64_t qCount,
                                                 int64_t &tileBegin, int64_t &tileEnd) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        const int64_t imageTileNum = (tiling_->imageSeqLen + tiling_->tileTokenNum - 1) / tiling_->tileTokenNum;
        tileBegin = imageTileNum;
        tileEnd = 0;

        for (int64_t row = 0; row < qCount; ++row) {
            const int64_t querySeq = querySeqBegin + row;
            if (querySeq >= validSeqLen) {
                continue;
            }
            if ((tiling_->hasText != 0) && (querySeq >= tiling_->imageSeqLen)) {
                tileBegin = 0;
                tileEnd = imageTileNum;
                return;
            }

            int64_t tBegin = 0;
            int64_t tEnd = 0;
            int64_t hBegin = 0;
            int64_t hEnd = 0;
            int64_t wBegin = 0;
            int64_t wEnd = 0;
            GetImageWindowRange(head, querySeq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
            tileBegin = MinInt64(tileBegin, TileLinearId(tBegin, hBegin, wBegin));
            tileEnd = MaxInt64(tileEnd, TileLinearId(tEnd - 1, hEnd - 1, wEnd - 1) + 1);
        }

        if (tileBegin >= tileEnd) {
            tileBegin = 0;
            tileEnd = 0;
        }
    }

    __aicore__ inline void GetTextScanRange(int64_t &textBegin, int64_t &textEnd) const
    {
        const int64_t validSeqLen = tiling_->imageSeqLen + tiling_->textLength;
        if (tiling_->hasText != 0 && tiling_->textLength > 0) {
            textBegin = tiling_->imageSeqLen;
            textEnd = validSeqLen;
        } else {
            textBegin = validSeqLen;
            textEnd = validSeqLen;
        }
    }

    __aicore__ inline void ProcessKvRange(int64_t taskIdx, int64_t head, int64_t seq, int64_t qCount,
                                          int64_t kvBegin, int64_t kvEnd, int64_t &tileLoop)
    {
        for (int64_t kvSeq = kvBegin; kvSeq < kvEnd; kvSeq += tiling_->kvBlock) {
            const int64_t kvCount = MinInt64(tiling_->kvBlock, kvEnd - kvSeq);
            if (!IsKvBlockRelevant(head, seq, qCount, kvSeq, kvCount)) {
                continue;
            }
            const uint32_t bufferIdx = static_cast<uint32_t>(tileLoop % BUFFER_NUM);
            if (tileLoop >= BUFFER_NUM) {
                CrossCoreWaitFlag(AivToAicSyncFlag(bufferIdx));
            }
            ComputeScoreTileTask(taskIdx, kvSeq, kvCount, bufferIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(AicToAivSyncFlag(bufferIdx));
            CrossCoreWaitFlag(AivProbToAicSyncFlag(bufferIdx));
            ComputeValueTileTask(taskIdx, kvSeq, kvCount, bufferIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(AicOutToAivSyncFlag(bufferIdx));
            ++tileLoop;
        }
    }

    __aicore__ inline void ProcessVisibleKvRange(int64_t taskIdx, int64_t head, int64_t seq, int64_t qCount,
                                                 int64_t kvBegin, int64_t kvEnd, int64_t &tileLoop)
    {
        for (int64_t kvSeq = kvBegin; kvSeq < kvEnd; kvSeq += tiling_->kvBlock) {
            const int64_t kvCount = MinInt64(tiling_->kvBlock, kvEnd - kvSeq);
            const uint32_t bufferIdx = static_cast<uint32_t>(tileLoop % BUFFER_NUM);
            if (tileLoop >= BUFFER_NUM) {
                CrossCoreWaitFlag(AivToAicSyncFlag(bufferIdx));
            }
            ComputeScoreTileTask(taskIdx, kvSeq, kvCount, bufferIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(AicToAivSyncFlag(bufferIdx));
            CrossCoreWaitFlag(AivProbToAicSyncFlag(bufferIdx));
            ComputeValueTileTask(taskIdx, kvSeq, kvCount, bufferIdx);
            CrossCoreSetFlag<2, PIPE_FIX>(AicOutToAivSyncFlag(bufferIdx));
            ++tileLoop;
        }
    }

    __aicore__ inline void ProcessImageKvTiles(int64_t taskIdx, int64_t head, int64_t seq, int64_t qCount,
                                               int64_t &tileLoop)
    {
        int64_t tileBegin = 0;
        int64_t tileEnd = 0;
        GetImageTileScanRange(head, seq, qCount, tileBegin, tileEnd);
        for (int64_t kvTileId = tileBegin; kvTileId < tileEnd; ++kvTileId) {
            if (!IsImageTileRelevant(head, seq, qCount, kvTileId)) {
                continue;
            }
            const int64_t kvBegin = kvTileId * tiling_->tileTokenNum;
            const int64_t kvEnd = MinInt64(kvBegin + tiling_->tileTokenNum, tiling_->imageSeqLen);
            ProcessVisibleKvRange(taskIdx, head, seq, qCount, kvBegin, kvEnd, tileLoop);
        }
    }

    __aicore__ inline LocalTensor<float> GetScoreHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + static_cast<int64_t>(bufferIdx) * scoreElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, scoreElems);
    }

    __aicore__ inline LocalTensor<DataType> GetProbHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t probElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + BUFFER_NUM * scoreElems * sizeof(float) +
                               static_cast<int64_t>(bufferIdx) * probElems * sizeof(DataType);
        return LocalTensor<DataType>(TPosition::VECIN, offset, probElems);
    }

    __aicore__ inline LocalTensor<float> GetOutHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t probElems = tiling_->qBlock * tiling_->kvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + BUFFER_NUM * scoreElems * sizeof(float) +
                               BUFFER_NUM * probElems * sizeof(DataType) +
                               static_cast<int64_t>(bufferIdx) * outElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, outElems);
    }

    __aicore__ inline void ComputeScoreTileTask(int64_t taskIdx, int64_t kvBegin, int64_t kvCount,
                                                uint32_t bufferIdx)
    {
        int64_t batch = 0;
        int64_t head = 0;
        int64_t seq = 0;
        DecodeCubeTask(taskIdx, batch, head, seq);
        const int64_t qCount = MinInt64(tiling_->qBlock, tiling_->seqLen - seq);
        if (qCount <= 0 || kvCount <= 0) {
            return;
        }

        const int64_t qBase = TensorOffset(batch, head, seq, 0);
        const int64_t kvBase = TensorOffset(batch, head, kvBegin, 0);
        const int64_t outOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->headDimAlign;

        qkMatmul_.SetOrgShape(qCount, kvCount, tiling_->headDim, tiling_->headDim, tiling_->kvBlock);
        qkMatmul_.SetTail(qCount, kvCount, tiling_->headDim);
        qkMatmul_.SetTensorA(qDataGm_[qBase]);
        qkMatmul_.SetTensorB(kDataGm_[kvBase], true);
#if STA_USE_VECIN_SCORE_HANDOFF
        LocalTensor<float> scoreLocal = GetScoreHandoffTensor(bufferIdx);
        qkMatmul_.template IterateAll<false>(scoreLocal, 0);
#else
        const int64_t scoreOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->kvBlock;
        qkMatmul_.template IterateAll<false>(scoreGm_[scoreOffset], false, false, true);
#endif
        qkMatmul_.WaitIterateAll();
        qkMatmul_.End();
        (void)outOffset;
    }

    __aicore__ inline void ComputeValueTileTask(int64_t taskIdx, int64_t kvBegin, int64_t kvCount,
                                                uint32_t bufferIdx)
    {
        int64_t batch = 0;
        int64_t head = 0;
        int64_t seq = 0;
        DecodeCubeTask(taskIdx, batch, head, seq);
        const int64_t qCount = MinInt64(tiling_->qBlock, tiling_->seqLen - seq);
        if (qCount <= 0 || kvCount <= 0) {
            return;
        }

        const int64_t kvBase = TensorOffset(batch, head, kvBegin, 0);
        pvMatmul_.SetOrgShape(qCount, tiling_->headDim, kvCount, tiling_->kvBlock, tiling_->headDim);
        pvMatmul_.SetTail(qCount, tiling_->headDim, kvCount);
#if STA_USE_VECIN_PROB_OUT_HANDOFF
        LocalTensor<DataType> probLocal = GetProbHandoffTensor(bufferIdx);
        LocalTensor<float> outLocal = GetOutHandoffTensor(bufferIdx);
        pvMatmul_.SetTensorA(probLocal);
        pvMatmul_.SetTensorB(vDataGm_[kvBase]);
        pvMatmul_.template IterateAll<false>(outLocal, 0);
#else
        const int64_t probOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->kvBlock;
        const int64_t outOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->headDimAlign;
        pvMatmul_.SetTensorA(probGm_[probOffset]);
        pvMatmul_.SetTensorB(vDataGm_[kvBase]);
        pvMatmul_.template IterateAll<false>(outGm_[outOffset], false, false, true);
#endif
        pvMatmul_.WaitIterateAll();
        pvMatmul_.End();
    }

private:
    GlobalTensor<DataType> qDataGm_;
    GlobalTensor<DataType> kDataGm_;
    GlobalTensor<DataType> vDataGm_;
    GlobalTensor<float> scoreGm_;
    GlobalTensor<DataType> probGm_;
    GlobalTensor<float> outGm_;
    Matmul<AType, BTypeTrans, ScoreCType, BiasType, MM_CFG> qkMatmul_;
    Matmul<ProbAType, BType, OutCType, BiasType, MM_CFG> pvMatmul_;
    const SlidingTileAttentionTilingData *tiling_ = nullptr;
};
} // namespace SlidingTileAttentionKernel

#endif // SLIDING_TILE_ATTENTION_H
