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
#define STA_USE_VECIN_PROB_OUT_HANDOFF 1
#endif

namespace SlidingTileAttentionKernel {
using namespace AscendC;
using namespace matmul;

constexpr float NEG_INF = -3.4028234663852886e+38F;
constexpr int64_t VECTOR_HEAD_DIM_LIMIT = 128;
constexpr int64_t VECTOR_ALIGN_ELEMS = 16;
constexpr int64_t VECTOR_FP32_BLOCK_ELEMS = 8;
constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t SCORE_HANDOFF_UB_OFFSET = 46 * 1024;
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

template <bool IS_BF16>
class SlidingTileAttention {
public:
    using DataType = typename DataTypeSelector<IS_BF16>::Type;

    __aicore__ inline SlidingTileAttention() = default;
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const SlidingTileAttentionTilingData *tilingData)
    {
        (void)q;
        (void)k;
        (void)v;
        tiling_ = tilingData;
        const int64_t total = tiling_->batchSize * tiling_->headNum * tiling_->seqLen * tiling_->headDim;
        yDataGm_.SetGlobalBuffer((__gm__ DataType *)y, total);
        scale_ = tiling_->scale;

        const int64_t cubeCoreIdx = GetCubeCoreIdx();
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        __gm__ uint8_t *base = (__gm__ uint8_t *)workspace + cubeCoreIdx * tiling_->cubeWorkspacePerCore;
        scoreGm_.SetGlobalBuffer((__gm__ float *)base, scoreElems * BUFFER_NUM);
        probGm_.SetGlobalBuffer((__gm__ DataType *)(base + scoreElems * BUFFER_NUM * sizeof(float)),
                                scoreElems * BUFFER_NUM);
        outGm_.SetGlobalBuffer((__gm__ float *)(base + scoreElems * BUFFER_NUM * 2 * sizeof(float)),
                               outElems * BUFFER_NUM);

        pipe_.InitBuffer(yDataBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(DataType));
        pipe_.InitBuffer(probDataBuf_, BUFFER_NUM * tiling_->maxKvBlock * sizeof(DataType));
        pipe_.InitBuffer(tmpFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(accFloatBuf_, BUFFER_NUM * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(accTileFloatBuf_, tiling_->qBlock * VECTOR_HEAD_DIM_LIMIT * sizeof(float));
        pipe_.InitBuffer(reduceFloatBuf_, BUFFER_NUM * tiling_->maxKvBlock * sizeof(float));
        pipe_.InitBuffer(reduceWorkBuf_, BUFFER_NUM * tiling_->maxKvBlock * sizeof(float));
        pipe_.InitBuffer(scoreFloatBuf_, BUFFER_NUM * tiling_->maxKvBlock * sizeof(float));
        pipe_.InitBuffer(weightFloatBuf_, BUFFER_NUM * tiling_->maxKvBlock * sizeof(float));
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
        const int64_t seqBlock = taskIdx % tiling_->seqBlockNum;
        const int64_t tmp = taskIdx / tiling_->seqBlockNum;
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
        return local[bufferIdx * tiling_->maxKvBlock];
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

    __aicore__ inline void DecodeTileCoord(int64_t tileId, int64_t &tileT, int64_t &tileH, int64_t &tileW) const
    {
        tileT = tileId / tiling_->tileNumHW;
        const int64_t rem = tileId - tileT * tiling_->tileNumHW;
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

    __aicore__ inline void GetTextScanRange(int64_t &textBegin, int64_t &textEnd) const
    {
        if (tiling_->hasText != 0 && tiling_->textLength > 0) {
            textBegin = tiling_->imageSeqLen;
            textEnd = tiling_->validSeqLen;
        } else {
            textBegin = tiling_->validSeqLen;
            textEnd = tiling_->validSeqLen;
        }
    }

    __aicore__ inline void CopyScoreRowToLocal(LocalTensor<float> &scoreLocal, uint32_t rowInBlock,
                                               int64_t kvCount, uint32_t bufferIdx)
    {
        const int64_t scoreOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->maxKvBlock +
                                    static_cast<int64_t>(rowInBlock) * tiling_->maxKvBlock;
        const uint32_t copyCount =
            static_cast<uint32_t>(AlignUpInt64(kvCount, VECTOR_FP32_BLOCK_ELEMS));
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(copyCount * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0F};
        DataCopyPad(scoreLocal, scoreGm_[scoreOffset], copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
    }

    __aicore__ inline LocalTensor<float> GetScoreHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t sharedElems = MaxInt64(scoreElems, outElems);
        const int64_t offset =
            SCORE_HANDOFF_UB_OFFSET + static_cast<int64_t>(bufferIdx) * sharedElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, scoreElems);
    }

    __aicore__ inline LocalTensor<DataType> GetProbHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t probElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t sharedElems = MaxInt64(scoreElems, outElems);
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + BUFFER_NUM * sharedElems * sizeof(float) +
                               static_cast<int64_t>(bufferIdx) * probElems * sizeof(DataType);
        return LocalTensor<DataType>(TPosition::VECIN, offset, probElems);
    }

    __aicore__ inline LocalTensor<float> GetOutHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t sharedElems = MaxInt64(scoreElems, outElems);
        const int64_t offset =
            SCORE_HANDOFF_UB_OFFSET + static_cast<int64_t>(bufferIdx) * sharedElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, outElems);
    }

    __aicore__ inline LocalTensor<float> GetScoreHandoffRow(uint32_t rowInBlock, int64_t kvCount,
                                                            uint32_t bufferIdx) const
    {
        (void)kvCount;
        LocalTensor<float> scoreTensor = GetScoreHandoffTensor(bufferIdx);
        const int64_t rowStride = tiling_->maxKvBlock;
        return scoreTensor[static_cast<int64_t>(rowInBlock) * rowStride];
    }

    __aicore__ inline LocalTensor<DataType> GetProbHandoffRow(uint32_t rowInBlock, uint32_t bufferIdx) const
    {
        LocalTensor<DataType> probTensor = GetProbHandoffTensor(bufferIdx);
        return probTensor[static_cast<int64_t>(rowInBlock) * tiling_->maxKvBlock];
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
            static_cast<uint32_t>(AlignUpInt64(kvCount, VECTOR_FP32_BLOCK_ELEMS));
        LocalTensor<float> scoreHandoff = GetScoreHandoffRow(rowInBlock, kvCount, bufferIdx);
        DataCopy(scoreLocal, scoreHandoff, copyCount);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0 + bufferIdx);
    }

    __aicore__ inline void StoreProbRow(uint32_t rowInBlock, LocalTensor<float> &weightLocal, int64_t kvCount,
                                        uint32_t bufferIdx)
    {
        LocalTensor<DataType> probLocal = GetKvBlockSlice<DataType>(probDataBuf_, bufferIdx);
        const uint32_t copyCount = static_cast<uint32_t>(AlignUpInt64(kvCount, VECTOR_ALIGN_ELEMS));
        if constexpr (IS_BF16) {
            Cast(probLocal, weightLocal, RoundMode::CAST_RINT, copyCount);
        } else {
            Cast(probLocal, weightLocal, RoundMode::CAST_ROUND, copyCount);
        }
        PipeBarrier<PIPE_V>();
#if STA_USE_VECIN_PROB_OUT_HANDOFF
        LocalTensor<DataType> probHandoff = GetProbHandoffRow(rowInBlock, bufferIdx);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2 + bufferIdx);
        DataCopy(probHandoff, probLocal, copyCount);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2 + bufferIdx);
#else
        const int64_t probOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->maxKvBlock +
                                   static_cast<int64_t>(rowInBlock) * tiling_->maxKvBlock;
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(copyCount * sizeof(DataType)), 0, 0, 0};
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

    __aicore__ inline float ExpScalarWithVector(float value, uint32_t bufferIdx)
    {
        LocalTensor<float> expLocal = GetBufferSlice<float>(tmpFloatBuf_, bufferIdx);
        Duplicate(expLocal, value, VECTOR_FP32_BLOCK_ELEMS);
        PipeBarrier<PIPE_V>();
        Exp(expLocal, expLocal, VECTOR_FP32_BLOCK_ELEMS);
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        return expLocal.GetValue(0);
    }

    __aicore__ inline uint32_t ScoreCalcCount(int64_t kvCount) const
    {
        return static_cast<uint32_t>(AlignUpInt64(kvCount, VECTOR_ALIGN_ELEMS));
    }

    __aicore__ inline void LoadScaledScoreRow(LocalTensor<float> &scoreLocal, int64_t kvCount,
                                              uint32_t rowInBlock, uint32_t bufferIdx)
    {
#if STA_USE_VECIN_SCORE_HANDOFF
        CopyScoreHandoffRowToLocal(scoreLocal, rowInBlock, kvCount, bufferIdx);
#else
        CopyScoreRowToLocal(scoreLocal, rowInBlock, kvCount, bufferIdx);
#endif
        const uint32_t calcCount = ScoreCalcCount(kvCount);
        const uint32_t validCalcCount =
            static_cast<uint32_t>(AlignUpInt64(kvCount, VECTOR_FP32_BLOCK_ELEMS));
        Muls(scoreLocal, scoreLocal, scale_, validCalcCount);
        PipeBarrier<PIPE_V>();
        DuplicateTailFp32(scoreLocal, kvCount, calcCount, NEG_INF);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumulateVisibleScoreBlock(LocalTensor<float> &accLocal, int64_t kvCount,
                                                       uint32_t rowInBlock, float &maxScore, float &denom,
                                                       uint32_t bufferIdx)
    {
        const uint32_t calcCount = ScoreCalcCount(kvCount);
        LocalTensor<float> scoreLocal = GetKvBlockSlice<float>(scoreFloatBuf_, bufferIdx);
        LoadScaledScoreRow(scoreLocal, kvCount, rowInBlock, bufferIdx);

        LocalTensor<float> reduceLocal = GetKvBlockSlice<float>(reduceFloatBuf_, bufferIdx);
        LocalTensor<float> reduceWorkLocal = GetKvBlockSlice<float>(reduceWorkBuf_, bufferIdx);
        ReduceMax(reduceLocal, scoreLocal, reduceWorkLocal, calcCount, false);
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        const float blockMaxScore = reduceLocal.GetValue(0);
        if (blockMaxScore == NEG_INF) {
            LocalTensor<float> weightLocal = GetKvBlockSlice<float>(weightFloatBuf_, bufferIdx);
            Duplicate(weightLocal, 0.0F, calcCount);
            PipeBarrier<PIPE_V>();
            StoreProbRow(rowInBlock, weightLocal, kvCount, bufferIdx);
            return;
        }

        if (blockMaxScore > maxScore) {
            const float oldScale = ExpScalarWithVector(maxScore - blockMaxScore, bufferIdx);
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
        StoreProbRow(rowInBlock, weightLocal, kvCount, bufferIdx);

        ReduceSum(reduceLocal, weightLocal, reduceWorkLocal, calcCount);
        SetFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        WaitFlag<HardEvent::V_S>(EVENT_ID1 + bufferIdx);
        denom += reduceLocal.GetValue(0);
    }

    __aicore__ inline void ProcessVisibleKvBlock(int64_t seq, int64_t qCount, int64_t kvCount,
                                                 int64_t &tileLoop, LocalTensor<float> &maxLocal,
                                                 LocalTensor<float> &denomLocal)
    {
        const uint32_t bufferIdx = static_cast<uint32_t>(tileLoop % BUFFER_NUM);
        CrossCoreWaitFlag(AicToAivSyncFlag(bufferIdx));
        for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
            const int64_t querySeq = seq + row;
            if (querySeq >= tiling_->validSeqLen) {
                continue;
            }
            LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
            float maxScore = maxLocal.GetValue(row);
            float denom = denomLocal.GetValue(row);
            AccumulateVisibleScoreBlock(accRow, kvCount, static_cast<uint32_t>(row), maxScore, denom, bufferIdx);
            maxLocal.SetValue(row, maxScore);
            denomLocal.SetValue(row, denom);
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(AivProbToAicSyncFlag(bufferIdx));
        CrossCoreWaitFlag(AicOutToAivSyncFlag(bufferIdx));
        for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
            const int64_t querySeq = seq + row;
            if (querySeq >= tiling_->validSeqLen) {
                continue;
            }
            LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
            AddPvOutRowToAcc(accRow, static_cast<uint32_t>(row), bufferIdx);
        }
        CrossCoreSetFlag<2, PIPE_MTE2>(AivToAicSyncFlag(bufferIdx));
        ++tileLoop;
    }

    __aicore__ inline void ProcessImageQueryImageTile(int64_t seq, int64_t qCount, int64_t &tileLoop,
                                                      LocalTensor<float> &maxLocal,
                                                      LocalTensor<float> &denomLocal)
    {
        ProcessVisibleKvBlock(seq, qCount, tiling_->tileTokenNum, tileLoop, maxLocal, denomLocal);
    }

    __aicore__ inline void ProcessTextQueryImageTile(int64_t seq, int64_t qCount, int64_t &tileLoop,
                                                     LocalTensor<float> &maxLocal,
                                                     LocalTensor<float> &denomLocal)
    {
        ProcessVisibleKvBlock(seq, qCount, tiling_->tileTokenNum, tileLoop, maxLocal, denomLocal);
    }

    __aicore__ inline void ProcessTextKvRange(int64_t seq, int64_t qCount, int64_t kvBegin, int64_t kvEnd,
                                              int64_t &tileLoop, LocalTensor<float> &maxLocal,
                                              LocalTensor<float> &denomLocal)
    {
        if (seq >= tiling_->validSeqLen) {
            return;
        }
        for (int64_t kvSeq = kvBegin; kvSeq < kvEnd; kvSeq += tiling_->kvBlock) {
            const int64_t kvCount = MinInt64(tiling_->kvBlock, kvEnd - kvSeq);
            ProcessVisibleKvBlock(seq, qCount, kvCount, tileLoop, maxLocal, denomLocal);
        }
    }

    __aicore__ inline void ProcessImageKvTiles(int64_t head, int64_t seq, int64_t qCount,
                                               int64_t &tileLoop, LocalTensor<float> &maxLocal,
                                               LocalTensor<float> &denomLocal)
    {
        if (seq >= tiling_->validSeqLen) {
            return;
        }
        if ((tiling_->hasText != 0) && (seq >= tiling_->imageSeqLen)) {
            for (int64_t kvTileId = 0; kvTileId < tiling_->imageTileNum; ++kvTileId) {
                ProcessTextQueryImageTile(seq, qCount, tileLoop, maxLocal, denomLocal);
            }
            return;
        }
        if (seq >= tiling_->imageSeqLen) {
            return;
        }

        int64_t tBegin = 0;
        int64_t tEnd = 0;
        int64_t hBegin = 0;
        int64_t hEnd = 0;
        int64_t wBegin = 0;
        int64_t wEnd = 0;
        GetImageWindowRange(head, seq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
        for (int64_t tileT = tBegin; tileT < tEnd; ++tileT) {
            for (int64_t tileH = hBegin; tileH < hEnd; ++tileH) {
                for (int64_t tileW = wBegin; tileW < wEnd; ++tileW) {
                    ProcessImageQueryImageTile(seq, qCount, tileLoop, maxLocal, denomLocal);
                }
            }
        }
    }

    __aicore__ inline void ProcessScoreTileTask(int64_t taskIdx)
    {
        int64_t batch = 0;
        int64_t head = 0;
        int64_t seq = 0;
        DecodeCubeTask(taskIdx, batch, head, seq);
        const int64_t qCount = MinInt64(tiling_->qBlock, tiling_->seqLen - seq);
        if (qCount <= 0 || tiling_->validSeqLen <= 0) {
            return;
        }

        LocalTensor<float> maxLocal = maxFloatBuf_.Get<float>();
        LocalTensor<float> denomLocal = denomFloatBuf_.Get<float>();
        for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
            const int64_t querySeq = seq + row;
            const int64_t qBase = TensorOffset(batch, head, querySeq, 0);
            LocalTensor<float> accRow = GetAccTileRow(static_cast<uint32_t>(row));
            if (querySeq >= tiling_->validSeqLen) {
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
        ProcessImageKvTiles(head, seq, qCount, tileLoop, maxLocal, denomLocal);
        ProcessTextKvRange(seq, qCount, textBegin, textEnd, tileLoop, maxLocal, denomLocal);

        for (int64_t row = GetSubBlockIdxValue(); row < qCount; row += GetTaskRatioValue()) {
            const int64_t querySeq = seq + row;
            if (querySeq >= tiling_->validSeqLen) {
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

private:
    GlobalTensor<DataType> yDataGm_;
    GlobalTensor<float> scoreGm_;
    GlobalTensor<DataType> probGm_;
    GlobalTensor<float> outGm_;
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> yDataBuf_;
    TBuf<QuePosition::VECCALC> probDataBuf_;
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
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
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
            (void)batch;
            int64_t textBegin = 0;
            int64_t textEnd = 0;
            GetTextScanRange(textBegin, textEnd);
            int64_t tileLoop = 0;
            ProcessImageKvTiles(taskIdx, head, seq, tileLoop);
            ProcessCubeTextKvRange(taskIdx, seq, textBegin, textEnd, tileLoop);
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
        const int64_t seqBlock = taskIdx % tiling_->seqBlockNum;
        const int64_t tmp = taskIdx / tiling_->seqBlockNum;
        head = tmp % tiling_->headNum;
        batch = tmp / tiling_->headNum;
        seq = seqBlock * tiling_->qBlock;
    }

    __aicore__ inline void DecodeTileCoord(int64_t tileId, int64_t &tileT, int64_t &tileH, int64_t &tileW) const
    {
        tileT = tileId / tiling_->tileNumHW;
        const int64_t rem = tileId - tileT * tiling_->tileNumHW;
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

    __aicore__ inline void GetTextScanRange(int64_t &textBegin, int64_t &textEnd) const
    {
        if (tiling_->hasText != 0 && tiling_->textLength > 0) {
            textBegin = tiling_->imageSeqLen;
            textEnd = tiling_->validSeqLen;
        } else {
            textBegin = tiling_->validSeqLen;
            textEnd = tiling_->validSeqLen;
        }
    }

    __aicore__ inline void ProcessCubeVisibleKvBlock(int64_t taskIdx, int64_t kvSeq, int64_t kvCount,
                                                     int64_t &tileLoop)
    {
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

    __aicore__ inline void ProcessCubeVisibleKvRange(int64_t taskIdx, int64_t kvBegin, int64_t kvEnd,
                                                     int64_t &tileLoop)
    {
        for (int64_t kvSeq = kvBegin; kvSeq < kvEnd; kvSeq += tiling_->kvBlock) {
            const int64_t kvCount = MinInt64(tiling_->kvBlock, kvEnd - kvSeq);
            ProcessCubeVisibleKvBlock(taskIdx, kvSeq, kvCount, tileLoop);
        }
    }

    __aicore__ inline void ProcessCubeImageQueryImageTile(int64_t taskIdx, int64_t kvBegin, int64_t &tileLoop)
    {
        ProcessCubeVisibleKvBlock(taskIdx, kvBegin, tiling_->tileTokenNum, tileLoop);
    }

    __aicore__ inline void ProcessCubeTextQueryImageTile(int64_t taskIdx, int64_t kvBegin, int64_t &tileLoop)
    {
        ProcessCubeVisibleKvBlock(taskIdx, kvBegin, tiling_->tileTokenNum, tileLoop);
    }

    __aicore__ inline void ProcessCubeTextKvRange(int64_t taskIdx, int64_t seq, int64_t kvBegin, int64_t kvEnd,
                                                  int64_t &tileLoop)
    {
        if (seq >= tiling_->validSeqLen) {
            return;
        }
        ProcessCubeVisibleKvRange(taskIdx, kvBegin, kvEnd, tileLoop);
    }

    __aicore__ inline void ProcessImageKvTiles(int64_t taskIdx, int64_t head, int64_t seq, int64_t &tileLoop)
    {
        if (seq >= tiling_->validSeqLen) {
            return;
        }
        if ((tiling_->hasText != 0) && (seq >= tiling_->imageSeqLen)) {
            for (int64_t kvTileId = 0; kvTileId < tiling_->imageTileNum; ++kvTileId) {
                const int64_t kvBegin = kvTileId * tiling_->tileTokenNum;
                ProcessCubeTextQueryImageTile(taskIdx, kvBegin, tileLoop);
            }
            return;
        }
        if (seq >= tiling_->imageSeqLen) {
            return;
        }

        int64_t tBegin = 0;
        int64_t tEnd = 0;
        int64_t hBegin = 0;
        int64_t hEnd = 0;
        int64_t wBegin = 0;
        int64_t wEnd = 0;
        GetImageWindowRange(head, seq, tBegin, tEnd, hBegin, hEnd, wBegin, wEnd);
        for (int64_t tileT = tBegin; tileT < tEnd; ++tileT) {
            for (int64_t tileH = hBegin; tileH < hEnd; ++tileH) {
                for (int64_t tileW = wBegin; tileW < wEnd; ++tileW) {
                    const int64_t kvTileId = TileLinearId(tileT, tileH, tileW);
                    const int64_t kvBegin = kvTileId * tiling_->tileTokenNum;
                    ProcessCubeImageQueryImageTile(taskIdx, kvBegin, tileLoop);
                }
            }
        }
    }

    __aicore__ inline LocalTensor<float> GetScoreHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t sharedElems = MaxInt64(scoreElems, outElems);
        const int64_t offset =
            SCORE_HANDOFF_UB_OFFSET + static_cast<int64_t>(bufferIdx) * sharedElems * sizeof(float);
        return LocalTensor<float>(TPosition::VECIN, offset, scoreElems);
    }

    __aicore__ inline LocalTensor<DataType> GetProbHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t probElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t sharedElems = MaxInt64(scoreElems, outElems);
        const int64_t offset = SCORE_HANDOFF_UB_OFFSET + BUFFER_NUM * sharedElems * sizeof(float) +
                               static_cast<int64_t>(bufferIdx) * probElems * sizeof(DataType);
        return LocalTensor<DataType>(TPosition::VECIN, offset, probElems);
    }

    __aicore__ inline LocalTensor<float> GetOutHandoffTensor(uint32_t bufferIdx) const
    {
        const int64_t scoreElems = tiling_->qBlock * tiling_->maxKvBlock;
        const int64_t outElems = tiling_->qBlock * tiling_->headDimAlign;
        const int64_t sharedElems = MaxInt64(scoreElems, outElems);
        const int64_t offset =
            SCORE_HANDOFF_UB_OFFSET + static_cast<int64_t>(bufferIdx) * sharedElems * sizeof(float);
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

        qkMatmul_.SetOrgShape(qCount, kvCount, tiling_->headDim, tiling_->headDim, tiling_->maxKvBlock);
        qkMatmul_.SetTail(qCount, kvCount, tiling_->headDim);
        qkMatmul_.SetTensorA(qDataGm_[qBase]);
        qkMatmul_.SetTensorB(kDataGm_[kvBase], true);
#if STA_USE_VECIN_SCORE_HANDOFF
        LocalTensor<float> scoreLocal = GetScoreHandoffTensor(bufferIdx);
        qkMatmul_.template IterateAll<false>(scoreLocal, 0);
#else
        const int64_t scoreOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->maxKvBlock;
        qkMatmul_.template IterateAll<false>(scoreGm_[scoreOffset], false, false, true);
#endif
        qkMatmul_.WaitIterateAll();
        qkMatmul_.End();
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
        pvMatmul_.SetOrgShape(qCount, tiling_->headDim, kvCount, tiling_->maxKvBlock, tiling_->headDim);
        pvMatmul_.SetTail(qCount, tiling_->headDim, kvCount);
#if STA_USE_VECIN_PROB_OUT_HANDOFF
        LocalTensor<DataType> probLocal = GetProbHandoffTensor(bufferIdx);
        LocalTensor<float> outLocal = GetOutHandoffTensor(bufferIdx);
        pvMatmul_.SetTensorA(probLocal);
        pvMatmul_.SetTensorB(vDataGm_[kvBase]);
        pvMatmul_.template IterateAll<false>(outLocal, 0);
#else
        const int64_t probOffset = static_cast<int64_t>(bufferIdx) * tiling_->qBlock * tiling_->maxKvBlock;
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
