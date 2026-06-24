#ifndef CAUSAL_CONV1D_DECODE_CUT_BH_H
#define CAUSAL_CONV1D_DECODE_CUT_BH_H

#include "kernel_operator.h"
#include "causal_conv1d_decode_cut_bh_struct.h"
#include "vf/causal_conv1d_decode_width4_vf.h"

using namespace AscendC;

constexpr int32_t CCDC_DOUBLE_BUFFER = 2;
constexpr int32_t CCDC_SINGLE_BUFFER = 1;
constexpr int32_t CCDC_ALIGN_BYTES = 32;
constexpr int32_t CCDC_MAX_WIDTH = 4;

class CausalConv1dDecodeCutBH {
public:
    __aicore__ inline explicit CausalConv1dDecodeCutBH(TPipe *pipe) : pipe_(pipe) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR bias, GM_ADDR numAcceptedTokens, GM_ADDR y,
                                const CausalConv1dDecodeCutBHTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams(const CausalConv1dDecodeCutBHTilingData *tilingData);
    __aicore__ inline void InitQueues();
    __aicore__ inline void CopyWeightBias();
    __aicore__ inline void CopyX(uint16_t blockCount, int64_t xOffset);
    __aicore__ inline void Compute(int32_t firstBatchIdxInLoop, uint16_t yBlockCount, int64_t yOffset,
                                   const LocalTensor<int32_t> &queryLocal,
                                   const LocalTensor<int32_t> &indicesLocal,
                                   const LocalTensor<int32_t> &acceptedLocal,
                                   const LocalTensor<bfloat16_t> &weightBiasLocal);
    __aicore__ inline int32_t GetHistoryBase(int32_t accepted) const;
    __aicore__ inline void UpdateConvStates(const LocalTensor<bfloat16_t> &xLocal,
                                            const LocalTensor<bfloat16_t> &stateLocal,
                                            int32_t curBatchUbOffset, int32_t tokenCount, int32_t accepted,
                                            int64_t writeSlot);
    template <HardEvent event>
    __aicore__ inline void SetWaitFlag(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> weightGm;
    GlobalTensor<bfloat16_t> convStatesInGm;
    GlobalTensor<bfloat16_t> convStatesOutGm;
    GlobalTensor<bfloat16_t> biasGm;
    GlobalTensor<bfloat16_t> yGm;
    GlobalTensor<int32_t> queryStartLocGm;
    GlobalTensor<int32_t> cacheIndicesGm;
    GlobalTensor<int32_t> acceptedGm;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> stateQueueIn;
    TQue<QuePosition::VECOUT, 1> yQueue;
    TBuf<TPosition::VECCALC> weightBiasBuf;
    TBuf<TPosition::VECCALC> indicesBuf;
    TBuf<TPosition::VECCALC> acceptedBuf;
    TBuf<TPosition::VECCALC> queryStartLocBuf;
    TPipe *pipe_;

    int64_t dimCoreCnt_;
    int64_t dimMainCoreCnt_;
    int64_t mainCoreDimLen_;
    int64_t tailCoreDimLen_;
    int64_t batchMainCoreCnt_;
    int64_t mainCoreBatchNum_;
    int64_t tailCoreBatchNum_;
    int64_t loopNumBS_;
    int64_t loopNumDim_;
    int64_t ubMainFactorBS_;
    int64_t ubTailFactorBS_;
    int64_t ubMainFactorDim_;
    int64_t ubTailFactorDim_;
    int64_t batchSize_;
    int64_t seqLen_;
    int64_t dim_;
    int64_t stateLen_;
    int64_t xStride_;
    int64_t cacheStride0_;
    int64_t cacheStride1_;
    int64_t padSlotId_;
    int64_t xInputMode_;
    int64_t hasBias_;
    int64_t hasQueryStartLoc_;
    int64_t hasCacheIndices_;
    int64_t hasNumAcceptedTokens_;
    int32_t firstBatchIdx_;
    int32_t dimOffset_;
    int32_t batchNumInLoop_;
    int32_t dimSizeInLoop_;
    int32_t dimOffsetPerLoop_;
    int32_t weightPartOffset_;
    int32_t biasPartOffset_;
};

__aicore__ inline void CausalConv1dDecodeCutBH::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR convStates, GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR bias,
    GM_ADDR numAcceptedTokens, GM_ADDR y, const CausalConv1dDecodeCutBHTilingData *tilingData)
{
    InitParams(tilingData);
    xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x);
    weightGm.SetGlobalBuffer((__gm__ bfloat16_t *)weight, CCDC_MAX_WIDTH * dim_);
    convStatesInGm.SetGlobalBuffer((__gm__ bfloat16_t *)convStates);
    convStatesOutGm.SetGlobalBuffer((__gm__ bfloat16_t *)convStates);
    if (hasQueryStartLoc_ != 0) {
        queryStartLocGm.SetGlobalBuffer((__gm__ int32_t *)queryStartLoc, batchSize_ + 1);
    }
    if (hasCacheIndices_ != 0) {
        cacheIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cacheIndices, batchSize_);
    }
    if (hasBias_ != 0) {
        biasGm.SetGlobalBuffer((__gm__ bfloat16_t *)bias, dim_);
    }
    if (hasNumAcceptedTokens_ != 0) {
        acceptedGm.SetGlobalBuffer((__gm__ int32_t *)numAcceptedTokens, batchSize_);
    }
    yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y);
    InitQueues();
}

__aicore__ inline void CausalConv1dDecodeCutBH::InitParams(const CausalConv1dDecodeCutBHTilingData *tilingData)
{
    dimCoreCnt_ = tilingData->dimCoreCnt;
    dimMainCoreCnt_ = tilingData->dimMainCoreCnt;
    mainCoreDimLen_ = tilingData->mainCoreDimLen;
    tailCoreDimLen_ = tilingData->tailCoreDimLen;
    batchMainCoreCnt_ = tilingData->batchMainCoreCnt;
    mainCoreBatchNum_ = tilingData->mainCoreBatchNum;
    tailCoreBatchNum_ = tilingData->tailCoreBatchNum;
    loopNumBS_ = tilingData->loopNumBS;
    loopNumDim_ = tilingData->loopNumDim;
    ubMainFactorBS_ = tilingData->ubMainFactorBS;
    ubTailFactorBS_ = tilingData->ubTailFactorBS;
    ubMainFactorDim_ = tilingData->ubMainFactorDim;
    ubTailFactorDim_ = tilingData->ubTailFactorDim;
    batchSize_ = tilingData->batchSize;
    seqLen_ = tilingData->seqLen;
    dim_ = tilingData->dim;
    stateLen_ = tilingData->stateLen;
    xStride_ = tilingData->xStride;
    cacheStride0_ = tilingData->cacheStride0;
    cacheStride1_ = tilingData->cacheStride1;
    padSlotId_ = tilingData->padSlotId;
    xInputMode_ = tilingData->xInputMode;
    hasBias_ = tilingData->hasBias;
    hasQueryStartLoc_ = tilingData->hasQueryStartLoc;
    hasCacheIndices_ = tilingData->hasCacheIndices;
    hasNumAcceptedTokens_ = tilingData->hasNumAcceptedTokens;

    int32_t blockIdx = GetBlockIdx();
    int32_t batchIdx = blockIdx / dimCoreCnt_;
    int32_t dimIdx = blockIdx - batchIdx * dimCoreCnt_;
    if (dimIdx < dimMainCoreCnt_) {
        dimOffset_ = dimIdx * static_cast<int32_t>(mainCoreDimLen_);
    } else {
        dimOffset_ = static_cast<int32_t>(dimMainCoreCnt_ * mainCoreDimLen_ +
                                          (dimIdx - dimMainCoreCnt_) * tailCoreDimLen_);
        loopNumDim_ = tilingData->lastCoreLoopNumDim;
        ubMainFactorDim_ = tilingData->lastCoreUbMainFactorDim;
        ubTailFactorDim_ = tilingData->lastCoreUbTailFactorDim;
    }
    if (dimIdx == dimCoreCnt_ - 1 && tilingData->dimRemainderElems > 0) {
        loopNumDim_ = tilingData->lastCoreLoopNumDim;
        ubMainFactorDim_ = tilingData->lastCoreUbMainFactorDim;
        ubTailFactorDim_ = tilingData->lastCoreUbTailFactorDim;
    }
    if (batchIdx < batchMainCoreCnt_) {
        firstBatchIdx_ = batchIdx * static_cast<int32_t>(mainCoreBatchNum_);
    } else {
        firstBatchIdx_ = static_cast<int32_t>(batchMainCoreCnt_ * mainCoreBatchNum_ +
                                              (batchIdx - batchMainCoreCnt_) * tailCoreBatchNum_);
        loopNumBS_ = tilingData->tailBlockLoopNumBS;
        ubMainFactorBS_ = tilingData->tailBlockUbFactorBS;
        ubTailFactorBS_ = tilingData->tailBlockUbTailFactorBS;
    }
    weightPartOffset_ = 0;
    biasPartOffset_ = static_cast<int32_t>(CCDC_MAX_WIDTH * ubMainFactorDim_);
}

__aicore__ inline void CausalConv1dDecodeCutBH::InitQueues()
{
    int32_t xElems = static_cast<int32_t>(ubMainFactorBS_ * seqLen_ * ubMainFactorDim_);
    int32_t xBytes = (xElems * sizeof(bfloat16_t) + CCDC_ALIGN_BYTES - 1) / CCDC_ALIGN_BYTES * CCDC_ALIGN_BYTES;
    int32_t weightBiasBytes = ((CCDC_MAX_WIDTH + 1) * ubMainFactorDim_ * sizeof(bfloat16_t) + CCDC_ALIGN_BYTES - 1) /
                              CCDC_ALIGN_BYTES * CCDC_ALIGN_BYTES;
    int32_t stateBytes = ((CCDC_MAX_WIDTH - 1) * ubMainFactorDim_ * sizeof(bfloat16_t) + CCDC_ALIGN_BYTES - 1) /
                         CCDC_ALIGN_BYTES * CCDC_ALIGN_BYTES;
    int32_t yBytes = (ubMainFactorBS_ * seqLen_ * ubMainFactorDim_ * sizeof(bfloat16_t) + CCDC_ALIGN_BYTES - 1) /
                     CCDC_ALIGN_BYTES * CCDC_ALIGN_BYTES;
    int32_t batchBytes = (batchSize_ * sizeof(int32_t) + CCDC_ALIGN_BYTES - 1) / CCDC_ALIGN_BYTES *
                         CCDC_ALIGN_BYTES;
    int32_t queryBytes = ((batchSize_ + 1) * sizeof(int32_t) + CCDC_ALIGN_BYTES - 1) / CCDC_ALIGN_BYTES *
                         CCDC_ALIGN_BYTES;
    // CopyX and Compute are issued serially per batch tile; the EnQue/DeQue pair is the x sync boundary.
    pipe_->InitBuffer(xQueue, CCDC_SINGLE_BUFFER, xBytes);
    pipe_->InitBuffer(stateQueueIn, CCDC_SINGLE_BUFFER, stateBytes);
    pipe_->InitBuffer(yQueue, CCDC_SINGLE_BUFFER, yBytes);
    pipe_->InitBuffer(weightBiasBuf, weightBiasBytes);
    pipe_->InitBuffer(indicesBuf, batchBytes);
    pipe_->InitBuffer(acceptedBuf, batchBytes);
    pipe_->InitBuffer(queryStartLocBuf, queryBytes);
}

__aicore__ inline void CausalConv1dDecodeCutBH::Process()
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams batchCopyParams;
    batchCopyParams.blockCount = 1;
    batchCopyParams.blockLen = batchSize_ * sizeof(int32_t);
    batchCopyParams.srcStride = 0;
    batchCopyParams.dstStride = 0;
    auto indicesLocal = indicesBuf.Get<int32_t>();
    auto acceptedLocal = acceptedBuf.Get<int32_t>();
    auto queryLocal = queryStartLocBuf.Get<int32_t>();
    for (int32_t b = 0; b < batchSize_; ++b) {
        indicesLocal.SetValue(b, b);
        acceptedLocal.SetValue(b, static_cast<int32_t>(seqLen_));
        queryLocal.SetValue(b, b * static_cast<int32_t>(seqLen_));
    }
    queryLocal.SetValue(static_cast<int32_t>(batchSize_), static_cast<int32_t>(batchSize_ * seqLen_));
    if (hasCacheIndices_ != 0) {
        DataCopyPad(indicesLocal, cacheIndicesGm, batchCopyParams, padParams);
    }
    // Default broadcast for the optional numAcceptedTokens input.
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    if (hasNumAcceptedTokens_ != 0) {
        DataCopyPad(acceptedLocal, acceptedGm, batchCopyParams, padParams);
    }
    if (hasQueryStartLoc_ != 0) {
        DataCopyParams queryCopyParams;
        queryCopyParams.blockCount = 1;
        queryCopyParams.blockLen = (batchSize_ + 1) * sizeof(int32_t);
        queryCopyParams.srcStride = 0;
        queryCopyParams.dstStride = 0;
        DataCopyPad(queryLocal, queryStartLocGm, queryCopyParams, padParams);
    }
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    auto weightBiasLocal = weightBiasBuf.Get<bfloat16_t>();
    for (int32_t dimLoop = 0; dimLoop < loopNumDim_; ++dimLoop) {
        dimSizeInLoop_ = (dimLoop == loopNumDim_ - 1) ? ubTailFactorDim_ : ubMainFactorDim_;
        dimOffsetPerLoop_ = dimLoop * static_cast<int32_t>(ubMainFactorDim_);
        CopyWeightBias();
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        for (int32_t batchLoop = 0; batchLoop < loopNumBS_; ++batchLoop) {
            batchNumInLoop_ = (batchLoop == loopNumBS_ - 1) ? ubTailFactorBS_ : ubMainFactorBS_;
            int32_t firstBatchIdxInLoop = firstBatchIdx_ + batchLoop * static_cast<int32_t>(ubMainFactorBS_);
            int32_t rowStart = queryLocal.GetValue(firstBatchIdxInLoop);
            int32_t rowEnd = queryLocal.GetValue(firstBatchIdxInLoop + batchNumInLoop_);
            uint16_t blockCount = static_cast<uint16_t>(rowEnd - rowStart);
            int64_t xOffset = static_cast<int64_t>(rowStart) * xStride_ + dimOffset_ + dimOffsetPerLoop_;
            int64_t yOffset = static_cast<int64_t>(rowStart) * dim_ + dimOffset_ + dimOffsetPerLoop_;
            CopyX(blockCount, xOffset);
            Compute(firstBatchIdxInLoop, blockCount, yOffset, queryLocal, indicesLocal, acceptedLocal,
                    weightBiasLocal);
        }
    }
}

__aicore__ inline void CausalConv1dDecodeCutBH::CopyWeightBias()
{
    auto weightBiasLocal = weightBiasBuf.Get<bfloat16_t>();
    auto weightLocal = weightBiasLocal[weightPartOffset_];
    auto biasLocal = weightBiasLocal[biasPartOffset_];
    DataCopyPadParams padParams{false, 0, 0, 0};
    int32_t dimOffset = dimOffset_ + dimOffsetPerLoop_;
    DataCopyParams weightParams;
    weightParams.blockCount = CCDC_MAX_WIDTH;
    weightParams.blockLen = dimSizeInLoop_ * sizeof(bfloat16_t);
    weightParams.srcStride = (dim_ - dimSizeInLoop_) * sizeof(bfloat16_t);
    weightParams.dstStride = 0;
    // Weight is fixed [4, dim] for the current decode-only baseline.
    DataCopyPad(weightLocal, weightGm[dimOffset], weightParams, padParams);
    if (hasBias_ != 0) {
        DataCopyParams biasParams;
        biasParams.blockCount = 1;
        biasParams.blockLen = dimSizeInLoop_ * sizeof(bfloat16_t);
        biasParams.srcStride = 0;
        biasParams.dstStride = 0;
        // Bias is logically broadcast over batch and sequence inside the VF call.
        DataCopyPad(biasLocal, biasGm[dimOffset], biasParams, padParams);
    }
}

__aicore__ inline void CausalConv1dDecodeCutBH::CopyX(uint16_t blockCount, int64_t xOffset)
{
    auto xLocal = xQueue.AllocTensor<bfloat16_t>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams xParams;
    xParams.blockCount = blockCount;
    xParams.blockLen = dimSizeInLoop_ * sizeof(bfloat16_t);
    xParams.srcStride = (xStride_ - dimSizeInLoop_) * sizeof(bfloat16_t);
    xParams.dstStride = 0;
    // Copy a [batchNumInLoop * seqLen, dimSizeInLoop] GM slice into compact UB rows.
    DataCopyPad(xLocal, xGm[xOffset], xParams, padParams);
    xQueue.EnQue(xLocal);
}

__aicore__ inline void CausalConv1dDecodeCutBH::Compute(
    int32_t firstBatchIdxInLoop, uint16_t yBlockCount, int64_t yOffset, const LocalTensor<int32_t> &queryLocal,
    const LocalTensor<int32_t> &indicesLocal, const LocalTensor<int32_t> &acceptedLocal,
    const LocalTensor<bfloat16_t> &weightBiasLocal)
{
    auto xLocal = xQueue.DeQue<bfloat16_t>();
    auto weightLocal = weightBiasLocal[weightPartOffset_];
    auto biasLocal = weightBiasLocal[biasPartOffset_];
    auto yLocal = yQueue.AllocTensor<bfloat16_t>();
    Duplicate(yLocal, static_cast<bfloat16_t>(0), yBlockCount * dimSizeInLoop_);
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams stateParams;
    stateParams.blockCount = CCDC_MAX_WIDTH - 1;
    stateParams.blockLen = dimSizeInLoop_ * sizeof(bfloat16_t);
    stateParams.srcStride = (cacheStride1_ - dimSizeInLoop_) * sizeof(bfloat16_t);
    stateParams.dstStride = 0;
    int32_t tileRowStart = queryLocal.GetValue(firstBatchIdxInLoop);

    for (int32_t b = 0; b < batchNumInLoop_; ++b) {
        int32_t curBatchIdx = firstBatchIdxInLoop + b;
        int64_t slot = (hasCacheIndices_ != 0) ? indicesLocal.GetValue(curBatchIdx) : curBatchIdx;
        int32_t curRowStart = queryLocal.GetValue(curBatchIdx);
        int32_t curRowEnd = queryLocal.GetValue(curBatchIdx + 1);
        int32_t tokenCount = curRowEnd - curRowStart;
        int32_t curBatchUbOffset = (curRowStart - tileRowStart) * dimSizeInLoop_;
        if (slot == padSlotId_) {
            continue;
        }
        int32_t accepted = acceptedLocal.GetValue(curBatchIdx);
        accepted = accepted < 0 ? 0 : accepted;
        accepted = accepted > tokenCount ? tokenCount : accepted;
        int32_t historyBase = GetHistoryBase(accepted);
        auto stateLocal = stateQueueIn.AllocTensor<bfloat16_t>();
        int64_t stateGmOffset = slot * cacheStride0_ + historyBase * cacheStride1_ + dimOffset_ + dimOffsetPerLoop_;
        // Copy only the W-1 history rows needed by decode into compact UB rows.
        DataCopyPad(stateLocal, convStatesInGm[stateGmOffset], stateParams, padParams);
        stateQueueIn.EnQue<bfloat16_t>(stateLocal);
        stateLocal = stateQueueIn.DeQue<bfloat16_t>();
        if (tokenCount >= 2) {
            int32_t base = CCDC_MAX_WIDTH - 1;
            auto line0 = stateLocal[0];
            auto line1 = stateLocal[dimSizeInLoop_];
            auto line2 = stateLocal[2 * dimSizeInLoop_];
            auto line3 = xLocal[curBatchUbOffset];
            auto line4 = (tokenCount >= 2) ? xLocal[curBatchUbOffset + dimSizeInLoop_] : xLocal[curBatchUbOffset];
            auto line5 = (tokenCount >= 3) ? xLocal[curBatchUbOffset + 2 * dimSizeInLoop_] : xLocal[curBatchUbOffset];
            auto line6 = (tokenCount >= 4) ? xLocal[curBatchUbOffset + 3 * dimSizeInLoop_] : xLocal[curBatchUbOffset];
            auto y0 = yLocal[curBatchUbOffset];
            auto y1 = yLocal[curBatchUbOffset + dimSizeInLoop_];
            auto y2 = (tokenCount >= 3) ? yLocal[curBatchUbOffset + 2 * dimSizeInLoop_] : y0;
            auto y3 = (tokenCount >= 4) ? yLocal[curBatchUbOffset + 3 * dimSizeInLoop_] : y0;
            Conv1dDecodeMultiS(line0, line1, line2, line3, line4, line5, line6, weightLocal, biasLocal, y0, y1, y2, y3,
                               dimSizeInLoop_, hasBias_, tokenCount);
            UpdateConvStates(xLocal, stateLocal, curBatchUbOffset, tokenCount, accepted, slot);
            stateQueueIn.FreeTensor<bfloat16_t>(stateLocal);
            continue;
        }
        for (int32_t s = 0; s < tokenCount; ++s) {
            int32_t base = CCDC_MAX_WIDTH - 1;
            int32_t rel0 = s - base;
            int32_t rel1 = s - base + 1;
            int32_t rel2 = s - base + 2;
            int32_t rel3 = s - base + 3;
            auto src0 = (rel0 < 0) ? stateLocal[(rel0 + base) * dimSizeInLoop_]
                                   : xLocal[curBatchUbOffset + rel0 * dimSizeInLoop_];
            auto src1 = (rel1 < 0) ? stateLocal[(rel1 + base) * dimSizeInLoop_]
                                   : xLocal[curBatchUbOffset + rel1 * dimSizeInLoop_];
            auto src2 = (rel2 < 0) ? stateLocal[(rel2 + base) * dimSizeInLoop_]
                                  : xLocal[curBatchUbOffset + rel2 * dimSizeInLoop_];
            auto src3 = (rel3 < 0) ? stateLocal[(rel3 + base) * dimSizeInLoop_]
                                  : xLocal[curBatchUbOffset + rel3 * dimSizeInLoop_];
            auto ySlice = yLocal[curBatchUbOffset + s * dimSizeInLoop_];
            Conv1dDecodeW4(src0, src1, src2, src3, weightLocal, biasLocal, ySlice, dimSizeInLoop_, hasBias_);
        }
        UpdateConvStates(xLocal, stateLocal, curBatchUbOffset, tokenCount, accepted, slot);
        stateQueueIn.FreeTensor<bfloat16_t>(stateLocal);
    }
    yQueue.EnQue<bfloat16_t>(yLocal);
    yLocal = yQueue.DeQue<bfloat16_t>();
    DataCopyParams yParams;
    yParams.blockCount = yBlockCount;
    yParams.blockLen = dimSizeInLoop_ * sizeof(bfloat16_t);
    yParams.srcStride = 0;
    yParams.dstStride = (dim_ - dimSizeInLoop_) * sizeof(bfloat16_t);
    // Write compact UB rows back to strided GM [batch, seq, dim].
    DataCopyPad(yGm[yOffset], yLocal, yParams);
    yQueue.FreeTensor<bfloat16_t>(yLocal);
    xQueue.FreeTensor<bfloat16_t>(xLocal);
}

__aicore__ inline int32_t CausalConv1dDecodeCutBH::GetHistoryBase(int32_t accepted) const
{
    int32_t base = CCDC_MAX_WIDTH - 1;
    if (hasNumAcceptedTokens_ == 0) {
        return 0;
    }
    int32_t historyBase = accepted - 1;
    int32_t maxBase = static_cast<int32_t>(stateLen_) - base;
    if (historyBase < 0) {
        historyBase = 0;
    }
    if (historyBase > maxBase) {
        historyBase = maxBase;
    }
    return historyBase;
}

__aicore__ inline void CausalConv1dDecodeCutBH::UpdateConvStates(
    const LocalTensor<bfloat16_t> &xLocal, const LocalTensor<bfloat16_t> &stateLocal, int32_t curBatchUbOffset,
    int32_t tokenCount, int32_t accepted, int64_t writeSlot)
{
    uint32_t blockLen = dimSizeInLoop_ * sizeof(bfloat16_t);
    uint32_t dstStride = (cacheStride1_ - dimSizeInLoop_) * sizeof(bfloat16_t);
    int64_t dstBase = writeSlot * cacheStride0_ + dimOffset_ + dimOffsetPerLoop_;

    if (hasNumAcceptedTokens_ != 0 && stateLen_ >= tokenCount + 2) {
        DataCopyParams keepParams;
        keepParams.blockCount = 2;
        keepParams.blockLen = blockLen;
        keepParams.srcStride = 0;
        keepParams.dstStride = dstStride;
        DataCopyPad(convStatesOutGm[dstBase], stateLocal[dimSizeInLoop_], keepParams);
        if (tokenCount > 0) {
            DataCopyParams xParams;
            xParams.blockCount = tokenCount;
            xParams.blockLen = blockLen;
            xParams.srcStride = 0;
            xParams.dstStride = dstStride;
            int64_t dstX = dstBase + 2 * cacheStride1_;
            DataCopyPad(convStatesOutGm[dstX], xLocal[curBatchUbOffset], xParams);
        }
        return;
    }

    int32_t historyRows = CCDC_MAX_WIDTH - 1;
    int32_t keepRows = historyRows - accepted;
    keepRows = keepRows < 0 ? 0 : keepRows;
    int32_t xRows = historyRows - keepRows;
    if (keepRows > 0) {
        DataCopyParams keepParams;
        keepParams.blockCount = keepRows;
        keepParams.blockLen = blockLen;
        keepParams.srcStride = 0;
        keepParams.dstStride = dstStride;
        // Keep the suffix of the W-1 decode history that survives accepted-token advancement.
        DataCopyPad(convStatesOutGm[dstBase], stateLocal[accepted * dimSizeInLoop_], keepParams);
    }
    if (xRows > 0) {
        DataCopyParams xParams;
        xParams.blockCount = xRows;
        xParams.blockLen = blockLen;
        xParams.srcStride = 0;
        xParams.dstStride = dstStride;
        int32_t srcX = curBatchUbOffset + (accepted - xRows) * dimSizeInLoop_;
        int64_t dstX = dstBase + keepRows * cacheStride1_;
        // Append the newest accepted x rows to finish the W-1 decode history window.
        DataCopyPad(convStatesOutGm[dstX], xLocal[srcX], xParams);
    }
}

#endif
