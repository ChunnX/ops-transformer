/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update.h
 * \brief Decode mode — sequence-tiled incremental causal conv1d.
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d.h"

namespace CausalConv1d {

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dUpdate : public CausalConv1d<T, inputModeKey, widthKey, hasBiasKey, activationKey> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR, GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut,
                                GM_ADDR y, GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
    {
        (void)workspace;
        this->InitGlobalBuffers(x, weight, bias, convStates, queryStartLoc, cacheIndices, y, convStatesOut, tilingData);
        this->numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(numAcceptedTokens));
        this->InitBuffers();
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsVarlenInputModeKey(inputModeKey)) {
            this->template ProcessImpl<SEQ_PARTITION_MODE_VARLEN>();
        } else {
            this->template ProcessImpl<SEQ_PARTITION_MODE_BATCH>();
        }
    }

protected:
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessImpl()
    {
        int32_t channelStart;
        int32_t curBaseDim;
        int64_t seqStart;
        int64_t seqEnd;
        if (!this->template InitBlock<kWindowMode>(channelStart, curBaseDim, seqStart, seqEnd)) {
            return;
        }
        this->template ProcessBlock<kWindowMode>(channelStart, curBaseDim, seqStart, seqEnd);
    }

    template <int32_t kWindowMode>
    __aicore__ inline bool InitBlock(int32_t &channelStart, int32_t &curBaseDim, int64_t &seqStart, int64_t &seqEnd)
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t batch = this->tilingData_->batch;
        const int32_t baseDim = this->tilingData_->baseDim;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const uint32_t blockIdx = GetBlockIdx();
        const int64_t tokensPerBlock = this->tilingData_->tokensPerBlock;
        const int64_t tokenBlockCnt = this->tilingData_->tokenBlockCnt;
        const int64_t coreCnt = tokenBlockCnt * baseDimCnt;
        if (static_cast<int64_t>(blockIdx) >= coreCnt) {
            return false;
        }

        const int32_t tokBlock = blockIdx / baseDimCnt;
        const int32_t baseDimIdx = blockIdx % baseDimCnt;
        channelStart = baseDimIdx * baseDim;
        curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);

        this->InitCalcBuf(curBaseDim);
        this->LoadWeightAndBias(channelStart, curBaseDim);

        const int64_t tokensPerGroupBlock = tokensPerBlock / coBatch;
        const int64_t batchGroups = batch / coBatch;
        seqStart = tokBlock * tokensPerGroupBlock;
        seqEnd = (seqStart + tokensPerGroupBlock < batchGroups) ? (seqStart + tokensPerGroupBlock) : batchGroups;
        return true;
    }

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessBlock(int32_t channelStart, int32_t curBaseDim, int64_t seqStart, int64_t seqEnd)
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t seqLen = this->tilingData_->seqLen;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);

        for (int64_t batchIdx = seqStart; batchIdx < seqEnd; ++batchIdx) {
            int32_t curSeqStart = 0;
            int32_t curSeqEnd = 0;
            this->template GetSeqWindow<kWindowMode>(static_cast<int32_t>(batchIdx), seqLen, coBatch, curSeqStart,
                                                     curSeqEnd);
            int32_t len;
            if constexpr (kWindowMode == SEQ_PARTITION_MODE_BATCH) {
                len = seqLen;
            } else {
                len = curSeqEnd - curSeqStart;
            }

            const int32_t cacheIdx = static_cast<int32_t>(batchIdx * coBatch);
            const int32_t stateTokenOffset = this->GetStateTokenOffset(static_cast<int32_t>(batchIdx * coBatch));

            this->InitRing(cacheIdx, stateTokenOffset, curSeqStart, len, channelStart, curBaseDim, dim);

            bool doVarLenStateWrite = false;
            if (this->tilingData_->hasNumAcceptedTokens && len > 0) {
                const int32_t keep = kernelWidth - 2;
                if (keep + len <= this->tilingData_->stateLen) {
                    doVarLenStateWrite = true;
                }
            }

            this->RunSeq(curSeqStart, len, channelStart, curBaseDim, cacheIdx, doVarLenStateWrite);

            if (doVarLenStateWrite) {
                this->WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim, len - 1);
            } else {
                this->WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
            }

            SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }

    __aicore__ inline int32_t GetStateTokenOffset(int32_t seq) const
    {
        if (!this->tilingData_->hasNumAcceptedTokens) {
            return 0;
        }
        int32_t offset = this->numAcceptedTokensGm.GetValue(seq) - 1;
        return (offset >= 0) ? offset : 0;
    }

    __aicore__ inline void InitRing(int32_t cacheIdx, int32_t stateTokenOffset, int32_t start, int32_t len,
                                    int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t dimBufferSize = (coBatch * baseDim);
        const int32_t seqLen = static_cast<int32_t>(this->tilingData_->seqLen);
        LocalTensor<T> ring = this->inBuf.template Get<T>();

        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);

        const uint32_t ubBlockBytes = Ops::Base::GetUbBlockSize();
        const uint32_t ringRowGapBlocks = static_cast<uint32_t>(dimBufferSize - baseDim) * sizeof(T) / ubBlockBytes;
        const int64_t stateBase = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                  static_cast<int64_t>(stateTokenOffset) * dim + channelStart;

        LoopModeParams loopParams;
        loopParams.loop1Size = static_cast<uint32_t>(coBatch);
        loopParams.loop2Size = 1;
        loopParams.loop1SrcStride = static_cast<uint64_t>(stateLen) * dim * sizeof(T);
        loopParams.loop1DstStride = static_cast<uint64_t>(baseDim) * sizeof(T);
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

        DataCopyExtParams copyExt;
        copyExt.blockCount = static_cast<uint16_t>(kernelWidth - 1);
        copyExt.blockLen = blockBytes;
        copyExt.srcStride = srcGap;
        copyExt.dstStride = ringRowGapBlocks;
        copyExt.rsv = 0;
        DataCopyPad(ring[0], this->convStatesGm[stateBase], copyExt, {false, 0, 0, 0});
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);

        const int32_t slot0 = SlotCurr(0, kernelWidth);
        const uint32_t srcGapBatch = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
        DataCopyPad(ring[slot0 * dimBufferSize], this->xGm[start * dim + channelStart],
                    {static_cast<uint16_t>(coBatch), blockBytes, srcGapBatch, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t cacheIdx,
                                  bool doVarLenStateWrite)
    {
        const int32_t dim = static_cast<int32_t>(this->tilingData_->dim);
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t dimBufferSize = (coBatch * baseDim);
        const int32_t seqLen = static_cast<int32_t>(this->tilingData_->seqLen);
        const int32_t stateLen = this->tilingData_->stateLen;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        LocalTensor<T> outT = this->outBuf.template Get<T>();

        for (int32_t t = 0; t < len; ++t) {
            WaitFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);

            if (t + 1 < len) {
                const int32_t slotNext = SlotPrefetch(t, kernelWidth);
                if (t > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>((t & 1) ? EVENT_ID1 : EVENT_ID0);
                }
                const int64_t xGmBase = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                const uint32_t srcGapBatchX = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
                DataCopyPad(ring[slotNext * dimBufferSize], this->xGm[xGmBase],
                            {static_cast<uint16_t>(coBatch), blockBytes, srcGapBatchX, 0, 0}, {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            const int32_t outSlot = t & 1;
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outT[outSlot * dimBufferSize],
                                dimBufferSize, t);

            SetFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);

            const int64_t outGmBase = static_cast<int64_t>(start + t) * dim + channelStart;
            const uint32_t dstGapBatchY = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
            DataCopyPad(this->yGm[outGmBase], outT[outSlot * dimBufferSize],
                        {static_cast<uint16_t>(coBatch), blockBytes, 0, dstGapBatchY, 0});
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            if (doVarLenStateWrite && t >= 1) {
                const int32_t deadSlot = t % (kernelWidth + 1);
                const int64_t stateBase =
                    static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(t - 1) * dim + channelStart;
                const uint32_t dstGapState = static_cast<uint32_t>(stateLen * dim - baseDim) * sizeof(T);
                DataCopyPad(this->convStatesOutGm[stateBase], ring[deadSlot * dimBufferSize],
                            {static_cast<uint16_t>(coBatch), blockBytes, 0, dstGapState, 0});
            }

            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_MTE2>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }
        }
    }
};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y,
                                             GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T, inputModeKey, widthKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    op.Process();
}

} // namespace CausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H
