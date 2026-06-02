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
 * \file causal_conv1d_fn.h
 * \brief Prefill mode — token-tiled full-sequence causal conv1d.
 */

#ifndef CAUSAL_CONV1D_FN_H
#define CAUSAL_CONV1D_FN_H

#include "causal_conv1d.h"

namespace CausalConv1d {

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dFn : public CausalConv1d<T, inputModeKey, widthKey, hasBiasKey, activationKey> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens,
                                GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        (void)numAcceptedTokens;
        this->InitGlobalBuffers(x, weight, bias, convStates, queryStartLoc, cacheIndices, y, convStatesOut, tilingData);
        this->initialStateModeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(initialStateMode));
        if (tilingData->hasInitStateWorkspace) {
            this->initStateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace));
        }
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
    GlobalTensor<T> initStateWorkspaceGm_;

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessImpl()
    {
        int32_t channelStart;
        int32_t curBaseDim;
        int32_t batchIdx;
        int32_t batchCnt;
        int32_t cursor;
        int32_t cursorEnd;
        if (!this->template InitBlock<kWindowMode>(channelStart, curBaseDim, batchIdx, batchCnt,
                                                    cursor, cursorEnd)) {
            return;
        }
        this->template ProcessBlock<kWindowMode>(channelStart, curBaseDim, batchIdx, batchCnt,
                                                  cursor, cursorEnd);
    }

    template <int32_t kWindowMode>
    __aicore__ inline bool InitBlock(int32_t &channelStart, int32_t &curBaseDim,
                                      int32_t &batchIdx, int32_t &batchCnt,
                                      int32_t &cursor, int32_t &cursorEnd)
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t batch = this->tilingData_->batch;
        const int32_t seqLen = this->tilingData_->seqLen;
        const int32_t cuSeqlen = this->tilingData_->cuSeqlen;
        const int32_t baseDim = this->tilingData_->baseDim;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const int32_t tokensPerBlock = static_cast<int32_t>(this->tilingData_->tokensPerBlock);
        const int32_t tokenBlockCnt = static_cast<int32_t>(this->tilingData_->tokenBlockCnt);

        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const int64_t coreCnt = static_cast<int64_t>(tokenBlockCnt) * baseDimCnt;
        if (static_cast<int64_t>(blockIdx) >= coreCnt) {
            return false;
        }

        const int32_t tokenTileId = blockIdx / baseDimCnt;
        const int32_t baseDimIdx = blockIdx % baseDimCnt;
        channelStart = baseDimIdx * baseDim;
        curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);
        const int32_t tokenStart = tokenTileId * tokensPerBlock;
        const int32_t tokenEndRaw = tokenStart + tokensPerBlock;
        const int32_t tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
        const bool valid = (tokenStart < cuSeqlen) && (curBaseDim > 0) && (tokenEnd > tokenStart);

        if (this->tilingData_->hasInitStateWorkspace) {
            if (valid && tokenTileId == 0) {
                this->PrefetchInitStatesToWorkspace(channelStart, curBaseDim);
            }
            SyncAll();
        }

        if (!valid) {
            return false;
        }

        this->InitCalcBuf(curBaseDim);
        this->LoadWeightAndBias(channelStart, curBaseDim);

        if constexpr (kWindowMode == SEQ_PARTITION_MODE_VARLEN) {
            batchIdx = this->LocateBatchByToken(tokenStart);
            batchCnt = batch;
            cursor = tokenStart;
            cursorEnd = tokenEnd;
        } else {
            batchIdx = tokenStart / (seqLen * coBatch);
            batchCnt = batch / coBatch;
            // cursor in time-step space: one step covers coBatch flat positions
            cursor = tokenStart / coBatch;
            cursorEnd = tokenEnd / coBatch;
        }
        return true;
    }

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessBlock(int32_t channelStart, int32_t curBaseDim,
                                         int32_t &batchIdx, int32_t batchCnt,
                                         int32_t &cursor, int32_t cursorEnd)
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t seqLen = this->tilingData_->seqLen;
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const bool hasCacheIndices = this->tilingData_->hasCacheIndices;
        const bool hasInitialStateMode = this->tilingData_->hasInitialStateMode;

        while (cursor < cursorEnd && batchIdx < batchCnt) {
            int32_t curSeqStart;
            int32_t curSeqEnd;
            this->template GetSeqWindow<kWindowMode>(batchIdx, seqLen, coBatch, curSeqStart, curSeqEnd);

            if (cursor >= curSeqEnd) {
                ++batchIdx;
                continue;
            }

            int32_t chunkEnd = (cursorEnd < curSeqEnd) ? cursorEnd : curSeqEnd;
            int32_t runLen = chunkEnd - cursor;
            if (runLen <= 0) {
                ++batchIdx;
                continue;
            }

            // flatPos = curSeqStart + (cursor % seqLen).
            // For coBatch=1: cursor is flat position, curSeqStart = batchIdx * seqLen,
            //   cursor % seqLen = cursor - curSeqStart, so flatPos = cursor.
            // For coBatch>1: cursor is time step, cursor % seqLen is batch-0 offset.
            // For varlen: cursor is flat position, flatPos = cursor.
            int32_t flatPos;
            if constexpr (kWindowMode == SEQ_PARTITION_MODE_BATCH) {
                flatPos = curSeqStart + (cursor % seqLen);
            } else {
                flatPos = cursor;
            }

            int32_t cacheIdx;
            if (!this->ResolveBatchCacheIndex(batchIdx * coBatch, hasCacheIndices, cacheIdx)) {
                cursor = chunkEnd;
                if (cursor >= curSeqEnd) ++batchIdx;
                continue;
            }

            const bool hasInit = this->ResolveBatchHasInit(batchIdx * coBatch, hasInitialStateMode);

            this->InitRing(cacheIdx, hasInit, curSeqStart, flatPos, channelStart, curBaseDim, dim);
            this->RunSeq(flatPos, runLen, channelStart, curBaseDim, hasInit, curSeqStart);

            if (cursor + runLen >= curSeqEnd) {
                for (int32_t b = 0; b < coBatch; ++b) {
                    this->WriteBackState(cacheIdx + b, runLen, channelStart, curBaseDim, dim, 0, b);
                }
            }

            SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
            SetEvent<HardEvent::MTE3_V>(HardEvent::MTE3_V);

            cursor = chunkEnd;
            if (cursor >= curSeqEnd) ++batchIdx;
        }
    }

    __aicore__ inline void PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize)
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t numCacheLines = this->tilingData_->numCacheLines;
        const int32_t batchRows = static_cast<int32_t>(this->tilingData_->kernelWidth) + 1;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDimSize) * sizeof(T);
        const uint32_t gapBytes = static_cast<uint32_t>(dim - baseDimSize) * sizeof(T);
        LocalTensor<T> ubBuf = this->inBuf.template Get<T>();

        const int64_t totalRows = static_cast<int64_t>(numCacheLines) * stateLen;
        for (int64_t row = 0; row < totalRows; row += batchRows) {
            const int64_t rowsThisBatch = (row + batchRows <= totalRows) ? batchRows : (totalRows - row);
            const int64_t rowOffset = row * dim + channelStart;

            DataCopyPad(ubBuf[0], this->convStatesGm[rowOffset],
                        {static_cast<uint16_t>(rowsThisBatch), blockBytes, gapBytes, 0, 0}, {false, 0, 0, 0});
            SetEvent<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

            DataCopyPad(this->initStateWorkspaceGm_[rowOffset], ubBuf[0],
                        {static_cast<uint16_t>(rowsThisBatch), blockBytes, 0, gapBytes, 0});
            SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }
    }

    __aicore__ inline int32_t LocateBatchByToken(int32_t tokenIdx) const
    {
        int32_t left = 0;
        int32_t right = static_cast<int32_t>(this->tilingData_->batch);
        while (left < right) {
            const int32_t mid = left + ((right - left) >> 1);
            const int32_t endVal = this->queryStartLocGm.GetValue(mid + 1);
            if (tokenIdx < endVal) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t chunkStart,
                                    int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t dimBufferSize = static_cast<int32_t>(this->tilingData_->dimBufferSize);
        const int32_t histBegin = chunkStart - (width - 1);
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);

        // padLen + loadLen = width - 1.
        int32_t padLen;
        if (seqStart <= histBegin) {
            padLen = 0;
        } else if (seqStart == chunkStart) {
            padLen = width - 1;
        } else {
            padLen = seqStart - histBegin;
        }
        const int32_t loadLen = width - 1 - padLen;

        bool hasGmHistoryCopy = false;

        // Pre-seqStart history: from conv_states (hasInit) or zeros (!hasInit).
        if (padLen > 0) {
            if (hasInit) {
                const int32_t statePos = histBegin - seqStart + width - 1;
                const uint32_t ubBlockBytes = Ops::Base::GetUbBlockSize();
                const uint32_t ringRowGapBlocks = static_cast<uint32_t>(dimBufferSize - baseDim) * sizeof(T) / ubBlockBytes;
                const int64_t stateSrcBase = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                              static_cast<int64_t>(statePos) * dim + channelStart;
                LoopModeParams loopParams;
                loopParams.loop1Size = static_cast<uint32_t>(coBatch);
                loopParams.loop2Size = 1;
                loopParams.loop1SrcStride = static_cast<uint64_t>(stateLen) * dim * sizeof(T);
                loopParams.loop1DstStride = static_cast<uint64_t>(baseDim) * sizeof(T);
                loopParams.loop2SrcStride = 0;
                loopParams.loop2DstStride = 0;
                SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
                DataCopyExtParams copyExt;
                copyExt.blockCount = static_cast<uint16_t>(padLen);
                copyExt.blockLen = blockBytes;
                copyExt.srcStride = srcGap;
                copyExt.dstStride = ringRowGapBlocks;
                copyExt.rsv = 0;
                if (this->tilingData_->hasInitStateWorkspace) {
                    DataCopyPad(ring[0], this->initStateWorkspaceGm_[stateSrcBase], copyExt, {false, 0, 0, 0});
                } else {
                    DataCopyPad(ring[0], this->convStatesGm[stateSrcBase], copyExt, {false, 0, 0, 0});
                }
                ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
                hasGmHistoryCopy = true;
            } else {
                Duplicate(ring[0], static_cast<T>(0), padLen * dimBufferSize);
            }
        }

        // Within-sequence history: load directly from xGm.
        if (loadLen > 0) {
            const int32_t seqLen = static_cast<int32_t>(this->tilingData_->seqLen);
            const uint32_t ubBlockBytes = Ops::Base::GetUbBlockSize();
            const uint32_t ringRowGapBlocks = static_cast<uint32_t>(dimBufferSize - baseDim) * sizeof(T) / ubBlockBytes;
            const int64_t xGmRow = static_cast<int64_t>(chunkStart - (width - 1)) + padLen;
            const int64_t xGmSrcBase = xGmRow * dim + channelStart;
            LoopModeParams loopParams;
            loopParams.loop1Size = static_cast<uint32_t>(coBatch);
            loopParams.loop2Size = 1;
            loopParams.loop1SrcStride = static_cast<uint64_t>(seqLen) * dim * sizeof(T);
            loopParams.loop1DstStride = static_cast<uint64_t>(baseDim) * sizeof(T);
            loopParams.loop2SrcStride = 0;
            loopParams.loop2DstStride = 0;
            SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
            DataCopyExtParams copyExt;
            copyExt.blockCount = static_cast<uint16_t>(loadLen);
            copyExt.blockLen = blockBytes;
            copyExt.srcStride = srcGap;
            copyExt.dstStride = ringRowGapBlocks;
            copyExt.rsv = 0;
            DataCopyPad(ring[padLen * dimBufferSize], this->xGm[xGmSrcBase], copyExt, {false, 0, 0, 0});
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
            hasGmHistoryCopy = true;
        }

        if (hasGmHistoryCopy) {
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        }

        const int32_t slot0 = SlotCurr(0, width);
        const int32_t seqLen = static_cast<int32_t>(this->tilingData_->seqLen);
        const uint32_t srcGapBatch = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
        DataCopyPad(ring[slot0 * dimBufferSize], this->xGm[chunkStart * dim + channelStart],
                    {static_cast<uint16_t>(coBatch), blockBytes, srcGapBatch, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim,
                                  bool hasInit = true, int32_t seqStart = 0)
    {
        const int32_t dim = static_cast<int32_t>(this->tilingData_->dim);
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const int32_t coBatch = static_cast<int32_t>(this->tilingData_->coBatch);
        const int32_t dimBufferSize = static_cast<int32_t>(this->tilingData_->dimBufferSize);
        const int32_t seqLen = static_cast<int32_t>(this->tilingData_->seqLen);
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
                SetFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }

            const int32_t outSlot = t & 1;
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            LocalTensor<T> outSlotT = outT[outSlot * dimBufferSize];
            if (!hasInit && (start + t) - seqStart < kernelWidth - 1) {
                Duplicate(outSlotT, static_cast<T>(0), dimBufferSize);
            } else {
                this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outSlotT, dimBufferSize, t);
            }

            SetFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);

            const int64_t outGmBase = static_cast<int64_t>(start + t) * dim + channelStart;
            const uint32_t dstGapBatchY = static_cast<uint32_t>(seqLen * dim - baseDim) * sizeof(T);
            DataCopyPad(this->yGm[outGmBase], outSlotT[0],
                        {static_cast<uint16_t>(coBatch), blockBytes, 0, dstGapBatchY, 0});
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_MTE2>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }
        }
    }

};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dFn(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                         GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                         const CausalConv1dTilingData *tilingData)
{
    CausalConv1dFn<T, inputModeKey, widthKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    op.Process();
}

} // namespace CausalConv1d

#endif // CAUSAL_CONV1D_FN_H
