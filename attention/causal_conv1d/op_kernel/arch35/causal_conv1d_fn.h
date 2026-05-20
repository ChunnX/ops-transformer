/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_fn.h
 * \brief
 */

#ifndef CAUSAL_CONV1D_FN_H
#define CAUSAL_CONV1D_FN_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dFn : public CausalConv1d<T, inputModeKey, widthKey, hasBiasKey, activationKey> {
    static constexpr int32_t kTemplateWidth = static_cast<int32_t>(widthKey);
    static constexpr bool kHasCompileTimeWidth =
        IsFnInputModeKey(inputModeKey) && (kTemplateWidth >= 2) && (kTemplateWidth <= MAX_WIDTH);
    static constexpr bool kHasBias = (hasBiasKey != 0);
    static constexpr bool kHasActivation = (activationKey != 0);

public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR initialStateMode, GM_ADDR numAcceptedTokens,
                                GM_ADDR convStatesOut, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        (void)numAcceptedTokens;
        this->ResetRuntimeState(tilingData);
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->convStatesOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStatesOut));
        this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
        this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cacheIndices));
        this->initialStateModeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(initialStateMode));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        if (tilingData->hasInitStateWorkspace) {
            const uint64_t syncElems = static_cast<uint64_t>(GetBlockNum()) * INIT_STATE_SYNCALL_NEED_SIZE;
            const uint64_t syncBytes = syncElems * sizeof(int32_t);
            const uint64_t workspaceElems =
                static_cast<uint64_t>(tilingData->numCacheLines) * tilingData->stateLen * tilingData->dim;
            this->initStateSyncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), syncElems);
            auto *workspaceBytes = reinterpret_cast<__gm__ uint8_t *>(workspace);
            this->initStateWorkspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspaceBytes + syncBytes),
                                                        workspaceElems);
        }
        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process()
    {
        if constexpr (IsVarlenInputModeKey(inputModeKey)) {
            this->template ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_VARLEN>();
        } else {
            this->template ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_BATCH>();
        }
    }

protected:
    GlobalTensor<int32_t> initStateSyncGm_;
    GlobalTensor<T> initStateWorkspaceGm_;

    __aicore__ inline int32_t FindVarlenSeqByToken(int32_t tokenIdx) const
    {
        int32_t left = 0;
        int32_t right = static_cast<int32_t>(this->tilingData_->batch);
        while (left < right) {
            const int32_t mid = left + ((right - left) >> 1);
            const int32_t endVal = static_cast<int32_t>(this->queryStartLocGm.GetValue(mid + 1));
            if (tokenIdx < endVal) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t tileStart,
                                            int32_t tileLen, int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const int32_t historyCount = width - 1;
        const int32_t historyStartTok = tileStart - historyCount;
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        bool hasGmHistoryCopy = false;
        bool hasVectorInit = false;
        const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);

        // Determine split: rows before seqStart need state/zero, rows >= seqStart come from xGm
        int32_t preSeqCount;
        if (seqStart <= historyStartTok) {
            preSeqCount = 0;
        } else if (seqStart > tileStart - 1) {
            preSeqCount = historyCount;
        } else {
            preSeqCount = seqStart - historyStartTok;
        }
        const int32_t xGmCount = historyCount - preSeqCount;

        // Pre-seqStart history: from conv_states (if hasInit) or zeros
        if (preSeqCount > 0) {
            if (hasInit) {
                const int32_t statePos = historyStartTok - seqStart + historyCount;
                const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(statePos) * dim;
                if (this->tilingData_->hasInitStateWorkspace) {
                    DataCopyPad(ring[0], this->initStateWorkspaceGm_[stateOffset],
                                {static_cast<uint16_t>(preSeqCount), blockBytes, srcGap, 0, 0}, {false, 0, 0, 0});
                } else {
                    DataCopyPad(ring[0], this->convStatesGm[stateOffset],
                                {static_cast<uint16_t>(preSeqCount), blockBytes, srcGap, 0, 0}, {false, 0, 0, 0});
                }
                hasGmHistoryCopy = true;
            } else {
                Duplicate(ring[0], static_cast<T>(0), preSeqCount * baseDim);
                hasVectorInit = true;
            }
        }

        // History from xGm
        if (xGmCount > 0) {
            const int64_t xGmOffset = static_cast<int64_t>(historyStartTok + preSeqCount) * dim + channelStart;
            DataCopyPad(ring[preSeqCount * baseDim], this->xGm[xGmOffset],
                        {static_cast<uint16_t>(xGmCount), blockBytes, srcGap, 0, 0}, {false, 0, 0, 0});
            hasGmHistoryCopy = true;
        }

        if (hasGmHistoryCopy) {
            SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        }
        if (hasVectorInit) {
            PipeBarrier<PIPE_V>();
        }

        if (tileLen > 0) {
            const int32_t slot0 = SlotCurr(0, width);
            const int64_t xOffset = static_cast<int64_t>(tileStart) * dim + channelStart;
            DataCopy(ring[slot0 * baseDim], this->xGm[xOffset], baseDim);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                           int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                           int32_t baseDim, int32_t dim)
    {
        if (chunkStart + chunkLen != seqStart + seqLen) {
            return;
        }

        this->WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim);
    }

    __aicore__ inline void PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize)
    {
        if (!this->tilingData_->hasInitStateWorkspace) {
            return;
        }

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

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessDefaultByWindowMode()
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t batch = this->tilingData_->batch;
        const int32_t seqLen = this->tilingData_->seqLen;
        const int32_t cuSeqlen = this->tilingData_->cuSeqlen;
        const int32_t baseDim = static_cast<int32_t>(this->tilingData_->baseDim);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const int32_t tokenBlockSize = static_cast<int32_t>(this->tilingData_->tokensPerBlock);
        const int32_t tokenBlockCnt = static_cast<int32_t>(this->tilingData_->tokenBlockCnt);
        const bool hasCacheIndices = this->tilingData_->hasCacheIndices;
        const bool hasInitialStateMode = this->tilingData_->hasInitialStateMode;

        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const int64_t coreCnt = static_cast<int64_t>(tokenBlockCnt) * baseDimCnt;
        if (static_cast<int64_t>(blockIdx) >= coreCnt) {
            return;
        }

        const int32_t tokenTileId = blockIdx / baseDimCnt;
        const int32_t baseDimIdx = blockIdx % baseDimCnt;
        const int32_t channelStart = baseDimIdx * baseDim;
        if (channelStart >= dim) {
            return;
        }

        const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);
        const int32_t tokenStart = tokenTileId * tokenBlockSize;
        const int32_t tokenEndRaw = tokenStart + tokenBlockSize;
        const int32_t tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
        const bool valid = (tokenStart < cuSeqlen) && (curBaseDim > 0) && (tokenEnd > tokenStart);

        if (this->tilingData_->hasInitStateWorkspace) {
            if (valid && tokenTileId == 0) {
                this->PrefetchInitStatesToWorkspace(channelStart, curBaseDim);
            }
            SyncAll();
        }

        if (!valid) {
            return;
        }

        this->InitCalcBuf(curBaseDim);
        this->LoadWeightAndBias(channelStart, curBaseDim);

        int32_t seq = 0;
        int32_t seqUpperBound = batch;
        if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_VARLEN) {
            seq = this->FindVarlenSeqByToken(tokenStart);
        } else {
            seq = (seqLen > 0) ? (tokenStart / seqLen) : 0;
        }

        int32_t cursor = tokenStart;
        while (cursor < tokenEnd && seq < seqUpperBound) {
            int32_t seqStart = 0;
            int32_t curSeqLen = 0;
            this->template GetSeqWindow<kWindowMode>(seq, seqLen, seqStart, curSeqLen);
            const int32_t curSeqEnd = seqStart + curSeqLen;
            if (cursor < seqStart) {
                cursor = seqStart;
            }
            if (cursor >= curSeqEnd) {
                ++seq;
                continue;
            }

            const int32_t tileEnd = (tokenEnd <= curSeqEnd) ? tokenEnd : curSeqEnd;
            const int32_t tileLen = tileEnd - cursor;
            if (tileLen <= 0) {
                ++seq;
                continue;
            }

            int32_t cacheIdx = 0;
            if (!this->ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
                cursor = tileEnd;
                ++seq;
                continue;
            }

            const bool hasInit = this->ResolveSeqHasInit(seq, hasInitialStateMode);

            this->InitRing(cacheIdx, hasInit, seqStart, cursor, tileLen, channelStart, curBaseDim, dim);
            this->RunSeq(cursor, tileLen, channelStart, curBaseDim, dim, hasInit, seqStart);
            this->MaybeWriteBackSeqSplitTailChunk(cursor, tileLen, seqStart, curSeqLen, cacheIdx, channelStart,
                                                  curBaseDim, dim);
            this->DrainTaskMte3();

            cursor = tileEnd;
            ++seq;
        }
    }

    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim,
                                  bool hasInit = true, int32_t seqStart = 0)
    {
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        LocalTensor<T> outT = this->outBuf.template Get<T>();

        for (int32_t t = 0; t < len; ++t) {
            WaitFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);

            // prefetch x[t+1] if available
            if (t + 1 < len) {
                const int32_t slotNext = SlotPrefetch(t, kernelWidth);
                const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                if (t > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>((t & 1) ? EVENT_ID1 : EVENT_ID0);
                }
                DataCopyPad(ring[slotNext * baseDim], this->xGm[xOffsetNext], {1, blockBytes, 0, 0, 0},
                            {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            const int32_t outSlot = t & 1;
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            LocalTensor<T> outSlotT = outT[outSlot * baseDim];
            if (!hasInit && (start + t) - seqStart < kernelWidth - 1) {
                Duplicate(outSlotT, static_cast<T>(0), baseDim);
            } else {
                this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outSlotT, baseDim, t);
            }

            SetFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);

            // copy output to GM
            const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
            DataCopyPad(this->yGm[outOffset], outSlotT, {1, blockBytes, 0, 0, 0});
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            // signal ring slot freed -> MTE2 can load next
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

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_FN_H
