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

struct FnDirectBlockTask {
    bool valid = false;
    int32_t tokenTileId = 0;
    int32_t baseDimIdx = 0;
    int32_t tokenStart = 0;
    int32_t tokenEnd = 0;
    int32_t channelStart = 0;
    int32_t baseDimSize = 0;
};

__aicore__ inline FnDirectBlockTask ResolveFnDirectBlockTask(int32_t blockIdx, int32_t tokenBlockCnt,
                                                             int32_t tokenBlockSize, int32_t cuSeqlen,
                                                             int32_t baseDimCnt, int32_t baseDim, int32_t dim)
{
    FnDirectBlockTask task;
    if (blockIdx < 0 || tokenBlockCnt <= 0 || tokenBlockSize <= 0 || cuSeqlen <= 0 || baseDimCnt <= 0 || baseDim <= 0 ||
        dim <= 0) {
        return task;
    }

    const int64_t phase1Grid = static_cast<int64_t>(tokenBlockCnt) * baseDimCnt;
    if (phase1Grid <= 0 || static_cast<int64_t>(blockIdx) >= phase1Grid) {
        return task;
    }

    task.tokenTileId = blockIdx / baseDimCnt;
    task.baseDimIdx = blockIdx % baseDimCnt;
    task.channelStart = task.baseDimIdx * baseDim;
    if (task.channelStart >= dim) {
        return task;
    }

    task.baseDimSize = (task.channelStart + baseDim <= dim) ? baseDim : (dim - task.channelStart);
    task.tokenStart = task.tokenTileId * tokenBlockSize;
    if (task.tokenStart >= cuSeqlen) {
        return task;
    }

    const int32_t tokenEndRaw = task.tokenStart + tokenBlockSize;
    task.tokenEnd = (tokenEndRaw <= cuSeqlen) ? tokenEndRaw : cuSeqlen;
    if (task.baseDimSize <= 0 || task.tokenEnd <= task.tokenStart) {
        return {};
    }

    task.valid = true;
    return task;
}

__aicore__ inline bool IsFnInitStateSnapshotOwnerBlock(const FnDirectBlockTask &task)
{
    return task.valid && task.tokenTileId == 0;
}

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
        if (tilingData->hasInitStateWorkspace != 0) {
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
        this->initSnapshotMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
        this->initSnapshotMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        this->initSyncVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        this->initSyncMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process()
    {
        this->ProcessVarlenTokenTiled();
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(this->initSnapshotMte2ToMte3Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(this->initSnapshotMte3ToMte2Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(this->initSyncVToMte3Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(this->initSyncMte3ToVEvent_);
        this->ReleaseEvents();
    }

protected:
    TEventID initSnapshotMte2ToMte3Event_;
    TEventID initSnapshotMte3ToMte2Event_;
    TEventID initSyncVToMte3Event_;
    TEventID initSyncMte3ToVEvent_;

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

    __aicore__ inline void InitRingSeqSplit(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t tileStart,
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
        int64_t xHistoryOffset = static_cast<int64_t>(historyStartTok) * dim + channelStart;

        for (int32_t i = 0, srcTok = historyStartTok; i < historyCount; ++i, ++srcTok, xHistoryOffset += dim) {
            LocalTensor<T> histSlot = ring[i * baseDim];
            if (srcTok >= seqStart) {
                DataCopy(histSlot, this->xGm[xHistoryOffset], baseDim);
                hasGmHistoryCopy = true;
            } else if (hasInit) {
                const int32_t statePos = srcTok - seqStart + historyCount;
                const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(statePos) * dim;
                if (this->tilingData_->hasInitStateWorkspace != 0) {
                    DataCopy(histSlot, this->initStateWorkspaceGm_[stateOffset], baseDim);
                } else {
                    DataCopy(histSlot, this->convStatesGm[stateOffset], baseDim);
                }
                hasGmHistoryCopy = true;
            } else {
                Duplicate(histSlot, static_cast<T>(0), baseDim);
                hasVectorInit = true;
            }
        }

        if (hasGmHistoryCopy) {
            SetFlag<HardEvent::MTE2_V>(this->stateMte2ToVEvent_);
            WaitFlag<HardEvent::MTE2_V>(this->stateMte2ToVEvent_);
        }
        if (hasVectorInit) {
            PipeBarrier<PIPE_V>();
        }

        if (tileLen > 0) {
            const int32_t slot0 = SlotCurr(0, width);
            const int64_t xOffset = static_cast<int64_t>(tileStart) * dim + channelStart;
            DataCopy(ring[slot0 * baseDim], this->xGm[xOffset], baseDim);
            SetFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slot0]);
        }

        if (tileLen > 1) {
            SetFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
        }
    }

    __aicore__ inline void ProcessFnChunk(int32_t cacheIdx, bool hasInit, int32_t seqStart, int32_t seqLen,
                                          int32_t chunkStart, int32_t chunkLen, int32_t channelStart, int32_t baseDim,
                                          int32_t dim)
    {
        this->InitCalcBuf(baseDim);
        this->LoadWeightAndBias(channelStart, baseDim);
        this->InitRingSeqSplit(cacheIdx, hasInit, seqStart, chunkStart, chunkLen, channelStart, baseDim, dim);

        this->RunSeq(chunkStart, chunkLen, channelStart, baseDim, dim, hasInit, seqStart);

        this->MaybeWriteBackSeqSplitTailChunk(chunkStart, chunkLen, seqStart, seqLen, cacheIdx, channelStart, baseDim,
                                              dim);
        this->DrainTaskMte3();
    }

    __aicore__ inline void MaybeWriteBackSeqSplitTailChunk(int32_t chunkStart, int32_t chunkLen, int32_t seqStart,
                                                           int32_t seqLen, int32_t cacheIdx, int32_t channelStart,
                                                           int32_t baseDim, int32_t dim)
    {
        if (chunkStart + chunkLen != seqStart + seqLen) {
            return;
        }

        this->DrainTaskMte3();
        this->WriteBackState(cacheIdx, chunkLen, channelStart, baseDim, dim);
    }

    __aicore__ inline void PrefetchInitStatesToWorkspace(int32_t channelStart, int32_t baseDimSize)
    {
        if (this->tilingData_->hasInitStateWorkspace == 0) {
            return;
        }

        const int32_t dim = this->tilingData_->dim;
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t numCacheLines = this->tilingData_->numCacheLines;
        LocalTensor<T> tmpBuf = this->inBuf.template Get<T>()[0];

        const int64_t totalRows = static_cast<int64_t>(numCacheLines) * stateLen;
        for (int64_t row = 0; row < totalRows; ++row) {
            const int64_t rowOffset = row * dim + channelStart;
            DataCopy(tmpBuf, this->convStatesGm[rowOffset], baseDimSize);
            SetFlag<HardEvent::MTE2_MTE3>(this->initSnapshotMte2ToMte3Event_);
            WaitFlag<HardEvent::MTE2_MTE3>(this->initSnapshotMte2ToMte3Event_);
            DataCopy(this->initStateWorkspaceGm_[rowOffset], tmpBuf, baseDimSize);
            SetFlag<HardEvent::MTE3_MTE2>(this->initSnapshotMte3ToMte2Event_);
            WaitFlag<HardEvent::MTE3_MTE2>(this->initSnapshotMte3ToMte2Event_);
        }
    }

    __aicore__ inline void ProcessVarlenTokenTiled()
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t batch = this->tilingData_->batch;
        const int32_t seqLen = this->tilingData_->seqLen;
        const int32_t cuSeqlen = this->tilingData_->cuSeqlen;
        const int32_t baseDim = static_cast<int32_t>(this->tilingData_->baseDim);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const int32_t tokenBlockSize = static_cast<int32_t>(this->tilingData_->tokensPerBlock);
        const int32_t tokenBlockCnt = static_cast<int32_t>(this->tilingData_->tokenBlockCnt);
        const bool hasCacheIndices = (this->tilingData_->hasCacheIndices != 0);
        const bool hasInitialStateMode = (this->tilingData_->hasInitialStateMode != 0);
        const bool isVarlenMode = IsVarlenInputModeKey(inputModeKey);

        const int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        const auto blockTask =
            ResolveFnDirectBlockTask(blockIdx, tokenBlockCnt, tokenBlockSize, cuSeqlen, baseDimCnt, baseDim, dim);
        if (this->tilingData_->hasInitStateWorkspace != 0) {
            if (IsFnInitStateSnapshotOwnerBlock(blockTask)) {
                this->PrefetchInitStatesToWorkspace(blockTask.channelStart, blockTask.baseDimSize);
            }
            SyncAll();
        }
        if (!blockTask.valid) {
            return;
        }

        int32_t seq = 0;
        int32_t seqUpperBound = batch;
        if (isVarlenMode) {
            seq = this->FindVarlenSeqByToken(blockTask.tokenStart);
        } else {
            seq = (seqLen > 0) ? (blockTask.tokenStart / seqLen) : 0;
        }

        int32_t cursor = blockTask.tokenStart;
        while (cursor < blockTask.tokenEnd && seq < seqUpperBound) {
            int32_t seqStart = 0;
            int32_t curSeqLen = 0;
            if constexpr (IsVarlenInputModeKey(inputModeKey)) {
                this->template GetSeqWindow<SEQ_TASK_WINDOW_MODE_VARLEN>(seq, seqLen, seqStart,
                                                                                       curSeqLen);
            } else {
                this->template GetSeqWindow<SEQ_TASK_WINDOW_MODE_BATCH>(seq, seqLen, seqStart,
                                                                                      curSeqLen);
            }
            const int32_t curSeqEnd = seqStart + curSeqLen;
            if (cursor < seqStart) {
                cursor = seqStart;
            }
            if (cursor >= curSeqEnd) {
                ++seq;
                continue;
            }

            const int32_t tileEnd = (blockTask.tokenEnd <= curSeqEnd) ? blockTask.tokenEnd : curSeqEnd;
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
            this->ProcessFnChunk(cacheIdx, hasInit, seqStart, curSeqLen, cursor, tileLen, blockTask.channelStart,
                                 blockTask.baseDimSize, dim);

            cursor = tileEnd;
            ++seq;
        }
    }

    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim,
                                   bool hasInit = true, int32_t seqStart = 0)
    {
        if constexpr (kHasCompileTimeWidth && !kHasBias) {
            RunSeqFnRolling(start, len, channelStart, baseDim, dim);
            return;
        }

        const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        LocalTensor<T> outT = this->outBuf.template Get<T>();

        for (int32_t t = 0; t < len; ++t) {
            const int32_t slotCurr = SlotCurr(t, width);

            WaitFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slotCurr]);

            if (t + 1 < len) {
                const int32_t slotNext = SlotPrefetch(t, width);
                const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                WaitFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
                DataCopy(ring[slotNext * baseDim], this->xGm[xOffsetNext], baseDim);
                SetFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slotNext]);
            }

            const int32_t outSlot = t & 1;
            LocalTensor<T> outSlotT = outT[outSlot * baseDim];
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>(this->outMte3ToVEvent_[outSlot]);
            }

            this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outSlotT, baseDim, t);

            this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outSlotT, baseDim, t);

            if (!hasInit && (start + t) - seqStart < static_cast<int32_t>(this->tilingData_->kernelWidth) - 1) {
                if constexpr (kHasBias) {
                    LocalTensor<float> &biasF = this->pool_.bias;
                    if constexpr (IsSameType<T, float>::value) {
                        DataCopy(outSlotT, biasF, baseDim);
                    } else {
                        Cast(outSlotT, biasF, RoundMode::CAST_RINT, baseDim);
                    }
                } else {
                    Duplicate(outSlotT, static_cast<T>(0), baseDim);
                }
            }

            SetFlag<HardEvent::V_MTE3>(this->outVToMte3Event_[outSlot]);

            const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
            WaitFlag<HardEvent::V_MTE3>(this->outVToMte3Event_[outSlot]);
            DataCopy(this->yGm[outOffset], outSlotT, baseDim);
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>(this->outMte3ToVEvent_[outSlot]);
            }

            if (t + 2 < len) {
                SetFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
            }
        }
    }

    __aicore__ inline void RestoreFnLocalPartials(int32_t baseDim)
    {
        if constexpr (!kHasCompileTimeWidth) {
            return;
        }

        auto pool = CalcBufPool::Init(this->calcBuf, kTemplateWidth, baseDim, true);
        LocalTensor<float> &weightF = pool.weight;
        LocalTensor<float> &state2F = pool.bias;
        LocalTensor<float> &state1F = pool.acc;
        LocalTensor<float> &state0F = pool.tmp;
        LocalTensor<float> &currF = pool.curr;
        LocalTensor<T> ring = this->inBuf.template Get<T>();

        if constexpr (kTemplateWidth == 2) {
            Duplicate(state2F, 0.0f, baseDim);
            Duplicate(state1F, 0.0f, baseDim);
            PipeBarrier<PIPE_V>();

            Cast(currF, ring[0], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            Mul(state0F, currF, weightF[0], baseDim);
            PipeBarrier<PIPE_V>();
        } else if constexpr (kTemplateWidth == 3) {
            Duplicate(state2F, 0.0f, baseDim);
            PipeBarrier<PIPE_V>();

            Cast(currF, ring[0], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            Mul(state0F, currF, weightF[0], baseDim);
            PipeBarrier<PIPE_V>();

            Cast(currF, ring[1 * baseDim], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            Mul(state1F, currF, weightF[0], baseDim);
            PipeBarrier<PIPE_V>();
            MulAddDst(state0F, currF, weightF[1 * baseDim], baseDim);
            PipeBarrier<PIPE_V>();
        } else if constexpr (kTemplateWidth == 4) {
            Cast(currF, ring[0], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            Mul(state0F, currF, weightF[0], baseDim);
            PipeBarrier<PIPE_V>();

            Cast(currF, ring[1 * baseDim], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            Mul(state1F, currF, weightF[0], baseDim);
            PipeBarrier<PIPE_V>();
            MulAddDst(state0F, currF, weightF[1 * baseDim], baseDim);
            PipeBarrier<PIPE_V>();

            Cast(currF, ring[2 * baseDim], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            Mul(state2F, currF, weightF[0], baseDim);
            PipeBarrier<PIPE_V>();
            MulAddDst(state1F, currF, weightF[1 * baseDim], baseDim);
            PipeBarrier<PIPE_V>();
            MulAddDst(state0F, currF, weightF[2 * baseDim], baseDim);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeFnRollingOutput(int32_t slotCurr, int32_t baseDim)
    {
        if constexpr (!kHasCompileTimeWidth) {
            return;
        }

        auto pool = CalcBufPool::Init(this->calcBuf, kTemplateWidth, baseDim, true);
        LocalTensor<float> &weightF = pool.weight;
        LocalTensor<float> &state0F = pool.tmp;
        LocalTensor<float> &currF = pool.curr;
        LocalTensor<T> ring = this->inBuf.template Get<T>();

                if constexpr (kHasActivation) {
            ComputeFnRollingOutputRegbase<T, true>(ring[slotCurr * baseDim], currF, state0F,
                                                   weightF[(kTemplateWidth - 1) * baseDim], baseDim);
        } else {
            ComputeFnRollingOutputRegbase<T, false>(ring[slotCurr * baseDim], currF, state0F,
                                                    weightF[(kTemplateWidth - 1) * baseDim], baseDim);
        }
    }

    __aicore__ inline void AdvanceFnLocalPartials(int32_t slotCurr, int32_t baseDim)
    {
        if constexpr (!kHasCompileTimeWidth) {
            return;
        }

        auto pool = CalcBufPool::Init(this->calcBuf, kTemplateWidth, baseDim, true);
        LocalTensor<float> &weightF = pool.weight;
        LocalTensor<float> &state2F = pool.bias;
        LocalTensor<float> &state1F = pool.acc;
        LocalTensor<float> &state0F = pool.tmp;
        LocalTensor<float> &currF = pool.curr;
        LocalTensor<T> ring = this->inBuf.template Get<T>();

        AdvanceFnLocalPartialsRegbase<T, kTemplateWidth>(ring[slotCurr * baseDim], weightF[0], state0F, state1F,
                                                         state2F, baseDim, baseDim);
    }

    __aicore__ inline void RunSeqFnRolling(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim,
                                           int32_t dim)
    {
        if constexpr (!kHasCompileTimeWidth) {
            return;
        }

        auto pool = CalcBufPool::Init(this->calcBuf, kTemplateWidth, baseDim, true);
        LocalTensor<float> &state0F = pool.tmp;
        LocalTensor<float> &currF = pool.curr;
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        LocalTensor<T> outT = this->outBuf.template Get<T>();
                RestoreFnLocalPartials(baseDim);

        for (int32_t t = 0; t < len; ++t) {
            const int32_t slotCurr = SlotCurr(t, kTemplateWidth);

            WaitFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slotCurr]);

            if (t + 1 < len) {
                const int32_t slotNext = SlotPrefetch(t, kTemplateWidth);
                const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                WaitFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
                DataCopy(ring[slotNext * baseDim], this->xGm[xOffsetNext], baseDim);
                SetFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slotNext]);
            }

            ComputeFnRollingOutput(slotCurr, baseDim);

            const int32_t outSlot = t & 1;
            LocalTensor<T> outSlotT = outT[outSlot * baseDim];
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>(this->outMte3ToVEvent_[outSlot]);
            }

            if constexpr (IsSameType<T, float>::value) {
                if constexpr (kHasActivation) {
                    DataCopy(outSlotT, currF, baseDim);
                } else {
                    DataCopy(outSlotT, state0F, baseDim);
                }
            } else {
                if constexpr (kHasActivation) {
                    Cast(outSlotT, currF, RoundMode::CAST_RINT, baseDim);
                } else {
                    Cast(outSlotT, state0F, RoundMode::CAST_RINT, baseDim);
                }
            }

            AdvanceFnLocalPartials(slotCurr, baseDim);

            SetFlag<HardEvent::V_MTE3>(this->outVToMte3Event_[outSlot]);

            const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
            WaitFlag<HardEvent::V_MTE3>(this->outVToMte3Event_[outSlot]);
            DataCopy(this->yGm[outOffset], outSlotT, baseDim);
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>(this->outMte3ToVEvent_[outSlot]);
            }

            if (t + 2 < len) {
                SetFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
            }
        }
    }
};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dFn(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                         GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                         GM_ADDR numAcceptedTokens, GM_ADDR convStatesOut, GM_ADDR y,
                                         GM_ADDR workspace, const CausalConv1dTilingData *tilingData)
{
    CausalConv1dFn<T, inputModeKey, widthKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens,
            convStatesOut, y, workspace, tilingData);
    op.Process();
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_FN_H
