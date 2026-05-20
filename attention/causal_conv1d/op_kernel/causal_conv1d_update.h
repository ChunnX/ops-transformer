/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T, uint32_t inputModeKey>
class CausalConv1dUpdate
    : public CausalConv1d<T, inputModeKey, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME> {
public:
    static constexpr bool kIsUpdateMode = IsUpdateInputModeKey(inputModeKey);

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates, GM_ADDR queryStartLoc,
                                GM_ADDR cacheIndices, GM_ADDR, GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                const CausalConv1dTilingData *tilingData)
    {
        (void)workspace;
        this->ResetRuntimeState(tilingData);
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
        this->weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight));
        this->biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias));
        this->convStatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(convStates));
        this->queryStartLocGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(queryStartLoc));
        this->cacheIndicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cacheIndices));
        this->numAcceptedTokensGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(numAcceptedTokens));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y));
        this->InitSharedBuffersAndEvents();
    }

    __aicore__ inline void Process()
    {
        const CausalConv1dTilingData *tilingData = this->GetTilingData();
        const int32_t dim = tilingData->dim;
        const int32_t width = static_cast<int32_t>(tilingData->kernelWidth);
        const int32_t baseDim = static_cast<int32_t>(tilingData->baseDim);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        if (baseDim <= 0 || width < 2 || width > MAX_WIDTH || dim <= 0 ||
            tilingData->batch <= 0) {
            this->ReleaseEvents();
            return;
        }

        this->ProcessDefault();
        this->ReleaseEvents();
    }

protected:
    __aicore__ inline void ProcessDefault();
    template <int32_t kWindowMode>
    __aicore__ inline void ProcessDefaultByWindowMode();
    __aicore__ inline void InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset, int32_t start,
                                    int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim);
    __aicore__ inline bool IsUpdateMode() const;
    __aicore__ inline bool IsUpdateSpecDecodingEnabled() const;
};

#define CAUSAL_CONV1D_TEMPLATE_ARGS typename T, uint32_t inputModeKey
#define CAUSAL_CONV1D_CLASS CausalConv1dUpdate<T, inputModeKey>

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessDefault()
{
    if constexpr (IsVarlenInputModeKey(inputModeKey)) {
        this->template ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_VARLEN>();
    } else {
        this->template ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_BATCH>();
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
template <int32_t kWindowMode>
__aicore__ inline void CAUSAL_CONV1D_CLASS::ProcessDefaultByWindowMode()
{
    const int32_t dim = this->tilingData_->dim;
    const int32_t batch = this->tilingData_->batch;
    const int32_t seqLen = this->tilingData_->seqLen;
    const int32_t baseDim = static_cast<int32_t>(this->tilingData_->baseDim);
    const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
    const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
    const bool hasCacheIndices = (this->tilingData_->hasCacheIndices != 0);
    const bool hasInit = true;
    const bool isSpecDecodingGlobal = this->IsUpdateSpecDecodingEnabled();

    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t blockNum = GetBlockNum();

    if (baseDim <= 0 || width < 2 || width > MAX_WIDTH) {
        this->ReleaseEvents();
        return;
    }

    const int64_t gridSize = static_cast<int64_t>(batch) * baseDimCnt;
    for (int64_t task = static_cast<int64_t>(blockIdx); task < gridSize; task += static_cast<int64_t>(blockNum)) {
        const int32_t seq = static_cast<int32_t>(task / baseDimCnt);
        const int32_t baseDimIdx = static_cast<int32_t>(task % baseDimCnt);
        const int32_t channelStart = baseDimIdx * baseDim;
        if (channelStart >= dim) {
            continue;
        }
        const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);

        int32_t start = 0;
        int32_t len = 0;
        if (!this->template ResolveSeqTaskWindowByMode<kWindowMode>(seq, seqLen, start, len)) {
            continue;
        }

        int32_t cacheIdx = 0;
        if (!this->ResolveSeqCacheIndex(seq, hasCacheIndices, cacheIdx)) {
            continue;
        }

        int32_t stateTokenOffset = 0;
        if (isSpecDecodingGlobal) {
            int32_t accepted = static_cast<int32_t>(this->numAcceptedTokensGm.GetValue(seq));
            stateTokenOffset = accepted - 1;
            const int32_t maxOffset = static_cast<int32_t>(this->tilingData_->stateLen - (width - 1));
            if (stateTokenOffset < 0) {
                stateTokenOffset = 0;
            } else if (stateTokenOffset > maxOffset) {
                stateTokenOffset = maxOffset;
            }
        }

        this->LoadWeightAndBias(channelStart, curBaseDim);

        this->InitRing(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        this->RunSeq(start, len, channelStart, curBaseDim, dim);

        if (isSpecDecodingGlobal) {
            this->DrainTaskMte3();
            this->WriteBackStateSpec(cacheIdx, hasInit, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
        } else {
            this->WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
        }

        this->DrainTaskMte3();
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::InitRing(int32_t cacheIdx, bool hasInit, int32_t stateTokenOffset,
                                                     int32_t start, int32_t len, int32_t channelStart,
                                                     int32_t baseDim, int32_t dim)
{
    const int32_t stateLen = this->tilingData_->stateLen;
    const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
    const int32_t ringStart = MAX_WIDTH - width;
    LocalTensor<T> ring = this->inBuf.template Get<T>();

    for (int32_t i = 0; i < ringStart; ++i) {
        Duplicate(ring[i * baseDim], static_cast<T>(0), baseDim);
    }
    if (ringStart > 0) {
        PipeBarrier<PIPE_V>();
    }

    if (hasInit) {
        for (int32_t i = 0; i < (width - 1); ++i) {
            const int32_t pos = stateTokenOffset + i;
            const int64_t stateOffset =
                static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(pos) * dim + channelStart;
            DataCopy(ring[(ringStart + i) * baseDim], this->convStatesGm[stateOffset], baseDim);
        }
        SetFlag<HardEvent::MTE2_V>(this->stateMte2ToVEvent_);
        WaitFlag<HardEvent::MTE2_V>(this->stateMte2ToVEvent_);
    } else {
        for (int32_t i = 0; i < (width - 1); ++i) {
            Duplicate(ring[(ringStart + i) * baseDim], static_cast<T>(0), baseDim);
        }
        PipeBarrier<PIPE_V>();
    }

    if (len > 0) {
        const int32_t slot0 = SlotCurr(0);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
        DataCopy(ring[slot0 * baseDim], this->xGm[xOffset], baseDim);
        SetFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slot0]);
    }

    if (len > 1) {
        SetFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
    }
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::IsUpdateMode() const
{
    return kIsUpdateMode;
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline bool CAUSAL_CONV1D_CLASS::IsUpdateSpecDecodingEnabled() const
{
    return IsUpdateInputModeKey(inputModeKey) && (this->tilingData_->hasNumAcceptedTokens != 0) &&
           (this->tilingData_->kernelWidth == 4);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
__aicore__ inline void CAUSAL_CONV1D_CLASS::RunSeq(int32_t start, int32_t len, int32_t channelStart,
                                                   int32_t baseDim, int32_t dim)
{
    const int32_t width = static_cast<int32_t>(this->tilingData_->kernelWidth);
    auto cl = CalcBufLayout::FromCalcBuf(this->calcBuf, width, baseDim);
    LocalTensor<float> &weightF = cl.weightF;
    LocalTensor<float> &biasF = cl.biasF;
    LocalTensor<float> &accF = cl.accF;
    LocalTensor<float> &tmpF = cl.tmpF;
    LocalTensor<T> ring = this->inBuf.template Get<T>();
    LocalTensor<T> outT = this->outBuf.template Get<T>();
    const bool hasBias = this->HasBias();
    const bool hasActivation = this->HasActivation();
    for (int32_t t = 0; t < len; ++t) {
        const int32_t slotCurr = SlotCurr(t);

        WaitFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slotCurr]);

        if (t + 1 < len) {
            const int32_t slotNext = SlotPrefetch(t);
            const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
            WaitFlag<HardEvent::V_MTE2>(this->inputVToMte2Event_);
            DataCopy(ring[slotNext * baseDim], this->xGm[xOffsetNext], baseDim);
            SetFlag<HardEvent::MTE2_V>(this->inputMte2ToVEvent_[slotNext]);
        }

        bool accInitialized = false;
        if (hasBias) {
            Adds(accF, biasF, 0.0f, baseDim);
            PipeBarrier<PIPE_V>();
            accInitialized = true;
        }

        for (int32_t j = 0; j < width; ++j) {
            const int32_t tap = (width - 1) - j;
            const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap);
            Cast(tmpF, ring[slot * baseDim], RoundMode::CAST_NONE, baseDim);
            PipeBarrier<PIPE_V>();
            if (!accInitialized) {
                Mul(accF, tmpF, weightF[j * baseDim], baseDim);
                accInitialized = true;
            } else {
                MulAddDst(accF, tmpF, weightF[j * baseDim], baseDim);
            }
        }

        PipeBarrier<PIPE_V>();

        if (hasActivation) {
            Silu(tmpF, accF, baseDim);
        }

        const int32_t outSlot = t & 1;
        LocalTensor<T> outSlotT = outT[outSlot * baseDim];
        if (t >= 2) {
            WaitFlag<HardEvent::MTE3_V>(this->outMte3ToVEvent_[outSlot]);
        }

        if constexpr (IsSameType<T, float>::value) {
            if (hasActivation) {
                DataCopy(outSlotT, tmpF, baseDim);
            } else {
                DataCopy(outSlotT, accF, baseDim);
            }
        } else {
            if (hasActivation) {
                Cast(outSlotT, tmpF, RoundMode::CAST_RINT, baseDim);
            } else {
                Cast(outSlotT, accF, RoundMode::CAST_RINT, baseDim);
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

#undef CAUSAL_CONV1D_CLASS
#undef CAUSAL_CONV1D_TEMPLATE_ARGS

template <typename T, uint32_t inputModeKey>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                             const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T, inputModeKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    op.Process();
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H
