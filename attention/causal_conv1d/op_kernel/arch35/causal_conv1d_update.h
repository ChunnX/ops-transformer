/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_update.h
 * \brief
 */

#ifndef CAUSAL_CONV1D_UPDATE_H
#define CAUSAL_CONV1D_UPDATE_H

#include "causal_conv1d.h"

namespace NsCausalConv1d {

template <typename T, uint32_t inputModeKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dUpdate : public CausalConv1d<T, inputModeKey, CAUSAL_CONV1D_TPL_WIDTH_RUNTIME, hasBiasKey, activationKey> {
    static constexpr bool kHasBias = (hasBiasKey != 0);
    static constexpr bool kHasActivation = (activationKey != 0);

public:
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
        if constexpr (IsVarlenInputModeKey(inputModeKey)) {
            this->template ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_VARLEN>();
        } else {
            this->template ProcessDefaultByWindowMode<SEQ_TASK_WINDOW_MODE_BATCH>();
        }
        this->ReleaseEvents();
    }

protected:

    template <int32_t kWindowMode>
    __aicore__ inline void ProcessDefaultByWindowMode()
    {
        const int32_t dim = this->tilingData_->dim;
        const int32_t batch = this->tilingData_->batch;
        const int32_t seqLen = this->tilingData_->seqLen;
        const int32_t baseDim = static_cast<int32_t>(this->tilingData_->baseDim);
        const int32_t baseDimCnt = (dim + baseDim - 1) / baseDim;
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const bool hasCacheIndices = (this->tilingData_->hasCacheIndices != 0);
        const uint32_t blockIdx = GetBlockIdx();
        const int64_t tokensPerBlock = this->tilingData_->tokensPerBlock;
        const int64_t tokenBlockCnt = this->tilingData_->tokenBlockCnt;

        const int64_t coreCnt = tokenBlockCnt * baseDimCnt;
        if (static_cast<int64_t>(blockIdx) >= coreCnt) {
            this->ReleaseEvents();
            return;
        }

        const int32_t tokBlock = static_cast<int32_t>(static_cast<int64_t>(blockIdx) / baseDimCnt);
        const int32_t baseDimIdx = static_cast<int32_t>(static_cast<int64_t>(blockIdx) % baseDimCnt);
        const int32_t channelStart = baseDimIdx * baseDim;
        const int32_t curBaseDim = (channelStart + baseDim <= dim) ? baseDim : (dim - channelStart);

        this->LoadWeightAndBias(channelStart, curBaseDim);

        const int64_t seqStart = tokBlock * tokensPerBlock;
        const int64_t seqEnd = (seqStart + tokensPerBlock < batch) ? (seqStart + tokensPerBlock) : batch;
        for (int64_t seq = seqStart; seq < seqEnd; ++seq) {
            int32_t start = 0;
            int32_t len = 0;
            this->template GetSeqWindow<kWindowMode>(static_cast<int32_t>(seq), seqLen, start, len);

            // 查出 seq 的物理 cache slot，填充 seq 直接跳过 continue
            int32_t cacheIdx = 0;
            if (!this->ResolveSeqCacheIndex(static_cast<int32_t>(seq), hasCacheIndices, cacheIdx)) {
                continue;
            }

            const int32_t stateTokenOffset = this->GetStateTokenOffset(static_cast<int32_t>(seq));

            this->InitRing(cacheIdx, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
            this->RunSeq(start, len, channelStart, curBaseDim, dim);

            if (this->tilingData_->hasNumAcceptedTokens) {
                this->DrainTaskMte3();
                this->WriteBackStateSpec(cacheIdx, stateTokenOffset, start, len, channelStart, curBaseDim,
                                         dim);
            } else {
                this->WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
            }

            this->DrainTaskMte3();
        }
    }

    // 算 spec decode 的 state 偏移。普通 decode 返回 0。
    __aicore__ inline int32_t GetStateTokenOffset(int32_t seq) const
    {
        if (!this->tilingData_->hasNumAcceptedTokens) {
            return 0;
        }
        int32_t offset = static_cast<int32_t>(this->numAcceptedTokensGm.GetValue(seq)) - 1;
        return (offset >= 0) ? offset : 0;
    }

    __aicore__ inline void InitRing(int32_t cacheIdx, int32_t stateTokenOffset, int32_t start,
                                    int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        LocalTensor<T> ring = this->inBuf.template Get<T>();

        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);
        const int64_t stateBaseOffset =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(stateTokenOffset) * dim + channelStart;
        DataCopyPad(ring[0], this->convStatesGm[stateBaseOffset],
                    {static_cast<uint16_t>(kernelWidth - 1), blockBytes, srcGap, 0, 0}, {false, 0, 0, 0});

        const int32_t slot0 = SlotCurr(0, kernelWidth);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
        DataCopy(ring[slot0 * baseDim], this->xGm[xOffset], baseDim);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

    }

    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        auto cl = CalcBufPool::Init(this->calcBuf, kernelWidth, baseDim, kHasBias);
        LocalTensor<float> &weightF = cl.weight;
        LocalTensor<float> &biasF = cl.bias;
        LocalTensor<float> &accF = cl.acc;
        LocalTensor<float> &tmpF = cl.tmp;
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        LocalTensor<T> outT = this->outBuf.template Get<T>();
        for (int32_t t = 0; t < len; ++t) {
            const int32_t slotCurr = SlotCurr(t, kernelWidth);

            WaitFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);

            if (t + 1 < len) {
                const int32_t slotNext = SlotPrefetch(t, kernelWidth);
                const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                if (t > 0) {
                    WaitFlag<HardEvent::V_MTE2>((t & 1) ? EVENT_ID1 : EVENT_ID0);
                }
                DataCopy(ring[slotNext * baseDim], this->xGm[xOffsetNext], baseDim);
                SetFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            bool accInitialized = false;
            if constexpr (kHasBias) {
                Adds(accF, biasF, 0.0f, baseDim);
                PipeBarrier<PIPE_V>();
                accInitialized = true;
            }

            for (int32_t j = 0; j < kernelWidth; ++j) {
                const int32_t tap = (kernelWidth - 1) - j;
                const int32_t slot = (tap == 0) ? slotCurr : SlotHist(t, tap, kernelWidth);
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

            if constexpr (kHasActivation) {
                Silu(tmpF, accF, baseDim);
            }

            const int32_t outSlot = t & 1;
            LocalTensor<T> outSlotT = outT[outSlot * baseDim];
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>(this->outMte3ToVEvent_[outSlot]);
            }

            if constexpr (IsSameType<T, float>::value) {
                if constexpr (kHasActivation) {
                    DataCopy(outSlotT, tmpF, baseDim);
                } else {
                    DataCopy(outSlotT, accF, baseDim);
                }
            } else {
                if constexpr (kHasActivation) {
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
                SetFlag<HardEvent::V_MTE2>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }
        }
    }
};

template <typename T, uint32_t inputModeKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                             const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T, inputModeKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    op.Process();
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H
