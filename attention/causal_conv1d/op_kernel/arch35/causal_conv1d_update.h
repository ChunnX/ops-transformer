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

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
class CausalConv1dUpdate : public CausalConv1d<T, inputModeKey, widthKey, hasBiasKey, activationKey> {
    static constexpr int32_t kTemplateWidth = static_cast<int32_t>(widthKey);
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

        this->InitCalcBuf(curBaseDim);
        this->LoadWeightAndBias(channelStart, curBaseDim);

        const int64_t seqStart = tokBlock * tokensPerBlock;
        const int64_t seqEnd = (seqStart + tokensPerBlock < batch) ? (seqStart + tokensPerBlock) : batch;
        for (int64_t seq = seqStart; seq < seqEnd; ++seq) {
            int32_t start = 0;
            int32_t len = 0;
            this->template GetSeqWindow<kWindowMode>(static_cast<int32_t>(seq), seqLen, start, len);

            int32_t cacheIdx = 0;
            if (!this->ResolveSeqCacheIndex(static_cast<int32_t>(seq), hasCacheIndices, cacheIdx)) {
                continue;
            }

            const int32_t stateTokenOffset = this->GetStateTokenOffset(static_cast<int32_t>(seq));

            this->InitRing(cacheIdx, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
            this->RunSeq(start, len, channelStart, curBaseDim, dim);

            if (this->tilingData_->hasNumAcceptedTokens) {
                this->DrainTaskMte3();
                this->WriteBackStateSpec(cacheIdx, stateTokenOffset, start, len, channelStart, curBaseDim, dim);
            } else {
                this->WriteBackState(cacheIdx, len, channelStart, curBaseDim, dim);
            }

            this->DrainTaskMte3();
        }
    }

    __aicore__ inline int32_t GetStateTokenOffset(int32_t seq) const
    {
        if (!this->tilingData_->hasNumAcceptedTokens) {
            return 0;
        }
        int32_t offset = static_cast<int32_t>(this->numAcceptedTokensGm.GetValue(seq)) - 1;
        return (offset >= 0) ? offset : 0;
    }

    __aicore__ inline void InitRing(int32_t cacheIdx, int32_t stateTokenOffset, int32_t start, int32_t len,
                                    int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t stateLen = this->tilingData_->stateLen;
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        LocalTensor<T> ring = this->inBuf.template Get<T>();

        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);
        const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                        static_cast<int64_t>(stateTokenOffset) * dim + channelStart;
        DataCopyPad(ring[0], this->convStatesGm[stateBaseOffset],
                    {static_cast<uint16_t>(kernelWidth - 1), blockBytes, srcGap, 0, 0}, {false, 0, 0, 0});

        const int32_t slot0 = SlotCurr(0, kernelWidth);
        const int64_t xOffset = static_cast<int64_t>(start) * dim + channelStart;
        DataCopyPad(ring[slot0 * baseDim], this->xGm[xOffset], {1, blockBytes, 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void ComputeConv1dUnroll(LocalTensor<T> ring, LocalTensor<float> weight, LocalTensor<float> bias,
                                               LocalTensor<T> out, int32_t baseDim, int32_t t, uint16_t colLoopTimes,
                                               uint32_t dataCount)
    {
        const uint16_t ringSize = static_cast<uint16_t>(kTemplateWidth + 1);
        __ubuf__ T *ringAddr = reinterpret_cast<__ubuf__ T *>(ring.GetPhyAddr());
        __ubuf__ float *weightAddr = reinterpret_cast<__ubuf__ float *>(weight.GetPhyAddr());
        __ubuf__ float *biasAddr = reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr());
        __ubuf__ T *outAddr = reinterpret_cast<__ubuf__ T *>(out.GetPhyAddr());
        __ubuf__ T *ring0 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 0) % ringSize) * baseDim;
        __ubuf__ T *ring1 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 1) % ringSize) * baseDim;
        __ubuf__ T *ring2 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 2) % ringSize) * baseDim;
        __ubuf__ T *ring3 = ringAddr + static_cast<int32_t>((static_cast<uint16_t>(t) + 3) % ringSize) * baseDim;
        __ubuf__ float *wPtr0 = weightAddr + static_cast<int32_t>(0) * baseDim;
        __ubuf__ float *wPtr1 = weightAddr + static_cast<int32_t>(1) * baseDim;
        __ubuf__ float *wPtr2 = weightAddr + static_cast<int32_t>(2) * baseDim;
        __ubuf__ float *wPtr3 = weightAddr + static_cast<int32_t>(3) * baseDim;
        __VEC_SCOPE__
        {
            RegTensor<T> ringTmp;
            RegTensor<float> ringF;
            RegTensor<float> weight;
            RegTensor<float> bias;
            RegTensor<float> tmp;
            RegTensor<T> outVal;
            MaskReg pregLoop;
            for (uint16_t col = 0; col < colLoopTimes; col++) {
                pregLoop = UpdateMask<float>(dataCount);
                if constexpr (kHasBias) {
                    LoadAlign(bias, biasAddr + col * V_LENGTH);
                } else {
                    Duplicate(bias, 0.0f, pregLoop);
                }
                // tap 0
                if constexpr (IsSameType<T, float>::value) {
                    LoadAlign(ringF, ring0 + col * V_LENGTH);
                } else {
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ring0 + col * V_LENGTH);
                    Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);
                }
                LoadAlign(weight, wPtr0 + col * V_LENGTH);
                Mul(tmp, ringF, weight, pregLoop);
                Add(bias, bias, tmp, pregLoop);
                // tap 1
                if constexpr (IsSameType<T, float>::value) {
                    LoadAlign(ringF, ring1 + col * V_LENGTH);
                } else {
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ring1 + col * V_LENGTH);
                    Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);
                }
                LoadAlign(weight, wPtr1 + col * V_LENGTH);
                Mul(tmp, ringF, weight, pregLoop);
                Add(bias, bias, tmp, pregLoop);
                // tap 2
                if constexpr (kTemplateWidth >= 3) {
                    if constexpr (IsSameType<T, float>::value) {
                        LoadAlign(ringF, ring2 + col * V_LENGTH);
                    } else {
                        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ring2 + col * V_LENGTH);
                        Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);
                    }
                    LoadAlign(weight, wPtr2 + col * V_LENGTH);
                    Mul(tmp, ringF, weight, pregLoop);
                    Add(bias, bias, tmp, pregLoop);
                }
                // tap 3
                if constexpr (kTemplateWidth >= 4) {
                    if constexpr (IsSameType<T, float>::value) {
                        LoadAlign(ringF, ring3 + col * V_LENGTH);
                    } else {
                        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ring3 + col * V_LENGTH);
                        Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);
                    }
                    LoadAlign(weight, wPtr3 + col * V_LENGTH);
                    Mul(tmp, ringF, weight, pregLoop);
                    Add(bias, bias, tmp, pregLoop);
                }
                if constexpr (kHasActivation) {
                    Muls(tmp, bias, -1.0f, pregLoop);
                    Exp(tmp, tmp, pregLoop);
                    Adds(tmp, tmp, 1.0f, pregLoop);
                    Div(bias, bias, tmp, pregLoop);
                }
                if constexpr (IsSameType<T, float>::value) {
                    StoreAlign(outAddr + col * V_LENGTH, bias, pregLoop);
                } else {
                    Cast<T, float, CastTraitB32ToB16>(outVal, bias, pregLoop);
                    StoreAlign<T, StoreDist::DIST_PACK_B32>(outAddr + col * V_LENGTH, outVal, pregLoop);
                }
            }
        }
    }

    __aicore__ inline void ComputeConv1d(LocalTensor<T> ring, LocalTensor<float> weight, LocalTensor<float> bias,
                                         LocalTensor<T> out, int32_t baseDim, int32_t t)
    {
        uint16_t colLoopTimes = static_cast<uint16_t>(Ceil(baseDim, V_LENGTH));
        uint32_t dataCount = static_cast<uint32_t>(baseDim);
        if constexpr (kTemplateWidth >= 2 && kTemplateWidth <= 4) {
            ComputeConv1dUnroll(ring, weight, bias, out, baseDim, t, colLoopTimes, dataCount);
        } else {
            __ubuf__ T *ringAddr = reinterpret_cast<__ubuf__ T *>(ring.GetPhyAddr());
            __ubuf__ float *weightAddr = reinterpret_cast<__ubuf__ float *>(weight.GetPhyAddr());
            __ubuf__ float *biasAddr = reinterpret_cast<__ubuf__ float *>(bias.GetPhyAddr());
            __ubuf__ T *outAddr = reinterpret_cast<__ubuf__ T *>(out.GetPhyAddr());
            const int32_t kernelWidth = kTemplateWidth;
            const uint16_t ringSize = static_cast<uint16_t>(kernelWidth + 1);
            __VEC_SCOPE__
            {
                RegTensor<T> ringTmp;
                RegTensor<float> ringF;
                RegTensor<float> weight;
                RegTensor<float> bias;
                RegTensor<float> tmp;
                RegTensor<T> outVal;
                MaskReg pregLoop;
                for (uint16_t col = 0; col < colLoopTimes; col++) {
                    pregLoop = UpdateMask<float>(dataCount);
                    if constexpr (kHasBias) {
                        LoadAlign(bias, biasAddr + col * V_LENGTH);
                    } else {
                        Duplicate(bias, 0.0f, pregLoop);
                    }
                    for (uint16_t j = 0; j < static_cast<uint16_t>(kernelWidth); j++) {
                        const int32_t ringOffset =
                            static_cast<int32_t>((static_cast<uint16_t>(t) + j) % ringSize) * baseDim + col * V_LENGTH;
                        const int32_t weightOffset = static_cast<int32_t>(j) * baseDim + col * V_LENGTH;
                        if constexpr (IsSameType<T, float>::value) {
                            LoadAlign(ringF, ringAddr + ringOffset);
                        } else {
                            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ringAddr + ringOffset);
                            Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);
                        }
                        LoadAlign(weight, weightAddr + weightOffset);
                        Mul(tmp, ringF, weight, pregLoop);
                        Add(bias, bias, tmp, pregLoop);
                    }
                    if constexpr (kHasActivation) {
                        Muls(tmp, bias, -1.0f, pregLoop);
                        Exp(tmp, tmp, pregLoop);
                        Adds(tmp, tmp, 1.0f, pregLoop);
                        Div(bias, bias, tmp, pregLoop);
                    }
                    if constexpr (IsSameType<T, float>::value) {
                        StoreAlign(outAddr + col * V_LENGTH, bias, pregLoop);
                    } else {
                        Cast<T, float, CastTraitB32ToB16>(outVal, bias, pregLoop);
                        StoreAlign<T, StoreDist::DIST_PACK_B32>(outAddr + col * V_LENGTH, outVal, pregLoop);
                    }
                }
            }
        }
    }

    __aicore__ inline void RunSeq(int32_t start, int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t kernelWidth = static_cast<int32_t>(this->tilingData_->kernelWidth);
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        LocalTensor<T> ring = this->inBuf.template Get<T>();
        LocalTensor<T> outT = this->outBuf.template Get<T>();

        // EVENT_ID0/1: hardware event ping-pong. t even→ID0, t odd→ID1.
        // MTE2_V: MTE→vec, ring[t] data ready.
        // V_MTE2: vec→MTE, ring[t+1] slot free to load.
        // V_MTE3: vec→MTE, output data ready for GM copy.
        // MTE3_V: MTE→vec, output buffer consumed, slot free to reuse.
        for (int32_t t = 0; t < len; ++t) {
            // wait for x[t] (loaded in InitRing or prev prefetch)
            WaitFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);

            // prefetch x[t+1] if available
            if (t + 1 < len) {
                const int32_t slotNext = SlotPrefetch(t, kernelWidth);
                const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                // wait for previous prefetch to release MTE pipeline
                if (t > 0) {
                    WaitFlag<HardEvent::V_MTE2>((t & 1) ? EVENT_ID1 : EVENT_ID0);
                }
                DataCopyPad(ring[slotNext * baseDim], this->xGm[xOffsetNext], {1, blockBytes, 0, 0, 0},
                            {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            // wait for output double-buffer slot to be consumed
            const int32_t outSlot = t & 1;
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            // compute conv1d for timestep t
            this->ComputeConv1d(ring, this->pool_.weight, this->pool_.bias, outT[outSlot * baseDim], baseDim, t);

            // signal output ready, then wait for it to complete before GM copy
            SetFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            WaitFlag<HardEvent::V_MTE3>((t & 1) ? EVENT_ID0 : EVENT_ID1);

            // copy output to GM
            const int64_t outOffset = static_cast<int64_t>(start + t) * dim + channelStart;
            LocalTensor<T> outSlotT = outT[outSlot * baseDim];
            DataCopyPad(this->yGm[outOffset], outSlotT, {1, blockBytes, 0, 0, 0});
            // signal output slot consumed → MTE3 free to reuse
            if (t + 2 < len) {
                SetFlag<HardEvent::MTE3_V>((t & 1) ? EVENT_ID1 : EVENT_ID0);
            }

            // signal ring slot freed → MTE2 can load next
            if (t + 2 < len) {
                SetFlag<HardEvent::V_MTE2>((t & 1) ? EVENT_ID0 : EVENT_ID1);
            }
        }
    }
};

template <typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey>
__aicore__ inline void RunCausalConv1dUpdate(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR convStates,
                                             GM_ADDR queryStartLoc, GM_ADDR cacheIndices, GM_ADDR initialStateMode,
                                             GM_ADDR numAcceptedTokens, GM_ADDR y, GM_ADDR workspace,
                                             const CausalConv1dTilingData *tilingData)
{
    CausalConv1dUpdate<T, inputModeKey, widthKey, hasBiasKey, activationKey> op;
    op.Init(x, weight, bias, convStates, queryStartLoc, cacheIndices, initialStateMode, numAcceptedTokens, y, workspace,
            tilingData);
    op.Process();
}

} // namespace NsCausalConv1d

#endif // CAUSAL_CONV1D_UPDATE_H
