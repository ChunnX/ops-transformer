/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d.h
 * \brief
 */

#ifndef CAUSAL_CONV1D_H
#define CAUSAL_CONV1D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "causal_conv1d_tiling_data.h"
#include "causal_conv1d_tiling_key.h"
#include "causal_conv1d_common.h"
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "causal_conv1d_regbase.h"
#endif

namespace NsCausalConv1d {

using namespace AscendC;
using namespace NsCausalConv1dCommon;

#define CAUSAL_CONV1D_TEMPLATE_ARGS typename T, uint32_t inputModeKey, uint32_t widthKey, uint32_t hasBiasKey, uint32_t activationKey

enum SeqTaskWindowMode : int32_t {
    SEQ_TASK_WINDOW_MODE_VARLEN = 0,
    SEQ_TASK_WINDOW_MODE_BATCH = 1,
};

inline constexpr int32_t INIT_STATE_SYNCALL_NEED_SIZE = 8;

__aicore__ inline int32_t RetreatRingSlot(int32_t slot, int32_t delta, int32_t ringSize)
{
    int32_t prev = slot - delta;
    return (prev >= 0) ? prev : (prev + ringSize);
}

template <CAUSAL_CONV1D_TEMPLATE_ARGS>
class CausalConv1d {
public:
    __aicore__ inline CausalConv1d() = default;

protected:
    static constexpr int32_t kTemplateWidth = static_cast<int32_t>(widthKey);
    static constexpr bool kHasCompileTimeWidth =
        IsFnInputModeKey(inputModeKey) && (kTemplateWidth >= 2) && (kTemplateWidth <= MAX_WIDTH);
    static constexpr bool kHasBias = (hasBiasKey != 0);
    static constexpr bool kHasActivation = (activationKey != 0);

    __aicore__ inline void ResetRuntimeState(const CausalConv1dTilingData *tilingData)
    {
        tilingData_ = tilingData;
    }

    __aicore__ inline void InitSharedBuffersAndEvents()
    {
        const int32_t rowStride = static_cast<int32_t>(tilingData_->baseDim);
        pipe.InitBuffer(inBuf, (static_cast<int32_t>(tilingData_->kernelWidth) + 1) * rowStride * sizeof(T));
        pipe.InitBuffer(outBuf, 2 * rowStride * sizeof(T));
        if constexpr (kHasCompileTimeWidth) {
            pipe.InitBuffer(calcBuf, (kTemplateWidth + 4) * rowStride * sizeof(float));
        } else {
            const int32_t width = static_cast<int32_t>(tilingData_->kernelWidth);
            const int32_t bufRows = width + (kHasBias ? 4 : 3);
            pipe.InitBuffer(calcBuf, bufRows * rowStride * sizeof(float));
        }
        AllocEvents();
    }

    __aicore__ inline void InitCalcBuf(int32_t baseDim)
    {
        pool_ = CalcBufPool::Init(calcBuf, static_cast<int32_t>(tilingData_->kernelWidth), baseDim, kHasBias);
    }

    __aicore__ inline void LoadWeightAndBias(int32_t channelStart, int32_t baseDim)
    {
        const int32_t dim = tilingData_->dim;
        const int32_t width = static_cast<int32_t>(tilingData_->kernelWidth);
        LocalTensor<float> &weightCalc = pool_.weight;
        LocalTensor<float> &biasCalc = pool_.bias;
        LocalTensor<T> weightDma;
        LocalTensor<T> biasDma;

        if constexpr (!std::is_same<T, float>::value) {
            weightDma = weightCalc.ReinterpretCast<T>();
            biasDma = biasCalc.ReinterpretCast<T>();
        }

        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t srcGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);
        const uint32_t ubBlockBytes = Ops::Base::GetUbBlockSize();
        const uint32_t dstGap = baseDim * (sizeof(float) - sizeof(T)) / ubBlockBytes;

        if constexpr (std::is_same<T, float>::value) {
            DataCopyPad(weightCalc[0], weightGm[channelStart],
                        {static_cast<uint16_t>(width), blockBytes, srcGap, dstGap, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(weightDma[baseDim], weightGm[channelStart],
                        {static_cast<uint16_t>(width), blockBytes, srcGap, dstGap, 0}, {false, 0, 0, 0});
        }

        if constexpr (kHasBias) {
            if constexpr (std::is_same<T, float>::value) {
                DataCopyPad(biasCalc, biasGm[channelStart], {1, blockBytes, 0, 0, 0}, {false, 0, 0, 0});
            } else {
                DataCopyPad(biasDma[baseDim], biasGm[channelStart], {1, blockBytes, 0, 0, 0}, {false, 0, 0, 0});
            }
        }

        SetEvent<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        if constexpr (!std::is_same<T, float>::value) {
            WeightBiasCast<T, kHasBias>(weightDma, biasDma, weightCalc, biasCalc, width, baseDim);
        }
    }

#define CONV1D_ACC(ringN, wPtrN)                                                                               \
    if constexpr (IsSameType<T, float>::value) {                                                             \
        LoadAlign(ringF, ringN + col * V_LENGTH);                                                            \
    } else {                                                                                                 \
        LoadAlign<T, LoadDist::DIST_UNPACK_B16>(ringTmp, ringN + col * V_LENGTH);                            \
        Cast<float, T, CastTraitB16ToB32>(ringF, ringTmp, pregLoop);                                        \
    }                                                                                                        \
    LoadAlign(weight, wPtrN + col * V_LENGTH);                                                               \
    Mul(tmp, ringF, weight, pregLoop);                                                                       \
    Add(bias, bias, tmp, pregLoop)

    __aicore__ inline void ComputeConv1dUnroll(LocalTensor<T> ring, LocalTensor<float> weight,
                                               LocalTensor<float> bias, LocalTensor<T> out, int32_t baseDim,
                                               int32_t t, uint16_t colLoopTimes, uint32_t dataCount)
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
                CONV1D_ACC(ring0, wPtr0);
                CONV1D_ACC(ring1, wPtr1);
                if constexpr (kTemplateWidth >= 3) {
                    CONV1D_ACC(ring2, wPtr2);
                }
                if constexpr (kTemplateWidth >= 4) {
                    CONV1D_ACC(ring3, wPtr3);
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

#undef CONV1D_ACC

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

    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart, int32_t baseDim,
                                          int32_t dim, int32_t destStart = 0)
    {
        const int32_t stateLen = tilingData_->stateLen;
        const int32_t width = static_cast<int32_t>(tilingData_->kernelWidth);
        if (len <= 0) {
            return;
        }

        LocalTensor<T> ring = inBuf.Get<T>();
        const int32_t ringSize = width + 1;
        const int32_t firstAliveSlot = len % ringSize;
        const int32_t lastAliveSlot = (len + width - 2) % ringSize;
        const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim +
                                        static_cast<int64_t>(destStart) * dim + channelStart;
        const uint32_t blockBytes = static_cast<uint32_t>(baseDim) * sizeof(T);
        const uint32_t dstGap = static_cast<uint32_t>(dim - baseDim) * sizeof(T);

        if (firstAliveSlot <= lastAliveSlot) {
            const uint16_t count = static_cast<uint16_t>(width - 1);
            DataCopyPad(convStatesOutGm[stateBaseOffset], ring[firstAliveSlot * baseDim],
                        {count, blockBytes, 0, dstGap, 0});
        } else {
            const uint16_t seg1 = static_cast<uint16_t>(ringSize - firstAliveSlot);
            const uint16_t seg2 = static_cast<uint16_t>(lastAliveSlot + 1);
            DataCopyPad(convStatesOutGm[stateBaseOffset], ring[firstAliveSlot * baseDim],
                        {seg1, blockBytes, 0, dstGap, 0});
            DataCopyPad(convStatesOutGm[stateBaseOffset + static_cast<int64_t>(seg1) * dim],
                        ring[0], {seg2, blockBytes, 0, dstGap, 0});
        }
    }

    __aicore__ inline void DrainTaskMte3()
    {
        SetEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }

    __aicore__ inline void AllocEvents()
    {
        const int32_t ringSize = static_cast<int32_t>(tilingData_->kernelWidth) + 1;
        stateMte2ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        for (int32_t i = 0; i < ringSize; ++i) {
            inputMte2ToVEvent_[i] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        }
        inputVToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        outMte3ToVEvent_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        outMte3ToVEvent_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        outVToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        outVToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void ReleaseEvents()
    {
        const int32_t ringSize = static_cast<int32_t>(tilingData_->kernelWidth) + 1;
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(stateMte2ToVEvent_);
        for (int32_t i = 0; i < ringSize; ++i) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(inputMte2ToVEvent_[i]);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(inputVToMte2Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(outMte3ToVEvent_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(outVToMte3Event_[1]);
    }

    template <int32_t kWindowMode>
    __aicore__ inline void GetSeqWindow(int32_t seq, int32_t seqLen, int32_t &start, int32_t &len) const
    {
        if constexpr (kWindowMode == SEQ_TASK_WINDOW_MODE_VARLEN) {
            start = queryStartLocGm.GetValue(seq);
            len = queryStartLocGm.GetValue(seq + 1) - start;
        } else {
            start = seq * seqLen;
            len = seqLen;
        }
    }

    // 查询 seq i 对应的物理 cache slot，用于多请求复用 conv_states 的场景。
    // cacheIndices 是 1D tensor，shape (batch,)，每个元素告知对应 seq 读哪个 slot。
    // 当 cacheIndices[i] == padSlotId 时返回 false，表示该 seq 为填充位应跳过。
    __aicore__ inline bool ResolveSeqCacheIndex(int32_t seq, bool hasCacheIndices, int32_t &cacheIdx) const
    {
        cacheIdx = seq;
        if (!hasCacheIndices) {
            return true;
        }

        const int64_t cacheIdx64 = cacheIndicesGm.GetValue(seq);
        if (cacheIdx64 == tilingData_->padSlotId) {
            return false;
        }
        cacheIdx = static_cast<int32_t>(cacheIdx64);
        return true;
    }

    __aicore__ inline bool ResolveSeqHasInit(int32_t seq, bool hasInitialStateMode) const
    {
        return hasInitialStateMode ? (initialStateModeGm.GetValue(seq) != 0) : false;
    }

    __aicore__ inline const CausalConv1dTilingData *GetTilingData() const
    {
        return tilingData_;
    }

protected:
    TPipe pipe;
    TBuf<QuePosition::VECIN> inBuf;
    TBuf<QuePosition::VECOUT> outBuf;
    TBuf<QuePosition::VECCALC> calcBuf;

    TEventID stateMte2ToVEvent_;
    TEventID inputMte2ToVEvent_[MAX_WIDTH + 1];
    TEventID inputVToMte2Event_;
    TEventID outMte3ToVEvent_[2];
    TEventID outVToMte3Event_[2];
    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<T> convStatesOutGm;
    GlobalTensor<int64_t> queryStartLocGm;
    GlobalTensor<int64_t> cacheIndicesGm;
    GlobalTensor<int64_t> initialStateModeGm;
    GlobalTensor<int64_t> numAcceptedTokensGm;
    GlobalTensor<T> yGm;

    const CausalConv1dTilingData *tilingData_{nullptr};

    CalcBufPool pool_;
};

#undef CAUSAL_CONV1D_TEMPLATE_ARGS

} // namespace NsCausalConv1d
#endif // CAUSAL_CONV1D_H
