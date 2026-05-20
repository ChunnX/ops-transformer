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

    __aicore__ inline void LoadWeightAndBias(int32_t channelStart, int32_t baseDim)
    {
        const int32_t dim = tilingData_->dim;
        const int32_t width = static_cast<int32_t>(tilingData_->kernelWidth);
        auto cl = CalcBufPool::Init(calcBuf, width, baseDim, kHasBias);
        LocalTensor<float> &weightCalc = cl.weight;
        LocalTensor<float> &biasCalc = cl.bias;
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

        if constexpr (!kHasBias) {
            Duplicate(biasCalc, 0.0f, baseDim);
        }
    }

    __aicore__ inline void WriteBackState(int32_t cacheIdx, int32_t len, int32_t channelStart, int32_t baseDim,
                                          int32_t dim)
    {
        const int32_t stateLen = tilingData_->stateLen;
        const int32_t width = static_cast<int32_t>(tilingData_->kernelWidth);
        if (len <= 0) {
            return;
        }

        const int32_t lastT = len - 1;
        LocalTensor<T> ring = inBuf.Get<T>();
        const int32_t lastSlot = SlotCurr(lastT, width);
        const int64_t stateBaseOffset = static_cast<int64_t>(cacheIdx) * stateLen * dim + channelStart;

        for (int32_t pos = 0; pos < (width - 1); ++pos) {
            const int32_t tap = (width - 2) - pos;
            const int32_t slot = RetreatRingSlot(lastSlot, tap, width + 1);
            const int64_t stateOffset = stateBaseOffset + static_cast<int64_t>(pos) * dim;
            DataCopy(convStatesGm[stateOffset], ring[slot * baseDim], baseDim);
        }
    }

    __aicore__ inline void WriteBackStateSpec(int32_t cacheIdx, int32_t stateTokenOffset, int32_t start,
                                              int32_t len, int32_t channelStart, int32_t baseDim, int32_t dim)
    {
        const int32_t width = static_cast<int32_t>(tilingData_->kernelWidth);
        const int32_t stateLen = tilingData_->stateLen;
        if (len <= 0) {
            return;
        }

        if (width != 4) {
            WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
            return;
        }

        const int32_t keep = width - 2;
        const int32_t reqStateLen = keep + len;
        if (reqStateLen > stateLen) {
            WriteBackState(cacheIdx, len, channelStart, baseDim, dim);
            return;
        }

        LocalTensor<T> ring = inBuf.Get<T>();
        LocalTensor<T> buf0 = ring[0 * baseDim];
        LocalTensor<T> buf1 = ring[1 * baseDim];

        const int32_t srcPos0 = stateTokenOffset + 1;
        const int32_t srcPos1 = stateTokenOffset + 2;
        const int64_t srcOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos0) * dim + channelStart;
        const int64_t srcOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(srcPos1) * dim + channelStart;
        DataCopy(buf0, convStatesGm[srcOffset0], baseDim);
        DataCopy(buf1, convStatesGm[srcOffset1], baseDim);
        SetFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        const int64_t dstOffset0 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(0) * dim + channelStart;
        const int64_t dstOffset1 =
            static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(1) * dim + channelStart;
        DataCopy(convStatesGm[dstOffset0], buf0, baseDim);
        DataCopy(convStatesGm[dstOffset1], buf1, baseDim);
        SetFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);

        const int64_t xOffset0 = static_cast<int64_t>(start) * dim + channelStart;
        DataCopy(buf0, xGm[xOffset0], baseDim);
        SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);

        for (int32_t t = 0; t < len; ++t) {
            const int32_t curr = t & 1;
            const int32_t next = curr ^ 1;
            LocalTensor<T> currBuf = (curr == 0) ? buf0 : buf1;
            LocalTensor<T> nextBuf = (next == 0) ? buf0 : buf1;

            WaitFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[curr]);

            if (t + 1 < len) {
                const int64_t xOffsetNext = static_cast<int64_t>(start + t + 1) * dim + channelStart;
                if (t > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[next]);
                }
                DataCopy(nextBuf, xGm[xOffsetNext], baseDim);
                SetFlag<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[next]);
            }

            const int64_t dstOffset =
                static_cast<int64_t>(cacheIdx) * stateLen * dim + static_cast<int64_t>(keep + t) * dim + channelStart;
            DataCopy(convStatesGm[dstOffset], currBuf, baseDim);
            SetFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[curr]);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
        if (len > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
        }
    }

    __aicore__ inline void DrainTaskMte3()
    {
        SetFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        WaitFlag<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        SetFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
        WaitFlag<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
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
        stateWritebackMte3ToVEvent_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        stateWritebackMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        stateShiftMte2ToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
        stateShiftVToMte3Event_ = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        stateShiftMte3ToMte2Event_ = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        specWritebackMte2ToMte3Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
        specWritebackMte2ToMte3Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>();
        specWritebackMte3ToMte2Event_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
        specWritebackMte3ToMte2Event_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>();
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
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(stateWritebackMte3ToVEvent_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateWritebackMte3ToMte2Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(stateShiftMte2ToMte3Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(stateShiftVToMte3Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(stateShiftMte3ToMte2Event_);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(specWritebackMte2ToMte3Event_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(specWritebackMte3ToMte2Event_[1]);
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
    TEventID stateWritebackMte3ToVEvent_;
    TEventID stateWritebackMte3ToMte2Event_;
    TEventID stateShiftMte2ToMte3Event_;
    TEventID stateShiftVToMte3Event_;
    TEventID stateShiftMte3ToMte2Event_;
    TEventID specWritebackMte2ToMte3Event_[2];
    TEventID specWritebackMte3ToMte2Event_[2];

    GlobalTensor<T> xGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> biasGm;
    GlobalTensor<T> convStatesGm;
    GlobalTensor<int64_t> queryStartLocGm;
    GlobalTensor<int64_t> cacheIndicesGm;
    GlobalTensor<int64_t> initialStateModeGm;
    GlobalTensor<int64_t> numAcceptedTokensGm;
    GlobalTensor<T> yGm;

    const CausalConv1dTilingData *tilingData_{nullptr};
};

#undef CAUSAL_CONV1D_TEMPLATE_ARGS

} // namespace NsCausalConv1d
#endif // CAUSAL_CONV1D_H
