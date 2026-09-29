/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "attention_block_vec.h"
#include "softmax_wide.h"
namespace Sage {
struct WideVectorLayout {
    static constexpr uint32_t SCORE_UB = 0, PV_UB = 4 * HALF * N * 2;
    static constexpr uint32_t P_PART_BYTES = HALF * N, PV_SLOT_BYTES = 2 * P_PART_BYTES,
                              ACCUM_UB = PV_UB + 2 * PV_SLOT_BYTES;
    static constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4;
    static constexpr uint32_t MU_UB = P_UB + P_SLOT_BYTES;
    static constexpr uint32_t KS_UB = MU_UB + 512, QS_UB = KS_UB + 2 * 512 * 2;
    static constexpr uint32_t MAX_UB = QS_UB + 512, DEN_UB = MAX_UB + 512, ALPHA_UB = DEN_UB + 512;
    static constexpr uint32_t SUM_UB = ALPHA_UB + 3 * 512;
    static constexpr uint32_t PARTIAL_SUMS_UB = P_UB, END = SUM_UB + HALF * 4;
    static constexpr uint32_t X_UB = P_UB, RED_UB = P_UB + 1024, WORK_UB = P_UB + 2048, QI_UB = P_UB + 4096;
    static_assert(PV_SLOT_BYTES >= HALF * 128 * 4, "PV slot cannot hold FP32 PV");
    static_assert(P_SLOT_BYTES >= HALF * 8 * 4, "output scratch cannot hold softmax partial sums");
    static_assert(END <= Arch::UB_SIZE, "wide softmax exceeds UB");
};

template <class T, uint32_t DP>
class AttentionBlockVecWide : public AttentionBlockVec<T, DP, WideVectorLayout> {
    using Base = AttentionBlockVec<T, DP, WideVectorLayout>;

public:
    __aicore__ inline __attribute__((always_inline)) void PrefetchWideScales(uint64_t kb, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        for (uint32_t pre = 0; pre < Min(count, uint32_t(2)); ++pre) {
            event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + pre);
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(resource_->ubBuf.template GetBufferByByte<half>(KS_UB + pre * 512 * 2), ksg_[kb + pre * 512], 512);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
    }
    template <bool CAUSAL = false, bool STREAM = false>
    __aicore__ inline __attribute__((always_inline)) void SoftmaxWide(uint32_t step, uint64_t kb, uint32_t count,
                                                                      uint32_t qstart = 0, uint32_t group = 0,
                                                                      uint64_t nextScale = ~uint64_t(0))
    {
        const uint32_t ring = STREAM ? group : step;
        auto scores = resource_->ubBuf.template GetBufferByByte<half>(SCORE_UB + (ring & 1) * 2 * HALF * N * 2);
        // Each FP8 NZ block replaces the consumed first FP16 stripe of its
        // 32-column score block. Keep the second stripe as a 2048-byte gap.
        auto p8 = scores.template ReinterpretCast<float8_e4m3_t>();
        auto alpha = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (ring % 3) * 512);
        CrossCoreWaitFlag<4, PIPE_V>(QK_READY);
        CrossCoreWaitFlag<4, PIPE_V>(QK_READY);
        // P_READY for fragment f follows its MTE3 copy. Cube waits for this
        // token before PV(f)'s MTE1->M, which precedes QK(f+4)'s M->FIX.
        // Thus score reuse cannot overwrite P before the copy completes.
        auto ksu = resource_->ubBuf.template GetBufferByByte<half>(KS_UB + (ring & 1) * 512 * 2);
        event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + (ring & 1));
        WaitFlag<HardEvent::MTE2_V>(ev);
        auto scoresNext =
            resource_->ubBuf.template GetBufferByByte<half>(SCORE_UB + ((ring & 1) * 2 + 1) * HALF * N * 2);
        auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
        // Output scratch is idle until the complete softmax group finishes.
        auto partialSums = resource_->ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB);
        if (CAUSAL && step * 512 + 512 > qstart + subBlockIdx_ * HALF + 1) {
            SoftmaxWideFirstVF<true>((__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                                     (__ubuf__ half *)maxima_.GetPhyAddr(), (__ubuf__ float *)denom_.GetPhyAddr(),
                                     (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                                     (__ubuf__ half *)qsu_.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                                     (__ubuf__ float *)scratch.GetPhyAddr(), (__ubuf__ float *)partialSums.GetPhyAddr(),
                                     int32_t(qstart + subBlockIdx_ * HALF) - int32_t(step * 512));
        } else {
            SoftmaxWideFirstVF((__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                               (__ubuf__ half *)maxima_.GetPhyAddr(), (__ubuf__ float *)denom_.GetPhyAddr(),
                               (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                               (__ubuf__ half *)qsu_.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                               (__ubuf__ float *)scratch.GetPhyAddr(), (__ubuf__ float *)partialSums.GetPhyAddr());
        }
        SetFlag<HardEvent::V_MTE2>(ev);
        if constexpr (STREAM) {
            if (nextScale != ~uint64_t(0)) {
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(ksu, ksg_[nextScale], 512);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        } else if (step + 2 < count) {
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(ksu, ksg_[kb + (step + 2) * 512], 512);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
        Sync<HardEvent::V_MTE3>();
        auto pl1 = resource_->l1Buf.template GetBufferByByte<float8_e4m3_t>(P_L1 + (ring & 1) * 2 * M * N);
        DataCopyParams cp{N / 32, HALF, HALF, HALF};
        DataCopy(pl1[subBlockIdx_ * HALF * 32], p8, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
        auto p8Next = scoresNext.template ReinterpretCast<float8_e4m3_t>();
        SoftmaxWideSecondVF((__ubuf__ half *)scoresNext.GetPhyAddr(), (__ubuf__ half *)maxima_.GetPhyAddr(),
                            (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                            (__ubuf__ float8_e4m3_t *)p8Next.GetPhyAddr(),
                            (__ubuf__ float *)scratch.GetPhyAddr(), (__ubuf__ float *)partialSums.GetPhyAddr());
        Sync<HardEvent::V_MTE3>();
        DataCopy(pl1[M * N + subBlockIdx_ * HALF * 32], p8Next, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateWideOutput(uint32_t step)
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        AccumulateWideFp32VF<DP>(
            (__ubuf__ float *)accum_.GetPhyAddr(),
            (__ubuf__ float *)resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (step & 1) * PV_SLOT_BYTES)
                .GetPhyAddr(),
            (__ubuf__ float *)resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (step % 3) * 512)
                .GetPhyAddr(),
            (__ubuf__ float *)denom_.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }

    __aicore__ inline __attribute__((always_inline)) void BeginStream(uint64_t firstScale, uint64_t secondScale)
    {
        SetFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            uint64_t offset = slot ? secondScale : firstScale;
            if (offset != ~uint64_t(0)) {
                event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + slot);
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(resource_->ubBuf.template GetBufferByByte<half>(KS_UB + slot * 1024), ksg_[offset], 512);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        }
    }
    __aicore__ inline __attribute__((always_inline)) void SelectStreamQuery(uint32_t ordinal, uint64_t qb,
                                                                            uint32_t qstart, bool first)
    {
        // MAX_UB already reserves 512 bytes: two independent 256-byte slots.
        maxima_ = resource_->ubBuf.template GetBufferByByte<half>(Base::MAX_UB + (ordinal & 1) * 256);
        if (first) {
            Sync<HardEvent::V_MTE2>();
            DataCopy(qsu_, Base::qsg_[qb + qstart + subBlockIdx_ * HALF], HALF);
            Sync<HardEvent::MTE2_V>();
            Duplicate(maxima_, half(-65504.0f), 128);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline __attribute__((always_inline)) void LoadLseMean(uint32_t b, uint32_t kh)
    {
        if (!Base::params_.returnLse || !Base::params_.smoothK)
            return;
        Sync<HardEvent::V_MTE2>();
        DataCopy(Base::mu_, Base::mg_[(uint64_t(b) * Base::params_.hk + kh) * DP], DP);
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamOutput(uint32_t group, bool first, bool last)
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        auto pv = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (group & 1) * PV_SLOT_BYTES);
        auto alpha = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (group % 3) * 512);
        if (last) {
            auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
            if (first)
                FinalizeWideFp32VF<T, DP, true>(
                    (__ubuf__ T *)Base::qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                    (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                    (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr());
            else
                FinalizeWideFp32VF<T, DP, false>(
                    (__ubuf__ T *)Base::qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                    (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                    (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr());
        } else if (first)
            InitializeWideFp32VF<DP>((__ubuf__ float *)accum_.GetPhyAddr(), (__ubuf__ float *)pv.GetPhyAddr(),
                                     (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float *)denom_.GetPhyAddr());
        else
            AccumulateWideFp32VF<DP>((__ubuf__ float *)accum_.GetPhyAddr(), (__ubuf__ float *)pv.GetPhyAddr(),
                                     (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float *)denom_.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }
    __aicore__ inline __attribute__((always_inline)) void AcquireStreamPv()
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
    }
    template <bool FIRST, bool LAST, bool FIRST_READY>
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamPairImpl(uint64_t group)
    {
        if constexpr (!FIRST_READY)
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        auto pv0 = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (group & 1) * PV_SLOT_BYTES);
        auto pv1 = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + ((group + 1) & 1) * PV_SLOT_BYTES);
        auto a0 = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (group % 3) * 512);
        auto a1 = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + ((group + 1) % 3) * 512);
        auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
        UpdateWidePairVF<T, DP, FIRST, LAST>(
            (__ubuf__ T *)Base::qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
            (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }
    template <bool FIRST_READY = false>
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamPair(uint64_t group, bool first, bool last)
    {
        if (first) {
            if (last)
                UpdateStreamPairImpl<true, true, FIRST_READY>(group);
            else
                UpdateStreamPairImpl<true, false, FIRST_READY>(group);
        } else {
            if (last)
                UpdateStreamPairImpl<false, true, FIRST_READY>(group);
            else
                UpdateStreamPairImpl<false, false, FIRST_READY>(group);
        }
    }
    __aicore__ inline __attribute__((always_inline)) void FinishStreamQuery(uint32_t ordinal, uint32_t qstart,
                                                                            uint32_t b, uint32_t h)
    {
        maxima_ = resource_->ubBuf.template GetBufferByByte<half>(Base::MAX_UB + (ordinal & 1) * 256);
        Base::StoreTileOutput(qstart, b, h);
    }
    __aicore__ inline __attribute__((always_inline)) void EndStream()
    {
        WaitFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
    }

private:
    using Base::resource_;
    using Base::ksg_;
    using Base::maxima_;
    using Base::denom_;
    using Base::qsu_;
    using Base::subBlockIdx_;
    using Base::accum_;
    using Base::KS_UB;
    using Base::SCORE_UB;
    using Base::PV_UB;
    using Base::PV_SLOT_BYTES;
    using Base::ALPHA_UB;
    using Base::SUM_UB;
};
} // namespace Sage
