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
    static constexpr uint32_t KS_UB = P_UB + P_SLOT_BYTES, QS_UB = KS_UB + 2 * 512 * 2;
    static constexpr uint32_t MAX_UB = QS_UB + 512, DEN_UB = MAX_UB + 512, ALPHA_UB = DEN_UB + 512;
    static constexpr uint32_t SUM_UB = ALPHA_UB + 3 * 512;
    static constexpr uint32_t VM_UB = SUM_UB + HALF * 4;
    static constexpr uint32_t PARTIAL_SUMS_UB = P_UB, END = VM_UB + 128 * sizeof(float);
    static_assert(PV_SLOT_BYTES >= HALF * 128 * 4, "PV slot cannot hold FP32 PV");
    static_assert(P_SLOT_BYTES >= HALF * 8 * 4, "output scratch cannot hold softmax partial sums");
    static_assert(END <= UB_HIGH_WATER, "wide softmax exceeds UB");
};

template <class T, uint32_t DP>
class AttentionBlockVecWide : public AttentionBlockVec<T, DP, WideVectorLayout> {
    using Base = AttentionBlockVec<T, DP, WideVectorLayout>;

public:
    template <bool CAUSAL>
    __aicore__ inline __attribute__((always_inline)) void SoftmaxWide(uint32_t step, uint32_t count, uint32_t qstart,
                                                                      uint32_t ring, uint64_t nextScale)
    {
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
        // Output scratch is idle until the complete softmax group finishes.
        auto partialSums = resource_->ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB);
        if constexpr (CAUSAL) {
            if (step + 1 == count) {
                SoftmaxWideFirstVF<true>((__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                                         (__ubuf__ half *)maxima_.GetPhyAddr(),
                                         (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                                         (__ubuf__ half *)qsu_.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                                         (__ubuf__ float *)partialSums.GetPhyAddr(),
                                         int32_t(qstart + subBlockIdx_ * HALF) - int32_t(step * 512));
            } else {
                SoftmaxWideFirstVF<false>((__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                                          (__ubuf__ half *)maxima_.GetPhyAddr(),
                                          (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                                          (__ubuf__ half *)qsu_.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                                          (__ubuf__ float *)partialSums.GetPhyAddr());
            }
        } else {
            SoftmaxWideFirstVF<false>((__ubuf__ half *)scores.GetPhyAddr(), (__ubuf__ half *)scoresNext.GetPhyAddr(),
                                      (__ubuf__ half *)maxima_.GetPhyAddr(),
                                      (__ubuf__ float *)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr(),
                                      (__ubuf__ half *)qsu_.GetPhyAddr(), (__ubuf__ half *)ksu.GetPhyAddr(),
                                      (__ubuf__ float *)partialSums.GetPhyAddr());
        }
        SetFlag<HardEvent::V_MTE2>(ev);
        if (nextScale != ~uint64_t(0)) {
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(ksu, ksg_[nextScale], 512);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
        Sync<HardEvent::V_MTE3>();
        auto pl1 = resource_->l1Buf.template GetBufferByByte<float8_e4m3_t>(P_L1 + (ring & 1) * 2 * M * N);
        DataCopyParams cp{N / 32, HALF, HALF, HALF};
        DataCopy(pl1[subBlockIdx_ * HALF * 32], p8, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
        auto p8Next = scoresNext.template ReinterpretCast<float8_e4m3_t>();
        SoftmaxWideSecondVF((__ubuf__ half *)scoresNext.GetPhyAddr(), (__ubuf__ half *)maxima_.GetPhyAddr(),
                            (__ubuf__ float *)alpha.GetPhyAddr(),
                            (__ubuf__ float8_e4m3_t *)p8Next.GetPhyAddr(),
                            (__ubuf__ float *)partialSums.GetPhyAddr());
        Sync<HardEvent::V_MTE3>();
        DataCopy(pl1[M * N + subBlockIdx_ * HALF * 32], p8Next, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
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
            Duplicate(maxima_, half(FP16_MIN), 128);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamOutput(uint32_t group, bool first,
                                                                            uint32_t b, uint32_t h)
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        Base::RoundPvSlot(group & 1);
        Base::LoadVMean(b, h);
        auto pv = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (group & 1) * PV_SLOT_BYTES);
        auto alpha = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (group % 3) * 512);
        auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
        if (first)
            FinalizeWideFp32VF<T, DP, true>(
                (__ubuf__ T *)Base::qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                (__ubuf__ float *)Base::vmean_.GetPhyAddr());
        else
            FinalizeWideFp32VF<T, DP, false>(
                (__ubuf__ T *)Base::qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                (__ubuf__ float *)pv.GetPhyAddr(), (__ubuf__ float *)alpha.GetPhyAddr(),
                (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
                (__ubuf__ float *)Base::vmean_.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }
    __aicore__ inline __attribute__((always_inline)) void AcquireStreamPv()
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
    }
    template <bool FIRST, bool LAST, bool FIRST_READY>
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamPairImpl(uint64_t group, uint32_t b,
                                                                              uint32_t h)
    {
        if constexpr (!FIRST_READY)
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        Base::RoundPvSlot(group & 1);
        Base::RoundPvSlot((group + 1) & 1);
        if constexpr (LAST)
            Base::LoadVMean(b, h);
        auto pv0 = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (group & 1) * PV_SLOT_BYTES);
        auto pv1 = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + ((group + 1) & 1) * PV_SLOT_BYTES);
        auto a0 = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (group % 3) * 512);
        auto a1 = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + ((group + 1) % 3) * 512);
        auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
        UpdateWidePairVF<T, DP, FIRST, LAST>(
            (__ubuf__ T *)Base::qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
            (__ubuf__ float *)pv0.GetPhyAddr(), (__ubuf__ float *)pv1.GetPhyAddr(),
            (__ubuf__ float *)a0.GetPhyAddr(), (__ubuf__ float *)a1.GetPhyAddr(),
            (__ubuf__ float *)denom_.GetPhyAddr(), (__ubuf__ float *)scratch.GetPhyAddr(),
            (__ubuf__ float *)Base::vmean_.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }
    template <bool FIRST_READY = false>
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamPair(uint64_t group, bool first, bool last,
                                                                           uint32_t b, uint32_t h)
    {
        if (first) {
            if (last)
                UpdateStreamPairImpl<true, true, FIRST_READY>(group, b, h);
            else
                UpdateStreamPairImpl<true, false, FIRST_READY>(group, b, h);
        } else {
            if (last)
                UpdateStreamPairImpl<false, true, FIRST_READY>(group, b, h);
            else
                UpdateStreamPairImpl<false, false, FIRST_READY>(group, b, h);
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
    using Base::RoundPvSlot;
    using Base::LoadVMean;
    using Base::KS_UB;
    using Base::SCORE_UB;
    using Base::PV_UB;
    using Base::PV_SLOT_BYTES;
    using Base::ALPHA_UB;
    using Base::SUM_UB;
};
} // namespace Sage
