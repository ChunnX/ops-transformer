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
#include "attention_common.h"
#include "softmax.h"
#include "output_update.h"
namespace Sage {
struct DefaultVectorLayout {
    static constexpr uint32_t ACCUM_UB = Sage::ACCUM_UB;
    static constexpr uint32_t MU_UB = Sage::MU_UB;
    static constexpr uint32_t P_UB = Sage::P_UB;
    static constexpr uint32_t QS_UB = Sage::QS_UB;
    static constexpr uint32_t MAX_UB = Sage::MAX_UB;
    static constexpr uint32_t DEN_UB = Sage::DEN_UB;
    static constexpr uint32_t KS_UB = Sage::KS_UB;
    static constexpr uint32_t SCORE_UB = Sage::SCORE_UB;
    static constexpr uint32_t ALPHA_UB = Sage::ALPHA_UB;
    static constexpr uint32_t SUM_UB = Sage::SUM_UB;
    static constexpr uint32_t PV_UB = Sage::PV_UB;
    static constexpr uint32_t PV_SLOT_BYTES = Sage::PV_SLOT_BYTES;
};
// AIV stages own scale prefetch, softmax/P transport, output update and LSE.
// All methods inline into the mixed entry to preserve AIC/AIV specialization.
template <class T, uint32_t DP, class Layout = DefaultVectorLayout>
class AttentionBlockVec {
public:
    __aicore__ inline __attribute__((always_inline)) void Init(FixedResource &resource, GM_ADDR q, GM_ADDR qs,
                                                               GM_ADDR ks, GM_ADDR mean, GM_ADDR out, GM_ADDR lse,
                                                               SageParams params)
    {
        resource_ = &resource;
        params_ = params;
        subBlockIdx_ = GetSubBlockIdx();
        qsg_.SetGlobalBuffer((__gm__ half *)qs);
        ksg_.SetGlobalBuffer((__gm__ half *)ks);
        mg_.SetGlobalBuffer((__gm__ float *)mean);
        lg_.SetGlobalBuffer((__gm__ float *)lse);
        qg_.SetGlobalBuffer((__gm__ T *)q);
        og_.SetGlobalBuffer((__gm__ T *)out);

        accum_ = resource_->ubBuf.GetBufferByByte<float>(ACCUM_UB);
        mu_ = resource_->ubBuf.GetBufferByByte<float>(MU_UB);
        qo_ = resource_->ubBuf.GetBufferByByte<T>(P_UB);
        qsu_ = resource_->ubBuf.GetBufferByByte<half>(QS_UB);
        maxima_ = resource_->ubBuf.GetBufferByByte<half>(MAX_UB);
        denom_ = resource_->ubBuf.GetBufferByByte<float>(DEN_UB);
    }
    __aicore__ inline __attribute__((always_inline)) void BeginTile(uint32_t b, uint32_t kh, uint64_t qb,
                                                                    uint32_t qstart)
    {
        Sync<HardEvent::V_MTE2>();
        Duplicate(accum_, 0.0f, HALF * params_.dp);
        DataCopy(mu_, mg_[(uint64_t(b) * params_.hk + kh) * params_.dp], params_.dp);
        DataCopy(qsu_, qsg_[qb + qstart + subBlockIdx_ * HALF], HALF);
        Sync<HardEvent::MTE2_S>();
        Duplicate(maxima_, half(-65504.0f), 128);
        Duplicate(denom_, 0.0f, 64);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline __attribute__((always_inline)) void PrefetchScales(uint64_t kb, uint32_t count)
    {
        // IDs 4/5 own the two immutable K-scale slots. Every prefetch
        // consumes a free token and publishes one MTE2->V ready token.
        SetFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        for (uint32_t pre = 0; pre < Min(count, uint32_t(2)); ++pre) {
            event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + pre);
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(resource_->ubBuf.GetBufferByByte<half>(KS_UB + pre * N * 2), ksg_[kb + pre * N], N);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
    }
    template <bool CAUSAL = false>
    __aicore__ inline __attribute__((always_inline)) void Softmax(uint32_t step, uint32_t qstart, uint64_t kb,
                                                                  uint32_t count)
    {
        uint32_t start = step * N, parity = step % PIPE_SLOTS;
        auto scores = resource_->ubBuf.GetBufferByByte<half>(SCORE_UB + parity * HALF * N * 2);
        auto p8 = resource_->ubBuf.GetBufferByByte<float8_e4m3_t>(P_UB + parity * P_SLOT_BYTES);
        auto alpha = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + (step % ALPHA_SLOTS) * 512);
        CrossCoreWaitFlag<4, PIPE_V>(QK_READY);
        auto ksu = resource_->ubBuf.GetBufferByByte<half>(KS_UB + (step & 1) * N * 2);
        event_t scaleEvent = static_cast<event_t>(SCALE_EVENT_BASE + (step & 1));
        WaitFlag<HardEvent::MTE2_V>(scaleEvent);
        const bool isFull = (qstart + subBlockIdx_ * HALF + HALF <= params_.sq && start + N <= params_.sk &&
                             (!CAUSAL || start + N <= qstart + subBlockIdx_ * HALF + 1));
        auto scoresPtr = (__ubuf__ half *)scores.GetPhyAddr();
        auto maxPtr = (__ubuf__ half *)maxima_.GetPhyAddr();
        auto denPtr = (__ubuf__ float *)denom_.GetPhyAddr();
        auto alphaPtr = (__ubuf__ float *)alpha.GetPhyAddr();
        auto p8Ptr = (__ubuf__ float8_e4m3_t *)p8.GetPhyAddr();
        auto qsPtr = (__ubuf__ half *)qsu_.GetPhyAddr();
        auto ksPtr = (__ubuf__ half *)ksu.GetPhyAddr();
        auto sumPtr = (__ubuf__ float *)resource_->ubBuf.template GetBufferByByte<float>(SUM_UB).GetPhyAddr();
        const uint32_t qOffset = qstart + subBlockIdx_ * HALF;

        if (isFull) {
            SoftmaxHalfVF<HALF, N, true, true, P_NZ_PAD_ROWS>(
                scoresPtr, maxPtr, denPtr, alphaPtr, p8Ptr, qsPtr, ksPtr, sumPtr,
                qOffset, params_.sq, start, params_.sk, CAUSAL);
        } else {
            SoftmaxHalfVF<HALF, N, false, true, P_NZ_PAD_ROWS>(
                scoresPtr, maxPtr, denPtr, alphaPtr, p8Ptr, qsPtr, ksPtr, sumPtr,
                qOffset, params_.sq, start, params_.sk, CAUSAL);
        }
        SetFlag<HardEvent::V_MTE2>(scaleEvent);
        if (step + 2 < count) {
            WaitFlag<HardEvent::V_MTE2>(scaleEvent);
            DataCopy(ksu, ksg_[kb + (step + 2) * N], N);
            SetFlag<HardEvent::MTE2_V>(scaleEvent);
        }
        Sync<HardEvent::V_MTE3>();
        auto pl1 = resource_->l1Buf.GetBufferByByte<float8_e4m3_t>(P_L1 + parity * M * N);
        DataCopyParams cp{N / 32, HALF, P_NZ_PAD_ROWS, HALF};
        DataCopy(pl1[subBlockIdx_ * HALF * 32], p8, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateOutput(uint32_t index, uint32_t count)
    {
        if (index & 1) {
            // Both PV slots remain owned by this pair until both reads finish.
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
            auto alpha0 = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + ((index - 1) % ALPHA_SLOTS) * 512);
            auto alpha1 = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
            AccumulateFp32PairVF<DP>(
                (__ubuf__ float *)accum_.GetPhyAddr(),
                (__ubuf__ float *)resource_->ubBuf.GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                (__ubuf__ float *)resource_->ubBuf.GetBufferByByte<float>(PV_UB + PV_SLOT_BYTES).GetPhyAddr(),
                (__ubuf__ float *)alpha0.GetPhyAddr(), (__ubuf__ float *)alpha1.GetPhyAddr());
            CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
            CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        } else if (index + 1 == count) {
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
            auto alpha = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
            AccumulateFp32VF<DP>((__ubuf__ float *)accum_.GetPhyAddr(),
                                 (__ubuf__ float *)resource_->ubBuf.GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                                 (__ubuf__ float *)alpha.GetPhyAddr(), HALF);
            CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        }
    }
    template <bool FusedCast = true, bool DrainScales = true>
    __aicore__ inline __attribute__((always_inline)) void FinalizeTile(uint32_t qstart, uint32_t b, uint32_t h)
    {
        if constexpr (DrainScales) {
            WaitFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        }
        if constexpr (FusedCast) {
            NormalizeCastVF<T, DP>(
                (__ubuf__ T *)qo_.GetPhyAddr(), (__ubuf__ float *)accum_.GetPhyAddr(),
                (__ubuf__ float *)denom_.GetPhyAddr(),
                (__ubuf__ float *)resource_->ubBuf.template GetBufferByByte<float>(SUM_UB).GetPhyAddr());
        } else {
            NormalizeVF((__ubuf__ float *)accum_.GetPhyAddr(), (__ubuf__ float *)denom_.GetPhyAddr(), params_.dp, HALF,
                        (__ubuf__ float *)resource_->ubBuf.template GetBufferByByte<float>(SUM_UB).GetPhyAddr());
            PipeBarrier<PIPE_V>();
            Cast(qo_, accum_, RoundMode::CAST_RINT, HALF * params_.dp);
        }
        StoreTileOutput(qstart, b, h);
    }
    __aicore__ inline __attribute__((always_inline)) void StoreTileOutput(uint32_t qstart, uint32_t b, uint32_t h)
    {
        PipeBarrier<PIPE_V>();
        uint32_t first = qstart + subBlockIdx_ * HALF;
        uint32_t validRows = first < params_.sq ? Min(HALF, params_.sq - first) : 0;
        if (params_.d == params_.dp) {
            if (validRows)
                WriteRow(og_, ((uint64_t(b) * params_.hq + h) * params_.sq + first) * params_.d, qo_,
                         validRows * params_.d);
        } else {
            for (uint32_t i = 0; i < validRows; ++i)
                WriteRow(og_, ((uint64_t(b) * params_.hq + h) * params_.sq + first + i) * params_.d,
                         qo_[i * params_.dp], params_.d);
        }
        // WriteRow fences output reuse; LSE owns its additional scalar fences.
        if (params_.returnLse)
            StoreTileLse(first, validRows, b, h);
    }

protected:
    __aicore__ inline __attribute__((always_inline)) void StoreTileLse(uint32_t first, uint32_t validRows,
                                                                       uint32_t b, uint32_t h)
    {
        // Vectorized epilogue: lse = float(maxima) + ln(denom) + sm_scale * <q, mean>.
        // One vector pass plus a single GM write replaces the former per-row loop
        // (two scalar UB reads, one Ln and one 4-byte GM store per row).
        // Scratch reuses the retired P0/output staging after its MTE3 fence; the
        // AIC may already be writing the next query tile's QK into the score area.
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::V_S>();
        if (!validRows) {
            return;
        }
        static_assert(2 * 8 * 128 * sizeof(float) + 1024 <= P_SLOT_BYTES, "LSE scratch exceeds retired P0");
        constexpr uint32_t CHUNK = 8;
        const uint32_t dp = params_.dp;
        auto raw = resource_->ubBuf.template GetBufferByByte<T>(P_UB);
        auto qf = resource_->ubBuf.template GetBufferByByte<float>(P_UB + CHUNK * dp * sizeof(T));
        auto lse = resource_->ubBuf.template GetBufferByByte<float>(P_UB + 2 * CHUNK * dp * sizeof(float));
        auto dot = resource_->ubBuf.template GetBufferByByte<float>(P_UB + 2 * CHUNK * dp * sizeof(float) + 256);
        auto work = resource_->ubBuf.template GetBufferByByte<float>(P_UB + 2 * CHUNK * dp * sizeof(float) + 512);
        // lse = ln(denom) + float(maxima) for all 64 rows in three vector ops.
        DataCopy(lse, denom_, HALF);
        Sync<HardEvent::MTE2_V>();
        Ln(lse, lse, HALF);
        PipeBarrier<PIPE_V>();
        Cast(dot, maxima_, RoundMode::CAST_NONE, HALF);
        PipeBarrier<PIPE_V>();
        Add(lse, lse, dot, HALF);
        PipeBarrier<PIPE_V>();
        if (params_.smoothK) {
            // Rows beyond validRows are never read from GM nor written back.
            for (uint32_t base = 0; base < HALF; base += CHUNK) {
                const uint32_t rows = Min(CHUNK, validRows - base);
                const uint32_t offset = Offset(params_.qs, b, h, first + base);
                if (params_.d == dp) {
                    Sync<HardEvent::V_MTE2>();
                    DataCopy(raw, qg_[offset], rows * params_.d);
                    Sync<HardEvent::MTE2_V>();
                } else {
                    for (uint32_t r = 0; r < rows; ++r)
                        ReadRow(raw[r * dp], qg_, offset + r * params_.qs[2], params_.d, dp);
                }
                Cast(qf, raw, RoundMode::CAST_NONE, rows * dp);
                PipeBarrier<PIPE_V>();
                for (uint32_t r = 0; r < rows; ++r) {
                    Mul(qf[r * dp], qf[r * dp], mu_, dp);
                    PipeBarrier<PIPE_V>();
                    ReduceSum(dot[base + r], qf[r * dp], work, dp);
                }
                PipeBarrier<PIPE_V>();
            }
            Muls(dot, dot, params_.smScale, HALF);
            PipeBarrier<PIPE_V>();
            Add(lse, lse, dot, HALF);
            PipeBarrier<PIPE_V>();
        }
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams cp{1, validRows * uint32_t(sizeof(float)), 0, 0, 0};
        DataCopyPad(lg_[(uint64_t(b) * params_.hq + h) * params_.sq + first], lse, cp);
        Sync<HardEvent::MTE3_S>();
    }
    static constexpr uint32_t ACCUM_UB = Layout::ACCUM_UB;
    static constexpr uint32_t MU_UB = Layout::MU_UB;
    static constexpr uint32_t P_UB = Layout::P_UB;
    static constexpr uint32_t QS_UB = Layout::QS_UB;
    static constexpr uint32_t MAX_UB = Layout::MAX_UB;
    static constexpr uint32_t DEN_UB = Layout::DEN_UB;
    static constexpr uint32_t KS_UB = Layout::KS_UB;
    static constexpr uint32_t SCORE_UB = Layout::SCORE_UB;
    static constexpr uint32_t ALPHA_UB = Layout::ALPHA_UB;
    static constexpr uint32_t SUM_UB = Layout::SUM_UB;
    static constexpr uint32_t PV_UB = Layout::PV_UB;
    static constexpr uint32_t PV_SLOT_BYTES = Layout::PV_SLOT_BYTES;
    FixedResource *resource_;
    SageParams params_;
    uint32_t subBlockIdx_;
    GlobalTensor<half> qsg_, ksg_;
    GlobalTensor<float> mg_, lg_;
    GlobalTensor<T> qg_, og_;
    LocalTensor<float> accum_, mu_, denom_;
    LocalTensor<half> qsu_, maxima_;
    LocalTensor<T> qo_;
};
} // namespace Sage
