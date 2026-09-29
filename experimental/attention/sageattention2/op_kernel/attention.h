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
#include "attention_block_cube.h"
#include "attention_block_vec.h"
#include "attention_wide_stream.h"
namespace Sage {
// Follow the repository's Init/Process kernel + Cube/Vector block organization.
// The shared kernel entry must inline architecture-dependent code for both sides.
template <class T, uint32_t DP, bool CAUSAL = false>
class AttentionKernel {
public:
    __aicore__ inline __attribute__((always_inline)) void Init(GM_ADDR q8, GM_ADDR k8, GM_ADDR v8,
                                                               GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
                                                               GM_ADDR out, GM_ADDR lse, SageParams params)
    {
        static_assert(DP == 64 || DP == 128, "unsupported padded head dimension");
        params_ = params;
        params_.dp = DP;
        q8_ = q8;
        k8_ = k8;
        v8_ = v8;
        vsmx_ = mxScale;
        vector_.Init(resource_, qs, ks, out, lse, params_);
    }
    __aicore__ inline __attribute__((always_inline)) void Process()
    {
        auto &r = resource_;
        MxScaleCache mxCache;
        SageParams p = params_;
        GM_ADDR q8 = q8_;
        GM_ADDR k8 = k8_;
        GM_ADDR v8 = v8_;
        uint32_t core = GetBlockIdx();
        if ASCEND_IS_AIV {
            core /= 2;
        }
        if ASCEND_IS_AIC {
            InitMxP(r);
            SetFlag<HardEvent::MTE1_MTE2>(QK_EVENT);
            SetFlag<HardEvent::MTE1_MTE2>(PV_EVENT);
            SetFlag<HardEvent::M_MTE1>(QK_EVENT);
            SetFlag<HardEvent::M_MTE1>(PV_EVENT);
            SetFlag<HardEvent::FIX_M>(QK_EVENT);
            SetFlag<HardEvent::FIX_M>(PV_EVENT);
        }
        uint32_t tiles = (p.sq + M - 1) / M;
        const uint64_t taskEnd = DieTaskEnd(p, uint64_t(p.b) * p.hq * tiles, core);
        for (uint64_t task = core; task < taskEnd; task += p.cores) {
            const uint64_t mapped = UseDieGroups(p) ? DieTask(task, uint64_t(p.b) * p.hq * tiles, p.cores) : task;
            uint32_t tile = mapped % tiles, h = (mapped / tiles) % p.hq, b = mapped / tiles / p.hq, kh = CalcKh(h, p.hq, p.hk);
            uint32_t qstart = tile * M;
            uint64_t qb = (uint64_t(b) * p.hq + h) * p.qp, kb = (uint64_t(b) * p.hk + kh) * p.kp;
            if ASCEND_IS_AIV {
                vector_.BeginTile(b, kh, qb, qstart);
            }
            uint32_t limit = CAUSAL ? Min(p.sk, qstart + M) : p.sk;
            // Two-block QK lookahead. Drain the final two PV/O updates.
            uint32_t count = (limit + N - 1) / N;
            if ASCEND_IS_AIV {
                vector_.PrefetchScales(kb, count);
            }
            for (uint32_t step = 0; step < count + LOOKAHEAD; ++step) {
                uint32_t start = step * N, parity = step % PIPE_SLOTS;
                uint32_t previous = (step + PIPE_SLOTS - LOOKAHEAD) % PIPE_SLOTS;
                if ASCEND_IS_AIC {
                    if (step < count) {
                        Gemm<int8_t, Catlass::layout::ColumnMajor>(r, q8 + (qb + qstart) * p.dp,
                                                                   k8 + (kb + start) * p.dp, p.dp, N,
                                                                   SCORE_UB + parity * HALF * N * 2, 0, step == 0);
                        Notify(QK_READY);
                    }
                    if (step >= LOOKAHEAD) {
                        WaitBothAiv<PIPE_MTE1>(P_READY);
                        // Before reusing slot i % PV_SLOTS, wait for Vec2(i-PV_SLOTS).
                        if (step >= LOOKAHEAD + PV_SLOTS) {
                            WaitBothAiv<PIPE_FIX>(PV_RELEASE);
                        }
                        Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true>(
                            r, v8, v8 + (kb + (step - LOOKAHEAD) * N) * p.dp, N, p.dp,
                            PV_UB + ((step - LOOKAHEAD) % PV_SLOTS) * PV_SLOT_BYTES, previous, true,
                            vsmx_ + (kb + (step - LOOKAHEAD) * N) / 32 * p.dp, &mxCache, p.kp, (step - LOOKAHEAD) * N);
                        Notify(PV_READY);
                    }
                } else {
                    if (step < count) {
                        vector_.template Softmax<CAUSAL>(step, qstart, kb, count);
                    }
                    if (step >= LOOKAHEAD) {
                        vector_.UpdateOutput(step - LOOKAHEAD, count);
                    }
                }
            }
            if ASCEND_IS_AIC {
                // Consume the outstanding release tokens before the next query tile.
                for (uint32_t drain = 0; drain < Min(count, PV_SLOTS); ++drain) {
                    WaitBothAiv<PIPE_FIX>(PV_RELEASE);
                }
            }
            if ASCEND_IS_AIV {
                vector_.FinalizeTile(qstart, b, h);
            }
        }
        if ASCEND_IS_AIC {
            WaitFlag<HardEvent::MTE1_MTE2>(QK_EVENT);
            WaitFlag<HardEvent::MTE1_MTE2>(PV_EVENT);
            WaitFlag<HardEvent::M_MTE1>(QK_EVENT);
            WaitFlag<HardEvent::M_MTE1>(PV_EVENT);
            WaitFlag<HardEvent::FIX_M>(QK_EVENT);
            WaitFlag<HardEvent::FIX_M>(PV_EVENT);
        }
        PipeBarrier<PIPE_ALL>();
    }

private:
    FixedResource resource_;
    AttentionBlockVec<T, DP> vector_;
    SageParams params_;
    GM_ADDR q8_;
    GM_ADDR k8_;
    GM_ADDR v8_;
    GM_ADDR vsmx_;
};
template <class T, uint32_t DP, bool CAUSAL = false>
__aicore__ inline __attribute__((always_inline)) void AttentionImpl(GM_ADDR q8, GM_ADDR k8, GM_ADDR v8,
                                                                    GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
                                                                    GM_ADDR out, GM_ADDR lse, SageParams params)
{
    AttentionKernel<T, DP, CAUSAL> kernel;
    kernel.Init(q8, k8, v8, qs, ks, mxScale, out, lse, params);
    kernel.Process();
}
template <class T, bool WIDE>
__aicore__ inline __attribute__((always_inline)) void Attention(GM_ADDR q8, GM_ADDR k8, GM_ADDR v8,
                                                                GM_ADDR qs, GM_ADDR ks, GM_ADDR mxScale,
                                                                GM_ADDR out, GM_ADDR lse, SageParams p)
{
    // Joint N512 uses NZ softmax on aligned prefill, including the causal frontier.
    if constexpr (WIDE) {
        if (p.causal)
            AttentionWideStream<T, true>(q8, k8, v8, qs, ks, mxScale, out, lse, p);
        else
            AttentionWideStream<T, false>(q8, k8, v8, qs, ks, mxScale, out, lse, p);
    } else {
        if (p.causal) {
            if (p.dp == 128)
                AttentionImpl<T, 128, true>(q8, k8, v8, qs, ks, mxScale, out, lse, p);
            else
                AttentionImpl<T, 64, true>(q8, k8, v8, qs, ks, mxScale, out, lse, p);
        } else {
            if (p.dp == 128)
                AttentionImpl<T, 128, false>(q8, k8, v8, qs, ks, mxScale, out, lse, p);
            else
                AttentionImpl<T, 64, false>(q8, k8, v8, qs, ks, mxScale, out, lse, p);
        }
    }
}
} // namespace Sage
