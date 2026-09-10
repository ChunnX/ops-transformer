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
#include "attention_cube_split_qk.h"
#include "attention_block_vec_wide.h"
namespace Sage {
// The ring belongs to the complete per-core stream, across query boundaries.
// A fragment is QK/PV256; two fragments share one softmax/O512 state.
struct WideFragmentInfo {
    uint64_t task, qb, kb;
    uint32_t ordinal, b, h, kh, qstart, count, part;
};
template <bool CAUSAL>
__aicore__ inline __attribute__((always_inline)) WideFragmentInfo DecodeWideQuery(const SageParams &p, uint64_t task,
                                                                                  uint32_t ordinal)
{
    WideFragmentInfo info;
    const uint32_t tiles = p.sq / M;
    info.task = task;
    if (UseDieGroups(p))
        task = DieTask(task, uint64_t(p.b) * p.hq * tiles, p.cores);
    info.ordinal = ordinal;
    info.qstart = (task % tiles) * M;
    info.h = (task / tiles) % p.hq;
    info.b = task / tiles / p.hq;
    info.kh = info.h / (p.hq / p.hk);
    info.qb = (uint64_t(info.b) * p.hq + info.h) * p.qp;
    info.kb = (uint64_t(info.b) * p.hk + info.kh) * p.kp;
    const uint32_t limit = CAUSAL ? Min(p.sk, info.qstart + M) : p.sk;
    info.count = (limit + 511) / 512;
    info.part = 0;
    return info;
}
template <bool CAUSAL>
__aicore__ inline __attribute__((always_inline)) uint64_t WideScaleAhead(const SageParams &p, WideFragmentInfo info,
                                                                         uint32_t ahead, uint64_t totalTasks)
{
    if (info.task >= totalTasks)
        return ~uint64_t(0);
    uint32_t group = info.part / 2 + ahead;
    while (group >= info.count) {
        group -= info.count;
        if (info.task + p.cores >= totalTasks)
            return ~uint64_t(0);
        info = DecodeWideQuery<CAUSAL>(p, info.task + p.cores, info.ordinal + 1);
    }
    return info.kb + group * 512;
}
template <class T, bool CAUSAL>
__aicore__ inline __attribute__((always_inline)) void ProcessWideStreamStep(
    FixedResource &r, AttentionBlockVecWide<T, 128> &vector, WideFragmentInfo *pending, uint64_t fragment, bool hasQk,
    GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR vsmx, MxScaleCache &mxCache, const SageParams &p, uint64_t totalTasks)
{
    constexpr uint32_t DP = 128, WIDE_B_L1 = P_L1 + 4 * M * N;
    if (hasQk) {
        const auto &current = pending[fragment & 3];
        if ASCEND_IS_AIC {
            GemmQkSplit<WIDE_B_L1>(r, q8 + (current.qb + current.qstart) * DP,
                                   k8 + (current.kb + current.part * N) * DP, (fragment & 3) * HALF * N * 2,
                                   current.part == 0);
        } else {
            if (fragment & 1) {
                vector.SelectStreamQuery(current.ordinal, current.qb, current.qstart, current.part == 1);
                const uint64_t nextScale = WideScaleAhead<CAUSAL>(p, current, 2, totalTasks);
                vector.template SoftmaxWide<CAUSAL, true>(current.part / 2, current.kb, current.count, current.qstart,
                                                          fragment / 2, nextScale);
            }
        }
    }
    if (fragment >= 3) {
        uint64_t previous = fragment - 3;
        const auto &old = pending[previous & 3];
        uint32_t group = previous / 2;
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag<4, PIPE_MTE1>(P_READY);
            CrossCoreWaitFlag<4, PIPE_MTE1>(P_READY + AIV_FLAG_OFFSET);
            uint32_t parity = previous & 3;
            uint32_t pvOffset = WideVectorLayout::PV_UB + (group & 1) * WideVectorLayout::PV_SLOT_BYTES;
            if (previous & 1) {
                if (group >= 2) {
                    CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE);
                    CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE + AIV_FLAG_OFFSET);
                }
                Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, true, false>(
                    r, v8, v8 + (old.kb + old.part * N) * DP, N, DP, pvOffset, parity, true,
                    vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N);
                Notify(PV_READY);
            } else {
                Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, false, true>(
                    r, v8, v8 + (old.kb + old.part * N) * DP, N, DP, pvOffset, parity, true,
                    vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N);
            }
        } else {
            if (previous & 1) {
                bool last = old.part + 1 == old.count * 2;
                if (last)
                    vector.PrepareStreamOutput(old.b, old.kh);
                vector.UpdateStreamOutput(group, old.part == 1, last);
                if (last)
                    vector.FinishStreamQuery(old.ordinal, old.qstart, old.b, old.h);
            }
        }
    }
}
template <class T, bool CAUSAL = false>
__aicore__ inline __attribute__((always_inline)) void AttentionWideStream(GM_ADDR q, GM_ADDR q8, GM_ADDR k8, GM_ADDR v8,
                                                                          GM_ADDR qs, GM_ADDR ks, GM_ADDR mean,
                                                                          GM_ADDR mxScale, GM_ADDR out, GM_ADDR lse,
                                                                          SageParams p)
{
    FixedResource r;
    MxScaleCache mxCache;
    AttentionBlockVecWide<T, 128> vector;
    vector.Init(r, q, qs, ks, mean, out, lse, p);
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= 2;
    }
    const uint64_t totalTasks = DieTaskEnd(p, uint64_t(p.b) * p.hq * (p.sq / M), core);
    if (core >= totalTasks)
        return;
    if ASCEND_IS_AIC {
        InitMxP(r);
        SetFlag<HardEvent::MTE1_MTE2>(QK_EVENT);
        SetFlag<HardEvent::MTE1_MTE2>(PV_EVENT);
        SetFlag<HardEvent::M_MTE1>(QK_EVENT);
        SetFlag<HardEvent::M_MTE1>(PV_EVENT);
        SetFlag<HardEvent::FIX_M>(QK_EVENT);
        SetFlag<HardEvent::FIX_M>(QK_SECOND_FIX_EVENT);
        SetFlag<HardEvent::FIX_M>(PV_EVENT);
    } else {
        auto first = DecodeWideQuery<CAUSAL>(p, core, 0);
        vector.BeginStream(first.kb, WideScaleAhead<CAUSAL>(p, first, 1, totalTasks));
    }
    WideFragmentInfo pending[4];
    uint64_t fragment = 0;
    uint32_t ordinal = 0;
    for (uint64_t task = core; task < totalTasks; task += p.cores, ++ordinal) {
        auto current = DecodeWideQuery<CAUSAL>(p, task, ordinal);
        for (uint32_t part = 0; part < current.count * 2; ++part, ++fragment) {
            current.part = part;
            pending[fragment & 3] = current;
            ProcessWideStreamStep<T, CAUSAL>(r, vector, pending, fragment, true, q8, k8, v8, mxScale, mxCache, p,
                                             totalTasks);
        }
    }
    // Drain once at the end of the complete stream, including a one-group core.
    for (uint32_t tail = 0; tail < 3; ++tail)
        ProcessWideStreamStep<T, CAUSAL>(r, vector, pending, fragment + tail, false, q8, k8, v8, mxScale, mxCache, p,
                                         totalTasks);
    if ASCEND_IS_AIC {
        for (uint32_t i = 0; i < Min(uint64_t(2), fragment / 2); ++i) {
            CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE);
            CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE + AIV_FLAG_OFFSET);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(QK_EVENT);
        WaitFlag<HardEvent::MTE1_MTE2>(PV_EVENT);
        WaitFlag<HardEvent::M_MTE1>(QK_EVENT);
        WaitFlag<HardEvent::M_MTE1>(PV_EVENT);
        WaitFlag<HardEvent::FIX_M>(QK_EVENT);
        WaitFlag<HardEvent::FIX_M>(QK_SECOND_FIX_EVENT);
        WaitFlag<HardEvent::FIX_M>(PV_EVENT);
    } else {
        vector.EndStream();
    }
    PipeBarrier<PIPE_ALL>();
}
} // namespace Sage
