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
// Query metadata is decoded once; Cube uses N256 fragments, Vector uses N512 groups.
struct WideQueryInfo {
    uint64_t task, qb, kb;
    uint32_t ordinal, b, h, kh, qstart, count;
};
template <bool CAUSAL>
__aicore__ inline __attribute__((always_inline)) WideQueryInfo DecodeWideQuery(const SageParams &p, uint64_t task,
                                                                                  uint32_t ordinal)
{
    WideQueryInfo info;
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
    return info;
}
template <bool CAUSAL>
__aicore__ inline __attribute__((always_inline)) uint64_t WideScaleAhead(const SageParams &p, WideQueryInfo info,
                                                                         uint32_t group, uint64_t totalTasks)
{
    if (info.task >= totalTasks)
        return ~uint64_t(0);
    while (group >= info.count) {
        group -= info.count;
        if (info.task + p.cores >= totalTasks)
            return ~uint64_t(0);
        info = DecodeWideQuery<CAUSAL>(p, info.task + p.cores, info.ordinal + 1);
    }
    return info.kb + group * 512;
}
// Only delayed O needs a record. O(g-2) retires before S(g) reuses its slot,
// including query boundaries, so two output records cover the complete stream.
struct WideOutputInfo {
    uint32_t ordinal, b, h, kh, qstart;
    bool first, last;
};
template <class T>
__aicore__ inline __attribute__((always_inline)) void FinishWideOutput(
    AttentionBlockVecWide<T, 128> &vector, const WideOutputInfo &info, uint64_t group)
{
    if (info.last)
        vector.LoadLseMean(info.b, info.kh);
    vector.UpdateStreamOutput(group, info.first, info.last);
    if (info.last)
        vector.FinishStreamQuery(info.ordinal, info.qstart, info.b, info.h);
}
// Cube retains only the fields consumed by delayed PV and V prefetch.
struct CubeFragmentInfo {
    uint64_t kb;
    uint32_t part;
};
template <uint32_t PHASE>
__aicore__ inline __attribute__((always_inline)) void ProcessWideCubeStep(
    FixedResource &r, CubeFragmentInfo *pending, uint64_t fragment, bool hasQk, GM_ADDR q,
    GM_ADDR k8, GM_ADDR v8, GM_ADDR vsmx, MxScaleCache &mxCache, const SageParams &p,
    uint32_t count, bool prefetchPv = true)
{
    constexpr uint32_t DP = 128, WIDE_B_L1 = P_L1 + 4 * M * N;
    if (hasQk) {
        const auto &current = pending[fragment & 3];
        GemmQkSplit<WIDE_B_L1>(r, q, (fragment & 3) * HALF * N * 2,
                               current.part == 0, PHASE);
    }
    if (fragment >= 3) {
        uint64_t previous = fragment - 3;
        const auto &old = pending[previous & 3];
        uint32_t group = previous / 2;
        CrossCoreWaitFlag<4, PIPE_MTE1>(P_READY);
        CrossCoreWaitFlag<4, PIPE_MTE1>(P_READY + AIV_FLAG_OFFSET);
        uint32_t parity = previous & 3;
        uint32_t pvOffset = WideVectorLayout::PV_UB + (group & 1) * WideVectorLayout::PV_SLOT_BYTES;
        if ((1 - PHASE)) {
            if (group >= 2) {
                CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE);
                CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE + AIV_FLAG_OFFSET);
            }
            Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, true, false, true>(
                r, v8, nullptr, N, DP, pvOffset, parity, true,
                vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N, (1 - PHASE));
            Notify(PV_READY);
        } else {
            Gemm<float8_e4m3_t, Catlass::layout::RowMajor, true, WIDE_B_L1, false, true, true>(
                r, v8, nullptr, N, DP, pvOffset, parity, true,
                vsmx + (old.kb + old.part * N) / 32 * DP, &mxCache, p.kp, old.part * N, (1 - PHASE));
        }
    }
    if (hasQk) {
        const auto &current = pending[fragment & 3];
        if (current.part + 2 < count * 2)
            PrefetchWideK<WIDE_B_L1>(r, k8 + (current.kb + (current.part + 2) * N) * DP, PHASE);
    }
    if (fragment >= 2 && prefetchPv) {
        const auto &next = pending[(fragment - 2) & 3];
        PrefetchWideV<WIDE_B_L1>(r, v8 + (next.kb + next.part * N) * DP, PHASE);
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
        SetFlag<HardEvent::MTE1_MTE2>(4);
        SetFlag<HardEvent::MTE1_MTE2>(5);
        SetFlag<HardEvent::MTE1_MTE2>(6);
        SetFlag<HardEvent::MTE1_MTE2>(7);
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
    WideOutputInfo outputPending[2];
    static_assert(sizeof(WideOutputInfo) == 24, "Vector ring stores only delayed output metadata");
    CubeFragmentInfo cubePending[4];
    static_assert(sizeof(CubeFragmentInfo) == 16, "Cube ring stores only K/V base and fragment index");
    uint64_t fragment = 0;
    uint32_t ordinal = 0;
    for (uint64_t task = core; task < totalTasks; task += p.cores, ++ordinal) {
        auto current = DecodeWideQuery<CAUSAL>(p, task, ordinal);
        if ASCEND_IS_AIC {
            constexpr uint32_t K_L1 = P_L1 + 4 * M * N;
            PrefetchWideK<K_L1>(r, k8 + current.kb * 128, 0);
            PrefetchWideK<K_L1>(r, k8 + (current.kb + N) * 128, 1);
        }
        GM_ADDR qTile = q8 + (current.qb + current.qstart) * 128;
        if ASCEND_IS_AIC {
            for (uint32_t part = 0; part < current.count * 2; part += 2, fragment += 2) {
                cubePending[fragment & 3] = {current.kb, part};
                ProcessWideCubeStep<0>(r, cubePending, fragment, true, qTile, k8, v8, mxScale, mxCache, p, current.count);
                cubePending[(fragment + 1) & 3] = {current.kb, part + 1};
                ProcessWideCubeStep<1>(r, cubePending, fragment + 1, true, qTile, k8, v8, mxScale, mxCache, p, current.count);
            }
        } else {
            for (uint32_t step = 0; step < current.count; ++step, fragment += 2) {
                const uint64_t group = fragment / 2;
                if (group >= 2)
                    FinishWideOutput(vector, outputPending[group & 1], group - 2);
                const bool first = step == 0, last = step + 1 == current.count;
                vector.SelectStreamQuery(current.ordinal, current.qb, current.qstart, first);
                const uint64_t nextScale = WideScaleAhead<CAUSAL>(p, current, step + 2, totalTasks);
                vector.template SoftmaxWide<CAUSAL, true>(step, current.kb, current.count, current.qstart,
                                                          group, nextScale);
                outputPending[group & 1] = {current.ordinal, current.b, current.h, current.kh,
                                            current.qstart, first, last};
            }
        }
    }
    if ASCEND_IS_AIC {
        ProcessWideCubeStep<0>(r, cubePending, fragment, false, nullptr, k8, v8, mxScale, mxCache, p, 0);
        ProcessWideCubeStep<1>(r, cubePending, fragment + 1, false, nullptr, k8, v8, mxScale, mxCache, p, 0);
        ProcessWideCubeStep<0>(r, cubePending, fragment + 2, false, nullptr, k8, v8, mxScale, mxCache, p, 0, false);
    } else {
        const uint64_t groups = fragment / 2;
        for (uint64_t group = groups - Min(uint64_t(2), groups); group < groups; ++group)
            FinishWideOutput(vector, outputPending[group & 1], group);
    }
    if ASCEND_IS_AIC {
        for (uint32_t i = 0; i < Min(uint64_t(2), fragment / 2); ++i) {
            CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE);
            CrossCoreWaitFlag<4, PIPE_FIX>(PV_RELEASE + AIV_FLAG_OFFSET);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(4);
        WaitFlag<HardEvent::MTE1_MTE2>(5);
        WaitFlag<HardEvent::MTE1_MTE2>(6);
        WaitFlag<HardEvent::MTE1_MTE2>(7);
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
