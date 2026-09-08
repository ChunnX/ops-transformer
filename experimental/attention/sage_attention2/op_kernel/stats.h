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
#include "common.h"
#include "stats_smooth_v.h"
namespace Sage {
// Accumulate a sequence batch; preserve per-channel FP32 summation order.
// Contiguous Stats uses 256 rows, with input/FP32/state buffers below 248 KiB UB.
template <uint32_t DP, bool MAXIMUM>
__simd_vf__ inline void StatsAccumulateImpl(__ubuf__ float *rows, __ubuf__ float *state, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> acc, x;
        LoadAlign(acc, state + col);
        for (uint32_t row = 0; row < count; ++row) {
            LoadAlign(x, rows + row * DP + col);
            if constexpr (MAXIMUM) {
                Abs(x, x, mask);
                Max(acc, acc, x, mask);
            } else {
                Add(acc, acc, x, mask);
            }
        }
        StoreAlign(state + col, acc, mask);
    }
}
__aicore__ inline void StatsAccumulateVF(__ubuf__ float *rows, __ubuf__ float *state, uint32_t count, uint32_t dp,
                                         uint32_t maximum)
{
    if (dp == 128) {
        if (maximum)
            StatsAccumulateImpl<128, true>(rows, state, count);
        else
            StatsAccumulateImpl<128, false>(rows, state, count);
    } else {
        if (maximum)
            StatsAccumulateImpl<64, true>(rows, state, count);
        else
            StatsAccumulateImpl<64, false>(rows, state, count);
    }
}
// One barrier separates sequence-partial writes from per-head reduction.
template <class T>
__aicore__ void SplitStats(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR scale, GM_ADDR partials, SageParams p)
{
    Catlass::Arch::Resource<Arch> r;
    GlobalTensor<T> kg, vg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    GlobalTensor<float> work, mg, sg;
    work.SetGlobalBuffer((__gm__ float *)partials);
    mg.SetGlobalBuffer((__gm__ float *)mean);
    sg.SetGlobalBuffer((__gm__ float *)scale);
    auto input = r.ubBuf.template GetBufferByByte<T>(0);
    auto x = r.ubBuf.template GetBufferByByte<float>(65536);
    auto sum = r.ubBuf.template GetBufferByByte<float>(196608);
    auto mx = r.ubBuf.template GetBufferByByte<float>(197120);
    auto divisor = r.ubBuf.template GetBufferByByte<float>(197632);
    const uint32_t chunk = (p.sk + SAGE_STATS_SPLITS - 1) / SAGE_STATS_SPLITS;
    for (uint64_t task = GetBlockIdx(); task < uint64_t(p.b) * p.hk * SAGE_STATS_SPLITS; task += (2 * p.cores)) {
        uint32_t bh = task / SAGE_STATS_SPLITS, part = task % SAGE_STATS_SPLITS;
        uint32_t b = bh / p.hk, h = bh % p.hk;
        uint32_t end = Min(p.sk, (part + 1) * chunk);
        Duplicate(sum, 0.0f, p.dp);
        Duplicate(mx, 0.0f, p.dp);
        for (uint32_t start = part * chunk; start < end; start += 256) {
            uint32_t rows = Min(uint32_t(256), end - start);
            if (p.smoothK) {
                Sync<HardEvent::V_MTE2>();
                DataCopy(input, kg[Offset(p.ks, b, h, start)], rows * p.dp);
                Sync<HardEvent::MTE2_V>();
                Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
                PipeBarrier<PIPE_V>();
                StatsAccumulateVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)sum.GetPhyAddr(), rows, p.dp, 0);
            }
            Sync<HardEvent::V_MTE2>();
            DataCopy(input, vg[Offset(p.vs, b, h, start)], rows * p.dp);
            Sync<HardEvent::MTE2_V>();
            Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
            PipeBarrier<PIPE_V>();
            StatsAccumulateVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)mx.GetPhyAddr(), rows, p.dp, 1);
        }
        WriteRow(work, task * 2 * p.dp, sum, p.dp);
        WriteRow(work, (task * 2 + 1) * p.dp, mx, p.dp);
    }
    SyncAll<true>();
    for (uint64_t bh = GetBlockIdx(); bh < uint64_t(p.b) * p.hk; bh += (2 * p.cores)) {
        Sync<HardEvent::V_MTE2>();
        DataCopy(x, work[bh * SAGE_STATS_SPLITS * 2 * p.dp], SAGE_STATS_SPLITS * 2 * p.dp);
        Sync<HardEvent::MTE2_V>();
        Duplicate(sum, 0.0f, p.dp);
        Duplicate(mx, 0.0f, p.dp);
        for (uint32_t part = 0; part < SAGE_STATS_SPLITS; ++part) {
            PipeBarrier<PIPE_V>();
            Add(sum, sum, x[part * 2 * p.dp], p.dp);
            Max(mx, mx, x[(part * 2 + 1) * p.dp], p.dp);
        }
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, 1.0f / p.sk, p.dp);
        Duplicate(divisor, 448.0f, p.dp);
        PipeBarrier<PIPE_V>();
        Div(mx, mx, divisor, p.dp);
        PipeBarrier<PIPE_V>();
        Maxs(mx, mx, 1.0e-30f, p.dp);
        PipeBarrier<PIPE_V>();
        WriteRow(mg, bh * p.dp, sum, p.dp);
        WriteRow(sg, bh * p.dp, mx, p.dp);
    }
}

template <class T>
__aicore__ void Stats(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR scale, SageParams p, GM_ADDR partials = nullptr)
{
    if (p.smoothV) {
        SmoothValueStats<T>(k, v, mean, scale, partials, p);
        return;
    }
    if (partials && p.d == p.dp && p.ks[2] == p.dp && p.vs[2] == p.dp && p.sk >= 1024) {
        SplitStats<T>(k, v, mean, scale, partials, p);
        return;
    }
    Catlass::Arch::Resource<Arch> r;
    GlobalTensor<T> kg, vg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    GlobalTensor<float> mg, sg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    sg.SetGlobalBuffer((__gm__ float *)scale);
    auto input = r.ubBuf.GetBufferByByte<T>(0);
    auto x = r.ubBuf.GetBufferByByte<float>(65536);
    auto sum = r.ubBuf.GetBufferByByte<float>(196608);
    auto mx = r.ubBuf.GetBufferByByte<float>(197120);
    auto ab = r.ubBuf.GetBufferByByte<float>(197632);
    for (uint32_t bh = GetBlockIdx(); bh < p.b * p.hk; bh += (2 * p.cores)) {
        uint32_t b = bh / p.hk, h = bh % p.hk;
        Duplicate(sum, 0.0f, p.dp);
        Duplicate(mx, 0.0f, p.dp);
        if (p.d == p.dp && p.ks[2] == p.dp && p.vs[2] == p.dp) {
            for (uint32_t start = 0; start < p.sk; start += 256) {
                uint32_t rows = Min(uint32_t(256), p.sk - start);
                if (p.smoothK) {
                    Sync<HardEvent::V_MTE2>();
                    DataCopy(input, kg[Offset(p.ks, b, h, start)], rows * p.dp);
                    Sync<HardEvent::MTE2_V>();
                    Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
                    PipeBarrier<PIPE_V>();
                    StatsAccumulateVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)sum.GetPhyAddr(), rows, p.dp,
                                      0);
                }
                Sync<HardEvent::V_MTE2>();
                DataCopy(input, vg[Offset(p.vs, b, h, start)], rows * p.dp);
                Sync<HardEvent::MTE2_V>();
                Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
                PipeBarrier<PIPE_V>();
                StatsAccumulateVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)mx.GetPhyAddr(), rows, p.dp, 1);
            }
            PipeBarrier<PIPE_V>();
        } else
            for (uint32_t s = 0; s < p.sk; ++s) {
                if (p.smoothK) {
                    ReadRow(input, kg, Offset(p.ks, b, h, s), p.d, p.dp);
                    Cast(x, input, RoundMode::CAST_NONE, p.dp);
                    PipeBarrier<PIPE_V>();
                    Add(sum, sum, x, p.dp);
                    PipeBarrier<PIPE_V>();
                }
                ReadRow(input, vg, Offset(p.vs, b, h, s), p.d, p.dp);
                Cast(x, input, RoundMode::CAST_NONE, p.dp);
                PipeBarrier<PIPE_V>();
                Abs(ab, x, p.dp);
                PipeBarrier<PIPE_V>();
                Max(mx, mx, ab, p.dp);
                PipeBarrier<PIPE_V>();
            }
        Muls(sum, sum, 1.0f / p.sk, p.dp);
        Duplicate(ab, 448.0f, p.dp);
        PipeBarrier<PIPE_V>();
        Div(mx, mx, ab, p.dp);
        PipeBarrier<PIPE_V>();
        // Zero V channels use a harmless positive scale; zero stays exactly zero.
        Maxs(mx, mx, 1.0e-30f, p.dp);
        PipeBarrier<PIPE_V>();
        WriteRow(mg, uint64_t(bh) * p.dp, sum, p.dp);
        WriteRow(sg, uint64_t(bh) * p.dp, mx, p.dp);
    }
}
} // namespace Sage
