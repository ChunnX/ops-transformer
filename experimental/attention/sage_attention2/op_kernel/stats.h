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
// Preserve per-channel FP32 summation order when accumulating K.
template <uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ float *rows, __ubuf__ float *state, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> acc, x;
        LoadAlign(acc, state + col);
        for (uint32_t row = 0; row < count; ++row) {
            LoadAlign(x, rows + row * DP + col);
            Add(acc, acc, x, mask);
        }
        StoreAlign(state + col, acc, mask);
    }
}
__aicore__ inline void StatsSumVF(__ubuf__ float *rows, __ubuf__ float *state, uint32_t count, uint32_t dp)
{
    if (dp == 128)
        StatsSumImpl<128>(rows, state, count);
    else
        StatsSumImpl<64>(rows, state, count);
}
// Compare input magnitudes in 16 bits; only the final channel maxima need FP32.
template <class T, uint32_t DP>
__simd_vf__ inline void StatsMax16Impl(__ubuf__ T *rows, __ubuf__ T *state, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<T, DP == 128 ? MaskPattern::ALL : MaskPattern::VL64>();
    RegTensor<T> acc, x;
    RegTensor<uint16_t> magnitudeMask;
    Duplicate(magnitudeMask, uint16_t(0x7fff));
    LoadAlign(acc, state);
    for (uint32_t row = 0; row < count; ++row) {
        LoadAlign(x, rows + row * DP);
        // Clearing the sign bit is exact for both FP16 and BF16.
        And((RegTensor<uint16_t> &)x, (RegTensor<uint16_t> &)x, magnitudeMask, mask);
        Max(acc, acc, x, mask);
    }
    StoreAlign(state, acc, mask);
}
template <class T>
__aicore__ inline void StatsMax16VF(__ubuf__ T *rows, __ubuf__ T *state, uint32_t count, uint32_t dp)
{
    if (dp == 128)
        StatsMax16Impl<T, 128>(rows, state, count);
    else
        StatsMax16Impl<T, 64>(rows, state, count);
}
// One barrier separates sequence-partial writes from per-head reduction.
template <class T>
__aicore__ void SplitStats(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR scale, GM_ADDR partials, SageParams p)
{
    FixedResource r;
    GlobalTensor<T> kg, vg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    GlobalTensor<float> work, mg, sg;
    work.SetGlobalBuffer((__gm__ float *)partials);
    mg.SetGlobalBuffer((__gm__ float *)mean);
    sg.SetGlobalBuffer((__gm__ float *)scale);
    // Two 128-row K/V input slots, one FP32 conversion buffer, and state.
    constexpr uint32_t BATCH_ROWS = 128;
    constexpr uint32_t SLOT_BYTES = 65536, V_INPUT = 32768, FLOAT_INPUT = 2 * SLOT_BYTES;
    constexpr uint32_t SUM = 196608, MAXIMUM = 197120, DIVISOR = 197632, MAXIMUM_16 = 198144;
    static_assert(MAXIMUM_16 + 128 * sizeof(T) <= 248 * 1024, "stats pipeline exceeds UB");
    auto x = r.ubBuf.template GetBufferByByte<float>(FLOAT_INPUT);
    auto sum = r.ubBuf.template GetBufferByByte<float>(SUM);
    auto mx = r.ubBuf.template GetBufferByByte<float>(MAXIMUM);
    auto max16 = r.ubBuf.template GetBufferByByte<T>(MAXIMUM_16);
    auto divisor = r.ubBuf.template GetBufferByByte<float>(DIVISOR);
    const uint32_t chunk = (p.sk + SAGE_STATS_SPLITS - 1) / SAGE_STATS_SPLITS;
    for (uint64_t task = GetBlockIdx(); task < uint64_t(p.b) * p.hk * SAGE_STATS_SPLITS; task += (2 * p.cores)) {
        uint32_t bh = task / SAGE_STATS_SPLITS, part = task % SAGE_STATS_SPLITS;
        uint32_t b = bh / p.hk, h = bh % p.hk;
        uint32_t end = Min(p.sk, (part + 1) * chunk);
        Duplicate(sum, 0.0f, p.dp);
        Duplicate(max16, T(0), p.dp);
        // Two 128-row K/V slots fit beside the FP32 conversion buffer.
        // Enqueue the next pair before consuming the current pair.
        const uint32_t begin = part * chunk;
        const uint32_t batches = (end - begin + BATCH_ROWS - 1) / BATCH_ROWS;
        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);
        for (uint32_t batch = 0; batch <= batches; ++batch) {
            if (batch < batches) {
                const uint32_t slot = batch & 1, start = begin + batch * BATCH_ROWS;
                const uint32_t rows = Min(BATCH_ROWS, end - start);
                const event_t event = static_cast<event_t>(1 + slot);
                WaitFlag<HardEvent::V_MTE2>(event);
                if (p.smoothK)
                    DataCopy(r.ubBuf.template GetBufferByByte<T>(slot * SLOT_BYTES),
                             kg[Offset(p.ks, b, h, start)], rows * p.dp);
                DataCopy(r.ubBuf.template GetBufferByByte<T>(slot * SLOT_BYTES + V_INPUT),
                         vg[Offset(p.vs, b, h, start)], rows * p.dp);
                SetFlag<HardEvent::MTE2_V>(event);
            }
            if (batch == 0)
                continue;
            const uint32_t slot = (batch - 1) & 1;
            const uint32_t rows = Min(BATCH_ROWS, end - begin - (batch - 1) * BATCH_ROWS);
            const event_t event = static_cast<event_t>(1 + slot);
            WaitFlag<HardEvent::MTE2_V>(event);
            if (p.smoothK) {
                Cast(x, r.ubBuf.template GetBufferByByte<T>(slot * SLOT_BYTES), RoundMode::CAST_NONE, rows * p.dp);
                PipeBarrier<PIPE_V>();
                StatsSumVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)sum.GetPhyAddr(), rows, p.dp);
            }
            StatsMax16VF((__ubuf__ T *)r.ubBuf.template GetBufferByByte<T>(slot * SLOT_BYTES + V_INPUT).GetPhyAddr(),
                         (__ubuf__ T *)max16.GetPhyAddr(), rows, p.dp);
            SetFlag<HardEvent::V_MTE2>(event);
        }
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);
        Cast(mx, max16, RoundMode::CAST_NONE, p.dp);
        PipeBarrier<PIPE_V>();
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
    FixedResource r;
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
    auto max16 = r.ubBuf.GetBufferByByte<T>(198144);
    auto ab = r.ubBuf.GetBufferByByte<float>(197632);
    for (uint32_t bh = GetBlockIdx(); bh < p.b * p.hk; bh += (2 * p.cores)) {
        uint32_t b = bh / p.hk, h = bh % p.hk;
        Duplicate(sum, 0.0f, p.dp);
        Duplicate(max16, T(0), p.dp);
        if (p.d == p.dp && p.ks[2] == p.dp && p.vs[2] == p.dp) {
            for (uint32_t start = 0; start < p.sk; start += 256) {
                uint32_t rows = Min(uint32_t(256), p.sk - start);
                if (p.smoothK) {
                    Sync<HardEvent::V_MTE2>();
                    DataCopy(input, kg[Offset(p.ks, b, h, start)], rows * p.dp);
                    Sync<HardEvent::MTE2_V>();
                    Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
                    PipeBarrier<PIPE_V>();
                    StatsSumVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)sum.GetPhyAddr(), rows, p.dp);
                }
                Sync<HardEvent::V_MTE2>();
                DataCopy(input, vg[Offset(p.vs, b, h, start)], rows * p.dp);
                Sync<HardEvent::MTE2_V>();
                StatsMax16VF((__ubuf__ T *)input.GetPhyAddr(), (__ubuf__ T *)max16.GetPhyAddr(), rows, p.dp);
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
                StatsMax16VF((__ubuf__ T *)input.GetPhyAddr(), (__ubuf__ T *)max16.GetPhyAddr(), 1, p.dp);
            }
        Cast(mx, max16, RoundMode::CAST_NONE, p.dp);
        PipeBarrier<PIPE_V>();
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
