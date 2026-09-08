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
namespace Sage {
// Per partition: K sum, V sum, V minimum, V maximum. The extrema determine
// max(abs(V-meanV)) after the mean is known, without a second read of V.
template <bool VALUE>
__simd_vf__ inline void AccumulateSmoothStatsVF(__ubuf__ float *rows, __ubuf__ float *state, uint32_t count,
                                                uint32_t dp)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t col = 0; col < dp; col += 64) {
        RegTensor<float> sum, low, high, x;
        LoadAlign(sum, state + (VALUE ? dp : 0) + col);
        if constexpr (VALUE) {
            LoadAlign(low, state + 2 * dp + col);
            LoadAlign(high, state + 3 * dp + col);
        }
        for (uint16_t row = 0; row < count; ++row) {
            LoadAlign(x, rows + row * dp + col);
            Add(sum, sum, x, f);
            if constexpr (VALUE) {
                Min(low, low, x, f);
                Max(high, high, x, f);
            }
        }
        StoreAlign(state + (VALUE ? dp : 0) + col, sum, f);
        if constexpr (VALUE) {
            StoreAlign(state + 2 * dp + col, low, f);
            StoreAlign(state + 3 * dp + col, high, f);
        }
    }
}
__simd_vf__ inline void FinalizeSmoothStatsVF(__ubuf__ float *partials, __ubuf__ float *state, uint32_t dp,
                                              float reciprocalLength)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> divisor;
    Duplicate(divisor, 448.0f);
    for (uint16_t col = 0; col < dp; col += 64) {
        RegTensor<float> ksum, vsum, low, high, x;
        Duplicate(ksum, 0.0f);
        Duplicate(vsum, 0.0f);
        Duplicate(low, __builtin_inff());
        Duplicate(high, -__builtin_inff());
        for (uint16_t part = 0; part < SAGE_STATS_SPLITS; ++part) {
            LoadAlign(x, partials + part * 4 * dp + col);
            Add(ksum, ksum, x, f);
            LoadAlign(x, partials + (part * 4 + 1) * dp + col);
            Add(vsum, vsum, x, f);
            LoadAlign(x, partials + (part * 4 + 2) * dp + col);
            Min(low, low, x, f);
            LoadAlign(x, partials + (part * 4 + 3) * dp + col);
            Max(high, high, x, f);
        }
        Muls(ksum, ksum, reciprocalLength, f);
        Muls(vsum, vsum, reciprocalLength, f);
        StoreAlign(state + col, ksum, f);
        StoreAlign(state + dp + col, vsum, f);
        Sub(low, low, vsum, f);
        Sub(high, high, vsum, f);
        Abs(low, low, f);
        Abs(high, high, f);
        Max(high, high, low, f);
        Div(high, high, divisor, f);
        Maxs(high, high, 1.0e-30f, f);
        StoreAlign(state + 2 * dp + col, high, f);
    }
}
template <class T>
__aicore__ void SmoothValueStats(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR scale, GM_ADDR partials, SageParams p)
{
    FixedResource r;
    GlobalTensor<T> kg, vg;
    GlobalTensor<float> work, mg, sg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    work.SetGlobalBuffer((__gm__ float *)partials);
    mg.SetGlobalBuffer((__gm__ float *)mean);
    sg.SetGlobalBuffer((__gm__ float *)scale);
    auto input = r.ubBuf.template GetBufferByByte<T>(0);
    auto x = r.ubBuf.template GetBufferByByte<float>(65536);
    auto state = r.ubBuf.template GetBufferByByte<float>(196608);
    const uint64_t heads = uint64_t(p.b) * p.hk;
    const uint32_t chunk = (p.sk + SAGE_STATS_SPLITS - 1) / SAGE_STATS_SPLITS;
    const bool contiguous = p.d == p.dp && p.ks[2] == p.dp && p.vs[2] == p.dp;
    for (uint64_t task = GetBlockIdx(); task < heads * SAGE_STATS_SPLITS; task += (2 * p.cores)) {
        uint32_t bh = task / SAGE_STATS_SPLITS, part = task % SAGE_STATS_SPLITS;
        uint32_t b = bh / p.hk, h = bh % p.hk;
        uint32_t end = Min(p.sk, (part + 1) * chunk);
        Duplicate(state, 0.0f, 2 * p.dp);
        Duplicate(state[2 * p.dp], __builtin_inff(), p.dp);
        Duplicate(state[3 * p.dp], -__builtin_inff(), p.dp);
        for (uint32_t start = part * chunk; start < end;) {
            uint32_t rows = contiguous ? Min(uint32_t(256), end - start) : 1;
            for (uint32_t value = p.smoothK ? 0 : 1; value < 2; ++value) {
                Sync<HardEvent::V_MTE2>();
                if (contiguous)
                    DataCopy(input, (value ? vg : kg)[Offset(value ? p.vs : p.ks, b, h, start)], rows * p.dp);
                else
                    ReadRow(input, value ? vg : kg, Offset(value ? p.vs : p.ks, b, h, start), p.d, p.dp);
                Sync<HardEvent::MTE2_V>();
                Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
                PipeBarrier<PIPE_V>();
                if (value)
                    AccumulateSmoothStatsVF<true>((__ubuf__ float *)x.GetPhyAddr(),
                                                  (__ubuf__ float *)state.GetPhyAddr(), rows, p.dp);
                else
                    AccumulateSmoothStatsVF<false>((__ubuf__ float *)x.GetPhyAddr(),
                                                   (__ubuf__ float *)state.GetPhyAddr(), rows, p.dp);
            }
            start += rows;
        }
        WriteRow(work, task * 4 * p.dp, state, 4 * p.dp);
    }
    SyncAll<true>();
    for (uint64_t bh = GetBlockIdx(); bh < heads; bh += (2 * p.cores)) {
        Sync<HardEvent::V_MTE2>();
        DataCopy(x, work[bh * SAGE_STATS_SPLITS * 4 * p.dp], SAGE_STATS_SPLITS * 4 * p.dp);
        Sync<HardEvent::MTE2_V>();
        FinalizeSmoothStatsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)state.GetPhyAddr(), p.dp,
                              1.0f / p.sk);
        WriteRow(mg, bh * p.dp, state, p.dp);
        WriteRow(mg, (heads + bh) * p.dp, state[p.dp], p.dp);
        WriteRow(sg, bh * p.dp, state[2 * p.dp], p.dp);
    }
}
} // namespace Sage
