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
// Shared stats UB layout: raw 16-bit rows, FP32 rows, then the column sums.
constexpr uint32_t STATS_RAW_UB = 0, STATS_F32_UB = 65536, STATS_SUM_UB = 131072;
// Load one 64-lane column slice of a 16-bit row as FP32 registers, unpacking
// and casting in the same sequence.
template <class T, uint32_t DP>
__simd_callee__ inline void LoadLaneF32(AscendC::MicroAPI::RegTensor<float> &x, __ubuf__ T *addr, uint32_t row,
                                        uint32_t col)
{
    using namespace AscendC::MicroAPI;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<T> raw;
    static constexpr CastTrait cast{
        RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, addr + row * DP + col);
    Cast<float, T, cast>(x, raw, mask);
}
// Preserve per-channel FP32 addition order. Raw 16-bit inputs are converted in
// registers.
// Optimized with 4-way independent accumulator unrolling and interleaved load/cast/add
// to completely saturate Vector ALU pipeline throughput and eliminate pipeline bubbles.
template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> acc0, acc1, acc2, acc3;
        LoadAlign(acc0, state + col);
        Duplicate(acc1, 0.0f);
        Duplicate(acc2, 0.0f);
        Duplicate(acc3, 0.0f);
        uint32_t row = 0;
        // 4-row unrolled accumulator pipeline to maximize Vector ALU throughput
        for (; row + 3 < count; row += 4) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<T> raw0, raw1, raw2, raw3;
            static constexpr CastTrait cast{
                RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw0, rows + (row + 0) * DP + col);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw1, rows + (row + 1) * DP + col);
            Cast<float, T, cast>(x0, raw0, mask);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw2, rows + (row + 2) * DP + col);
            Cast<float, T, cast>(x1, raw1, mask);
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw3, rows + (row + 3) * DP + col);
            Cast<float, T, cast>(x2, raw2, mask);
            Add(acc0, acc0, x0, mask);
            Cast<float, T, cast>(x3, raw3, mask);
            Add(acc1, acc1, x1, mask);
            Add(acc2, acc2, x2, mask);
            Add(acc3, acc3, x3, mask);
        }
        // 2-row cleanup loop
        for (; row + 1 < count; row += 2) {
            RegTensor<float> x0, x1;
            LoadLaneF32<T, DP>(x0, rows, row, col);
            LoadLaneF32<T, DP>(x1, rows, row + 1, col);
            Add(acc0, acc0, x0, mask);
            Add(acc1, acc1, x1, mask);
        }
        // 1-row tail
        if (row < count) {
            RegTensor<float> x_tail;
            LoadLaneF32<T, DP>(x_tail, rows, row, col);
            Add(acc0, acc0, x_tail, mask);
        }
        // Tree reduction across 4 accumulators
        Add(acc0, acc0, acc1, mask);
        Add(acc2, acc2, acc3, mask);
        Add(acc0, acc0, acc2, mask);
        StoreAlign(state + col, acc0, mask);
    }
}

template <class T>
__aicore__ inline void StatsSumVF(LocalTensor<T> rows, LocalTensor<float> state, uint32_t count, uint32_t dp)
{
    auto input = (__ubuf__ T *)rows.GetPhyAddr();
    auto sum = (__ubuf__ float *)state.GetPhyAddr();
    if (dp == 128)
        StatsSumImpl<T, 128>(input, sum, count);
    else
        StatsSumImpl<T, 64>(input, sum, count);
}
// Assign consecutive 128-row batches to each AIV within its die group.
// A complete head has a single writer; only heads crossing an AIV boundary
// use FP32 atomic adds. The caller synchronizes before consuming mean.
// IS_V selects the sequence-mean of V (smooth_v) instead of the one of K
// (smooth_k); both reduce the Sk rows of a key-value head.
template <class T, bool IS_V>
__aicore__ void StreamKStats(GM_ADDR src, GM_ADDR mean, SageParams p)
{
    const uint64_t *const strides = IS_V ? p.vs : p.ks;
    FixedResource r;
    GlobalTensor<T> kg;
    kg.SetGlobalBuffer((__gm__ T *)src);
    GlobalTensor<float> mg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    constexpr uint32_t BATCH_ROWS = 128, SLOT_BYTES = 32768;
    auto sum = r.ubBuf.template GetBufferByByte<float>(STATS_SUM_UB);
    const bool grouped = UseDieGroups(p);
    const uint32_t workers = grouped ? p.cores : 2 * p.cores;
    const uint32_t rank = GetBlockIdx() % workers;
    const uint32_t group = GetBlockIdx() / workers;
    const uint64_t heads = uint64_t(p.b) * p.hk / (grouped ? 2 : 1);
    const uint64_t firstHead = group * heads;
    const uint32_t headBatches = (p.sk + BATCH_ROWS - 1) / BATCH_ROWS;
    const uint64_t totalBatches = heads * headBatches;
    const uint64_t first = totalBatches * rank / workers;
    const uint64_t last = totalBatches * (rank + 1) / workers;

    // Workspace is uninitialized/reused. Publish zeros before any atomics.
    Duplicate(sum, 0.0f, p.dp);
    for (uint64_t head = rank; head < heads; head += workers)
        WriteRow(mg, (firstHead + head) * p.dp, sum, p.dp);
    SyncAll<true>();

    const float invSk = 1.0f / float(p.sk);
    LocalTensor<T> slotBuf[2] = {
        r.ubBuf.template GetBufferByByte<T>(0),
        r.ubBuf.template GetBufferByByte<T>(SLOT_BYTES)
    };
    for (uint64_t cursor = first; cursor < last;) {
        const uint64_t head = cursor / headBatches;
        const uint64_t stop = Min(last, (head + 1) * headBatches);
        const uint64_t bh = firstHead + head;
        const uint32_t b = bh / p.hk, h = bh % p.hk;
        const uint64_t headBase = Offset(p.ks, b, h, 0);
        const uint32_t begin = (cursor - head * headBatches) * BATCH_ROWS;
        const uint32_t end = Min(p.sk, uint32_t(stop - head * headBatches) * BATCH_ROWS);
        const bool exclusive = begin == 0 && end == p.sk;
        Duplicate(sum, 0.0f, p.dp);
        const uint32_t batches = (end - begin + BATCH_ROWS - 1) / BATCH_ROWS;
        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);
        for (uint32_t batch = 0; batch <= batches; ++batch) {
            if (batch < batches) {
                const uint32_t slot = batch & 1, start = begin + batch * BATCH_ROWS;
                const uint32_t rows = Min(BATCH_ROWS, end - start);
                const event_t event = static_cast<event_t>(1 + slot);
                WaitFlag<HardEvent::V_MTE2>(event);
                DataCopy(slotBuf[slot], kg[headBase + uint64_t(start) * p.dp], rows * p.dp);
                SetFlag<HardEvent::MTE2_V>(event);
            }
            if (batch == 0)
                continue;
            const uint32_t slot = (batch - 1) & 1;
            const uint32_t rows = Min(BATCH_ROWS, end - begin - (batch - 1) * BATCH_ROWS);
            const event_t event = static_cast<event_t>(1 + slot);
            WaitFlag<HardEvent::MTE2_V>(event);
            StatsSumVF(slotBuf[slot], sum, rows, p.dp);
            SetFlag<HardEvent::V_MTE2>(event);
        }
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);

        PipeBarrier<PIPE_V>();
        Muls(sum, sum, invSk, p.dp);
        if (!exclusive)
            SetAtomicAdd<float>();
        WriteRow(mg, bh * p.dp, sum, p.dp);
        if (!exclusive)
            SetAtomicNone();
        cursor = stop;
    }
}

template <class T>
__aicore__ void Stats(GM_ADDR k, GM_ADDR mean, SageParams p)
{
    if (p.smoothK && p.d == p.dp && p.sk >= 1024) {
        StreamKStats<T>(k, mean, p);
        return;
    }
    const bool grouped = UseDieGroups(p);
    const uint32_t workers = grouped ? p.cores : 2 * p.cores;
    const uint64_t heads = uint64_t(p.b) * p.hk / (grouped ? 2 : 1);
    FixedResource r;
    GlobalTensor<T> kg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    GlobalTensor<float> mg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    auto input = r.ubBuf.template GetBufferByByte<T>(STATS_RAW_UB);
    auto x = r.ubBuf.template GetBufferByByte<float>(STATS_F32_UB);
    auto sum = r.ubBuf.template GetBufferByByte<float>(STATS_SUM_UB);
    const uint32_t rank = GetBlockIdx() % workers;
    const uint64_t firstHead = grouped ? GetBlockIdx() / p.cores * heads : 0;
    const float invSk = 1.0f / float(p.sk);
    for (uint64_t local = rank; local < heads; local += workers) {
        const uint64_t bh = firstHead + local;
        uint32_t b = bh / p.hk, h = bh % p.hk;
        Duplicate(sum, 0.0f, p.dp);
        if (p.smoothK) {
            if (p.d == p.dp) {
                constexpr uint32_t BATCH_ROWS = 128, SLOT_BYTES = 32768;
                LocalTensor<T> slotBuf[2] = {
                    r.ubBuf.template GetBufferByByte<T>(0),
                    r.ubBuf.template GetBufferByByte<T>(SLOT_BYTES)
                };
                const uint32_t batches = (p.sk + BATCH_ROWS - 1) / BATCH_ROWS;
                SetFlag<HardEvent::V_MTE2>(1);
                SetFlag<HardEvent::V_MTE2>(2);
                for (uint32_t batch = 0; batch <= batches; ++batch) {
                    if (batch < batches) {
                        const uint32_t slot = batch & 1, start = batch * BATCH_ROWS;
                        const uint32_t rows = Min(BATCH_ROWS, p.sk - start);
                        const event_t event = static_cast<event_t>(1 + slot);
                        WaitFlag<HardEvent::V_MTE2>(event);
                        DataCopy(slotBuf[slot], kg[Offset(p.ks, b, h, start)], rows * p.dp);
                        SetFlag<HardEvent::MTE2_V>(event);
                    }
                    if (batch == 0)
                        continue;
                    const uint32_t slot = (batch - 1) & 1;
                    const uint32_t rows = Min(BATCH_ROWS, p.sk - (batch - 1) * BATCH_ROWS);
                    const event_t event = static_cast<event_t>(1 + slot);
                    WaitFlag<HardEvent::MTE2_V>(event);
                    StatsSumVF(slotBuf[slot], sum, rows, p.dp);
                    SetFlag<HardEvent::V_MTE2>(event);
                }
                WaitFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::V_MTE2>(2);
            } else {
                for (uint32_t row = 0; row < p.sk; ++row) {
                    ReadRow(input, kg, Offset(p.ks, b, h, row), p.d, p.dp);
                    Cast(x, input, RoundMode::CAST_NONE, p.dp);
                    PipeBarrier<PIPE_V>();
                    Add(sum, sum, x, p.dp);
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(sum, sum, invSk, p.dp);
        }
        WriteRow(mg, bh * p.dp, sum, p.dp);
    }
}
} // namespace Sage
