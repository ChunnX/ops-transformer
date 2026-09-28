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
// Preserve per-channel FP32 addition order. Raw inputs are converted in
// registers; the fallback can pass its already-converted FP32 rows.
template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t col = 0; col < DP; col += 64) {
        RegTensor<float> acc, x;
        LoadAlign(acc, state + col);
        for (uint32_t row = 0; row < count; ++row) {
            if constexpr (AscendC::IsSameType<T, float>::value) {
                LoadAlign(x, rows + row * DP + col);
            } else {
                RegTensor<T> raw;
                static constexpr CastTrait cast{
                    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, rows + row * DP + col);
                Cast<float, T, cast>(x, raw, mask);
            }
            Add(acc, acc, x, mask);
        }
        StoreAlign(state + col, acc, mask);
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
template <class T>
__aicore__ void StreamKStats(GM_ADDR k, GM_ADDR mean, SageParams p)
{
    FixedResource r;
    GlobalTensor<T> kg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    GlobalTensor<float> mg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    constexpr uint32_t BATCH_ROWS = 128, SLOT_BYTES = 32768;
    auto sum = r.ubBuf.template GetBufferByByte<float>(131072);
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

    for (uint64_t cursor = first; cursor < last;) {
        const uint64_t head = cursor / headBatches;
        const uint64_t stop = Min(last, (head + 1) * headBatches);
        const uint64_t bh = firstHead + head;
        const uint32_t b = bh / p.hk, h = bh % p.hk;
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
                DataCopy(r.ubBuf.template GetBufferByByte<T>(slot * SLOT_BYTES), kg[Offset(p.ks, b, h, start)],
                         rows * p.dp);
                SetFlag<HardEvent::MTE2_V>(event);
            }
            if (batch == 0)
                continue;
            const uint32_t slot = (batch - 1) & 1;
            const uint32_t rows = Min(BATCH_ROWS, end - begin - (batch - 1) * BATCH_ROWS);
            const event_t event = static_cast<event_t>(1 + slot);
            WaitFlag<HardEvent::MTE2_V>(event);
            auto input = r.ubBuf.template GetBufferByByte<T>(slot * SLOT_BYTES);
            StatsSumVF(input, sum, rows, p.dp);
            SetFlag<HardEvent::V_MTE2>(event);
        }
        WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::V_MTE2>(2);

        PipeBarrier<PIPE_V>();
        Muls(sum, sum, 1.0f / p.sk, p.dp);
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
    FixedResource r;
    GlobalTensor<T> kg;
    kg.SetGlobalBuffer((__gm__ T *)k);
    GlobalTensor<float> mg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    auto input = r.ubBuf.GetBufferByByte<T>(0);
    auto x = r.ubBuf.GetBufferByByte<float>(65536);
    auto sum = r.ubBuf.GetBufferByByte<float>(131072);
    const bool grouped = UseDieGroups(p);
    const uint32_t workers = grouped ? p.cores : 2 * p.cores;
    const uint32_t rank = GetBlockIdx() % workers;
    const uint64_t heads = uint64_t(p.b) * p.hk / (grouped ? 2 : 1);
    const uint64_t firstHead = grouped ? GetBlockIdx() / p.cores * heads : 0;
    for (uint64_t local = rank; local < heads; local += workers) {
        const uint64_t bh = firstHead + local;
        uint32_t b = bh / p.hk, h = bh % p.hk;
        Duplicate(sum, 0.0f, p.dp);
        if (p.smoothK) {
            if (p.d == p.dp) {
                for (uint32_t start = 0; start < p.sk; start += 128) {
                    const uint32_t rows = Min(uint32_t(128), p.sk - start);
                    Sync<HardEvent::V_MTE2>();
                    DataCopy(input, kg[Offset(p.ks, b, h, start)], rows * p.dp);
                    Sync<HardEvent::MTE2_V>();
                    Cast(x, input, RoundMode::CAST_NONE, rows * p.dp);
                    PipeBarrier<PIPE_V>();
                    StatsSumVF(x, sum, rows, p.dp);
                }
            } else {
                for (uint32_t row = 0; row < p.sk; ++row) {
                    ReadRow(input, kg, Offset(p.ks, b, h, row), p.d, p.dp);
                    Cast(x, input, RoundMode::CAST_NONE, p.dp);
                    PipeBarrier<PIPE_V>();
                    Add(sum, sum, x, p.dp);
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(sum, sum, 1.0f / p.sk, p.dp);
        }
        WriteRow(mg, bh * p.dp, sum, p.dp);
    }
}
} // namespace Sage
