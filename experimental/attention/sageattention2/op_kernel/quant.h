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
#include "stats.h"
#include "scale.h"
#include "quant_v_mx.h"
namespace Sage {
template <uint32_t DP, bool RECIPROCAL>
__simd_callee__ inline void QuantPackRows(__ubuf__ float *values, __ubuf__ float *rowScales, __ubuf__ int8_t *output)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t row = 0; row < 64; ++row) {
        RegTensor<float> scale, value;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(scale, rowScales + row);
#pragma unroll
        for (uint16_t col = 0; col < DP; col += 64) {
            RegTensor<int32_t> packed;
            LoadAlign(value, values + row * DP + col);
            if constexpr (RECIPROCAL)
                Mul(value, value, scale, full);
            else
                Div(value, value, scale, full);
            Cast<int32_t, float, trait>(packed, value, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(output + row * DP + col, (RegTensor<int8_t> &)packed, full);
        }
    }
}
template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                      __ubuf__ int8_t *out, __ubuf__ half *halfScales, __ubuf__ float *factor,
                                      __ubuf__ float *inverseScales, uint32_t valid, uint32_t reciprocal)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> a, b, ma, mb, ab, bb, m, zero, one, divisor;
    Duplicate(zero, 0.0f);
    Duplicate(one, 1.0f);
    Duplicate(divisor, 127.0f);
    if (valid < 64)
        StoreAlign(scales, zero, full);
    if constexpr (SMOOTH) {
        LoadAlign(ma, mean);
        if constexpr (DP == 128)
            LoadAlign(mb, mean + 64);
    }
    auto dst = scales;
    UnalignReg compact;
    for (uint16_t row = 0; row < valid; ++row) {
        LoadAlign(a, x + row * DP);
        if constexpr (SMOOTH)
            Sub(a, a, ma, full);
        Abs(ab, a, full);
        if constexpr (SMOOTH)
            StoreAlign(x + row * DP, a, full);
        if constexpr (DP == 128) {
            LoadAlign(b, x + row * DP + 64);
            if constexpr (SMOOTH)
                Sub(b, b, mb, full);
            Abs(bb, b, full);
            Max(ab, ab, bb, full);
            if constexpr (SMOOTH)
                StoreAlign(x + row * DP + 64, b, full);
        }
        ReduceMax(m, ab, full);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m, compact, 1);
    }
    if (valid)
        StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // All 64 row maxima fit in one FP32 register. Form token scales together
    // with the same division and zero-row rule as the scalar-row schedule.
    LoadAlign(m, scales);
    MaskReg positive;
    if (reciprocal) {
        Div(m, m, divisor, full);
        Compare<float, CMPMODE::GT>(positive, m, zero, full);
    } else {
        Compare<float, CMPMODE::GT>(positive, m, zero, full);
        Div(m, m, divisor, full);
    }
    Select(m, m, one, positive);
    StoreAlign(scales, m, full);
    if (reciprocal) {
        RegTensor<float> inverse;
        Div(inverse, one, m, full);
        StoreAlign(inverseScales, inverse, full);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    if (reciprocal)
        QuantPackRows<DP, true>(x, inverseScales, out);
    else
        QuantPackRows<DP, false>(x, scales, out);
    PackFixedScaleBlock(scales, halfScales, factor, valid);
}
__aicore__ inline void QuantRowsVF(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                   __ubuf__ int8_t *out, __ubuf__ half *halfScales, __ubuf__ float *factor,
                                   __ubuf__ float *inverseScales, uint32_t dp,
                                   uint32_t smooth, uint32_t valid, uint32_t reciprocal)
{
    if (dp == 128) {
        if (smooth)
            QuantRowsImpl<128, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, reciprocal);
        else
            QuantRowsImpl<128, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, reciprocal);
    } else {
        if (smooth)
            QuantRowsImpl<64, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, reciprocal);
        else
            QuantRowsImpl<64, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, reciprocal);
    }
}
template <class T>
__aicore__ inline void LoadQuantRows(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t offset,
                                    uint32_t rows, uint32_t dim, uint32_t paddedDim)
{
    if (rows == 0)
        return;
    if (dim == paddedDim) {
        DataCopy(dst, src[offset], rows * paddedDim);
        return;
    }
    constexpr uint32_t blockElements = 32 / sizeof(T);
    const uint32_t alignedDim = (dim + blockElements - 1) / blockElements * blockElements;
    DataCopyExtParams copy{static_cast<uint16_t>(rows), dim * uint32_t(sizeof(T)),
                           0, (paddedDim - alignedDim) / blockElements, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignedDim - dim), T(0)};
    DataCopyPad(dst, src[offset], copy, pad);
}
template <class T, bool PERMUTE_V = false>
__aicore__ void QuantBatch(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR mxScale, GM_ADDR q8, GM_ADDR k8,
                           GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, SageParams p)
{
    FixedResource r;
    GlobalTensor<T> qg, kg, vg;
    qg.SetGlobalBuffer((__gm__ T *)q);
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    qg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    vg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    GlobalTensor<uint16_t> mxg;
    mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);
    GlobalTensor<float> mg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    GlobalTensor<half> qsg, ksg;
    qsg.SetGlobalBuffer((__gm__ half *)qs);
    ksg.SetGlobalBuffer((__gm__ half *)ks);
    GlobalTensor<int8_t> qo, ko;
    qo.SetGlobalBuffer((__gm__ int8_t *)q8);
    ko.SetGlobalBuffer((__gm__ int8_t *)k8);
    GlobalTensor<float8_e4m3_t> vo;
    vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);
    // Two independent input/output slots let MTE2(i+1), Vector(i), and
    // MTE3(i-1) overlap. Q/K and V outputs must not alias: both remain live
    // until the slot's output-complete event is consumed on its next use.
    constexpr uint32_t SLOT_BYTES = 96 * 1024;
    constexpr uint32_t V_INPUT = 16384, FLOAT_INPUT = 32768;
    constexpr uint32_t QK_OUTPUT = 65536, V_OUTPUT = 73728;
    constexpr uint32_t MEAN = 81920, MX_SCALE = 82432, RAW_SCALE = 82944, HALF_SCALE = 83200;
    constexpr uint32_t INVERSE_SCALE = 83456;
    static_assert(INVERSE_SCALE + 64 * sizeof(float) <= SLOT_BYTES, "inverse scale exceeds slot");
    static_assert(2 * SLOT_BYTES + 512 <= Arch::UB_SIZE, "quant pipeline exceeds UB");
    auto factors = r.ubBuf.GetBufferByByte<float>(2 * SLOT_BYTES);
    FixedScaleFactorsVF((__ubuf__ float *)factors.GetPhyAddr(), p.smScale);
    const uint64_t nq = uint64_t(p.b) * p.hq * p.qp / 64;
    const uint64_t nk = uint64_t(p.b) * p.hk * p.kp / 64;
    const bool grouped = UseDieGroups(p);
    const uint32_t cores = grouped ? p.cores : 2 * p.cores;
    const uint32_t core = GetBlockIdx() % cores, group = GetBlockIdx() / cores;
    const uint64_t localNq = nq / (grouped ? 2 : 1), localNk = nk / (grouped ? 2 : 1);
    if (core >= localNq + localNk)
        return;
    const uint64_t iterations = (localNq + localNk - core + cores - 1) / cores;
    const bool reciprocal = p.d == p.dp && p.sq % 64 == 0 && p.sk % 64 == 0;
    if (p.d != p.dp) {
        for (uint32_t slot = 0; slot < 2; ++slot) {
            Duplicate(r.ubBuf.GetBufferByByte<T>(slot * SLOT_BYTES), T(0), 64 * p.dp);
            Duplicate(r.ubBuf.GetBufferByByte<T>(slot * SLOT_BYTES + V_INPUT), T(0), 64 * p.dp);
        }
    }
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::V_MTE2>(2);
    struct BatchInfo {
        uint64_t idx;
        bool isq;
        uint32_t valid;
    } pending[2];
    for (uint64_t iteration = 0; iteration <= iterations; ++iteration) {
        // Enqueue next input before waiting for the current input. MTE2 must
        // consume this slot's Vector release independently of the MTE3 pipe;
        // the latter only reads the disjoint output regions of this slot.
        if (iteration < iterations) {
            const uint32_t slot = iteration & 1, base = slot * SLOT_BYTES;
            const uint64_t task = core + iteration * cores;
            const bool isq = task < localNq;
            const uint64_t idx = (isq ? group * localNq + task : group * localNk + task - localNq) * 64;
            const uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
            const uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
            const uint32_t length = isq ? p.sq : p.sk;
            const uint32_t valid = token < length ? Min(uint32_t(64), length - token) : 0;
            pending[slot] = {idx, isq, valid};
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(1 + slot));
            LoadQuantRows(r.ubBuf.GetBufferByByte<T>(base), isq ? qg : kg,
                           Offset(isq ? p.qs : p.ks, b, h, token), valid, p.d, p.dp);
            if (!isq) {
                LoadQuantRows(r.ubBuf.GetBufferByByte<T>(base + V_INPUT), vg,
                               Offset(p.vs, b, h, token), valid, p.d, p.dp);
                DataCopy(r.ubBuf.GetBufferByByte<float>(base + MEAN), mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
            }
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1 + slot));
        }
        if (iteration == 0)
            continue;
        const uint32_t slot = (iteration - 1) & 1, base = slot * SLOT_BYTES;
        const event_t event = static_cast<event_t>(1 + slot);
        const BatchInfo info = pending[slot];
        WaitFlag<HardEvent::MTE2_V>(event);
        if (iteration > 2)
            WaitFlag<HardEvent::MTE3_V>(event);
        auto input = r.ubBuf.GetBufferByByte<T>(base);
        auto x = r.ubBuf.GetBufferByByte<float>(base + FLOAT_INPUT);
        auto mu = r.ubBuf.GetBufferByByte<float>(base + MEAN);
        auto vs = r.ubBuf.GetBufferByByte<float>(base + MX_SCALE);
        auto out = r.ubBuf.GetBufferByByte<int8_t>(base + QK_OUTPUT);
        auto fp8 = r.ubBuf.GetBufferByByte<float8_e4m3_t>(base + V_OUTPUT);
        auto scales = r.ubBuf.GetBufferByByte<float>(base + RAW_SCALE);
        auto halfScales = r.ubBuf.GetBufferByByte<half>(base + HALF_SCALE);
        auto inverseScales = r.ubBuf.GetBufferByByte<float>(base + INVERSE_SCALE);
        if (info.valid < 64) {
            Duplicate(input[info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
            PipeBarrier<PIPE_V>();
        }
        Cast(x, input, RoundMode::CAST_NONE, 64 * p.dp);
        PipeBarrier<PIPE_V>();
        QuantRowsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)mu.GetPhyAddr(),
                    (__ubuf__ float *)scales.GetPhyAddr(), (__ubuf__ int8_t *)out.GetPhyAddr(),
                    (__ubuf__ half *)halfScales.GetPhyAddr(), (__ubuf__ float *)factors[info.isq ? 0 : 64].GetPhyAddr(),
                    (__ubuf__ float *)inverseScales.GetPhyAddr(), p.dp, !info.isq && p.smoothK && info.valid,
                    info.valid, reciprocal);
        if (!info.isq) {
            PipeBarrier<PIPE_V>();
            auto vInput = r.ubBuf.GetBufferByByte<T>(base + V_INPUT);
            if (info.valid < 64)
                Duplicate(vInput[info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
            PipeBarrier<PIPE_V>();
            QuantVRowsMxVF<T, PERMUTE_V>((__ubuf__ T *)vInput.GetPhyAddr(), (__ubuf__ T *)x.GetPhyAddr(),
                                         (__ubuf__ uint16_t *)vs.GetPhyAddr(),
                                         (__ubuf__ float8_e4m3_t *)fp8.GetPhyAddr(), p.dp);
        }
        SetFlag<HardEvent::V_MTE2>(event);
        Sync<HardEvent::V_MTE3>();
        DataCopy((info.isq ? qo : ko)[info.idx * p.dp], out, 64 * p.dp);
        DataCopy((info.isq ? qsg : ksg)[info.idx], halfScales, 64);
        if (!info.isq) {
            DataCopy(vo[info.idx * p.dp], fp8, 64 * p.dp);
            DataCopy(mxg[info.idx / 64 * p.dp], vs.ReinterpretCast<uint16_t>(), p.dp);
        }
        SetFlag<HardEvent::MTE3_V>(event);
    }
    for (uint32_t slot = 0; slot < Min(uint64_t(2), iterations); ++slot)
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1 + slot));
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::V_MTE2>(2);
}
template <class T, bool PERMUTE_V = false>
__aicore__ void Quant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR mxScale, GM_ADDR q8, GM_ADDR k8,
                      GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, SageParams p)
{
    QuantBatch<T, PERMUTE_V>(q, k, v, mean, mxScale, q8, k8, v8, qs, ks, p);
}
} // namespace Sage
