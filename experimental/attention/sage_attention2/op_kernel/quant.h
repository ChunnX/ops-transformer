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
namespace Sage {
template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                      __ubuf__ int8_t *out, __ubuf__ half *halfScales, __ubuf__ float *factor,
                                      __ubuf__ float *inverseScales, uint32_t valid)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> a, b, ma, mb, ab, bb, m, zero, one, divisor;
    Duplicate(zero, 0.0f);
    Duplicate(one, 1.0f);
    Duplicate(divisor, 127.0f);
    if constexpr (SMOOTH) {
        LoadAlign(ma, mean);
        if constexpr (DP == 128)
            LoadAlign(mb, mean + 64);
    }
    auto dst = scales;
    UnalignReg compact;
    for (uint16_t row = 0; row < 64; ++row) {
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
    StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // All 64 row maxima fit in one FP32 register. Form token scales together
    // with the same division and zero-row rule as the scalar-row schedule.
    LoadAlign(m, scales);
    Div(m, m, divisor, full);
    MaskReg positive;
    Compare<float, CMPMODE::GT>(positive, m, zero, full);
    Select(m, m, one, positive);
    StoreAlign(scales, m, full);
    // Keep the original scales for dequantization; reuse one reciprocal per token.
    RegTensor<float> inverse;
    Div(inverse, one, m, full);
    StoreAlign(inverseScales, inverse, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t row = 0; row < 64; ++row) {
        LoadAlign<float, LoadDist::DIST_BRC_B32>(m, inverseScales + row);
#pragma unroll
        for (uint16_t col = 0; col < DP; col += 64) {
            RegTensor<int32_t> packed;
            LoadAlign(a, x + row * DP + col);
            Mul(a, a, m, full);
            Cast<int32_t, float, trait>(packed, a, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + row * DP + col, (RegTensor<int8_t> &)packed, full);
        }
    }
    PackFixedScaleBlock(scales, halfScales, factor, valid);
}
__aicore__ inline void QuantRowsVF(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                   __ubuf__ int8_t *out, __ubuf__ half *halfScales, __ubuf__ float *factor,
                                   __ubuf__ float *inverseScales, uint32_t dp,
                                   uint32_t smooth, uint32_t valid)
{
    if (dp == 128) {
        if (smooth)
            QuantRowsImpl<128, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid);
        else
            QuantRowsImpl<128, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid);
    } else {
        if (smooth)
            QuantRowsImpl<64, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid);
        else
            QuantRowsImpl<64, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid);
    }
}
template <bool PERMUTE_V = false, bool SMOOTH = false>
__simd_vf__ inline void QuantVRowsVF(__ubuf__ float *x, __ubuf__ float *scales, __ubuf__ float8_e4m3_t *out,
                                     uint32_t dp, __ubuf__ float *mean = nullptr)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t col = 0; col < dp; col += 64) {
        RegTensor<float> scale, a, mu, one;
        LoadAlign(scale, scales + col);
        Duplicate(one, 1.0f);
        // Each channel reciprocal is shared by all 64 input rows.
        Div(scale, one, scale, full);
        if constexpr (SMOOTH)
            LoadAlign(mu, mean + col);
        if constexpr (PERMUTE_V) {
            // Emit each interleaved pair in destination order. Both source
            // rows and destinations are affine; no per-row bit permutation.
            for (uint16_t group = 0; group < 64; group += 32) {
                for (uint16_t row = 0; row < 16; ++row) {
                    RegTensor<float> a0, a1;
                    RegTensor<float8_e4m3_t> p0, p1;
                    LoadAlign(a0, x + (group + row) * dp + col);
                    LoadAlign(a1, x + (group + row + 16) * dp + col);
                    if constexpr (SMOOTH) {
                        Sub(a0, a0, mu, full);
                        Sub(a1, a1, mu, full);
                    }
                    Mul(a0, a0, scale, full);
                    Mul(a1, a1, scale, full);
                    Cast<float8_e4m3_t, float, trait>(p0, a0, full);
                    Cast<float8_e4m3_t, float, trait>(p1, a1, full);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + (group + row * 2) * dp + col, p0, full);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + (group + row * 2 + 1) * dp + col, p1,
                                                                         full);
                }
            }
        } else {
            for (uint16_t row = 0; row < 64; ++row) {
                RegTensor<float8_e4m3_t> packed;
                LoadAlign(a, x + row * dp + col);
                if constexpr (SMOOTH)
                    Sub(a, a, mu, full);
                Mul(a, a, scale, full);
                Cast<float8_e4m3_t, float, trait>(packed, a, full);
                StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * dp + col, packed, full);
            }
        }
    }
}
template <class T, bool PERMUTE_V = false, bool RAW_SCALES = false>
__aicore__ void QuantBatch(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vscale, GM_ADDR q8, GM_ADDR k8,
                           GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, SageParams p, GM_ADDR rawQs, GM_ADDR rawKs)
{
    FixedResource r;
    GlobalTensor<T> qg, kg, vg;
    qg.SetGlobalBuffer((__gm__ T *)q);
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    GlobalTensor<float> mg, sg;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    sg.SetGlobalBuffer((__gm__ float *)vscale);
    GlobalTensor<half> qsg, ksg;
    qsg.SetGlobalBuffer((__gm__ half *)qs);
    ksg.SetGlobalBuffer((__gm__ half *)ks);
    GlobalTensor<float> rawQg, rawKg;
    if constexpr (RAW_SCALES) {
        rawQg.SetGlobalBuffer((__gm__ float *)rawQs);
        rawKg.SetGlobalBuffer((__gm__ float *)rawKs);
    }
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
    constexpr uint32_t MEAN = 81920, V_SCALE = 82432, RAW_SCALE = 82944, HALF_SCALE = 83200;
    constexpr uint32_t V_MEAN = 83456;
    constexpr uint32_t INVERSE_SCALE = V_MEAN + 512;
    static_assert(INVERSE_SCALE + 64 * sizeof(float) <= SLOT_BYTES, "inverse scale exceeds slot");
    static_assert(V_MEAN + 512 <= SLOT_BYTES, "quant slot overflow");
    static_assert(2 * SLOT_BYTES + 512 <= 248 * 1024, "quant pipeline exceeds UB");
    auto factors = r.ubBuf.GetBufferByByte<float>(2 * SLOT_BYTES);
    FixedScaleFactorsVF((__ubuf__ float *)factors.GetPhyAddr(), p.smScale);
    const uint64_t nq = uint64_t(p.b) * p.hq * p.qp / 64;
    const uint64_t nk = uint64_t(p.b) * p.hk * p.kp / 64;
    const uint32_t core = GetBlockIdx(), cores = (2 * p.cores);
    if (core >= nq + nk)
        return;
    const uint64_t iterations = (nq + nk - core + cores - 1) / cores;
    SetFlag<HardEvent::V_MTE2>(1);
    SetFlag<HardEvent::V_MTE2>(2);
    struct BatchInfo {
        uint64_t idx;
        bool isq, valid;
    } pending[2];
    for (uint64_t iteration = 0; iteration <= iterations; ++iteration) {
        // Enqueue next input before waiting for the current input. MTE2 must
        // consume this slot's Vector release independently of the MTE3 pipe;
        // the latter only reads the disjoint output regions of this slot.
        if (iteration < iterations) {
            const uint32_t slot = iteration & 1, base = slot * SLOT_BYTES;
            const uint64_t task = core + iteration * cores;
            const bool isq = task < nq;
            const uint64_t idx = (isq ? task : task - nq) * 64;
            const uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
            const uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
            const bool valid = token < (isq ? p.sq : p.sk);
            pending[slot] = {idx, isq, valid};
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(1 + slot));
            if (valid)
                DataCopy(r.ubBuf.GetBufferByByte<T>(base), (isq ? qg : kg)[Offset(isq ? p.qs : p.ks, b, h, token)],
                         64 * p.dp);
            if (!isq) {
                if (valid)
                    DataCopy(r.ubBuf.GetBufferByByte<T>(base + V_INPUT), vg[Offset(p.vs, b, h, token)], 64 * p.dp);
                DataCopy(r.ubBuf.GetBufferByByte<float>(base + MEAN), mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                DataCopy(r.ubBuf.GetBufferByByte<float>(base + V_SCALE), sg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                if (p.smoothV)
                    DataCopy(r.ubBuf.GetBufferByByte<float>(base + V_MEAN),
                             mg[(uint64_t(p.b) * p.hk + uint64_t(b) * p.hk + h) * p.dp], p.dp);
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
        auto vs = r.ubBuf.GetBufferByByte<float>(base + V_SCALE);
        auto out = r.ubBuf.GetBufferByByte<int8_t>(base + QK_OUTPUT);
        auto fp8 = r.ubBuf.GetBufferByByte<float8_e4m3_t>(base + V_OUTPUT);
        auto scales = r.ubBuf.GetBufferByByte<float>(base + RAW_SCALE);
        auto halfScales = r.ubBuf.GetBufferByByte<half>(base + HALF_SCALE);
        auto inverseScales = r.ubBuf.GetBufferByByte<float>(base + INVERSE_SCALE);
        if (info.valid)
            Cast(x, input, RoundMode::CAST_NONE, 64 * p.dp);
        else
            Duplicate(x, 0.0f, 64 * p.dp);
        PipeBarrier<PIPE_V>();
        QuantRowsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)mu.GetPhyAddr(),
                    (__ubuf__ float *)scales.GetPhyAddr(), (__ubuf__ int8_t *)out.GetPhyAddr(),
                    (__ubuf__ half *)halfScales.GetPhyAddr(), (__ubuf__ float *)factors[info.isq ? 0 : 64].GetPhyAddr(),
                    (__ubuf__ float *)inverseScales.GetPhyAddr(), p.dp, !info.isq && p.smoothK && info.valid,
                    info.valid ? 64 : 0);
        if (!info.isq) {
            PipeBarrier<PIPE_V>();
            if (info.valid)
                Cast(x, r.ubBuf.GetBufferByByte<T>(base + V_INPUT), RoundMode::CAST_NONE, 64 * p.dp);
            else
                Duplicate(x, 0.0f, 64 * p.dp);
            PipeBarrier<PIPE_V>();
            if (p.smoothV && info.valid)
                QuantVRowsVF<PERMUTE_V, true>(
                    (__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)vs.GetPhyAddr(),
                    (__ubuf__ float8_e4m3_t *)fp8.GetPhyAddr(), p.dp,
                    (__ubuf__ float *)r.ubBuf.GetBufferByByte<float>(base + V_MEAN).GetPhyAddr());
            else
                QuantVRowsVF<PERMUTE_V>((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)vs.GetPhyAddr(),
                                        (__ubuf__ float8_e4m3_t *)fp8.GetPhyAddr(), p.dp);
        }
        SetFlag<HardEvent::V_MTE2>(event);
        Sync<HardEvent::V_MTE3>();
        DataCopy((info.isq ? qo : ko)[info.idx * p.dp], out, 64 * p.dp);
        DataCopy((info.isq ? qsg : ksg)[info.idx], halfScales, 64);
        if constexpr (RAW_SCALES)
            DataCopy((info.isq ? rawQg : rawKg)[info.idx], scales, 64);
        if (!info.isq)
            DataCopy(vo[info.idx * p.dp], fp8, 64 * p.dp);
        SetFlag<HardEvent::MTE3_V>(event);
    }
    for (uint32_t slot = 0; slot < Min(uint64_t(2), iterations); ++slot)
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1 + slot));
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::V_MTE2>(2);
}
template <class T, bool PERMUTE_V = false, bool RAW_SCALES = false>
__aicore__ void Quant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vscale, GM_ADDR q8, GM_ADDR k8, GM_ADDR v8,
                      GM_ADDR qs, GM_ADDR ks, SageParams p, GM_ADDR rawQs = nullptr, GM_ADDR rawKs = nullptr)
{
    if (p.d == p.dp && p.sq % 64 == 0 && p.sk % 64 == 0 && p.qs[2] == p.dp && p.ks[2] == p.dp && p.vs[2] == p.dp) {
        QuantBatch<T, PERMUTE_V, RAW_SCALES>(q, k, v, mean, vscale, q8, k8, v8, qs, ks, p, rawQs, rawKs);
        return;
    }
    FixedResource r;
    GlobalTensor<T> qg, kg, vg;
    qg.SetGlobalBuffer((__gm__ T *)q);
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
    GlobalTensor<float> mg, vs;
    mg.SetGlobalBuffer((__gm__ float *)mean);
    vs.SetGlobalBuffer((__gm__ float *)vscale);
    GlobalTensor<half> qsg, ksg;
    qsg.SetGlobalBuffer((__gm__ half *)qs);
    ksg.SetGlobalBuffer((__gm__ half *)ks);
    GlobalTensor<float> rawQg, rawKg;
    if constexpr (RAW_SCALES) {
        rawQg.SetGlobalBuffer((__gm__ float *)rawQs);
        rawKg.SetGlobalBuffer((__gm__ float *)rawKs);
    }
    GlobalTensor<int8_t> qo, ko;
    qo.SetGlobalBuffer((__gm__ int8_t *)q8);
    ko.SetGlobalBuffer((__gm__ int8_t *)k8);
    GlobalTensor<float8_e4m3_t> vo;
    vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);
    auto input = r.ubBuf.GetBufferByByte<T>(0);
    auto x = r.ubBuf.GetBufferByByte<float>(512);
    auto aux = r.ubBuf.GetBufferByByte<float>(1024);
    auto ab = r.ubBuf.GetBufferByByte<float>(1536);
    auto red = r.ubBuf.GetBufferByByte<float>(2048);
    auto work = r.ubBuf.GetBufferByByte<float>(2560);
    auto out = r.ubBuf.GetBufferByByte<int8_t>(4096);
    auto fp8 = out.template ReinterpretCast<float8_e4m3_t>();
    auto halfScales = r.ubBuf.GetBufferByByte<half>(4352);
    auto factors = r.ubBuf.GetBufferByByte<float>(4608);
    FixedScaleFactorsVF((__ubuf__ float *)factors.GetPhyAddr(), p.smScale);
    uint64_t nq = uint64_t(p.b) * p.hq * p.qp, nk = uint64_t(p.b) * p.hk * p.kp;
    for (uint64_t row = GetBlockIdx(); row < nq + nk; row += (2 * p.cores)) {
        bool isq = row < nq;
        uint64_t idx = isq ? row : row - nq;
        uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
        uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
        bool valid = token < (isq ? p.sq : p.sk);
        if (valid) {
            ReadRow(input, isq ? qg : kg, Offset(isq ? p.qs : p.ks, b, h, token), p.d, p.dp);
            Cast(x, input, RoundMode::CAST_NONE, p.dp);
            PipeBarrier<PIPE_V>();
            if (!isq && p.smoothK) {
                DataCopy(aux, mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
                Sub(x, x, aux, p.dp);
                PipeBarrier<PIPE_V>();
            }
        } else {
            Duplicate(x, 0.0f, p.dp);
        }
        Abs(ab, x, p.dp);
        PipeBarrier<PIPE_V>();
        ReduceMax(red, ab, work, p.dp, false);
        float a = Scalar(red);
        float s = a > 0 ? a / 127.0f : 1.0f;
        Duplicate(aux, s, p.dp);
        PipeBarrier<PIPE_V>();
        Div(x, x, aux, p.dp);
        PipeBarrier<PIPE_V>();
        Pack8((__ubuf__ int8_t *)out.GetPhyAddr(), (__ubuf__ float *)x.GetPhyAddr(), p.dp);
        WriteRow(isq ? qo : ko, idx * p.dp, out, p.dp);
        red.SetValue(0, s);
        if constexpr (RAW_SCALES) {
            Sync<HardEvent::S_MTE3>();
            DataCopyExtParams cp{1, 4, 0, 0, 0};
            DataCopyPad((isq ? rawQg : rawKg)[idx], red, cp);
            Sync<HardEvent::MTE3_S>();
        }
        Sync<HardEvent::S_V>();
        PackFixedScaleVF((__ubuf__ float *)red.GetPhyAddr(), (__ubuf__ half *)halfScales.GetPhyAddr(),
                         (__ubuf__ float *)factors[isq ? 0 : 64].GetPhyAddr(), valid ? 1 : 0);
        WriteRow(isq ? qsg : ksg, idx, halfScales, 1);
        if (!isq) {
            if (valid) {
                ReadRow(input, vg, Offset(p.vs, b, h, token), p.d, p.dp);
                Cast(x, input, RoundMode::CAST_NONE, p.dp);
                PipeBarrier<PIPE_V>();
                if (p.smoothV) {
                    DataCopy(aux, mg[(uint64_t(p.b) * p.hk + uint64_t(b) * p.hk + h) * p.dp], p.dp);
                    Sync<HardEvent::MTE2_V>();
                    Sub(x, x, aux, p.dp);
                    PipeBarrier<PIPE_V>();
                    Sync<HardEvent::V_MTE2>();
                }
                DataCopy(aux, vs[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
                Div(x, x, aux, p.dp);
                PipeBarrier<PIPE_V>();
            } else {
                Duplicate(x, 0.0f, p.dp);
            }
            Pack8((__ubuf__ float8_e4m3_t *)fp8.GetPhyAddr(), (__ubuf__ float *)x.GetPhyAddr(), p.dp);
            uint64_t vrow = PERMUTE_V ? ((idx & ~uint64_t(31)) | ((idx & 15) << 1) | ((idx & 16) >> 4)) : idx;
            WriteRow(vo, vrow * p.dp, fp8, p.dp);
        }
    }
}
} // namespace Sage
