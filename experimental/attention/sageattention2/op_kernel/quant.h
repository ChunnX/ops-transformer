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
#include "scale.h"
#include "quant_v_mx.h"
namespace Sage {

template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, [[maybe_unused]] __ubuf__ float *mean, __ubuf__ float *scales,
                                      __ubuf__ int8_t *out, __ubuf__ half *halfScales, __ubuf__ float *factor,
                                      __ubuf__ float *inverseScales, uint32_t valid)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ma, mb, zero, one, invDivisor;
    Duplicate(zero, 0.0f);
    Duplicate(one, 1.0f);
    Duplicate(invDivisor, INV_INT8_MAX);
    if constexpr (SMOOTH) {
        LoadAlign(ma, mean);
        if constexpr (DP == 128)
            LoadAlign(mb, mean + 64);
    }
    auto dst = scales;
    UnalignReg compact;
    // 2-row interleaved reduction pipeline hiding RAW dependency latency
    for (uint16_t row = 0; row < 64; row += 2) {
        RegTensor<float> a0, a1, ab0, ab1, m0, m1;
        LoadAlign(a0, x + row * DP);
        LoadAlign(a1, x + (row + 1) * DP);
        if constexpr (SMOOTH) {
            Sub(a0, a0, ma, full);
            Sub(a1, a1, ma, full);
        }
        Abs(ab0, a0, full);
        Abs(ab1, a1, full);
        if constexpr (SMOOTH) {
            StoreAlign(x + row * DP, a0, full);
            StoreAlign(x + (row + 1) * DP, a1, full);
        }
        if constexpr (DP == 128) {
            RegTensor<float> b0, b1, bb0, bb1;
            LoadAlign(b0, x + row * DP + 64);
            LoadAlign(b1, x + (row + 1) * DP + 64);
            if constexpr (SMOOTH) {
                Sub(b0, b0, mb, full);
                Sub(b1, b1, mb, full);
            }
            Abs(bb0, b0, full);
            Abs(bb1, b1, full);
            Max(ab0, ab0, bb0, full);
            Max(ab1, ab1, bb1, full);
            if constexpr (SMOOTH) {
                StoreAlign(x + row * DP + 64, b0, full);
                StoreAlign(x + (row + 1) * DP + 64, b1, full);
            }
        }
        ReduceMax(m0, ab0, full);
        ReduceMax(m1, ab1, full);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m0, compact, 1);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m1, compact, 1);
    }
    StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // All 64 row maxima fit in one FP32 register. Form token scales together
    // with the same division and zero-row rule as the scalar-row schedule.
    RegTensor<float> m;
    LoadAlign(m, scales);
    Mul(m, m, invDivisor, full);
    MaskReg positive;
    Compare<float, CMPMODE::GT>(positive, m, zero, full);
    Select(m, m, one, positive);
    StoreAlign(scales, m, full);
    // Keep the original scales for dequantization; reuse one reciprocal per token.
    // Use Newton-Raphson FastReciprocal to eliminate 32-cycle hardware vector division.
    RegTensor<float> inverse;
    FastReciprocal(inverse, m, full);
    StoreAlign(inverseScales, inverse, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t row = 0; row < 64; row += 2) {
        RegTensor<float> m0, m1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(m0, inverseScales + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(m1, inverseScales + row + 1);
        if constexpr (DP == 128) {
            RegTensor<float> a0_0, a0_1, a1_0, a1_1;
            RegTensor<int32_t> p0_0, p0_1, p1_0, p1_1;
            LoadAlign(a0_0, x + row * DP);
            LoadAlign(a0_1, x + row * DP + 64);
            LoadAlign(a1_0, x + (row + 1) * DP);
            LoadAlign(a1_1, x + (row + 1) * DP + 64);
            Mul(a0_0, a0_0, m0, full);
            Mul(a0_1, a0_1, m0, full);
            Mul(a1_0, a1_0, m1, full);
            Mul(a1_1, a1_1, m1, full);
            Cast<int32_t, float, trait>(p0_0, a0_0, full);
            Cast<int32_t, float, trait>(p0_1, a0_1, full);
            Cast<int32_t, float, trait>(p1_0, a1_0, full);
            Cast<int32_t, float, trait>(p1_1, a1_1, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + row * DP, (RegTensor<int8_t> &)p0_0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + row * DP + 64, (RegTensor<int8_t> &)p0_1, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP, (RegTensor<int8_t> &)p1_0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP + 64, (RegTensor<int8_t> &)p1_1, full);
        } else {
            RegTensor<float> a0, a1;
            RegTensor<int32_t> p0, p1;
            LoadAlign(a0, x + row * DP);
            LoadAlign(a1, x + (row + 1) * DP);
            Mul(a0, a0, m0, full);
            Mul(a1, a1, m1, full);
            Cast<int32_t, float, trait>(p0, a0, full);
            Cast<int32_t, float, trait>(p1, a1, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + row * DP, (RegTensor<int8_t> &)p0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP, (RegTensor<int8_t> &)p1, full);
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
template <class T, bool PERMUTE_V>
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
    static_assert(2 * SLOT_BYTES + 512 <= 248 * 1024, "quant pipeline exceeds UB");
    auto factors = r.ubBuf.GetBufferByByte<float>(2 * SLOT_BYTES);
    FixedScaleFactorsVF((__ubuf__ float *)factors.GetPhyAddr(), p.smScale);
    __ubuf__ float *factorPtr[2] = {(__ubuf__ float *)factors[0].GetPhyAddr(),
                                   (__ubuf__ float *)factors[64].GetPhyAddr()};

    // Pre-allocate slot buffers to eliminate per-iteration UB address resolution overhead
    LocalTensor<T> input[2], vInput[2];
    LocalTensor<float> x[2], mu[2], vs[2], scales[2], inverseScales[2];
    LocalTensor<int8_t> out[2];
    LocalTensor<float8_e4m3_t> fp8[2];
    LocalTensor<half> halfScales[2];
    for (uint32_t s = 0; s < 2; ++s) {
        const uint32_t b = s * SLOT_BYTES;
        input[s] = r.ubBuf.template GetBufferByByte<T>(b);
        vInput[s] = r.ubBuf.template GetBufferByByte<T>(b + V_INPUT);
        x[s] = r.ubBuf.template GetBufferByByte<float>(b + FLOAT_INPUT);
        mu[s] = r.ubBuf.template GetBufferByByte<float>(b + MEAN);
        vs[s] = r.ubBuf.template GetBufferByByte<float>(b + MX_SCALE);
        out[s] = r.ubBuf.template GetBufferByByte<int8_t>(b + QK_OUTPUT);
        fp8[s] = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(b + V_OUTPUT);
        scales[s] = r.ubBuf.template GetBufferByByte<float>(b + RAW_SCALE);
        halfScales[s] = r.ubBuf.template GetBufferByByte<half>(b + HALF_SCALE);
        inverseScales[s] = r.ubBuf.template GetBufferByByte<float>(b + INVERSE_SCALE);
    }

    const uint64_t nq = uint64_t(p.b) * p.hq * p.qp / 64;
    const uint64_t nk = uint64_t(p.b) * p.hk * p.kp / 64;
    const bool grouped = UseDieGroups(p);
    const uint32_t cores = grouped ? p.cores : 2 * p.cores;
    const uint32_t core = GetBlockIdx() % cores, group = GetBlockIdx() / cores;
    const uint64_t localNq = nq / (grouped ? 2 : 1), localNk = nk / (grouped ? 2 : 1);
    if (core >= localNq + localNk)
        return;
    const uint64_t iterations = (localNq + localNk - core + cores - 1) / cores;
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
            const uint32_t slot = iteration & 1;
            const uint64_t task = core + iteration * cores;
            const bool iskv = task < localNk;
            const bool isq = !iskv;
            const uint64_t idx = (iskv ? group * localNk + task : group * localNq + (task - localNk)) * 64;
            const uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
            const uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
            const uint32_t valid = token < (isq ? p.sq : p.sk) ? 64 : 0;
            pending[slot] = {idx, isq, valid};
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(1 + slot));
            if (valid)
                DataCopy(input[slot], (isq ? qg : kg)[Offset(isq ? p.qs : p.ks, b, h, token)], 64 * p.dp);
            if (!isq) {
                if (valid)
                    DataCopy(vInput[slot], vg[Offset(p.vs, b, h, token)], 64 * p.dp);
                DataCopy(mu[slot], mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
            }
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1 + slot));
        }
        if (iteration == 0)
            continue;
        const uint32_t slot = (iteration - 1) & 1;
        const event_t event = static_cast<event_t>(1 + slot);
        const BatchInfo info = pending[slot];
        WaitFlag<HardEvent::MTE2_V>(event);
        if (iteration > 2)
            WaitFlag<HardEvent::MTE3_V>(event);
        if (info.valid < 64) {
            Duplicate(input[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
            PipeBarrier<PIPE_V>();
        }
        Cast(x[slot], input[slot], RoundMode::CAST_NONE, 64 * p.dp);
        PipeBarrier<PIPE_V>();
        QuantRowsVF((__ubuf__ float *)x[slot].GetPhyAddr(), (__ubuf__ float *)mu[slot].GetPhyAddr(),
                    (__ubuf__ float *)scales[slot].GetPhyAddr(), (__ubuf__ int8_t *)out[slot].GetPhyAddr(),
                    (__ubuf__ half *)halfScales[slot].GetPhyAddr(), factorPtr[info.isq ? 0 : 1],
                    (__ubuf__ float *)inverseScales[slot].GetPhyAddr(), p.dp, !info.isq && p.smoothK && info.valid,
                    info.valid);
        if (!info.isq) {
            PipeBarrier<PIPE_V>();
            if (info.valid < 64)
                Duplicate(vInput[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
            PipeBarrier<PIPE_V>();
            QuantVRowsMxVF<T, PERMUTE_V>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
                                         (__ubuf__ T *)x[slot].GetPhyAddr(),
                                         (__ubuf__ uint16_t *)vs[slot].GetPhyAddr(),
                                         (__ubuf__ float8_e4m3_t *)fp8[slot].GetPhyAddr(), p.dp);
        }
        SetFlag<HardEvent::V_MTE2>(event);
        Sync<HardEvent::V_MTE3>();
        DataCopy((info.isq ? qo : ko)[info.idx * p.dp], out[slot], 64 * p.dp);
        DataCopy((info.isq ? qsg : ksg)[info.idx], halfScales[slot], 64);
        if (!info.isq) {
            DataCopy(vo[info.idx * p.dp], fp8[slot], 64 * p.dp);
            DataCopy(mxg[info.idx / 64 * p.dp], vs[slot].ReinterpretCast<uint16_t>(), p.dp);
        }
        SetFlag<HardEvent::MTE3_V>(event);
    }
    for (uint32_t slot = 0; slot < Min(uint64_t(2), iterations); ++slot)
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(1 + slot));
    WaitFlag<HardEvent::V_MTE2>(1);
    WaitFlag<HardEvent::V_MTE2>(2);
}
template <class T, bool PERMUTE_V>
__aicore__ void Quant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR mxScale, GM_ADDR q8, GM_ADDR k8,
                      GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, SageParams p)
{
    if (p.d == p.dp && p.sq % 64 == 0 && p.sk % 64 == 0 && p.qs[2] == p.dp && p.ks[2] == p.dp && p.vs[2] == p.dp) {
        QuantBatch<T, PERMUTE_V>(q, k, v, mean, mxScale, q8, k8, v8, qs, ks, p);
        return;
    }
    FixedResource r;
    GlobalTensor<T> qg, kg, vg;
    qg.SetGlobalBuffer((__gm__ T *)q);
    kg.SetGlobalBuffer((__gm__ T *)k);
    vg.SetGlobalBuffer((__gm__ T *)v);
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
    auto input = r.ubBuf.GetBufferByByte<T>(0);
    auto x = r.ubBuf.GetBufferByByte<float>(512);
    auto aux = r.ubBuf.GetBufferByByte<float>(1024);
    auto ab = r.ubBuf.GetBufferByByte<float>(1536);
    auto red = r.ubBuf.GetBufferByByte<float>(2048);
    auto work = r.ubBuf.GetBufferByByte<float>(2560);
    auto out = r.ubBuf.GetBufferByByte<int8_t>(4096);
    auto halfScales = r.ubBuf.GetBufferByByte<half>(4352);
    auto factors = r.ubBuf.GetBufferByByte<float>(4608);
    FixedScaleFactorsVF((__ubuf__ float *)factors.GetPhyAddr(), p.smScale);
    uint64_t nq = uint64_t(p.b) * p.hq * p.qp;
    uint64_t nk = uint64_t(p.b) * p.hk * p.kp;
    for (uint64_t row = GetBlockIdx(); row < nq + nk; row += (2 * p.cores)) {
        bool iskv = row < nk;
        bool isq = !iskv;
        uint64_t idx = iskv ? row : row - nk;
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
        float s = a > 0 ? a * INV_INT8_MAX : 1.0f;
        float inv_s = a > 0 ? INT8_MAX_FLOAT / a : 1.0f;
        Duplicate(aux, inv_s, p.dp);
        PipeBarrier<PIPE_V>();
        Mul(x, x, aux, p.dp);
        PipeBarrier<PIPE_V>();
        Pack8((__ubuf__ int8_t *)out.GetPhyAddr(), (__ubuf__ float *)x.GetPhyAddr(), p.dp);
        WriteRow(isq ? qo : ko, idx * p.dp, out, p.dp);
        red.SetValue(0, s);
        Sync<HardEvent::S_V>();
        PackFixedScaleVF((__ubuf__ float *)red.GetPhyAddr(), (__ubuf__ half *)halfScales.GetPhyAddr(),
                         (__ubuf__ float *)factors[isq ? 0 : 64].GetPhyAddr(), valid ? 1 : 0);
        WriteRow(isq ? qsg : ksg, idx, halfScales, 1);
    }
    // Preserve the scalar Q/K rounding path for padded inputs. V still uses
    // exactly the same 32-token MX quantizer as the aligned batch path.
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<uint16_t> mxg;
    mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);
    auto vInput = r.ubBuf.GetBufferByByte<T>(8192);
    auto maxima = r.ubBuf.GetBufferByByte<T>(32768);
    auto vOut = r.ubBuf.GetBufferByByte<float8_e4m3_t>(33792);
    auto mx = r.ubBuf.GetBufferByByte<uint16_t>(43008);
    for (uint64_t tile = GetBlockIdx(); tile < nk / 64; tile += 2 * p.cores) {
        const uint64_t idx = tile * 64;
        const uint32_t token = idx % p.kp, h = (idx / p.kp) % p.hk, b = idx / p.kp / p.hk;
        const uint32_t valid = token < p.sk ? Min(uint32_t(64), p.sk - token) : 0;
        if (p.d == p.dp && valid) {
            Sync<HardEvent::V_MTE2>();
            DataCopy(vInput, vg[Offset(p.vs, b, h, token)], valid * p.dp);
            Sync<HardEvent::MTE2_V>();
        } else {
            for (uint32_t row = 0; row < valid; ++row)
                ReadRow(vInput[row * p.dp], vg, Offset(p.vs, b, h, token + row), p.d, p.dp);
        }
        if (valid < 64)
            Duplicate(vInput[valid * p.dp], T(0), (64 - valid) * p.dp);
        PipeBarrier<PIPE_V>();
        QuantVRowsMxVF<T, PERMUTE_V>((__ubuf__ T *)vInput.GetPhyAddr(), (__ubuf__ T *)maxima.GetPhyAddr(),
                                     (__ubuf__ uint16_t *)mx.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)vOut.GetPhyAddr(),
                                     p.dp);
        WriteRow(vo, idx * p.dp, vOut, 64 * p.dp);
        WriteRow(mxg, tile * p.dp, mx, p.dp);
    }
}
} // namespace Sage
