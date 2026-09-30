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

#include "../types.h"

namespace sage::compute {

// Broadcast grouped INT8 scale across token groups
__simd_callee__ inline void BroadcastGroupMaxima(__ubuf__ float *maxima, uint32_t group) {
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> m, shifted;
    LoadAlign(m, maxima);
    for (uint32_t step = 1; step < group; step <<= 1) {
        StoreAlign(maxima, m, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign(shifted, maxima + step);
        Max(m, m, shifted, full);
    }
    StoreAlign(maxima, m, full);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto keep = UpdateMask<float>(group);
    for (uint32_t start = 0; start < 64; start += group) {
        RegTensor<float> value;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(value, maxima + start);
        StoreAlign(maxima + start, value, keep);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Convert FP32 scales to half precision with pre-fused factor
__simd_callee__ inline void PackFixedScaleBlock(__ubuf__ float *scales, __ubuf__ half *halfScales,
                                                float factor, uint32_t valid) {
    using namespace AscendC::MicroAPI;
    auto mask = UpdateMask<float>(valid);
    RegTensor<float> r0, r1, rf;
    RegTensor<half> h0, h1;
    Duplicate(rf, factor);
    LoadAlign(r0, scales);
    LoadAlign(r1, scales + 64);
    Mul(r0, r0, rf, mask);
    Mul(r1, r1, rf, mask);
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Cast<half, float, trait>(h0, r0, mask);
    Cast<half, float, trait>(h1, r1, mask);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(halfScales, h0, mask);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(halfScales + 64, h1, mask);
}

// Symmetrical INT8 Quantization microkernel with 4-way unrolling to saturate Vector ALU
template <uint32_t DP, bool SMOOTH>
__simd_vf__ inline void QuantRowsImpl(__ubuf__ float *x, [[maybe_unused]] __ubuf__ float *mean,
                                      __ubuf__ float *scales, __ubuf__ int8_t *out,
                                      __ubuf__ half *halfScales, float factor,
                                      __ubuf__ float *inverseScales, uint32_t valid, uint32_t group) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> ma, mb, zero, one, invDivisor;
    Duplicate(zero, 0.0f);
    Duplicate(one, 1.0f);
    Duplicate(invDivisor, INV_INT8_MAX);

    if constexpr (SMOOTH) {
        LoadAlign(ma, mean);
        if constexpr (DP == 128) LoadAlign(mb, mean + 64);
    }

    auto dst = scales;
    UnalignReg compact;

    if constexpr (DP == 128) {
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
            ReduceMax(m0, ab0, full);
            ReduceMax(m1, ab1, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m0, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m1, compact, 1);
        }
    } else {
        // 4-way unrolling for DP == 64 to saturate Vector ALU pipeline
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> a0, a1, a2, a3;
            RegTensor<float> ab0, ab1, ab2, ab3;
            RegTensor<float> m0, m1, m2, m3;

            LoadAlign(a0, x + (row + 0) * DP);
            LoadAlign(a1, x + (row + 1) * DP);
            LoadAlign(a2, x + (row + 2) * DP);
            LoadAlign(a3, x + (row + 3) * DP);

            if constexpr (SMOOTH) {
                Sub(a0, a0, ma, full);
                Sub(a1, a1, ma, full);
                Sub(a2, a2, ma, full);
                Sub(a3, a3, ma, full);
            }

            Abs(ab0, a0, full);
            Abs(ab1, a1, full);
            Abs(ab2, a2, full);
            Abs(ab3, a3, full);

            if constexpr (SMOOTH) {
                StoreAlign(x + (row + 0) * DP, a0, full);
                StoreAlign(x + (row + 1) * DP, a1, full);
                StoreAlign(x + (row + 2) * DP, a2, full);
                StoreAlign(x + (row + 3) * DP, a3, full);
            }

            ReduceMax(m0, ab0, full);
            ReduceMax(m1, ab1, full);
            ReduceMax(m2, ab2, full);
            ReduceMax(m3, ab3, full);

            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m0, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m1, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m2, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m3, compact, 1);
        }
    }

    StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    if (group > 1) {
        BroadcastGroupMaxima(scales, group);
    }

    RegTensor<float> s, inverse;
    LoadAlign(s, scales);
    Mul(s, s, invDivisor, full);
    StoreAlign(scales, s, full);

    MaskReg active;
    Compare<float, CMPMODE::GT>(active, s, zero, full);
    Select(s, s, one, active);
    FastReciprocal(inverse, s, full);
    Select(inverse, inverse, zero, active);
    StoreAlign(inverseScales, inverse, full);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> m0, m1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m0, inverseScales + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m1, inverseScales + row + 1);
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
        }
    } else {
        // 4-way unrolling for DP == 64 in Phase 2
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> m0, m1, m2, m3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m0, inverseScales + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m1, inverseScales + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m2, inverseScales + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(m3, inverseScales + row + 3);

            RegTensor<float> a0, a1, a2, a3;
            RegTensor<int32_t> p0, p1, p2, p3;
            LoadAlign(a0, x + (row + 0) * DP);
            LoadAlign(a1, x + (row + 1) * DP);
            LoadAlign(a2, x + (row + 2) * DP);
            LoadAlign(a3, x + (row + 3) * DP);

            Mul(a0, a0, m0, full);
            Mul(a1, a1, m1, full);
            Mul(a2, a2, m2, full);
            Mul(a3, a3, m3, full);

            Cast<int32_t, float, trait>(p0, a0, full);
            Cast<int32_t, float, trait>(p1, a1, full);
            Cast<int32_t, float, trait>(p2, a2, full);
            Cast<int32_t, float, trait>(p3, a3, full);

            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 0) * DP, (RegTensor<int8_t> &)p0, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 1) * DP, (RegTensor<int8_t> &)p1, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 2) * DP, (RegTensor<int8_t> &)p2, full);
            StoreAlign<int8_t, StoreDist::DIST_PACK4_B32>(out + (row + 3) * DP, (RegTensor<int8_t> &)p3, full);
        }
    }

    PackFixedScaleBlock(scales, halfScales, factor, valid);
}

__aicore__ inline void QuantRowsVF(__ubuf__ float *x, __ubuf__ float *mean, __ubuf__ float *scales,
                                   __ubuf__ int8_t *out, __ubuf__ half *halfScales, float factor,
                                   __ubuf__ float *inverseScales, uint32_t dp,
                                   uint32_t smooth, uint32_t valid, uint32_t group) {
    if (dp == 128) {
        if (smooth)
            QuantRowsImpl<128, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        else
            QuantRowsImpl<128, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
    } else {
        if (smooth)
            QuantRowsImpl<64, true>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
        else
            QuantRowsImpl<64, false>(x, mean, scales, out, halfScales, factor, inverseScales, valid, group);
    }
}

// MXFP8 Value Quantization with 4-way unrolling
template <class T, bool PERMUTE_V>
__simd_vf__ inline void QuantVRowsMxVF(__ubuf__ T *x, __ubuf__ T *maxima, __ubuf__ uint16_t *scales,
                                       __ubuf__ float8_e4m3_t *out, uint32_t dp) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    uint32_t count = dp;
    auto h = UpdateMask<T>(count);
    RegTensor<uint16_t> sign;
    Duplicate(sign, FP16_ABS_MASK);

    for (uint16_t group = 0; group < 2; ++group) {
        RegTensor<T> mx0, mx1, mx2, mx3;
        Duplicate(mx0, T(0));
        Duplicate(mx1, T(0));
        Duplicate(mx2, T(0));
        Duplicate(mx3, T(0));
        __ubuf__ T *grpBase = x + (group * 32) * dp;
        for (uint16_t row = 0; row < 32; row += 4) {
            RegTensor<T> a0, a1, a2, a3;
            LoadAlign(a0, grpBase + (row + 0) * dp);
            LoadAlign(a1, grpBase + (row + 1) * dp);
            LoadAlign(a2, grpBase + (row + 2) * dp);
            LoadAlign(a3, grpBase + (row + 3) * dp);
            And((RegTensor<uint16_t> &)a0, (RegTensor<uint16_t> &)a0, sign, h);
            And((RegTensor<uint16_t> &)a1, (RegTensor<uint16_t> &)a1, sign, h);
            And((RegTensor<uint16_t> &)a2, (RegTensor<uint16_t> &)a2, sign, h);
            And((RegTensor<uint16_t> &)a3, (RegTensor<uint16_t> &)a3, sign, h);
            Max(mx0, mx0, a0, h);
            Max(mx1, mx1, a1, h);
            Max(mx2, mx2, a2, h);
            Max(mx3, mx3, a3, h);
        }
        Max(mx0, mx0, mx1, h);
        Max(mx2, mx2, mx3, h);
        Max(mx0, mx0, mx2, h);
        StoreAlign(maxima + group * dp, mx0, h);
    }

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait toFloat{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<uint32_t> expMask, expBias;
    Duplicate(expMask, EXPONENT_MASK);
    Duplicate(expBias, RECIPROCAL_EXP_BIAS);

    for (uint16_t col = 0; col < dp; col += 64) {
        RegTensor<uint32_t> pair;
        Duplicate(pair, uint32_t(0));
        for (uint16_t group = 0; group < 2; ++group) {
            RegTensor<float> mx, inv;
            RegTensor<T> mx16;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(mx16, maxima + group * dp + col);
            Cast<float, T, toFloat>(mx, mx16, f);
            Muls(mx, mx, INV_MX_FP8_MAX, f);
            Maxs(mx, mx, MIN_POSITIVE_FLOAT, f);

            RegTensor<uint32_t> bits, rounded, exponent, inverseBits;
            Adds(bits, (RegTensor<uint32_t> &)mx, EXPONENT_ROUND_BIAS, f);
            And(rounded, bits, expMask, f);
            ShiftRights(exponent, rounded, int16_t(23), f);
            if (group == 1)
                ShiftLefts(exponent, exponent, int16_t(8), f);
            Or(pair, pair, exponent, f);

            Subs(inverseBits, expBias, rounded, f);
            inv = (RegTensor<float> &)inverseBits;

            __ubuf__ T *xBase0 = x + (group * 32) * dp + col;
            __ubuf__ T *xBase1 = x + (group * 32 + 16) * dp + col;

            if constexpr (PERMUTE_V) {
                __ubuf__ float8_e4m3_t *outBase = out + (group * 32) * dp + col;
                for (uint16_t row = 0; row < 16; row += 2) {
                    RegTensor<T> x0, x1, x2, x3;
                    RegTensor<float> a0, a1, a2, a3;
                    RegTensor<float8_e4m3_t> p0, p1, p2, p3;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x0, xBase0 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1, xBase1 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x2, xBase0 + (row + 1) * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x3, xBase1 + (row + 1) * dp);
                    Cast<float, T, toFloat>(a0, x0, f);
                    Cast<float, T, toFloat>(a1, x1, f);
                    Cast<float, T, toFloat>(a2, x2, f);
                    Cast<float, T, toFloat>(a3, x3, f);
                    Mul(a0, a0, inv, f);
                    Mul(a1, a1, inv, f);
                    Mul(a2, a2, inv, f);
                    Mul(a3, a3, inv, f);
                    Cast<float8_e4m3_t, float, cast>(p0, a0, f);
                    Cast<float8_e4m3_t, float, cast>(p1, a1, f);
                    Cast<float8_e4m3_t, float, cast>(p2, a2, f);
                    Cast<float8_e4m3_t, float, cast>(p3, a3, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + (row * 2) * dp, p0, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + (row * 2 + 1) * dp, p1, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + ((row + 1) * 2) * dp, p2, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + ((row + 1) * 2 + 1) * dp, p3, f);
                }
            } else {
                __ubuf__ float8_e4m3_t *outBase0 = out + (group * 32) * dp + col;
                __ubuf__ float8_e4m3_t *outBase1 = out + (group * 32 + 16) * dp + col;
                for (uint16_t row = 0; row < 16; row += 2) {
                    RegTensor<T> x0, x1, x2, x3;
                    RegTensor<float> a0, a1, a2, a3;
                    RegTensor<float8_e4m3_t> p0, p1, p2, p3;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x0, xBase0 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1, xBase1 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x2, xBase0 + (row + 1) * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x3, xBase1 + (row + 1) * dp);
                    Cast<float, T, toFloat>(a0, x0, f);
                    Cast<float, T, toFloat>(a1, x1, f);
                    Cast<float, T, toFloat>(a2, x2, f);
                    Cast<float, T, toFloat>(a3, x3, f);
                    Mul(a0, a0, inv, f);
                    Mul(a1, a1, inv, f);
                    Mul(a2, a2, inv, f);
                    Mul(a3, a3, inv, f);
                    Cast<float8_e4m3_t, float, cast>(p0, a0, f);
                    Cast<float8_e4m3_t, float, cast>(p1, a1, f);
                    Cast<float8_e4m3_t, float, cast>(p2, a2, f);
                    Cast<float8_e4m3_t, float, cast>(p3, a3, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase0 + row * dp, p0, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase1 + row * dp, p1, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase0 + (row + 1) * dp, p2, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase1 + (row + 1) * dp, p3, f);
                }
            }
        }
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(scales + col, (RegTensor<uint16_t> &)pair, f);
    }
}

} // namespace sage::compute
