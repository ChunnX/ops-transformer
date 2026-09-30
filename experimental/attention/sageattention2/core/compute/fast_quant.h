/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../types.h"
#include "../fast_math.h"
#if defined(__CCE__)
#include "../device_resource.h"
#endif

namespace sage::compute {

#if defined(__CCE__)
using AscendC::RoundMode;
using AscendC::CMPMODE;
#endif

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
            if constexpr (SMOOTH) {
                StoreAlign(x + row * DP + 64, b0, full);
                StoreAlign(x + (row + 1) * DP + 64, b1, full);
            }
            Max(m0, ab0, bb0, full);
            Max(m1, ab1, bb1, full);
            ReduceDataBlock<ReduceType::MAX>(m0, m0, full);
            ReduceDataBlock<ReduceType::MAX>(m1, m1, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m0, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, m1, compact, 1);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> a0, a1, a2, a3;
            RegTensor<float> ab0, ab1, ab2, ab3;
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
            ReduceDataBlock<ReduceType::MAX>(ab0, ab0, full);
            ReduceDataBlock<ReduceType::MAX>(ab1, ab1, full);
            ReduceDataBlock<ReduceType::MAX>(ab2, ab2, full);
            ReduceDataBlock<ReduceType::MAX>(ab3, ab3, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab0, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab1, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab2, compact, 1);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(dst, ab3, compact, 1);
        }
    }

    StoreUnAlignPost(dst, compact, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    BroadcastGroupMaxima(scales, group);

    RegTensor<float> m;
    LoadAlign(m, scales);
    Mul(m, m, invDivisor, full);
    MaskReg positive;
    Compare<float, CMPMODE::GT>(positive, m, zero, full);
    Select(m, m, one, positive);
    StoreAlign(scales, m, full);

    // Two-stage Taylor-Newton reciprocal
    RegTensor<float> inverse;
    FastReciprocal(inverse, m, full);
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

            Sub(inverseBits, expBias, rounded, f);
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

// Smooth-V sequence mean subtraction
template <class T, uint32_t DP>
__simd_vf__ inline void CenterVRowsImpl(__ubuf__ T *v, __ubuf__ float *meanV, uint32_t rows) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait toFloat{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait back{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t col = 0; col < DP; col += 64) {
        RegTensor<float> mv;
        LoadAlign(mv, meanV + col);
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<T> raw, packed;
            RegTensor<float> x;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, v + row * DP + col);
            Cast<float, T, toFloat>(x, raw, f);
            Sub(x, x, mv, f);
            Cast<T, float, back>(packed, x, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(v + row * DP + col, packed, f);
        }
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

template <class T>
__aicore__ inline void CenterVRowsVF(__ubuf__ T *v, __ubuf__ float *meanV, uint32_t dp, uint32_t rows) {
    if (dp == 128)
        CenterVRowsImpl<T, 128>(v, meanV, rows);
    else
        CenterVRowsImpl<T, 64>(v, meanV, rows);
}

// 4-way unrolled reduction for sequence-level channel stats
template <class T, uint32_t DP>
__simd_callee__ inline void LoadLaneF32(AscendC::MicroAPI::RegTensor<float> &x, __ubuf__ T *addr, uint32_t row,
                                        uint32_t col) {
    using namespace AscendC::MicroAPI;
    auto mask = CreateMask<float, MaskPattern::ALL>();
    RegTensor<T> raw;
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, addr + row * DP + col);
    Cast<float, T, cast>(x, raw, mask);
}

template <class T, uint32_t DP>
__simd_vf__ inline void StatsSumImpl(__ubuf__ T *rows, __ubuf__ float *state, uint32_t count) {
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
        for (; row + 3 < count; row += 4) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<T> raw0, raw1, raw2, raw3;
            static constexpr CastTrait cast{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
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
        for (; row + 1 < count; row += 2) {
            RegTensor<float> x0, x1;
            LoadLaneF32<T, DP>(x0, rows, row, col);
            LoadLaneF32<T, DP>(x1, rows, row + 1, col);
            Add(acc0, acc0, x0, mask);
            Add(acc1, acc1, x1, mask);
        }
        if (row < count) {
            RegTensor<float> x_tail;
            LoadLaneF32<T, DP>(x_tail, rows, row, col);
            Add(acc0, acc0, x_tail, mask);
        }
        Add(acc0, acc0, acc1, mask);
        Add(acc2, acc2, acc3, mask);
        Add(acc0, acc0, acc2, mask);
        StoreAlign(state + col, acc0, mask);
    }
}

template <class T>
__aicore__ inline void StatsSumVF(AscendC::LocalTensor<T> rows, AscendC::LocalTensor<float> state, uint32_t count, uint32_t dp) {
    auto input = (__ubuf__ T *)rows.GetPhyAddr();
    auto sum = (__ubuf__ float *)state.GetPhyAddr();
    if (dp == 128)
        StatsSumImpl<T, 128>(input, sum, count);
    else
        StatsSumImpl<T, 64>(input, sum, count);
}

// Clean-Room FastQuant Engine encapsulating Stats & Quant microkernels
template <class T, bool WIDE>
class FastQuant {
public:
    __aicore__ static inline void ProcessStats(GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, const SageParams &p) {
        if (p.smoothK) {
            Stats<false>(k, mean, p);
        }
        if (p.smoothV) {
            Stats<true>(v, vMean, p);
        }
    }

    template <bool IS_V>
    __aicore__ static void Stats(GM_ADDR src, GM_ADDR mean, const SageParams &p) {
        using namespace AscendC;
        const uint64_t *const strides = IS_V ? p.vs : p.ks;
        const bool center = IS_V ? bool(p.smoothV) : bool(p.smoothK);
        if (!center) return;

        const bool grouped = UseDieGroups(p);
        const uint32_t workers = grouped ? p.cores : 2 * p.cores;
        const uint64_t heads = uint64_t(p.b) * p.hk / (grouped ? 2 : 1);
        FixedResource r;
        GlobalTensor<T> kg;
        kg.SetGlobalBuffer((__gm__ T *)src);
        GlobalTensor<float> mg;
        mg.SetGlobalBuffer((__gm__ float *)mean);

        constexpr uint32_t STATS_BATCH_ROWS = 128, STATS_DP_MAX = 128;
        constexpr uint32_t STATS_SLOT_BYTES = STATS_BATCH_ROWS * STATS_DP_MAX * 2U;
        constexpr uint32_t STATS_F32_BYTES = STATS_BATCH_ROWS * STATS_DP_MAX * 4U;
        constexpr uint32_t STATS_RAW_UB = 0, STATS_F32_UB = 2 * STATS_SLOT_BYTES;
        constexpr uint32_t STATS_SUM_UB = STATS_F32_UB + STATS_F32_BYTES;

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
            if (p.d == p.dp) {
                LocalTensor<T> slotBuf[2] = {
                    r.ubBuf.template GetBufferByByte<T>(0),
                    r.ubBuf.template GetBufferByByte<T>(STATS_SLOT_BYTES)
                };
                const uint32_t batches = (p.sk + STATS_BATCH_ROWS - 1) / STATS_BATCH_ROWS;
                SetFlag<HardEvent::V_MTE2>(1);
                SetFlag<HardEvent::V_MTE2>(2);
                for (uint32_t batch = 0; batch <= batches; ++batch) {
                    if (batch < batches) {
                        const uint32_t slot = batch & 1, start = batch * STATS_BATCH_ROWS;
                        const uint32_t rows = Min(STATS_BATCH_ROWS, p.sk - start);
                        const event_t event = static_cast<event_t>(1 + slot);
                        WaitFlag<HardEvent::V_MTE2>(event);
                        DataCopy(slotBuf[slot], kg[Offset(strides, b, h, start)], rows * p.dp);
                        SetFlag<HardEvent::MTE2_V>(event);
                    }
                    if (batch == 0) continue;
                    const uint32_t slot = (batch - 1) & 1;
                    const uint32_t rows = Min(STATS_BATCH_ROWS, p.sk - (batch - 1) * STATS_BATCH_ROWS);
                    const event_t event = static_cast<event_t>(1 + slot);
                    WaitFlag<HardEvent::MTE2_V>(event);
                    StatsSumVF(slotBuf[slot], sum, rows, p.dp);
                    SetFlag<HardEvent::V_MTE2>(event);
                }
                WaitFlag<HardEvent::V_MTE2>(1);
                WaitFlag<HardEvent::V_MTE2>(2);
            } else {
                for (uint32_t row = 0; row < p.sk; ++row) {
                    ReadRow(input, kg, Offset(strides, b, h, row), p.d, p.dp);
                    Cast(x, input, RoundMode::CAST_NONE, p.dp);
                    PipeBarrier<PIPE_V>();
                    Add(sum, sum, x, p.dp);
                }
            }
            PipeBarrier<PIPE_V>();
            Muls(sum, sum, invSk, p.dp);
            WriteRow(mg, bh * p.dp, sum, p.dp);
        }
    }

    __aicore__ static void QuantBatch(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        using namespace AscendC;
        FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer((__gm__ T *)q);
        kg.SetGlobalBuffer((__gm__ T *)k);
        vg.SetGlobalBuffer((__gm__ T *)v);
        qg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        vg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer((__gm__ float *)mean);
        vmg.SetGlobalBuffer((__gm__ float *)vMean);
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer((__gm__ half *)qs);
        ksg.SetGlobalBuffer((__gm__ half *)ks);
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer((__gm__ int8_t *)q8);
        ko.SetGlobalBuffer((__gm__ int8_t *)k8);
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);

        constexpr uint32_t SLOT_BYTES = 96 * 1024;
        constexpr uint32_t V_INPUT = 16384, FLOAT_INPUT = 32768;
        constexpr uint32_t QK_OUTPUT = 65536, V_OUTPUT = 73728;
        constexpr uint32_t MEAN = 81920, MX_SCALE = 82432, RAW_SCALE = 82944, HALF_SCALE = 83200;
        constexpr uint32_t INVERSE_SCALE = 83456, V_MEAN = 83712;

        LocalTensor<T> input[2], vInput[2];
        LocalTensor<float> x[2], mu[2], vmean[2], vs[2], scales[2], inverseScales[2];
        LocalTensor<int8_t> out[2];
        LocalTensor<float8_e4m3_t> fp8[2];
        LocalTensor<half> halfScales[2];
        for (uint32_t s = 0; s < 2; ++s) {
            const uint32_t b = s * SLOT_BYTES;
            input[s] = r.ubBuf.template GetBufferByByte<T>(b);
            vInput[s] = r.ubBuf.template GetBufferByByte<T>(b + V_INPUT);
            x[s] = r.ubBuf.template GetBufferByByte<float>(b + FLOAT_INPUT);
            mu[s] = r.ubBuf.template GetBufferByByte<float>(b + MEAN);
            vmean[s] = r.ubBuf.template GetBufferByByte<float>(b + V_MEAN);
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
        const uint64_t localNq = WIDE ? 0 : (nq / (grouped ? 2 : 1)), localNk = nk / (grouped ? 2 : 1);
        if (core >= localNq + localNk) return;
        const uint64_t iterations = (localNq + localNk - core + cores - 1) / cores;

        SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::V_MTE2>(2);
        struct BatchInfo {
            uint64_t idx;
            bool isq;
            uint32_t valid;
        } pending[2];

        for (uint64_t iteration = 0; iteration <= iterations; ++iteration) {
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
                    if (p.smoothV)
                        DataCopy(vmean[slot], vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                }
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(1 + slot));
            }
            if (iteration == 0) continue;
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
                        (__ubuf__ half *)halfScales[slot].GetPhyAddr(), info.isq ? p.qFactor : p.kFactor,
                        (__ubuf__ float *)inverseScales[slot].GetPhyAddr(), p.dp, !info.isq && p.smoothK && info.valid,
                        info.valid, info.isq ? p.qg : p.kg);
            if (!info.isq) {
                PipeBarrier<PIPE_V>();
                if (info.valid < 64)
                    Duplicate(vInput[slot][info.valid * p.dp], T(0), (64 - info.valid) * p.dp);
                PipeBarrier<PIPE_V>();
                if (p.smoothV && info.valid) {
                    CenterVRowsVF<T>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
                                     (__ubuf__ float *)vmean[slot].GetPhyAddr(), p.dp, info.valid);
                    PipeBarrier<PIPE_V>();
                }
                QuantVRowsMxVF<T, WIDE>((__ubuf__ T *)vInput[slot].GetPhyAddr(),
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

    __aicore__ static void QuantFallback(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        using namespace AscendC;
        FixedResource r;
        GlobalTensor<T> qg, kg, vg;
        qg.SetGlobalBuffer((__gm__ T *)q);
        kg.SetGlobalBuffer((__gm__ T *)k);
        vg.SetGlobalBuffer((__gm__ T *)v);
        GlobalTensor<float> mg, vmg;
        mg.SetGlobalBuffer((__gm__ float *)mean);
        vmg.SetGlobalBuffer((__gm__ float *)vMean);
        GlobalTensor<half> qsg, ksg;
        qsg.SetGlobalBuffer((__gm__ half *)qs);
        ksg.SetGlobalBuffer((__gm__ half *)ks);
        GlobalTensor<int8_t> qo, ko;
        qo.SetGlobalBuffer((__gm__ int8_t *)q8);
        ko.SetGlobalBuffer((__gm__ int8_t *)k8);
        GlobalTensor<float8_e4m3_t> vo;
        vo.SetGlobalBuffer((__gm__ float8_e4m3_t *)v8);

        constexpr uint32_t X = 16384, MU = 49152, RAW_SCALE = 49664, INVERSE_SCALE = 49920;
        constexpr uint32_t HALF_SCALE = 50176, QK_OUTPUT = 50304;
        constexpr uint32_t V_INPUT = 65536, V_MAXIMA = 81920, V_OUTPUT = 82944, MX_SCALE = 91136;
        constexpr uint32_t V_MEAN = 92160;

        auto input = r.ubBuf.template GetBufferByByte<T>(0);
        auto x = r.ubBuf.template GetBufferByByte<float>(X);
        auto mu = r.ubBuf.template GetBufferByByte<float>(MU);
        auto scales = r.ubBuf.template GetBufferByByte<float>(RAW_SCALE);
        auto inverseScales = r.ubBuf.template GetBufferByByte<float>(INVERSE_SCALE);
        auto halfScales = r.ubBuf.template GetBufferByByte<half>(HALF_SCALE);
        auto out = r.ubBuf.template GetBufferByByte<int8_t>(QK_OUTPUT);

        const uint64_t nq = WIDE ? 0 : (uint64_t(p.b) * p.hq * p.qp);
        const uint64_t nk = uint64_t(p.b) * p.hk * p.kp;

        for (uint64_t tile = GetBlockIdx(); tile < (nq + nk) / 64; tile += 2 * p.cores) {
            const bool iskv = tile < nk / 64;
            const bool isq = !iskv;
            const uint64_t idx = (iskv ? tile : tile - nk / 64) * 64;
            const uint32_t seq = isq ? p.qp : p.kp, heads = isq ? p.hq : p.hk;
            const uint32_t token = idx % seq, h = (idx / seq) % heads, b = idx / seq / heads;
            const uint32_t limit = isq ? p.sq : p.sk;
            const uint32_t valid = token < limit ? Min(uint32_t(64), limit - token) : 0;
            for (uint32_t row = 0; row < valid; ++row)
                ReadRow(input[row * p.dp], isq ? qg : kg, Offset(isq ? p.qs : p.ks, b, h, token + row), p.d, p.dp);
            if (valid < 64)
                Duplicate(input[valid * p.dp], T(0), (64 - valid) * p.dp);
            Cast(x, input, RoundMode::CAST_NONE, 64 * p.dp);
            PipeBarrier<PIPE_V>();
            if (!isq && p.smoothK && valid) {
                DataCopy(mu, mg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
            }
            QuantRowsVF((__ubuf__ float *)x.GetPhyAddr(), (__ubuf__ float *)mu.GetPhyAddr(),
                        (__ubuf__ float *)scales.GetPhyAddr(), (__ubuf__ int8_t *)out.GetPhyAddr(),
                        (__ubuf__ half *)halfScales.GetPhyAddr(), isq ? p.qFactor : p.kFactor,
                        (__ubuf__ float *)inverseScales.GetPhyAddr(), p.dp, !isq && p.smoothK && valid, valid,
                        isq ? p.qg : p.kg);
            WriteRow(isq ? qo : ko, idx * p.dp, out, 64 * p.dp);
            WriteRow(isq ? qsg : ksg, idx, halfScales, 64);
        }

        PipeBarrier<PIPE_ALL>();
        GlobalTensor<uint16_t> mxg;
        mxg.SetGlobalBuffer((__gm__ uint16_t *)mxScale);
        auto vInput = r.ubBuf.template GetBufferByByte<T>(V_INPUT);
        auto maxima = r.ubBuf.template GetBufferByByte<T>(V_MAXIMA);
        auto vOut = r.ubBuf.template GetBufferByByte<float8_e4m3_t>(V_OUTPUT);
        auto mx = r.ubBuf.template GetBufferByByte<uint16_t>(MX_SCALE);
        auto vmean = r.ubBuf.template GetBufferByByte<float>(V_MEAN);

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
            if (p.smoothV && valid) {
                DataCopy(vmean, vmg[(uint64_t(b) * p.hk + h) * p.dp], p.dp);
                Sync<HardEvent::MTE2_V>();
                CenterVRowsVF<T>((__ubuf__ T *)vInput.GetPhyAddr(), (__ubuf__ float *)vmean.GetPhyAddr(),
                                 p.dp, valid);
                PipeBarrier<PIPE_V>();
            }
            QuantVRowsMxVF<T, WIDE>((__ubuf__ T *)vInput.GetPhyAddr(), (__ubuf__ T *)maxima.GetPhyAddr(),
                                    (__ubuf__ uint16_t *)mx.GetPhyAddr(), (__ubuf__ float8_e4m3_t *)vOut.GetPhyAddr(),
                                    p.dp);
            WriteRow(vo, idx * p.dp, vOut, 64 * p.dp);
            WriteRow(mxg, tile * p.dp, mx, p.dp);
        }
    }

    __aicore__ static inline void Quantize(
        GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean, GM_ADDR vMean, GM_ADDR mxScale,
        GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs, GM_ADDR ks, const SageParams &p)
    {
        if (p.d == p.dp && p.sq % 64 == 0 && p.sk % 64 == 0) {
            QuantBatch(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks, p);
        } else {
            QuantFallback(q, k, v, mean, vMean, mxScale, q8, k8, v8, qs, ks, p);
        }
    }
};

} // namespace sage::compute
