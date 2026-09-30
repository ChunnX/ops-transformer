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
// smooth_v removes the per-channel sequence mean before the MX scale is
// derived. The subtraction happens in FP32 registers and only touches the
// valid rows, so the sequence padding and the channel padding stay zero.
template <class T, uint32_t DP>
__simd_vf__ inline void CenterVRowsImpl(__ubuf__ T *v, __ubuf__ float *meanV, uint32_t rows)
{
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
__aicore__ inline void CenterVRowsVF(__ubuf__ T *v, __ubuf__ float *meanV, uint32_t dp, uint32_t rows)
{
    if (dp == 128)
        CenterVRowsImpl<T, 128>(v, meanV, rows);
    else
        CenterVRowsImpl<T, 64>(v, meanV, rows);
}
// One 64-row tile contains two independent MX groups per value channel.
// Scales are stored as [channel, two consecutive sequence groups].
template <class T, bool PERMUTE_V>
__simd_vf__ inline void QuantVRowsMxVF(__ubuf__ T *x, __ubuf__ T *maxima, __ubuf__ uint16_t *scales,
                                       __ubuf__ float8_e4m3_t *out, uint32_t dp)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    uint32_t count = dp;
    auto h = UpdateMask<T>(count);
    RegTensor<uint16_t> sign;
    Duplicate(sign, FP16_ABS_MASK);
    for (uint16_t group = 0; group < 2; ++group) {
        RegTensor<T> mx0, mx1;
        Duplicate(mx0, T(0));
        Duplicate(mx1, T(0));
        for (uint16_t row = 0; row < 32; row += 2) {
            RegTensor<T> a0, a1;
            LoadAlign(a0, x + (group * 32 + row) * dp);
            LoadAlign(a1, x + (group * 32 + row + 1) * dp);
            And((RegTensor<uint16_t> &)a0, (RegTensor<uint16_t> &)a0, sign, h);
            And((RegTensor<uint16_t> &)a1, (RegTensor<uint16_t> &)a1, sign, h);
            Max(mx0, mx0, a0, h);
            Max(mx1, mx1, a1, h);
        }
        Max(mx0, mx0, mx1, h);
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
            // Round amax / 448 upward to an exact power of two. Construct
            // its reciprocal from exponent bits, without a vector divide.
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

            // Hoist VAG base pointers outside the row loop to ensure VAG hoisting compliance
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
} // namespace Sage
