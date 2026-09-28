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
            RegTensor<float> a, mx, inv;
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
                for (uint16_t row = 0; row < 16; ++row) {
                    RegTensor<T> x0, x1;
                    RegTensor<float> a0, a1;
                    RegTensor<float8_e4m3_t> p0, p1;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x0, xBase0 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1, xBase1 + row * dp);
                    Cast<float, T, toFloat>(a0, x0, f);
                    Cast<float, T, toFloat>(a1, x1, f);
                    Mul(a0, a0, inv, f);
                    Mul(a1, a1, inv, f);
                    Cast<float8_e4m3_t, float, cast>(p0, a0, f);
                    Cast<float8_e4m3_t, float, cast>(p1, a1, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + (row * 2) * dp, p0, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase + (row * 2 + 1) * dp, p1, f);
                }
            } else {
                __ubuf__ float8_e4m3_t *outBase0 = out + (group * 32) * dp + col;
                __ubuf__ float8_e4m3_t *outBase1 = out + (group * 32 + 16) * dp + col;
                for (uint16_t row = 0; row < 16; ++row) {
                    RegTensor<T> x0, x1;
                    RegTensor<float> a0, a1;
                    RegTensor<float8_e4m3_t> p0, p1;
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x0, xBase0 + row * dp);
                    LoadAlign<T, LoadDist::DIST_UNPACK_B16>(x1, xBase1 + row * dp);
                    Cast<float, T, toFloat>(a0, x0, f);
                    Cast<float, T, toFloat>(a1, x1, f);
                    Mul(a0, a0, inv, f);
                    Mul(a1, a1, inv, f);
                    Cast<float8_e4m3_t, float, cast>(p0, a0, f);
                    Cast<float8_e4m3_t, float, cast>(p1, a1, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase0 + row * dp, p0, f);
                    StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(outBase1 + row * dp, p1, f);
                }
            }
        }
        StoreAlign<uint16_t, StoreDist::DIST_PACK_B32>(scales + col, (RegTensor<uint16_t> &)pair, f);
    }
}
} // namespace Sage
