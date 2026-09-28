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

// Update running denominator from single alpha slot: den = isFirst ? sum : (den * ratio + sum)
// Returns the computed den register to avoid redundant UB store-reload roundtrips.
__simd_callee__ inline AscendC::MicroAPI::RegTensor<float> UpdateWideDenominator(__ubuf__ float *denom,
                                                                                  __ubuf__ float *alpha,
                                                                                  AscendC::MicroAPI::MaskReg &f,
                                                                                  bool isFirst)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> den, sum;
    LoadAlign(sum, alpha + 64);
    if (isFirst) {
        den = sum;
    } else {
        RegTensor<float> ratio;
        LoadAlign(den, denom);
        LoadAlign(ratio, alpha);
        MulDstAdd(den, ratio, sum, f);
    }
    StoreAlign(denom, den, f);
    return den;
}

// Update running denominator across wide pair: den = (isFirst ? sum0 : den * a0 + sum0) * a1 + sum1
__simd_callee__ inline AscendC::MicroAPI::RegTensor<float> UpdateWidePairDenominator(
    __ubuf__ float *denom, __ubuf__ float *alpha0, __ubuf__ float *alpha1, AscendC::MicroAPI::MaskReg &f,
    bool isFirst)
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> den, a, sum;
    LoadAlign(sum, alpha0 + 64);
    if (isFirst) {
        den = sum;
    } else {
        LoadAlign(den, denom);
        LoadAlign(a, alpha0);
        MulDstAdd(den, a, sum, f);
    }
    LoadAlign(a, alpha1);
    LoadAlign(sum, alpha1 + 64);
    MulDstAdd(den, a, sum, f);
    StoreAlign(denom, den, f);
    return den;
}

// Reusable tile scaler and converter: multiplies source elements by reciprocal denominators
// and casts directly to the target output format T.
// Unrolled 4 rows across columns to ensure >= 3 independent instructions between
// Load -> Mul -> Cast -> StoreAlign, eliminating all Vector ALU RAW hazard bubbles.
template <class T, uint32_t DP>
__simd_callee__ inline void ScaleAndCastTileVF(__ubuf__ T *out, __ubuf__ float *src, __ubuf__ float *invDenom)
{
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    // Loop interchange: outer loop over rows avoids reloading invDenom broadcast factors across column blocks.
    // Halves DIST_BRC_B32 UB memory traffic from 128 to 64 loads for DP128.
    for (uint16_t row = 0; row < 64; row += 4) {
        RegTensor<float> ratio0, ratio1, ratio2, ratio3;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, invDenom + row + 0);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, invDenom + row + 1);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, invDenom + row + 2);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, invDenom + row + 3);

        for (uint16_t col = 0; col < DP; col += 64) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<T> p0, p1, p2, p3;

            LoadAlign(x0, src + (row + 0) * DP + col);
            LoadAlign(x1, src + (row + 1) * DP + col);
            LoadAlign(x2, src + (row + 2) * DP + col);
            LoadAlign(x3, src + (row + 3) * DP + col);

            // 4-way independent Mul/Cast/Store eliminates RAW pipeline bubbles
            Mul(x0, x0, ratio0, f);
            Mul(x1, x1, ratio1, f);
            Mul(x2, x2, ratio2, f);
            Mul(x3, x3, ratio3, f);

            Cast<T, float, cast>(p0, x0, f);
            Cast<T, float, cast>(p1, x1, f);
            Cast<T, float, cast>(p2, x2, f);
            Cast<T, float, cast>(p3, x3, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 0) * DP + col, p0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + col, p1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 2) * DP + col, p2, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 3) * DP + col, p3, f);
        }
    }
}

// Interleaved FMA accumulation, reciprocal scaling, and casting to target format T for single tile update.
// Implements lazy softmax rescaling: when alpha == 1.0f (or ALPHA_ONE == true or alpha == nullptr),
// bypasses broadcast load and converts MulDstAdd to Add.
// Unrolled 4 rows with outer row iteration to minimize broadcast loads and eliminate RAW hazard bubbles.
template <class T, uint32_t DP, bool ALPHA_ONE = false>
__simd_callee__ inline void AccumulateAndCastTileVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                                    [[maybe_unused]] __ubuf__ float *alpha, __ubuf__ float *invDenom)
{
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < 64; row += 4) {
        RegTensor<float> inv0, inv1, inv2, inv3;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(inv0, invDenom + row + 0);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(inv1, invDenom + row + 1);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(inv2, invDenom + row + 2);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(inv3, invDenom + row + 3);

        RegTensor<float> r0, r1, r2, r3;
        if constexpr (!ALPHA_ONE) {
            if (alpha != nullptr) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r0, alpha + row + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r1, alpha + row + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r2, alpha + row + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r3, alpha + row + 3);
            }
        }

        for (uint16_t col = 0; col < DP; col += 64) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<float> y0, y1, y2, y3;
            RegTensor<T> p0, p1, p2, p3;

            LoadAlign(x0, accum + (row + 0) * DP + col);
            LoadAlign(x1, accum + (row + 1) * DP + col);
            LoadAlign(x2, accum + (row + 2) * DP + col);
            LoadAlign(x3, accum + (row + 3) * DP + col);

            LoadAlign(y0, partial + (row + 0) * DP + col);
            LoadAlign(y1, partial + (row + 1) * DP + col);
            LoadAlign(y2, partial + (row + 2) * DP + col);
            LoadAlign(y3, partial + (row + 3) * DP + col);

            if constexpr (ALPHA_ONE) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
                Add(x2, x2, y2, f);
                Add(x3, x3, y3, f);
            } else if (alpha == nullptr) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
                Add(x2, x2, y2, f);
                Add(x3, x3, y3, f);
            } else {
                MulDstAdd(x0, r0, y0, f);
                MulDstAdd(x1, r1, y1, f);
                MulDstAdd(x2, r2, y2, f);
                MulDstAdd(x3, r3, y3, f);
            }

            Mul(x0, x0, inv0, f);
            Mul(x1, x1, inv1, f);
            Mul(x2, x2, inv2, f);
            Mul(x3, x3, inv3, f);

            Cast<T, float, cast>(p0, x0, f);
            Cast<T, float, cast>(p1, x1, f);
            Cast<T, float, cast>(p2, x2, f);
            Cast<T, float, cast>(p3, x3, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 0) * DP + col, p0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + col, p1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 2) * DP + col, p2, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 3) * DP + col, p3, f);
        }
    }
}

// Two-row interleaved computation for wide pair update, eliminating RAW pipeline stall bubbles.
// Implements lazy softmax rescaling: when ALPHA0_ONE / ALPHA1_ONE == true or alpha0 / alpha1 == nullptr,
// bypasses broadcast loads and converts MulDstAdd to Add. Store ordering is reordered row-0-first
// to maximize latency hiding.
template <class T, uint32_t DP, bool FIRST, bool LAST, bool ALPHA0_ONE = false, bool ALPHA1_ONE = false>
__simd_callee__ inline void UpdateWidePairTwoRows([[maybe_unused]] __ubuf__ T *out,
                                                  [[maybe_unused]] __ubuf__ float *accum,
                                                  __ubuf__ float *partial0,
                                                  __ubuf__ float *partial1,
                                                  [[maybe_unused]] __ubuf__ float *alpha0,
                                                  [[maybe_unused]] __ubuf__ float *alpha1,
                                                  [[maybe_unused]] __ubuf__ float *scratch,
                                                  uint16_t row,
                                                  AscendC::MicroAPI::MaskReg &f)
{
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> x0_0, x1_0, x0_1, x1_1, z0_0, z1_0, z0_1, z1_1;

    if constexpr (FIRST) {
        LoadAlign(x0_0, partial0 + row * DP);
        LoadAlign(x1_0, partial0 + row * DP + 64);
        LoadAlign(x0_1, partial0 + (row + 1) * DP);
        LoadAlign(x1_1, partial0 + (row + 1) * DP + 64);
    } else {
        RegTensor<float> y0_0, y1_0, y0_1, y1_1, a0_0, a0_1;
        LoadAlign(x0_0, accum + row * DP);
        LoadAlign(x1_0, accum + row * DP + 64);
        LoadAlign(x0_1, accum + (row + 1) * DP);
        LoadAlign(x1_1, accum + (row + 1) * DP + 64);
        LoadAlign(y0_0, partial0 + row * DP);
        LoadAlign(y1_0, partial0 + row * DP + 64);
        LoadAlign(y0_1, partial0 + (row + 1) * DP);
        LoadAlign(y1_1, partial0 + (row + 1) * DP + 64);

        if constexpr (ALPHA0_ONE) {
            Add(x0_0, x0_0, y0_0, f);
            Add(x1_0, x1_0, y1_0, f);
            Add(x0_1, x0_1, y0_1, f);
            Add(x1_1, x1_1, y1_1, f);
        } else if (alpha0 == nullptr) {
            Add(x0_0, x0_0, y0_0, f);
            Add(x1_0, x1_0, y1_0, f);
            Add(x0_1, x0_1, y0_1, f);
            Add(x1_1, x1_1, y1_1, f);
        } else {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);
            MulDstAdd(x0_0, a0_0, y0_0, f);
            MulDstAdd(x0_1, a0_1, y0_1, f);
            MulDstAdd(x1_0, a0_0, y1_0, f);
            MulDstAdd(x1_1, a0_1, y1_1, f);
        }
    }

    LoadAlign(z0_0, partial1 + row * DP);
    LoadAlign(z1_0, partial1 + row * DP + 64);
    LoadAlign(z0_1, partial1 + (row + 1) * DP);
    LoadAlign(z1_1, partial1 + (row + 1) * DP + 64);

    if constexpr (ALPHA1_ONE) {
        Add(x0_0, x0_0, z0_0, f);
        Add(x1_0, x1_0, z1_0, f);
        Add(x0_1, x0_1, z0_1, f);
        Add(x1_1, x1_1, z1_1, f);
    } else if (alpha1 == nullptr) {
        Add(x0_0, x0_0, z0_0, f);
        Add(x1_0, x1_0, z1_0, f);
        Add(x0_1, x0_1, z0_1, f);
        Add(x1_1, x1_1, z1_1, f);
    } else {
        RegTensor<float> a1_0, a1_1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);
        MulDstAdd(x0_0, a1_0, z0_0, f);
        MulDstAdd(x0_1, a1_1, z0_1, f);
        MulDstAdd(x1_0, a1_0, z1_0, f);
        MulDstAdd(x1_1, a1_1, z1_1, f);
    }

    if constexpr (LAST) {
        RegTensor<float> inv0, inv1;
        RegTensor<T> p0_0, p1_0, p0_1, p1_1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(inv0, scratch + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(inv1, scratch + row + 1);
        Mul(x0_0, x0_0, inv0, f);
        Mul(x0_1, x0_1, inv1, f);
        Mul(x1_0, x1_0, inv0, f);
        Mul(x1_1, x1_1, inv1, f);
        Cast<T, float, cast>(p0_0, x0_0, f);
        Cast<T, float, cast>(p0_1, x0_1, f);
        Cast<T, float, cast>(p1_0, x1_0, f);
        Cast<T, float, cast>(p1_1, x1_1, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
    } else {
        StoreAlign(accum + row * DP, x0_0, f);
        StoreAlign(accum + row * DP + 64, x1_0, f);
        StoreAlign(accum + (row + 1) * DP, x0_1, f);
        StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
    }
}

template <uint32_t DP, bool ALPHA_ONE = false>
__simd_vf__ inline void AccumulateFp32VF(__ubuf__ float *accum, __ubuf__ float *partial,
                                         [[maybe_unused]] __ubuf__ float *alpha, uint32_t rows = 8)
{
    static_assert(DP == 64 || DP == 128, "unsupported padded dimension");
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    if constexpr (DP == 128) {
        uint32_t row = 0;
        for (; row + 3 < (rows & ~1U); row += 4) {
            RegTensor<float> x0_0, x1_0, x2_0, x3_0;
            RegTensor<float> x0_1, x1_1, x2_1, x3_1;
            RegTensor<float> y0_0, y1_0, y2_0, y3_0;
            RegTensor<float> y0_1, y1_1, y2_1, y3_1;

            LoadAlign(x0_0, accum + (row + 0) * DP);
            LoadAlign(x1_0, accum + (row + 1) * DP);
            LoadAlign(x2_0, accum + (row + 2) * DP);
            LoadAlign(x3_0, accum + (row + 3) * DP);

            LoadAlign(x0_1, accum + (row + 0) * DP + 64);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);
            LoadAlign(x2_1, accum + (row + 2) * DP + 64);
            LoadAlign(x3_1, accum + (row + 3) * DP + 64);

            LoadAlign(y0_0, partial + (row + 0) * DP);
            LoadAlign(y1_0, partial + (row + 1) * DP);
            LoadAlign(y2_0, partial + (row + 2) * DP);
            LoadAlign(y3_0, partial + (row + 3) * DP);

            LoadAlign(y0_1, partial + (row + 0) * DP + 64);
            LoadAlign(y1_1, partial + (row + 1) * DP + 64);
            LoadAlign(y2_1, partial + (row + 2) * DP + 64);
            LoadAlign(y3_1, partial + (row + 3) * DP + 64);

            if constexpr (ALPHA_ONE) {
                Add(x0_0, x0_0, y0_0, f);
                Add(x1_0, x1_0, y1_0, f);
                Add(x2_0, x2_0, y2_0, f);
                Add(x3_0, x3_0, y3_0, f);
                Add(x0_1, x0_1, y0_1, f);
                Add(x1_1, x1_1, y1_1, f);
                Add(x2_1, x2_1, y2_1, f);
                Add(x3_1, x3_1, y3_1, f);
            } else if (alpha == nullptr) {
                Add(x0_0, x0_0, y0_0, f);
                Add(x1_0, x1_0, y1_0, f);
                Add(x2_0, x2_0, y2_0, f);
                Add(x3_0, x3_0, y3_0, f);
                Add(x0_1, x0_1, y0_1, f);
                Add(x1_1, x1_1, y1_1, f);
                Add(x2_1, x2_1, y2_1, f);
                Add(x3_1, x3_1, y3_1, f);
            } else {
                RegTensor<float> ratio0, ratio1, ratio2, ratio3;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, alpha + row + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, alpha + row + 3);

                MulDstAdd(x0_0, ratio0, y0_0, f);
                MulDstAdd(x1_0, ratio1, y1_0, f);
                MulDstAdd(x2_0, ratio2, y2_0, f);
                MulDstAdd(x3_0, ratio3, y3_0, f);
                MulDstAdd(x0_1, ratio0, y0_1, f);
                MulDstAdd(x1_1, ratio1, y1_1, f);
                MulDstAdd(x2_1, ratio2, y2_1, f);
                MulDstAdd(x3_1, ratio3, y3_1, f);
            }

            StoreAlign(accum + (row + 0) * DP, x0_0, f);
            StoreAlign(accum + (row + 1) * DP, x1_0, f);
            StoreAlign(accum + (row + 2) * DP, x2_0, f);
            StoreAlign(accum + (row + 3) * DP, x3_0, f);
            StoreAlign(accum + (row + 0) * DP + 64, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
            StoreAlign(accum + (row + 2) * DP + 64, x2_1, f);
            StoreAlign(accum + (row + 3) * DP + 64, x3_1, f);
        }
        for (; row < (rows & ~1U); row += 2) {
            RegTensor<float> x0_0, y0_0, x0_1, y0_1;
            RegTensor<float> x1_0, y1_0, x1_1, y1_1;
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);
            LoadAlign(y0_0, partial + row * DP);
            LoadAlign(y0_1, partial + (row + 1) * DP);
            LoadAlign(y1_0, partial + row * DP + 64);
            LoadAlign(y1_1, partial + (row + 1) * DP + 64);

            if constexpr (ALPHA_ONE) {
                Add(x0_0, x0_0, y0_0, f);
                Add(x1_0, x1_0, y1_0, f);
                Add(x0_1, x0_1, y0_1, f);
                Add(x1_1, x1_1, y1_1, f);
            } else if (alpha == nullptr) {
                Add(x0_0, x0_0, y0_0, f);
                Add(x1_0, x1_0, y1_0, f);
                Add(x0_1, x0_1, y0_1, f);
                Add(x1_1, x1_1, y1_1, f);
            } else {
                RegTensor<float> ratio0, ratio1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);
                MulDstAdd(x0_0, ratio0, y0_0, f);
                MulDstAdd(x1_0, ratio0, y1_0, f);
                MulDstAdd(x0_1, ratio1, y0_1, f);
                MulDstAdd(x1_1, ratio1, y1_1, f);
            }

            StoreAlign(accum + row * DP, x0_0, f);
            StoreAlign(accum + row * DP + 64, x1_0, f);
            StoreAlign(accum + (row + 1) * DP, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        }
    } else {
        uint32_t row = 0;
        for (; row + 3 < (rows & ~1U); row += 4) {
            RegTensor<float> x0, x1, x2, x3, y0, y1, y2, y3;
            LoadAlign(x0, accum + (row + 0) * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);
            LoadAlign(y0, partial + (row + 0) * DP);
            LoadAlign(y1, partial + (row + 1) * DP);
            LoadAlign(y2, partial + (row + 2) * DP);
            LoadAlign(y3, partial + (row + 3) * DP);

            if constexpr (ALPHA_ONE) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
                Add(x2, x2, y2, f);
                Add(x3, x3, y3, f);
            } else if (alpha == nullptr) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
                Add(x2, x2, y2, f);
                Add(x3, x3, y3, f);
            } else {
                RegTensor<float> r0, r1, r2, r3;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r0, alpha + row + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r1, alpha + row + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r2, alpha + row + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(r3, alpha + row + 3);
                MulDstAdd(x0, r0, y0, f);
                MulDstAdd(x1, r1, y1, f);
                MulDstAdd(x2, r2, y2, f);
                MulDstAdd(x3, r3, y3, f);
            }

            StoreAlign(accum + (row + 0) * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
            StoreAlign(accum + (row + 2) * DP, x2, f);
            StoreAlign(accum + (row + 3) * DP, x3, f);
        }
        for (; row < (rows & ~1U); row += 2) {
            RegTensor<float> x0, x1, y0, y1;
            LoadAlign(x0, accum + row * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(y0, partial + row * DP);
            LoadAlign(y1, partial + (row + 1) * DP);

            if constexpr (ALPHA_ONE) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
            } else if (alpha == nullptr) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
            } else {
                RegTensor<float> ratio0, ratio1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);
                MulDstAdd(x0, ratio0, y0, f);
                MulDstAdd(x1, ratio1, y1, f);
            }

            StoreAlign(accum + row * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
        }
    }
    if (rows & 1) {
        uint32_t row = rows - 1;
        RegTensor<float> x0, y0;
        LoadAlign(x0, accum + row * DP);
        LoadAlign(y0, partial + row * DP);
        if constexpr (DP == 128) {
            RegTensor<float> x1, y1;
            LoadAlign(x1, accum + row * DP + 64);
            LoadAlign(y1, partial + row * DP + 64);
            if constexpr (ALPHA_ONE) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
            } else if (alpha == nullptr) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
            } else {
                RegTensor<float> ratio;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
                MulDstAdd(x0, ratio, y0, f);
                MulDstAdd(x1, ratio, y1, f);
            }
            StoreAlign(accum + row * DP, x0, f);
            StoreAlign(accum + row * DP + 64, x1, f);
        } else {
            if constexpr (ALPHA_ONE) {
                Add(x0, x0, y0, f);
            } else if (alpha == nullptr) {
                Add(x0, x0, y0, f);
            } else {
                RegTensor<float> ratio;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
                MulDstAdd(x0, ratio, y0, f);
            }
            StoreAlign(accum + row * DP, x0, f);
        }
    }
}

// Wide P1 publishes its FP32 block sum in the unused half of its alpha slot.
// Consume it with O in group order, after the matching PV_READY notification.
// Implements lazy softmax rescaling and row-0-first store ordering.
template <uint32_t DP, bool ALPHA_ONE = false>
__simd_vf__ inline void AccumulateWideFp32VF(__ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha,
                                             __ubuf__ float *denom)
{
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    UpdateWideDenominator(denom, alpha, f, false);
    for (uint16_t row = 0; row < 64; row += 4) {
        RegTensor<float> x0_0, x1_0, x2_0, x3_0;
        RegTensor<float> x0_1, x1_1, x2_1, x3_1;
        RegTensor<float> y0_0, y1_0, y2_0, y3_0;
        RegTensor<float> y0_1, y1_1, y2_1, y3_1;

        LoadAlign(x0_0, accum + (row + 0) * DP);
        LoadAlign(x1_0, accum + (row + 1) * DP);
        LoadAlign(x2_0, accum + (row + 2) * DP);
        LoadAlign(x3_0, accum + (row + 3) * DP);

        LoadAlign(x0_1, accum + (row + 0) * DP + 64);
        LoadAlign(x1_1, accum + (row + 1) * DP + 64);
        LoadAlign(x2_1, accum + (row + 2) * DP + 64);
        LoadAlign(x3_1, accum + (row + 3) * DP + 64);

        LoadAlign(y0_0, partial + (row + 0) * DP);
        LoadAlign(y1_0, partial + (row + 1) * DP);
        LoadAlign(y2_0, partial + (row + 2) * DP);
        LoadAlign(y3_0, partial + (row + 3) * DP);

        LoadAlign(y0_1, partial + (row + 0) * DP + 64);
        LoadAlign(y1_1, partial + (row + 1) * DP + 64);
        LoadAlign(y2_1, partial + (row + 2) * DP + 64);
        LoadAlign(y3_1, partial + (row + 3) * DP + 64);

        if constexpr (ALPHA_ONE) {
            Add(x0_0, x0_0, y0_0, f);
            Add(x1_0, x1_0, y1_0, f);
            Add(x2_0, x2_0, y2_0, f);
            Add(x3_0, x3_0, y3_0, f);
            Add(x0_1, x0_1, y0_1, f);
            Add(x1_1, x1_1, y1_1, f);
            Add(x2_1, x2_1, y2_1, f);
            Add(x3_1, x3_1, y3_1, f);
        } else if (alpha == nullptr) {
            Add(x0_0, x0_0, y0_0, f);
            Add(x1_0, x1_0, y1_0, f);
            Add(x2_0, x2_0, y2_0, f);
            Add(x3_0, x3_0, y3_0, f);
            Add(x0_1, x0_1, y0_1, f);
            Add(x1_1, x1_1, y1_1, f);
            Add(x2_1, x2_1, y2_1, f);
            Add(x3_1, x3_1, y3_1, f);
        } else {
            RegTensor<float> ratio0, ratio1, ratio2, ratio3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, alpha + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, alpha + row + 3);

            MulDstAdd(x0_0, ratio0, y0_0, f);
            MulDstAdd(x1_0, ratio1, y1_0, f);
            MulDstAdd(x2_0, ratio2, y2_0, f);
            MulDstAdd(x3_0, ratio3, y3_0, f);
            MulDstAdd(x0_1, ratio0, y0_1, f);
            MulDstAdd(x1_1, ratio1, y1_1, f);
            MulDstAdd(x2_1, ratio2, y2_1, f);
            MulDstAdd(x3_1, ratio3, y3_1, f);
        }

        StoreAlign(accum + (row + 0) * DP, x0_0, f);
        StoreAlign(accum + (row + 1) * DP, x1_0, f);
        StoreAlign(accum + (row + 2) * DP, x2_0, f);
        StoreAlign(accum + (row + 3) * DP, x3_0, f);
        StoreAlign(accum + (row + 0) * DP + 64, x0_1, f);
        StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        StoreAlign(accum + (row + 2) * DP + 64, x2_1, f);
        StoreAlign(accum + (row + 3) * DP + 64, x3_1, f);
    }
}

// First PV of a query initializes O and its denominator. The previous query
// remains live until its final Vec2; Vec1 must not clear this shared O buffer.
template <uint32_t DP>
__simd_vf__ inline void InitializeWideFp32VF(__ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha,
                                             __ubuf__ float *denom)
{
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> value;
    LoadAlign(value, alpha + 64);
    StoreAlign(denom, value, f);
    for (uint16_t offset = 0; offset < 64 * DP; offset += 64) {
        LoadAlign(value, partial + offset);
        StoreAlign(accum + offset, value, f);
    }
}

template <uint32_t DP, bool ALPHA0_ONE = false, bool ALPHA1_ONE = false>
__simd_vf__ inline void AccumulateFp32PairVF(__ubuf__ float *accum, __ubuf__ float *partial0, __ubuf__ float *partial1,
                                             [[maybe_unused]] __ubuf__ float *alpha0,
                                             [[maybe_unused]] __ubuf__ float *alpha1)
{
    static_assert(DP == 64 || DP == 128, "paired O requires a supported padded dimension");
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            RegTensor<float> y0_0, y1_0, y0_1, y1_1;
            RegTensor<float> z0_0, z1_0, z0_1, z1_1;

            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            LoadAlign(y0_0, partial0 + row * DP);
            LoadAlign(y1_0, partial0 + row * DP + 64);
            LoadAlign(y0_1, partial0 + (row + 1) * DP);
            LoadAlign(y1_1, partial0 + (row + 1) * DP + 64);

            // Phase 1: FMA or Add with alpha0 and partial0
            if constexpr (ALPHA0_ONE) {
                Add(x0_0, x0_0, y0_0, f);
                Add(x1_0, x1_0, y1_0, f);
                Add(x0_1, x0_1, y0_1, f);
                Add(x1_1, x1_1, y1_1, f);
            } else if (alpha0 == nullptr) {
                Add(x0_0, x0_0, y0_0, f);
                Add(x1_0, x1_0, y1_0, f);
                Add(x0_1, x0_1, y0_1, f);
                Add(x1_1, x1_1, y1_1, f);
            } else {
                RegTensor<float> a0_0, a0_1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);
                MulDstAdd(x0_0, a0_0, y0_0, f);
                MulDstAdd(x1_0, a0_0, y1_0, f);
                MulDstAdd(x0_1, a0_1, y0_1, f);
                MulDstAdd(x1_1, a0_1, y1_1, f);
            }

            LoadAlign(z0_0, partial1 + row * DP);
            LoadAlign(z1_0, partial1 + row * DP + 64);
            LoadAlign(z0_1, partial1 + (row + 1) * DP);
            LoadAlign(z1_1, partial1 + (row + 1) * DP + 64);

            // Phase 2: FMA or Add with alpha1 and partial1
            if constexpr (ALPHA1_ONE) {
                Add(x0_0, x0_0, z0_0, f);
                Add(x1_0, x1_0, z1_0, f);
                Add(x0_1, x0_1, z0_1, f);
                Add(x1_1, x1_1, z1_1, f);
            } else if (alpha1 == nullptr) {
                Add(x0_0, x0_0, z0_0, f);
                Add(x1_0, x1_0, z1_0, f);
                Add(x0_1, x0_1, z0_1, f);
                Add(x1_1, x1_1, z1_1, f);
            } else {
                RegTensor<float> a1_0, a1_1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);
                MulDstAdd(x0_0, a1_0, z0_0, f);
                MulDstAdd(x1_0, a1_0, z1_0, f);
                MulDstAdd(x0_1, a1_1, z0_1, f);
                MulDstAdd(x1_1, a1_1, z1_1, f);
            }

            StoreAlign(accum + row * DP, x0_0, f);
            StoreAlign(accum + row * DP + 64, x1_0, f);
            StoreAlign(accum + (row + 1) * DP, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        }
    } else {
        uint16_t row = 0;
        for (; row + 3 < 64; row += 4) {
            RegTensor<float> x0, x1, x2, x3;
            RegTensor<float> y0, y1, y2, y3;
            RegTensor<float> z0, z1, z2, z3;

            LoadAlign(x0, accum + (row + 0) * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            LoadAlign(y0, partial0 + (row + 0) * DP);
            LoadAlign(y1, partial0 + (row + 1) * DP);
            LoadAlign(y2, partial0 + (row + 2) * DP);
            LoadAlign(y3, partial0 + (row + 3) * DP);

            if constexpr (ALPHA0_ONE) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
                Add(x2, x2, y2, f);
                Add(x3, x3, y3, f);
            } else if (alpha0 == nullptr) {
                Add(x0, x0, y0, f);
                Add(x1, x1, y1, f);
                Add(x2, x2, y2, f);
                Add(x3, x3, y3, f);
            } else {
                RegTensor<float> a0_0, a0_1, a0_2, a0_3;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_2, alpha0 + row + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_3, alpha0 + row + 3);
                MulDstAdd(x0, a0_0, y0, f);
                MulDstAdd(x1, a0_1, y1, f);
                MulDstAdd(x2, a0_2, y2, f);
                MulDstAdd(x3, a0_3, y3, f);
            }

            LoadAlign(z0, partial1 + (row + 0) * DP);
            LoadAlign(z1, partial1 + (row + 1) * DP);
            LoadAlign(z2, partial1 + (row + 2) * DP);
            LoadAlign(z3, partial1 + (row + 3) * DP);

            if constexpr (ALPHA1_ONE) {
                Add(x0, x0, z0, f);
                Add(x1, x1, z1, f);
                Add(x2, x2, z2, f);
                Add(x3, x3, z3, f);
            } else if (alpha1 == nullptr) {
                Add(x0, x0, z0, f);
                Add(x1, x1, z1, f);
                Add(x2, x2, z2, f);
                Add(x3, x3, z3, f);
            } else {
                RegTensor<float> a1_0, a1_1, a1_2, a1_3;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_2, alpha1 + row + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_3, alpha1 + row + 3);
                MulDstAdd(x0, a1_0, z0, f);
                MulDstAdd(x1, a1_1, z1, f);
                MulDstAdd(x2, a1_2, z2, f);
                MulDstAdd(x3, a1_3, z3, f);
            }

            StoreAlign(accum + (row + 0) * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
            StoreAlign(accum + (row + 2) * DP, x2, f);
            StoreAlign(accum + (row + 3) * DP, x3, f);
        }
        for (; row < 64; ++row) {
            RegTensor<float> x0, y0, z0;
            LoadAlign(x0, accum + row * DP);
            LoadAlign(y0, partial0 + row * DP);
            LoadAlign(z0, partial1 + row * DP);

            if constexpr (ALPHA0_ONE) {
                Add(x0, x0, y0, f);
            } else if (alpha0 == nullptr) {
                Add(x0, x0, y0, f);
            } else {
                RegTensor<float> a0;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
                MulDstAdd(x0, a0, y0, f);
            }

            if constexpr (ALPHA1_ONE) {
                Add(x0, x0, z0, f);
            } else if (alpha1 == nullptr) {
                Add(x0, x0, z0, f);
            } else {
                RegTensor<float> a1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
                MulDstAdd(x0, a1, z0, f);
            }

            StoreAlign(accum + row * DP, x0, f);
        }
    }
}

__simd_vf__ inline void NormalizeVF(__ubuf__ float *accum, __ubuf__ float *denom, uint32_t dp, uint32_t rows,
                                    __ubuf__ float *scratch = nullptr)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    if (scratch != nullptr) {
        RegTensor<float> den, inv;
        LoadAlign(den, denom);
        FastReciprocal(inv, den, full);
        StoreAlign(scratch, inv, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if (dp == 128) {
            uint32_t i = 0;
            for (; i + 3 < (rows & ~1U); i += 4) {
                RegTensor<float> ratio0, ratio1, ratio2, ratio3;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + i + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + i + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, scratch + i + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, scratch + i + 3);
                RegTensor<float> x0_0, x1_0, x2_0, x3_0;
                RegTensor<float> x0_1, x1_1, x2_1, x3_1;
                LoadAlign(x0_0, accum + (i + 0) * dp + 0);
                LoadAlign(x1_0, accum + (i + 1) * dp + 0);
                LoadAlign(x2_0, accum + (i + 2) * dp + 0);
                LoadAlign(x3_0, accum + (i + 3) * dp + 0);
                LoadAlign(x0_1, accum + (i + 0) * dp + 64);
                LoadAlign(x1_1, accum + (i + 1) * dp + 64);
                LoadAlign(x2_1, accum + (i + 2) * dp + 64);
                LoadAlign(x3_1, accum + (i + 3) * dp + 64);

                Mul(x0_0, x0_0, ratio0, full);
                Mul(x1_0, x1_0, ratio1, full);
                Mul(x2_0, x2_0, ratio2, full);
                Mul(x3_0, x3_0, ratio3, full);
                Mul(x0_1, x0_1, ratio0, full);
                Mul(x1_1, x1_1, ratio1, full);
                Mul(x2_1, x2_1, ratio2, full);
                Mul(x3_1, x3_1, ratio3, full);

                StoreAlign(accum + (i + 0) * dp + 0, x0_0, full);
                StoreAlign(accum + (i + 1) * dp + 0, x1_0, full);
                StoreAlign(accum + (i + 2) * dp + 0, x2_0, full);
                StoreAlign(accum + (i + 3) * dp + 0, x3_0, full);
                StoreAlign(accum + (i + 0) * dp + 64, x0_1, full);
                StoreAlign(accum + (i + 1) * dp + 64, x1_1, full);
                StoreAlign(accum + (i + 2) * dp + 64, x2_1, full);
                StoreAlign(accum + (i + 3) * dp + 64, x3_1, full);
            }
            for (; i < (rows & ~1U); i += 2) {
                RegTensor<float> ratio0, ratio1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + i + 1);
                RegTensor<float> x0_0, x0_1, x1_0, x1_1;
                LoadAlign(x0_0, accum + i * dp + 0);
                LoadAlign(x0_1, accum + (i + 1) * dp + 0);
                LoadAlign(x1_0, accum + i * dp + 64);
                LoadAlign(x1_1, accum + (i + 1) * dp + 64);
                Mul(x0_0, x0_0, ratio0, full);
                Mul(x0_1, x0_1, ratio1, full);
                Mul(x1_0, x1_0, ratio0, full);
                Mul(x1_1, x1_1, ratio1, full);
                StoreAlign(accum + i * dp + 0, x0_0, full);
                StoreAlign(accum + i * dp + 64, x1_0, full);
                StoreAlign(accum + (i + 1) * dp + 0, x0_1, full);
                StoreAlign(accum + (i + 1) * dp + 64, x1_1, full);
            }
        } else {
            uint32_t i = 0;
            for (; i + 3 < rows; i += 4) {
                RegTensor<float> ratio0, ratio1, ratio2, ratio3;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + i + 0);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + i + 1);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, scratch + i + 2);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, scratch + i + 3);
                RegTensor<float> x0, x1, x2, x3;
                LoadAlign(x0, accum + (i + 0) * dp);
                LoadAlign(x1, accum + (i + 1) * dp);
                LoadAlign(x2, accum + (i + 2) * dp);
                LoadAlign(x3, accum + (i + 3) * dp);
                Mul(x0, x0, ratio0, full);
                Mul(x1, x1, ratio1, full);
                Mul(x2, x2, ratio2, full);
                Mul(x3, x3, ratio3, full);
                StoreAlign(accum + (i + 0) * dp, x0, full);
                StoreAlign(accum + (i + 1) * dp, x1, full);
                StoreAlign(accum + (i + 2) * dp, x2, full);
                StoreAlign(accum + (i + 3) * dp, x3, full);
            }
            for (; i < (rows & ~1U); i += 2) {
                RegTensor<float> ratio0, ratio1;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + i + 1);
                RegTensor<float> x0, x1;
                LoadAlign(x0, accum + i * dp);
                LoadAlign(x1, accum + (i + 1) * dp);
                Mul(x0, x0, ratio0, full);
                Mul(x1, x1, ratio1, full);
                StoreAlign(accum + i * dp, x0, full);
                StoreAlign(accum + (i + 1) * dp, x1, full);
            }
        }
        if (rows & 1) {
            uint32_t i = rows - 1;
            RegTensor<float> ratio, x;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, scratch + i);
            for (uint32_t col = 0; col < dp; col += 64) {
                LoadAlign(x, accum + i * dp + col);
                Mul(x, x, ratio, full);
                StoreAlign(accum + i * dp + col, x, full);
            }
        }
    } else {
        for (uint32_t i = 0; i < rows; ++i) {
            RegTensor<float> den, ratio;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(den, denom + i);
            FastReciprocal(ratio, den, full);
            for (uint32_t col = 0; col < dp; col += 64) {
                RegTensor<float> x;
                LoadAlign(x, accum + i * dp + col);
                Mul(x, x, ratio, full);
                StoreAlign(accum + i * dp + col, x, full);
            }
        }
    }
}

// Finalize the complete 64-row wide tile. Compute all reciprocals together,
// and cast directly into the output buffer.
template <class T, uint32_t DP>
__simd_vf__ inline void NormalizeCastVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                        __ubuf__ float *scratch)
{
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, inv;
    LoadAlign(den, denom);
    FastReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    ScaleAndCastTileVF<T, DP>(out, accum, scratch);
}

// The last PV never needs to store the updated FP32 O back to UB. Keep the
// FMA -> reciprocal multiply -> cast rounding sequence.
// Bypasses denom reload by directly using the returned den register.
template <class T, uint32_t DP, bool FIRST, bool ALPHA_ONE = false>
__simd_vf__ inline void FinalizeWideFp32VF(__ubuf__ T *out, [[maybe_unused]] __ubuf__ float *accum,
                                           __ubuf__ float *partial, __ubuf__ float *alpha, __ubuf__ float *denom,
                                           __ubuf__ float *scratch)
{
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den = UpdateWideDenominator(denom, alpha, f, FIRST);
    RegTensor<float> inv;
    FastReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    if constexpr (FIRST) {
        ScaleAndCastTileVF<T, DP>(out, partial, scratch);
    } else {
        AccumulateAndCastTileVF<T, DP, ALPHA_ONE>(out, accum, partial, alpha, scratch);
    }
}

// Two consecutive groups of the same query, preserving both FP32 FMA roundings.
template <class T, uint32_t DP, bool FIRST, bool LAST, bool ALPHA0_ONE = false, bool ALPHA1_ONE = false>
__simd_vf__ inline void UpdateWidePairVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                         __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                         __ubuf__ float *denom, __ubuf__ float *scratch)
{
    using namespace AscendC::MicroAPI;
    static_assert(DP == 128, "paired wide update requires DP128");
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den = UpdateWidePairDenominator(denom, alpha0, alpha1, f, FIRST);
    if constexpr (LAST) {
        RegTensor<float> inv;
        FastReciprocal(inv, den, f);
        StoreAlign(scratch, inv, f);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    for (uint16_t row = 0; row < 64; row += 2) {
        UpdateWidePairTwoRows<T, DP, FIRST, LAST, ALPHA0_ONE, ALPHA1_ONE>(
            out, accum, partial0, partial1, alpha0, alpha1, scratch, row, f);
    }
}

} // namespace Sage
