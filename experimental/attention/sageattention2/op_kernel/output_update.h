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
template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32VF(__ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha,
                                         uint32_t rows = 8)
{
    static_assert(DP == 64 || DP == 128, "unsupported padded dimension");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t row = 0; row < rows; ++row) {
        RegTensor<float> ratio, x0, y0;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
        LoadAlign(x0, accum + row * DP);
        LoadAlign(y0, partial + row * DP);
        if constexpr (DP == 128) {
            RegTensor<float> x1, y1;
            LoadAlign(x1, accum + row * DP + 64);
            LoadAlign(y1, partial + row * DP + 64);
            MulDstAdd(x0, ratio, y0, f);
            MulDstAdd(x1, ratio, y1, f);
            StoreAlign(accum + row * DP, x0, f);
            StoreAlign(accum + row * DP + 64, x1, f);
        } else {
            MulDstAdd(x0, ratio, y0, f);
            StoreAlign(accum + row * DP, x0, f);
        }
    }
}
// Wide P1 publishes its FP32 block sum in the unused half of its alpha slot.
// Consume it with O in group order, after the matching PV_READY notification.
template <uint32_t DP>
__simd_vf__ inline void AccumulateWideFp32VF(__ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha,
                                             __ubuf__ float *denom)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    {
        RegTensor<float> den, ratio, sum;
        LoadAlign(den, denom);
        LoadAlign(ratio, alpha);
        LoadAlign(sum, alpha + 64);
        MulDstAdd(den, ratio, sum, f);
        StoreAlign(denom, den, f);
    }
    for (uint16_t row = 0; row < 64; ++row) {
        RegTensor<float> ratio, x0, y0, x1, y1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
        LoadAlign(x0, accum + row * DP);
        LoadAlign(y0, partial + row * DP);
        LoadAlign(x1, accum + row * DP + 64);
        LoadAlign(y1, partial + row * DP + 64);
        MulDstAdd(x0, ratio, y0, f);
        MulDstAdd(x1, ratio, y1, f);
        StoreAlign(accum + row * DP, x0, f);
        StoreAlign(accum + row * DP + 64, x1, f);
    }
}
// First PV of a query initializes O and its denominator. The previous query
// remains live until its final Vec2; Vec1 must not clear this shared O buffer.
template <uint32_t DP>
__simd_vf__ inline void InitializeWideFp32VF(__ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha,
                                             __ubuf__ float *denom)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> value;
    LoadAlign(value, alpha + 64);
    StoreAlign(denom, value, f);
    for (uint16_t offset = 0; offset < 64 * DP; offset += 64) {
        LoadAlign(value, partial + offset);
        StoreAlign(accum + offset, value, f);
    }
}
template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32PairVF(__ubuf__ float *accum, __ubuf__ float *partial0, __ubuf__ float *partial1,
                                             __ubuf__ float *alpha0, __ubuf__ float *alpha1)
{
    static_assert(DP == 64 || DP == 128, "paired O requires a supported padded dimension");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < 64; ++row) {
        RegTensor<float> a0, a1, x0, y0, z0;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
        LoadAlign(x0, accum + row * DP);
        LoadAlign(y0, partial0 + row * DP);
        LoadAlign(z0, partial1 + row * DP);
        if constexpr (DP == 128) {
            RegTensor<float> x1, y1, z1;
            LoadAlign(x1, accum + row * DP + 64);
            LoadAlign(y1, partial0 + row * DP + 64);
            LoadAlign(z1, partial1 + row * DP + 64);
            // Keep both FP32 FMA rounding points, with no PV storage rounding.
            MulDstAdd(x0, a0, y0, f);
            MulDstAdd(x1, a0, y1, f);
            MulDstAdd(x0, a1, z0, f);
            MulDstAdd(x1, a1, z1, f);
            StoreAlign(accum + row * DP, x0, f);
            StoreAlign(accum + row * DP + 64, x1, f);
        } else {
            MulDstAdd(x0, a0, y0, f);
            MulDstAdd(x0, a1, z0, f);
            StoreAlign(accum + row * DP, x0, f);
        }
    }
}
__simd_vf__ inline void NormalizeVF(__ubuf__ float *accum, __ubuf__ float *denom, uint32_t dp, uint32_t rows)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> one;
    Duplicate(one, 1.0f);
    for (uint32_t i = 0; i < rows; ++i) {
        RegTensor<float> den, ratio;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(den, denom + i);
        Div(ratio, one, den, full);
        for (uint32_t col = 0; col < dp; col += 64) {
            RegTensor<float> x;
            LoadAlign(x, accum + i * dp + col);
            Mul(x, x, ratio, full);

            StoreAlign(accum + i * dp + col, x, full);
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
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> one, den, inv;
    Duplicate(one, 1.0f);
    LoadAlign(den, denom);
    Div(inv, one, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t col = 0; col < DP; col += 64) {
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<float> ratio, x;
            RegTensor<T> packed;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, scratch + row);
            LoadAlign(x, accum + row * DP + col);
            Mul(x, x, ratio, f);
            Cast<T, float, cast>(packed, x, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + col, packed, f);
        }
    }
}
// The last PV never needs to store the updated FP32 O back to UB. Keep the
// FMA -> reciprocal multiply -> cast rounding sequence.
template <class T, uint32_t DP, bool FIRST>
__simd_vf__ inline void FinalizeWideFp32VF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                           __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, sum, one, inv;
    LoadAlign(sum, alpha + 64);
    if constexpr (FIRST) {
        den = sum;
    } else {
        RegTensor<float> ratio;
        LoadAlign(den, denom);
        LoadAlign(ratio, alpha);
        MulDstAdd(den, ratio, sum, f);
    }
    StoreAlign(denom, den, f); // LSE still uses the final FP32 denominator.
    Duplicate(one, 1.0f);
    Div(inv, one, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t col = 0; col < DP; col += 64) {
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<float> x, ratio;
            RegTensor<T> packed;
            if constexpr (FIRST) {
                LoadAlign(x, partial + row * DP + col);
            } else {
                RegTensor<float> y;
                LoadAlign(x, accum + row * DP + col);
                LoadAlign(y, partial + row * DP + col);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
                MulDstAdd(x, ratio, y, f);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, scratch + row);
            Mul(x, x, ratio, f);
            Cast<T, float, cast>(packed, x, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + col, packed, f);
        }
    }
}

// Two consecutive groups of the same query, preserving both FP32 FMA roundings.
template <class T, uint32_t DP, bool FIRST, bool LAST>
__simd_vf__ inline void UpdateWidePairVF(__ubuf__ T *out, __ubuf__ float *accum,
                                         __ubuf__ float *partial0, __ubuf__ float *partial1, __ubuf__ float *alpha0,
                                         __ubuf__ float *alpha1, __ubuf__ float *denom, __ubuf__ float *scratch)
{
    using namespace AscendC::MicroAPI;
    static_assert(DP == 128, "paired wide update requires DP128");
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, a, sum;
    LoadAlign(sum, alpha0 + 64);
    if constexpr (FIRST) {
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
    if constexpr (LAST) {
        RegTensor<float> one, inv;
        Duplicate(one, 1.0f);
        Div(inv, one, den, f);
        StoreAlign(scratch, inv, f);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    for (uint16_t row = 0; row < 64; ++row) {
        RegTensor<float> x0, x1, z0, z1, a0, a1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
        if constexpr (FIRST) {
            LoadAlign(x0, partial0 + row * DP);
            LoadAlign(x1, partial0 + row * DP + 64);
        } else {
            RegTensor<float> y0, y1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
            LoadAlign(x0, accum + row * DP);
            LoadAlign(x1, accum + row * DP + 64);
            LoadAlign(y0, partial0 + row * DP);
            LoadAlign(y1, partial0 + row * DP + 64);
            MulDstAdd(x0, a0, y0, f);
            MulDstAdd(x1, a0, y1, f);
        }
        LoadAlign(z0, partial1 + row * DP);
        LoadAlign(z1, partial1 + row * DP + 64);
        MulDstAdd(x0, a1, z0, f);
        MulDstAdd(x1, a1, z1, f);
        if constexpr (LAST) {
            RegTensor<float> inv;
            RegTensor<T> p0, p1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(inv, scratch + row);
            Mul(x0, x0, inv, f);
            Mul(x1, x1, inv, f);
            Cast<T, float, cast>(p0, x0, f);
            Cast<T, float, cast>(p1, x1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1, f);
        } else {
            StoreAlign(accum + row * DP, x0, f);
            StoreAlign(accum + row * DP + 64, x1, f);
        }
    }
}
} // namespace Sage
