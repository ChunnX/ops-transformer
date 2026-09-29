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

// pv_accum_dtype "fp32+fp16" keeps each per-block PV partial in half precision
// before folding it into the FP32 running state. Narrowing and widening share
// the same 64-element register footprint, so the pair applies exactly one FP16
// rounding to the slot in place.
__simd_vf__ inline void RoundPartialFp16VF(__ubuf__ float *partial, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait narrow{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait widen{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    for (uint32_t i = 0; i < count; i += 64) {
        RegTensor<float> x;
        RegTensor<half> packed;
        LoadAlign(x, partial + i);
        Cast<half, float, narrow>(packed, x, f);
        Cast<float, half, widen>(x, packed, f);
        StoreAlign(partial + i, x, f);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Accumulate a 64-row PV tile into the running FP32 O with the alpha ratio FMA.
template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32VF(__ubuf__ float *accum, __ubuf__ float *partial, __ubuf__ float *alpha)
{
    static_assert(DP == 64 || DP == 128, "unsupported padded dimension");
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t row = 0; row < 64; ++row) {
        RegTensor<float> ratio, x0, y0;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
        LoadAlign(x0, accum + row * DP);
        LoadAlign(y0, partial + row * DP);
        MulDstAdd(x0, ratio, y0, f);
        StoreAlign(accum + row * DP, x0, f);
        if constexpr (DP == 128) {
            RegTensor<float> x1, y1;
            LoadAlign(x1, accum + row * DP + 64);
            LoadAlign(y1, partial + row * DP + 64);
            MulDstAdd(x1, ratio, y1, f);
            StoreAlign(accum + row * DP + 64, x1, f);
        }
    }
}

// Accumulate two consecutive PV tiles of the same query in one pass, preserving
// both FP32 FMA roundings (O = (O * a0 + P0) * a1 + P1).
template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32PairVF(__ubuf__ float *accum, __ubuf__ float *partial0, __ubuf__ float *partial1,
                                             __ubuf__ float *alpha0, __ubuf__ float *alpha1)
{
    static_assert(DP == 64 || DP == 128, "paired O requires a supported padded dimension");
    using namespace AscendC::MicroAPI;
    auto f = CreateMask<float, MaskPattern::ALL>();
    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<float> x0, x1, y0, y1, z0, z1, a0, a1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
            LoadAlign(x0, accum + row * DP);
            LoadAlign(x1, accum + row * DP + 64);
            LoadAlign(y0, partial0 + row * DP);
            LoadAlign(y1, partial0 + row * DP + 64);
            MulDstAdd(x0, a0, y0, f);
            MulDstAdd(x1, a0, y1, f);

            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
            LoadAlign(z0, partial1 + row * DP);
            LoadAlign(z1, partial1 + row * DP + 64);
            MulDstAdd(x0, a1, z0, f);
            MulDstAdd(x1, a1, z1, f);

            StoreAlign(accum + row * DP, x0, f);
            StoreAlign(accum + row * DP + 64, x1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<float> x, y, z, a0, a1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha0 + row);
            LoadAlign(x, accum + row * DP);
            LoadAlign(y, partial0 + row * DP);
            MulDstAdd(x, a0, y, f);

            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha1 + row);
            LoadAlign(z, partial1 + row * DP);
            MulDstAdd(x, a1, z, f);

            StoreAlign(accum + row * DP, x, f);
        }
    }
}

// Finalize the complete 64-row wide tile. Compute all reciprocals together with FastReciprocal,
// and cast directly into the output buffer. meanV carries the per-channel V
// centering term (all zeros when smooth_v is off) and is added in FP32 before
// the output cast.
template <class T, uint32_t DP>
__simd_vf__ inline void NormalizeCastVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                        __ubuf__ float *scratch, __ubuf__ float *meanV)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, inv;
    LoadAlign(den, denom);
    FastReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if constexpr (DP == 128) {
        RegTensor<float> mv0, mv1;
        LoadAlign(mv0, meanV);
        LoadAlign(mv1, meanV + 64);
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> ratio0, ratio1, x0_0, x1_0, x0_1, x1_1;
            RegTensor<T> p0_0, p1_0, p0_1, p1_1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + row + 1);
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);
            Mul(x0_0, x0_0, ratio0, f);
            Mul(x1_0, x1_0, ratio0, f);
            Mul(x0_1, x0_1, ratio1, f);
            Mul(x1_1, x1_1, ratio1, f);
            Add(x0_0, x0_0, mv0, f);
            Add(x1_0, x1_0, mv1, f);
            Add(x0_1, x0_1, mv0, f);
            Add(x1_1, x1_1, mv1, f);
            Cast<T, float, cast>(p0_0, x0_0, f);
            Cast<T, float, cast>(p1_0, x1_0, f);
            Cast<T, float, cast>(p0_1, x0_1, f);
            Cast<T, float, cast>(p1_1, x1_1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
        }
    } else {
        RegTensor<float> mv;
        LoadAlign(mv, meanV);
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> ratio0, ratio1, x0, x1;
            RegTensor<T> packed0, packed1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + row + 1);
            LoadAlign(x0, accum + row * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            Mul(x0, x0, ratio0, f);
            Mul(x1, x1, ratio1, f);
            Add(x0, x0, mv, f);
            Add(x1, x1, mv, f);
            Cast<T, float, cast>(packed0, x0, f);
            Cast<T, float, cast>(packed1, x1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, packed0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, packed1, f);
        }
    }
}

// The last PV never needs to store the updated FP32 O back to UB. Keep the
// FMA -> reciprocal multiply -> cast rounding sequence.
template <class T, uint32_t DP, bool FIRST>
__simd_vf__ inline void FinalizeWideFp32VF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                           __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch,
                                           __ubuf__ float *meanV)
{
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, sum, inv;
    LoadAlign(sum, alpha + 64);
    if constexpr (FIRST) {
        den = sum;
    } else {
        RegTensor<float> ratio;
        LoadAlign(den, denom);
        LoadAlign(ratio, alpha);
        MulDstAdd(den, ratio, sum, f);
    }
    StoreAlign(denom, den, f);
    FastReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if constexpr (DP == 128) {
        RegTensor<float> mv0, mv1;
        LoadAlign(mv0, meanV);
        LoadAlign(mv1, meanV + 64);
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<float> x0, x1, ratio, rinv;
            RegTensor<T> p0, p1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rinv, scratch + row);
            if constexpr (FIRST) {
                LoadAlign(x0, partial + row * DP);
                LoadAlign(x1, partial + row * DP + 64);
            } else {
                RegTensor<float> y0, y1;
                LoadAlign(x0, accum + row * DP);
                LoadAlign(x1, accum + row * DP + 64);
                LoadAlign(y0, partial + row * DP);
                LoadAlign(y1, partial + row * DP + 64);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
                MulDstAdd(x0, ratio, y0, f);
                MulDstAdd(x1, ratio, y1, f);
            }
            Mul(x0, x0, rinv, f);
            Mul(x1, x1, rinv, f);
            Add(x0, x0, mv0, f);
            Add(x1, x1, mv1, f);
            Cast<T, float, cast>(p0, x0, f);
            Cast<T, float, cast>(p1, x1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1, f);
        }
    } else {
        RegTensor<float> mv;
        LoadAlign(mv, meanV);
        for (uint16_t row = 0; row < 64; ++row) {
            RegTensor<float> x, ratio, rinv;
            RegTensor<T> packed;
            if constexpr (FIRST) {
                LoadAlign(x, partial + row * DP);
            } else {
                RegTensor<float> y;
                LoadAlign(x, accum + row * DP);
                LoadAlign(y, partial + row * DP);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio, alpha + row);
                MulDstAdd(x, ratio, y, f);
            }
            LoadAlign<float, LoadDist::DIST_BRC_B32>(rinv, scratch + row);
            Mul(x, x, rinv, f);
            Add(x, x, mv, f);
            Cast<T, float, cast>(packed, x, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, packed, f);
        }
    }
}

// Two consecutive groups of the same query, preserving both FP32 FMA roundings.
// Register-efficient tight loop (<= 8 live registers) avoiding any spilling to UB.
template <class T, uint32_t DP, bool FIRST, bool LAST>
__simd_vf__ inline void UpdateWidePairVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                         __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                         __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV)
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
        RegTensor<float> inv;
        FastReciprocal(inv, den, f);
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
            RegTensor<float> inv, mv0, mv1;
            RegTensor<T> p0, p1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(inv, scratch + row);
            LoadAlign(mv0, meanV);
            LoadAlign(mv1, meanV + 64);
            Mul(x0, x0, inv, f);
            Mul(x1, x1, inv, f);
            Add(x0, x0, mv0, f);
            Add(x1, x1, mv1, f);
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
