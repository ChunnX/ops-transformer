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
#endif

// Reciprocal of softmax denominator, undoing the PV MMA P-scale shift 2^-8
__simd_callee__ inline void NormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg &mask) {
    using namespace AscendC::MicroAPI;
    RegTensor<float> undo;
    FastReciprocal(inv, den, mask);
    Duplicate(undo, PV_PARTIAL_UNDO);
    Mul(inv, inv, undo, mask);
}

// 4-way unrolled FP32 accumulator for Online Softmax PV update
template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32VF(__ubuf__ float *accum, __ubuf__ float *partial,
                                         __ubuf__ float *alpha) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();

    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> ratio0, ratio1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);

            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            RegTensor<float> y0_0, y1_0, y0_1, y1_1;

            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            LoadAlign(y0_0, partial + row * DP);
            LoadAlign(y1_0, partial + row * DP + 64);
            LoadAlign(y0_1, partial + (row + 1) * DP);
            LoadAlign(y1_1, partial + (row + 1) * DP + 64);

            MulDstAdd(x0_0, ratio0, y0_0, f);
            MulDstAdd(x1_0, ratio0, y1_0, f);
            MulDstAdd(x0_1, ratio1, y0_1, f);
            MulDstAdd(x1_1, ratio1, y1_1, f);

            StoreAlign(accum + row * DP, x0_0, f);
            StoreAlign(accum + row * DP + 64, x1_0, f);
            StoreAlign(accum + (row + 1) * DP, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> ratio0, ratio1, ratio2, ratio3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, alpha + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, alpha + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, alpha + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, alpha + row + 3);

            RegTensor<float> x0, x1, x2, x3;
            RegTensor<float> y0, y1, y2, y3;
            LoadAlign(x0, accum + (row + 0) * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            LoadAlign(y0, partial + (row + 0) * DP);
            LoadAlign(y1, partial + (row + 1) * DP);
            LoadAlign(y2, partial + (row + 2) * DP);
            LoadAlign(y3, partial + (row + 3) * DP);

            MulDstAdd(x0, ratio0, y0, f);
            MulDstAdd(x1, ratio1, y1, f);
            MulDstAdd(x2, ratio2, y2, f);
            MulDstAdd(x3, ratio3, y3, f);

            StoreAlign(accum + (row + 0) * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
            StoreAlign(accum + (row + 2) * DP, x2, f);
            StoreAlign(accum + (row + 3) * DP, x3, f);
        }
    }
}

// 4-way unrolled Paired FP32 accumulator
template <uint32_t DP>
__simd_vf__ inline void AccumulateFp32PairVF(__ubuf__ float *accum, __ubuf__ float *partial0,
                                             __ubuf__ float *partial1, __ubuf__ float *alpha0,
                                             __ubuf__ float *alpha1) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();

    if constexpr (DP == 128) {
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> a0_0, a0_1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);

            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            RegTensor<float> y0_0, y1_0, y0_1, y1_1;
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            LoadAlign(y0_0, partial0 + row * DP);
            LoadAlign(y1_0, partial0 + row * DP + 64);
            LoadAlign(y0_1, partial0 + (row + 1) * DP);
            LoadAlign(y1_1, partial0 + (row + 1) * DP + 64);

            MulDstAdd(x0_0, a0_0, y0_0, f);
            MulDstAdd(x1_0, a0_0, y1_0, f);
            MulDstAdd(x0_1, a0_1, y0_1, f);
            MulDstAdd(x1_1, a0_1, y1_1, f);

            RegTensor<float> a1_0, a1_1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);

            RegTensor<float> z0_0, z1_0, z0_1, z1_1;
            LoadAlign(z0_0, partial1 + row * DP);
            LoadAlign(z1_0, partial1 + row * DP + 64);
            LoadAlign(z0_1, partial1 + (row + 1) * DP);
            LoadAlign(z1_1, partial1 + (row + 1) * DP + 64);

            MulDstAdd(x0_0, a1_0, z0_0, f);
            MulDstAdd(x1_0, a1_0, z1_0, f);
            MulDstAdd(x0_1, a1_1, z0_1, f);
            MulDstAdd(x1_1, a1_1, z1_1, f);

            StoreAlign(accum + row * DP, x0_0, f);
            StoreAlign(accum + row * DP + 64, x1_0, f);
            StoreAlign(accum + (row + 1) * DP, x0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, x1_1, f);
        }
    } else {
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> a0_0, a0_1, a0_2, a0_3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_2, alpha0 + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_3, alpha0 + row + 3);

            RegTensor<float> x0, x1, x2, x3;
            RegTensor<float> y0, y1, y2, y3;
            LoadAlign(x0, accum + (row + 0) * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            LoadAlign(y0, partial0 + (row + 0) * DP);
            LoadAlign(y1, partial0 + (row + 1) * DP);
            LoadAlign(y2, partial0 + (row + 2) * DP);
            LoadAlign(y3, partial0 + (row + 3) * DP);

            MulDstAdd(x0, a0_0, y0, f);
            MulDstAdd(x1, a0_1, y1, f);
            MulDstAdd(x2, a0_2, y2, f);
            MulDstAdd(x3, a0_3, y3, f);

            RegTensor<float> a1_0, a1_1, a1_2, a1_3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row + 0);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_2, alpha1 + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_3, alpha1 + row + 3);

            RegTensor<float> z0, z1, z2, z3;
            LoadAlign(z0, partial1 + (row + 0) * DP);
            LoadAlign(z1, partial1 + (row + 1) * DP);
            LoadAlign(z2, partial1 + (row + 2) * DP);
            LoadAlign(z3, partial1 + (row + 3) * DP);

            MulDstAdd(x0, a1_0, z0, f);
            MulDstAdd(x1, a1_1, z1, f);
            MulDstAdd(x2, a1_2, z2, f);
            MulDstAdd(x3, a1_3, z3, f);

            StoreAlign(accum + (row + 0) * DP, x0, f);
            StoreAlign(accum + (row + 1) * DP, x1, f);
            StoreAlign(accum + (row + 2) * DP, x2, f);
            StoreAlign(accum + (row + 3) * DP, x3, f);
        }
    }
}

// Round per-block PV partial to FP16 for fp32+fp16 accumulation
__simd_vf__ inline void RoundPartialFp16VF(__ubuf__ float *partial, uint32_t count) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    static constexpr CastTrait narrow{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait widen{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    for (uint32_t i = 0; i < count; i += 128) {
        RegTensor<float> x0, x1;
        RegTensor<half> packed0, packed1;
        LoadAlign(x0, partial + i);
        LoadAlign(x1, partial + i + 64);
        Cast<half, float, narrow>(packed0, x0, f);
        Cast<half, float, narrow>(packed1, x1, f);
        Cast<float, half, widen>(x0, packed0, f);
        Cast<float, half, widen>(x1, packed1, f);
        StoreAlign(partial + i, x0, f);
        StoreAlign(partial + i + 64, x1, f);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

// Normalize and cast output (standard path)
template <class T, uint32_t DP>
__simd_vf__ inline void NormalizeCastVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                        __ubuf__ float *scratch, __ubuf__ float *meanV) {
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, inv;
    LoadAlign(den, denom);
    NormalizerReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if constexpr (DP == 128) {
        RegTensor<float> mv0, mv1;
        LoadAlign(mv0, meanV);
        LoadAlign(mv1, meanV + 64);
        for (uint16_t row = 0; row < 64; row += 2) {
            RegTensor<float> ratio0, ratio1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + row + 1);

            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
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

            RegTensor<T> p0_0, p1_0, p0_1, p1_1;
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
        for (uint16_t row = 0; row < 64; row += 4) {
            RegTensor<float> ratio0, ratio1, ratio2, ratio3;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio1, scratch + row + 1);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio2, scratch + row + 2);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ratio3, scratch + row + 3);

            RegTensor<float> x0, x1, x2, x3;
            LoadAlign(x0, accum + row * DP);
            LoadAlign(x1, accum + (row + 1) * DP);
            LoadAlign(x2, accum + (row + 2) * DP);
            LoadAlign(x3, accum + (row + 3) * DP);

            Mul(x0, x0, ratio0, f);
            Mul(x1, x1, ratio1, f);
            Mul(x2, x2, ratio2, f);
            Mul(x3, x3, ratio3, f);

            Add(x0, x0, mv, f);
            Add(x1, x1, mv, f);
            Add(x2, x2, mv, f);
            Add(x3, x3, mv, f);

            RegTensor<T> packed0, packed1, packed2, packed3;
            Cast<T, float, cast>(packed0, x0, f);
            Cast<T, float, cast>(packed1, x1, f);
            Cast<T, float, cast>(packed2, x2, f);
            Cast<T, float, cast>(packed3, x3, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, packed0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, packed1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 2) * DP, packed2, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 3) * DP, packed3, f);
        }
    }
}

// Wide finalize & update routines
template <class T, uint32_t DP, bool FIRST>
__simd_vf__ inline void FinalizeWideFp32VF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                           __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch,
                                           __ubuf__ float *meanV) {
    static_assert(DP == 128, "FinalizeWideFp32VF is specialized for DP == 128");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> den, inv;
    LoadAlign(den, denom);
    NormalizerReciprocal(inv, den, f);
    StoreAlign(scratch, inv, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> mv0, mv1;
    LoadAlign(mv0, meanV);
    LoadAlign(mv1, meanV + 64);
    for (uint16_t row = 0; row < 64; row += 2) {
        RegTensor<float> r0, r1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(r0, scratch + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(r1, scratch + row + 1);

        RegTensor<float> x0_0, x1_0, x0_1, x1_1;
        RegTensor<float> y0_0, y1_0, y0_1, y1_1;
        LoadAlign(x0_0, accum + row * DP);
        LoadAlign(x1_0, accum + row * DP + 64);
        LoadAlign(x0_1, accum + (row + 1) * DP);
        LoadAlign(x1_1, accum + (row + 1) * DP + 64);

        LoadAlign(y0_0, partial + row * DP);
        LoadAlign(y1_0, partial + row * DP + 64);
        LoadAlign(y0_1, partial + (row + 1) * DP);
        LoadAlign(y1_1, partial + (row + 1) * DP + 64);

        if constexpr (!FIRST) {
            RegTensor<float> a0, a1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a0, alpha + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(a1, alpha + row + 1);
            MulDstAdd(y0_0, a0, x0_0, f);
            MulDstAdd(y1_0, a0, x1_0, f);
            MulDstAdd(y0_1, a1, x0_1, f);
            MulDstAdd(y1_1, a1, x1_1, f);
        }

        Mul(y0_0, y0_0, r0, f);
        Mul(y1_0, y1_0, r0, f);
        Mul(y0_1, y0_1, r1, f);
        Mul(y1_1, y1_1, r1, f);

        Add(y0_0, y0_0, mv0, f);
        Add(y1_0, y1_0, mv1, f);
        Add(y0_1, y0_1, mv0, f);
        Add(y1_1, y1_1, mv1, f);

        RegTensor<T> p0_0, p1_0, p0_1, p1_1;
        Cast<T, float, cast>(p0_0, y0_0, f);
        Cast<T, float, cast>(p1_0, y1_0, f);
        Cast<T, float, cast>(p0_1, y0_1, f);
        Cast<T, float, cast>(p1_1, y1_1, f);

        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
        StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
    }
}

template <class T, uint32_t DP, bool FIRST, bool LAST>
__simd_vf__ inline void UpdateWidePairVF(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                         __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                         __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV) {
    static_assert(DP == 128, "UpdateWidePairVF is specialized for DP == 128");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    if constexpr (LAST) {
        RegTensor<float> den, inv;
        LoadAlign(den, denom);
        NormalizerReciprocal(inv, den, f);
        StoreAlign(scratch, inv, f);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> mv0, mv1;
    if constexpr (LAST) {
        LoadAlign(mv0, meanV);
        LoadAlign(mv1, meanV + 64);
    }
    for (uint16_t row = 0; row < 64; row += 2) {
        RegTensor<float> a0_0, a0_1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_0, alpha0 + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a0_1, alpha0 + row + 1);

        RegTensor<float> a1_0, a1_1;
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_0, alpha1 + row);
        LoadAlign<float, LoadDist::DIST_BRC_B32>(a1_1, alpha1 + row + 1);

        RegTensor<float> y0_0, y1_0, y0_1, y1_1;
        RegTensor<float> z0_0, z1_0, z0_1, z1_1;
        LoadAlign(y0_0, partial0 + row * DP);
        LoadAlign(y1_0, partial0 + row * DP + 64);
        LoadAlign(y0_1, partial0 + (row + 1) * DP);
        LoadAlign(y1_1, partial0 + (row + 1) * DP + 64);

        LoadAlign(z0_0, partial1 + row * DP);
        LoadAlign(z1_0, partial1 + row * DP + 64);
        LoadAlign(z0_1, partial1 + (row + 1) * DP);
        LoadAlign(z1_1, partial1 + (row + 1) * DP + 64);

        if constexpr (FIRST) {
            MulDstAdd(z0_0, a1_0, y0_0, f);
            MulDstAdd(z1_0, a1_0, y1_0, f);
            MulDstAdd(z0_1, a1_1, y0_1, f);
            MulDstAdd(z1_1, a1_1, y1_1, f);
        } else {
            RegTensor<float> x0_0, x1_0, x0_1, x1_1;
            LoadAlign(x0_0, accum + row * DP);
            LoadAlign(x1_0, accum + row * DP + 64);
            LoadAlign(x0_1, accum + (row + 1) * DP);
            LoadAlign(x1_1, accum + (row + 1) * DP + 64);

            MulDstAdd(y0_0, a0_0, x0_0, f);
            MulDstAdd(y1_0, a0_0, x1_0, f);
            MulDstAdd(y0_1, a0_1, x0_1, f);
            MulDstAdd(y1_1, a0_1, x1_1, f);

            MulDstAdd(z0_0, a1_0, y0_0, f);
            MulDstAdd(z1_0, a1_0, y1_0, f);
            MulDstAdd(z0_1, a1_1, y0_1, f);
            MulDstAdd(z1_1, a1_1, y1_1, f);
        }

        if constexpr (LAST) {
            RegTensor<float> r0, r1;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(r0, scratch + row);
            LoadAlign<float, LoadDist::DIST_BRC_B32>(r1, scratch + row + 1);

            Mul(z0_0, z0_0, r0, f);
            Mul(z1_0, z1_0, r0, f);
            Mul(z0_1, z0_1, r1, f);
            Mul(z1_1, z1_1, r1, f);

            Add(z0_0, z0_0, mv0, f);
            Add(z1_0, z1_0, mv1, f);
            Add(z0_1, z0_1, mv0, f);
            Add(z1_1, z1_1, mv1, f);

            RegTensor<T> p0_0, p1_0, p0_1, p1_1;
            Cast<T, float, cast>(p0_0, z0_0, f);
            Cast<T, float, cast>(p1_0, z1_0, f);
            Cast<T, float, cast>(p0_1, z0_1, f);
            Cast<T, float, cast>(p1_1, z1_1, f);

            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP, p0_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + row * DP + 64, p1_0, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP, p0_1, f);
            StoreAlign<T, StoreDist::DIST_PACK_B32>(out + (row + 1) * DP + 64, p1_1, f);
        } else {
            StoreAlign(accum + row * DP, z0_0, f);
            StoreAlign(accum + row * DP + 64, z1_0, f);
            StoreAlign(accum + (row + 1) * DP, z0_1, f);
            StoreAlign(accum + (row + 1) * DP + 64, z1_1, f);
        }
    }
}

// Clean-Room OutAccumulator Engine encapsulating output normalization & LSE writeback
class OutAccumulator {
public:
    template <uint32_t DP>
    __aicore__ static inline void AccumulateFp32(__ubuf__ float *accum, __ubuf__ float *partial,
                                                  __ubuf__ float *alpha) {
        AccumulateFp32VF<DP>(accum, partial, alpha);
    }

    template <uint32_t DP>
    __aicore__ static inline void AccumulateFp32Pair(__ubuf__ float *accum, __ubuf__ float *partial0,
                                                      __ubuf__ float *partial1, __ubuf__ float *alpha0,
                                                      __ubuf__ float *alpha1) {
        AccumulateFp32PairVF<DP>(accum, partial0, partial1, alpha0, alpha1);
    }

    __aicore__ static inline void RoundPartialFp16(__ubuf__ float *partial, uint32_t count) {
        RoundPartialFp16VF(partial, count);
    }

    template <class T, uint32_t DP>
    __aicore__ static inline void NormalizeCast(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *denom,
                                                 __ubuf__ float *scratch, __ubuf__ float *meanV) {
        NormalizeCastVF<T, DP>(out, accum, denom, scratch, meanV);
    }

    template <class T, uint32_t DP, bool FIRST>
    __aicore__ static inline void FinalizeWideFp32(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial,
                                                    __ubuf__ float *alpha, __ubuf__ float *denom, __ubuf__ float *scratch,
                                                    __ubuf__ float *meanV) {
        FinalizeWideFp32VF<T, DP, FIRST>(out, accum, partial, alpha, denom, scratch, meanV);
    }

    template <class T, uint32_t DP, bool FIRST, bool LAST>
    __aicore__ static inline void UpdateWidePair(__ubuf__ T *out, __ubuf__ float *accum, __ubuf__ float *partial0,
                                                  __ubuf__ float *partial1, __ubuf__ float *alpha0, __ubuf__ float *alpha1,
                                                  __ubuf__ float *denom, __ubuf__ float *scratch, __ubuf__ float *meanV) {
        UpdateWidePairVF<T, DP, FIRST, LAST>(out, accum, partial0, partial1, alpha0, alpha1, denom, scratch, meanV);
    }

    __aicore__ static inline void StoreTileLse(
        AscendC::GlobalTensor<float> &lg, AscendC::LocalTensor<float> denom, AscendC::LocalTensor<half> maxima,
        AscendC::LocalTensor<float> scratch, uint32_t first, uint32_t validRows,
        uint32_t b, uint32_t h, const SageParams &p)
    {
        using namespace AscendC;
        Sync<HardEvent::MTE3_S>();
        Sync<HardEvent::V_S>();
        if (!validRows) return;

        auto lse = scratch;
        auto maxF = scratch[HALF];
        DataCopy(lse, denom, HALF);
        Sync<HardEvent::MTE2_V>();
        Ln(lse, lse, HALF);
        PipeBarrier<PIPE_V>();
        Cast(maxF, maxima, RoundMode::CAST_NONE, HALF);
        PipeBarrier<PIPE_V>();
        Add(lse, lse, maxF, HALF);
        PipeBarrier<PIPE_V>();
        Muls(lse, lse, LOG2_E, HALF);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams cp{1, validRows * uint32_t(sizeof(float)), 0, 0, 0};
        DataCopyPad(lg[(uint64_t(b) * p.hq + h) * p.sq + first], lse, cp);
        Sync<HardEvent::MTE3_S>();
    }
};

} // namespace sage::compute
