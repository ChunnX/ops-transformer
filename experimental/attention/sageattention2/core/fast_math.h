/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "types.h"
#include <cstring>
#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
#include "kernel_operator.h"
#endif

namespace sage {
namespace math {

// Portable two-stage Taylor-Newton Reciprocal: eliminates 32-cycle hardware vector division.
// Initial reciprocal seed magic constant: 0x7EEF370B
inline float FastReciprocalScalar(float d) noexcept
{
    if (d == 0.0f) return 1.0f;
    uint32_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    uint32_t estBits = 0x7EEF370BU - bits;
    float x;
    std::memcpy(&x, &estBits, sizeof(x));
    // Newton-Raphson Step 1: x = x * (2 - d * x)
    x = x * (2.0f - d * x);
    // Newton-Raphson Step 2: x = x * (2 - d * x)
    x = x * (2.0f - d * x);
    return x;
}

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
// AscendC MicroAPI FastReciprocal with magic seed 0x7EEF370B and two Newton-Raphson iterations
__simd_callee__ inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::RegTensor<float> &src,
                                           AscendC::MicroAPI::MaskReg &mask)
{
    using namespace AscendC::MicroAPI;
    RegTensor<uint32_t> magic, ix, iy;
    RegTensor<float> y0, term, two;
    Duplicate(two, 2.0f);
    Duplicate(magic, FAST_RECIPROCAL_MAGIC);
    ix = (RegTensor<uint32_t> &)src;
    Sub(iy, magic, ix, mask);
    y0 = (RegTensor<float> &)iy;

    // First Newton-Raphson iteration: y1 = y0 * (2.0f - src * y0)
    Mul(term, src, y0, mask);
    Sub(term, two, term, mask);
    Mul(dst, y0, term, mask);

    // Second Newton-Raphson iteration
    Mul(term, src, dst, mask);
    Sub(term, two, term, mask);
    Mul(dst, dst, term, mask);
}

// AscendC Vector SIMD implementation of two-stage Taylor-Newton Reciprocal
__simd_callee__ inline void FastVectorReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg &mask) noexcept
{
    FastReciprocal(inv, den, mask);
}

// Normalizer reciprocal with fused 2^8 exponent undo scale
__simd_callee__ inline void FastNormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                     AscendC::MicroAPI::RegTensor<float> &den,
                                                     AscendC::MicroAPI::MaskReg &mask) noexcept
{
    using namespace AscendC::MicroAPI;
    FastReciprocal(inv, den, mask);
    RegTensor<float> undo;
    Duplicate(undo, PV_PARTIAL_UNDO);
    Mul(inv, inv, undo, mask);
}
#endif

} // namespace math

using math::FastReciprocalScalar;
#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
using math::FastReciprocal;
using math::FastNormalizerReciprocal;
#endif

} // namespace sage
