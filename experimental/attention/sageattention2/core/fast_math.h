/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "types.h"
#include <cstring>

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
// AscendC Vector SIMD implementation of two-stage Taylor-Newton Reciprocal
__simd_vf__ inline void FastVectorReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                             const AscendC::MicroAPI::RegTensor<float> &den,
                                             AscendC::MicroAPI::MaskReg mask) noexcept
{
    using namespace AscendC::MicroAPI;
    RegTensor<float> two, nr;
    Duplicate(two, 2.0f);
    // Single-cycle initial estimate
    Reciprocal(inv, den, mask);
    // Step 1: nr = 2 - den * inv; inv = inv * nr
    Mul(nr, den, inv, mask);
    Sub(nr, two, nr, mask);
    Mul(inv, inv, nr, mask);
    // Step 2: nr = 2 - den * inv; inv = inv * nr
    Mul(nr, den, inv, mask);
    Sub(nr, two, nr, mask);
    Mul(inv, inv, nr, mask);
}

// Normalizer reciprocal with fused 2^8 exponent undo scale
__simd_vf__ inline void FastNormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 const AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg mask) noexcept
{
    using namespace AscendC::MicroAPI;
    FastVectorReciprocal(inv, den, mask);
    RegTensor<float> undo;
    Duplicate(undo, PV_PARTIAL_UNDO);
    Mul(inv, inv, undo, mask);
}
#endif

} // namespace math
} // namespace sage
