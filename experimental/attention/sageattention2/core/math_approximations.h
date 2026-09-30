#pragma once

#include <cstring>
#include "platform_traits.h"
#include "numerical_types.h"

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
#include "kernel_operator.h"
#endif

namespace sage::core {

// Portable Two-Stage Taylor-Newton Reciprocal (Host and Device)
// Replaces 32-cycle hardware vector division instruction `vdiv` with
// an initial seed lookup (0x7EEF370B) followed by 2 iterations of Newton-Raphson:
// x_0 = as_float(0x7EEF370B - as_uint(d))
// x_{n+1} = x_n * (2 - d * x_n)
// Relative precision error strictly <= 2^-14 (< 1e-4).
inline float FastReciprocalScalar(float d) noexcept {
    if (d == 0.0f) return 1.0f;
    uint32_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    uint32_t estBits = FAST_RECIPROCAL_MAGIC - bits;
    float x;
    std::memcpy(&x, &estBits, sizeof(x));
    // Newton-Raphson Iteration 1
    x = x * (2.0f - d * x);
    // Newton-Raphson Iteration 2
    x = x * (2.0f - d * x);
    return x;
}

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
// AscendC MicroAPI Vectorized Fast Reciprocal Primitives

__simd_callee__ inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::RegTensor<float> &src,
                                           AscendC::MicroAPI::MaskReg &mask) {
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

    // Second Newton-Raphson iteration: y2 = y1 * (2.0f - src * y1)
    Mul(term, src, dst, mask);
    Sub(term, two, term, mask);
    Mul(dst, dst, term, mask);
}

__simd_callee__ inline void FastVectorReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg &mask) noexcept {
    FastReciprocal(inv, den, mask);
}

// Normalizer reciprocal with fused 2^8 exponent undo scale
__simd_callee__ inline void FastNormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                     AscendC::MicroAPI::RegTensor<float> &den,
                                                     AscendC::MicroAPI::MaskReg &mask) noexcept {
    using namespace AscendC::MicroAPI;
    FastReciprocal(inv, den, mask);
    RegTensor<float> undo;
    Duplicate(undo, PV_PARTIAL_UNDO);
    Mul(inv, inv, undo, mask);
}

__simd_callee__ inline void NormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg &mask) noexcept {
    FastNormalizerReciprocal(inv, den, mask);
}
#endif

} // namespace sage::core

namespace sage {
using core::FastReciprocalScalar;
#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
using core::FastReciprocal;
using core::FastVectorReciprocal;
using core::FastNormalizerReciprocal;
using core::NormalizerReciprocal;
#endif
} // namespace sage
