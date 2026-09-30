#pragma once

#include <cstring>
#include <cstdint>
#include "platform_traits.h"

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
#include "kernel_operator.h"
#endif

namespace sage::core {

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
namespace vapi = AscendC::MicroAPI;
#endif

// One magic-number seed plus two Newton-Raphson steps: the pair reaches ~2^-16
// relative error, which is below the INT8/FP8 quantization noise it normalizes.
inline float FastReciprocalScalar(float denom) noexcept {
    if (denom == 0.0f) {
        return 1.0f;
    }
    uint32_t val_bits = 0;
    std::memcpy(&val_bits, &denom, sizeof(float));
    const uint32_t seed_bits = FAST_RECIPROCAL_MAGIC - val_bits;
    float est_val = 0.0f;
    std::memcpy(&est_val, &seed_bits, sizeof(float));

    const float res0 = 2.0f - denom * est_val;
    est_val = est_val * res0;
    const float res1 = 2.0f - denom * est_val;
    est_val = est_val * res1;
    return est_val;
}

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
__simd_callee__ __attribute__((always_inline)) inline void FastReciprocal(vapi::RegTensor<float> &dst,
                                           vapi::RegTensor<float> &src,
                                           vapi::MaskReg &mask) {
    vapi::RegTensor<uint32_t> magic_reg, u32_den, u32_est;
    vapi::Duplicate(magic_reg, FAST_RECIPROCAL_MAGIC);
    u32_den = (vapi::RegTensor<uint32_t> &)src;
    vapi::Sub(u32_est, magic_reg, u32_den, mask);

    vapi::RegTensor<float> est_f32, two_reg, step_res;
    est_f32 = (vapi::RegTensor<float> &)u32_est;
    vapi::Duplicate(two_reg, 2.0f);

    // Step 1: y1 = y0 * (2 - d * y0)
    vapi::Mul(step_res, src, est_f32, mask);
    vapi::Sub(step_res, two_reg, step_res, mask);
    vapi::Mul(est_f32, est_f32, step_res, mask);

    // Step 2: y2 = y1 * (2 - d * y1)
    vapi::Mul(step_res, src, est_f32, mask);
    vapi::Sub(step_res, two_reg, step_res, mask);
    vapi::Mul(dst, est_f32, step_res, mask);
}

// Softmax normalizer: 1/denom in the PV_PARTIAL_UNDO-scaled domain the P tile was
// accumulated in, so the scaling cancels without an extra pass over the tile.
__simd_callee__ __attribute__((always_inline)) inline void FastNormalizerReciprocal(vapi::RegTensor<float> &inv,
                                                     vapi::RegTensor<float> &den,
                                                     vapi::MaskReg &mask) noexcept {
    FastReciprocal(inv, den, mask);
    vapi::RegTensor<float> undo_reg;
    vapi::Duplicate(undo_reg, PV_PARTIAL_UNDO);
    vapi::Mul(inv, inv, undo_reg, mask);
}
#endif

} // namespace sage::core
