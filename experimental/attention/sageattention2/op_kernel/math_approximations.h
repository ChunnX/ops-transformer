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
template <int N_STEPS = 2>
__simd_callee__ __attribute__((always_inline)) static inline void VectorReciprocalNewton(
    vapi::RegTensor<float> &out_inv,
    vapi::RegTensor<float> &in_den,
    vapi::MaskReg &exec_mask) noexcept {

    vapi::RegTensor<uint32_t> magic_reg, u32_den, u32_est;
    vapi::Duplicate(magic_reg, FAST_RECIPROCAL_MAGIC);
    u32_den = (vapi::RegTensor<uint32_t> &)in_den;
    vapi::Sub(u32_est, magic_reg, u32_den, exec_mask);

    vapi::RegTensor<float> est_f32, two_reg, step_res;
    est_f32 = (vapi::RegTensor<float> &)u32_est;
    vapi::Duplicate(two_reg, 2.0f);

    // Step 1: y1 = y0 * (2 - d * y0)
    vapi::Mul(step_res, in_den, est_f32, exec_mask);
    vapi::Sub(step_res, two_reg, step_res, exec_mask);
    vapi::Mul(est_f32, est_f32, step_res, exec_mask);

    // Step 2: y2 = y1 * (2 - d * y1)
    vapi::Mul(step_res, in_den, est_f32, exec_mask);
    vapi::Sub(step_res, two_reg, step_res, exec_mask);
    vapi::Mul(out_inv, est_f32, step_res, exec_mask);
}

__simd_callee__ __attribute__((always_inline)) inline void FastReciprocal(vapi::RegTensor<float> &dst,
                                           vapi::RegTensor<float> &src,
                                           vapi::MaskReg &mask) {
    VectorReciprocalNewton<2>(dst, src, mask);
}

__simd_callee__ __attribute__((always_inline)) inline void FastNormalizerReciprocal(vapi::RegTensor<float> &inv,
                                                     vapi::RegTensor<float> &den,
                                                     vapi::MaskReg &mask) noexcept {
    VectorReciprocalNewton<2>(inv, den, mask);
    vapi::RegTensor<float> undo_reg;
    vapi::Duplicate(undo_reg, PV_PARTIAL_UNDO);
    vapi::Mul(inv, inv, undo_reg, mask);
}
#endif

} // namespace sage::core

namespace sage {
inline float FastReciprocalScalar(float d) noexcept {
    return core::FastReciprocalScalar(d);
}
#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
__simd_callee__ __attribute__((always_inline)) inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::RegTensor<float> &src,
                                           AscendC::MicroAPI::MaskReg &mask) {
    core::VectorReciprocalNewton<2>(dst, src, mask);
}

__simd_callee__ __attribute__((always_inline)) inline void FastNormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                     AscendC::MicroAPI::RegTensor<float> &den,
                                                     AscendC::MicroAPI::MaskReg &mask) noexcept {
    core::FastNormalizerReciprocal(inv, den, mask);
}
#endif
} // namespace sage
