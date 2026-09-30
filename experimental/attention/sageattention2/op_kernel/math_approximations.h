#pragma once

#include <cstring>
#include <cstdint>
#include "platform_traits.h"
#include "sage_attention2_tiling_data.h"

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
#include "kernel_operator.h"
#endif

namespace sage::core {

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
namespace vapi = AscendC::MicroAPI;
#endif

template <typename TScalar = float>
struct ReciprocalApproximationPipeline {
    static inline float EvaluateScalar(float denom) noexcept {
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
    __simd_callee__ static inline void VectorApproximation(
        vapi::RegTensor<float> &out_inv,
        vapi::RegTensor<float> &in_den,
        vapi::MaskReg &exec_mask,
        float post_scale = 1.0f) noexcept {

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

        if (post_scale != 1.0f) {
            vapi::RegTensor<float> undo_reg;
            vapi::Duplicate(undo_reg, post_scale);
            vapi::Mul(out_inv, out_inv, undo_reg, exec_mask);
        }
    }
#endif
};

inline float FastReciprocalScalar(float d) noexcept {
    return ReciprocalApproximationPipeline<float>::EvaluateScalar(d);
}

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
__simd_callee__ inline void FastReciprocal(vapi::RegTensor<float> &dst,
                                           vapi::RegTensor<float> &src,
                                           vapi::MaskReg &mask) {
    ReciprocalApproximationPipeline<float>::VectorApproximation<2>(dst, src, mask, 1.0f);
}

__simd_callee__ inline void FastVectorReciprocal(vapi::RegTensor<float> &inv,
                                                 vapi::RegTensor<float> &den,
                                                 vapi::MaskReg &mask) noexcept {
    ReciprocalApproximationPipeline<float>::VectorApproximation<2>(inv, den, mask, 1.0f);
}

__simd_callee__ inline void FastNormalizerReciprocal(vapi::RegTensor<float> &inv,
                                                     vapi::RegTensor<float> &den,
                                                     vapi::MaskReg &mask) noexcept {
    ReciprocalApproximationPipeline<float>::VectorApproximation<2>(inv, den, mask, PV_PARTIAL_UNDO);
}

__simd_callee__ inline void NormalizerReciprocal(vapi::RegTensor<float> &inv,
                                                 vapi::RegTensor<float> &den,
                                                 vapi::MaskReg &mask) noexcept {
    ReciprocalApproximationPipeline<float>::VectorApproximation<2>(inv, den, mask, PV_PARTIAL_UNDO);
}
#endif

} // namespace sage::core

namespace sage {
inline float FastReciprocalScalar(float d) noexcept {
    return core::FastReciprocalScalar(d);
}
#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
__simd_callee__ inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::RegTensor<float> &src,
                                           AscendC::MicroAPI::MaskReg &mask) {
    core::ReciprocalApproximationPipeline<float>::VectorApproximation<2>(dst, src, mask, 1.0f);
}

__simd_callee__ inline void FastVectorReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg &mask) noexcept {
    core::ReciprocalApproximationPipeline<float>::VectorApproximation<2>(inv, den, mask, 1.0f);
}

__simd_callee__ inline void FastNormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                     AscendC::MicroAPI::RegTensor<float> &den,
                                                     AscendC::MicroAPI::MaskReg &mask) noexcept {
    core::ReciprocalApproximationPipeline<float>::VectorApproximation<2>(inv, den, mask, core::PV_PARTIAL_UNDO);
}

__simd_callee__ inline void NormalizerReciprocal(AscendC::MicroAPI::RegTensor<float> &inv,
                                                 AscendC::MicroAPI::RegTensor<float> &den,
                                                 AscendC::MicroAPI::MaskReg &mask) noexcept {
    core::ReciprocalApproximationPipeline<float>::VectorApproximation<2>(inv, den, mask, core::PV_PARTIAL_UNDO);
}
#endif
} // namespace sage
