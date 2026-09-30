#pragma once

#include <cstdint>
#include <type_traits>
#include <string_view>
#include "platform_traits.h"

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
#include "kernel_operator.h"
#define SAGE_DEVICE_FUNC __aicore__ inline
#else
#define SAGE_DEVICE_FUNC inline
#endif

namespace sage::core {

// Quantization Granularity and PV Accumulation Mode Enumerations

enum class SageQuantGran : int64_t {
    PER_THREAD = 0,
    PER_WARP   = 1,
};

enum class SagePvAccum : int64_t {
    FP32            = 0,
    FP32_PLUS_FP32  = 1,
    FP32_PLUS_FP16  = 2,
};

// Type Validation and String Parser Helpers

constexpr bool IsValidQuantGran(int64_t val) noexcept {
    return val == static_cast<int64_t>(SageQuantGran::PER_THREAD) ||
           val == static_cast<int64_t>(SageQuantGran::PER_WARP);
}

constexpr bool IsValidPvAccum(int64_t val) noexcept {
    return val == static_cast<int64_t>(SagePvAccum::FP32) ||
           val == static_cast<int64_t>(SagePvAccum::FP32_PLUS_FP32) ||
           val == static_cast<int64_t>(SagePvAccum::FP32_PLUS_FP16);
}

constexpr uint32_t QTokensPerScale(bool isPerWarp) noexcept {
    return isPerWarp ? 32U : 4U;
}

constexpr uint32_t KTokensPerScale(bool isPerWarp) noexcept {
    return isPerWarp ? 32U : 4U;
}

// Compile-time string parser for Quantization Granularity
constexpr SageQuantGran ParseQuantGran(std::string_view s) {
    if (s == "per_warp") return SageQuantGran::PER_WARP;
    return SageQuantGran::PER_THREAD;
}

// Compile-time string parser for PV Accumulation mode
constexpr SagePvAccum ParsePvAccum(std::string_view s) {
    if (s == "fp32") return SagePvAccum::FP32;
    if (s == "fp32+fp32") return SagePvAccum::FP32_PLUS_FP32;
    return SagePvAccum::FP32_PLUS_FP16; // default
}

// Precision & Supported Input Types Traits

template <class T>
inline constexpr bool is_supported_input_type_v = false;

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
template <>
inline constexpr bool is_supported_input_type_v<half> = true;

template <>
inline constexpr bool is_supported_input_type_v<bfloat16_t> = true;
#else
template <>
inline constexpr bool is_supported_input_type_v<uint16_t> = true;

template <>
inline constexpr bool is_supported_input_type_v<float> = true;
#endif

// Alignment, Padding & Index Calculation Helpers

SAGE_DEVICE_FUNC constexpr bool IsPowerOfTwo(uint32_t v) noexcept {
    return v != 0U && (v & (v - 1U)) == 0U;
}

SAGE_DEVICE_FUNC constexpr uint32_t RoundUpTo(uint32_t value, uint32_t align) noexcept {
    return (value + align - 1U) & ~(align - 1U);
}

SAGE_DEVICE_FUNC constexpr uint64_t RoundUpTo(uint64_t value, uint64_t align) noexcept {
    return (value + align - 1ULL) & ~(align - 1ULL);
}

// Head-dimension padding envelope: raw D in (0, 128], padded to 64 or 128
SAGE_DEVICE_FUNC constexpr uint32_t PaddedHeadDim(uint32_t d) noexcept {
    return d <= 64U ? 64U : 128U;
}

SAGE_DEVICE_FUNC constexpr uint64_t Offset(const uint64_t *s, uint64_t b, uint64_t h, uint64_t t) noexcept {
    return b * s[0] + h * s[1] + t * s[2];
}

// Core Execution Parameter Structs

// Runtime shape, layout, and execution parameters shared across Host and Device.
// Trivially copyable without tail padding.
struct SageParams {
    uint32_t b, hq, hk, sq, sk, d, dp, qp, kp, cores;
    uint32_t causal, smoothK, returnLse;
    uint32_t qg, kg;                // Tokens per shared INT8 scale (Q and K)
    SagePvAccum accum;
    uint32_t smoothV;
    float qFactor, kFactor;         // Precomputed FP16 dequantization balancing factors
    uint64_t qs[3], ks[3], vs[3];   // Canonical B, H, S strides, in elements
};

// Byte offsets relative to GetUserWorkspace(), aligned to 512 bytes
struct SageAttention2TilingData {
    SageParams params;
    uint64_t mean;
    uint64_t vMean;
    uint64_t vMxScale;
    uint64_t q8;
    uint64_t k8;
    uint64_t v8;
    uint64_t qScale;
    uint64_t kScale;
};

// Tiling key encoding helper for template compilation (dtype in low 8 bits, wide in bit 8)
constexpr uint64_t GET_TPL_TILING_KEY(uint32_t dtype, bool isWide) noexcept {
    return static_cast<uint64_t>(dtype) | (static_cast<uint64_t>(isWide ? 1U : 0U) << 8);
}

} // namespace sage::core

namespace sage {
using namespace core;
}

// CANN kernel codegen (DFX/meta sections and GET_TILING_DATA macros) references the
// tiling struct from the global namespace; the file-scope alias is required by the
// generated kernel translation unit, not by downstream code.
using SageAttention2TilingData = sage::core::SageAttention2TilingData;
using SageParams = sage::core::SageParams;
