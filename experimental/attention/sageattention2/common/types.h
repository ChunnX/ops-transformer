/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <type_traits>
#include "constants.h"

#if defined(__CCE__) || defined(__AICORE__) || defined(ASCENDC_CPU_DEBUG)
#define SAGE_DEVICE_FUNC __aicore__
#else
#define SAGE_DEVICE_FUNC
#endif

namespace sage {

// ============================================================================
// Quantization and Accumulation Enums
// ============================================================================

// Q/K INT8 quantization granularity
enum class SageQuantGran : uint32_t {
    PER_THREAD = 0,
    PER_WARP = 1,
    COUNT = 2
};

// PV accumulation format:
// FP32_FP16: rounds the per-block partial before folding into FP32 running state
// FP32_FP32: folds intermediate partial in full precision (shares datapath with FP32)
// FP32: native FP32 accumulator
enum class SagePvAccum : uint32_t {
    FP32_FP16 = 0,
    FP32_FP32 = 1,
    FP32 = 2,
    COUNT = 3
};

// ============================================================================
// Granularity and Tile Validation Helpers
// ============================================================================

SAGE_DEVICE_FUNC constexpr uint32_t QTokensPerScale(bool perWarp) noexcept {
    return perWarp ? 32U : 4U;
}

SAGE_DEVICE_FUNC constexpr uint32_t KTokensPerScale(bool perWarp) noexcept {
    return perWarp ? 64U : 16U;
}

SAGE_DEVICE_FUNC constexpr bool DividesTile(uint32_t gran) noexcept {
    return QUANT_TILE_ROWS % gran == 0U;
}

static_assert(DividesTile(QTokensPerScale(false)) && DividesTile(QTokensPerScale(true)) &&
              DividesTile(KTokensPerScale(false)) && DividesTile(KTokensPerScale(true)),
              "Quantization granularity must divide the 64-row tile");

SAGE_DEVICE_FUNC constexpr bool IsValidQuantGran(int64_t gran) noexcept {
    return gran >= 0 && gran < static_cast<int64_t>(SageQuantGran::COUNT);
}

SAGE_DEVICE_FUNC constexpr bool IsValidPvAccum(int64_t accum) noexcept {
    return accum >= 0 && accum < static_cast<int64_t>(SagePvAccum::COUNT);
}

// Host-only constexpr string parsing (prevents device string GM address space errors)
#ifndef __CCE__
SAGE_DEVICE_FUNC constexpr bool StrEq(const char *a, const char *b) noexcept {
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

SAGE_DEVICE_FUNC constexpr int64_t ParseQuantGran(const char *str) noexcept {
    if (str == nullptr) return -1;
    if (StrEq(str, "per_thread")) return static_cast<int64_t>(SageQuantGran::PER_THREAD);
    if (StrEq(str, "per_warp")) return static_cast<int64_t>(SageQuantGran::PER_WARP);
    return -1;
}

SAGE_DEVICE_FUNC constexpr int64_t ParsePvAccum(const char *str) noexcept {
    if (str == nullptr) return -1;
    if (StrEq(str, "fp32")) return static_cast<int64_t>(SagePvAccum::FP32);
    if (StrEq(str, "fp32+fp16")) return static_cast<int64_t>(SagePvAccum::FP32_FP16);
    if (StrEq(str, "fp32+fp32")) return static_cast<int64_t>(SagePvAccum::FP32_FP32);
    return -1;
}
#endif // __CCE__

// ============================================================================
// Mathematical and Alignment Utilities
// ============================================================================

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

// ============================================================================
// Core Execution Parameter Structs
// ============================================================================

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
    uint64_t mean, vMean, vMxScale, q8, k8, v8, qScale, kScale;
};

// ============================================================================
// Type Traits
// ============================================================================

#if defined(__CCE__) || defined(__AICORE__) || defined(ASCENDC_CPU_DEBUG)
template <typename T>
struct is_supported_input_type : std::false_type {};
template <>
struct is_supported_input_type<half> : std::true_type {};
template <>
struct is_supported_input_type<bfloat16_t> : std::true_type {};
template <typename T>
inline constexpr bool is_supported_input_type_v = is_supported_input_type<T>::value;
#else
template <typename T>
inline constexpr bool is_supported_input_type_v = true;
#endif

} // namespace sage
