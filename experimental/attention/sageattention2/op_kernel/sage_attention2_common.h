/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_COMMON_H_
#define SAGE_ATTENTION2_COMMON_H_

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>
#include "kernel_operator.h"

// CANN 9.0 beta exposes these data-type identifiers as macros, while the pinned Ascend 950 CATLASS
// revision expects namespace constants. Normalize the legacy headers locally without changing CATLASS.
#ifdef DT_FLOAT
#undef DT_FLOAT
#undef DT_FLOAT16
#undef DT_BF16
namespace AscendC {
constexpr uint32_t DT_FLOAT = 0U;
constexpr uint32_t DT_FLOAT16 = 1U;
constexpr uint32_t DT_BF16 = 27U;
} // namespace AscendC
#endif

namespace SageAttention2Kernel {

// Match one attention task to the native 128-row SageAttention2 quantization block. Each vector
// sub-block owns 64 rows. Keep the 256-column KV tile to avoid doubling online-softmax work.
constexpr uint32_t Q_TILE = 128U;
constexpr uint32_t KV_TILE = 256U;
constexpr uint32_t Q_QUANT_BLOCK = 128U;
constexpr uint32_t K_QUANT_BLOCK = 64U;
constexpr uint32_t WARP_ROWS = 32U;
constexpr uint32_t THREAD_ROWS = 8U;
constexpr float INT8_MAX_VALUE = 127.0F;
constexpr float FP8_MAX_VALUE = 448.0F;
constexpr float FP8_FP16_ACCUM_MAX_VALUE = 2.25F;
constexpr float LOG2_E = 1.4426950408889634F;
constexpr float MIN_SCORE = -3.0e38F;

enum class Layout : uint32_t { HND = 0U, NHD = 1U };
enum class QuantGran : uint32_t { PER_WARP = 0U, PER_THREAD = 1U };
enum class AccumMode : uint32_t { FP32 = 0U, FP32_FP32 = 1U, FP32_FP16 = 2U };

union Float32Bits {
    float value;
    uint32_t bits;
};

union Float16Bits {
    half value;
    uint16_t bits;
};

// Scalar BF16 casts are not supported by the Ascend 950 beta compiler backend. Keep all source/output
// tensors as their 16-bit representation and perform the BF16 conversion in integer space.
template <typename T>
__aicore__ inline float DecodeInput16(uint16_t raw)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        Float32Bits value;
        value.bits = static_cast<uint32_t>(raw) << 16U;
        return value.value;
    } else {
        Float16Bits value;
        value.bits = raw;
        return static_cast<float>(value.value);
    }
}

template <typename T>
__aicore__ inline uint16_t EncodeInput16(float raw)
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        Float32Bits value;
        value.value = raw;
        const uint32_t roundBias = 0x7FFFU + ((value.bits >> 16U) & 1U);
        return static_cast<uint16_t>((value.bits + roundBias) >> 16U);
    } else {
        Float16Bits value;
        value.value = static_cast<half>(raw);
        return value.bits;
    }
}

template <typename T>
__aicore__ inline T Min(T lhs, T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

template <typename T>
__aicore__ inline T Max(T lhs, T rhs)
{
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__aicore__ inline T Abs(T value)
{
    return value < static_cast<T>(0) ? -value : value;
}

__aicore__ inline int8_t RoundSaturateInt8(float value)
{
    value = Max(-INT8_MAX_VALUE, Min(INT8_MAX_VALUE, value));
    const int32_t rounded = value >= 0.0F ? static_cast<int32_t>(value + 0.5F) : static_cast<int32_t>(value - 0.5F);
    return static_cast<int8_t>(rounded);
}

// Ascend 950 FP8 types are pointer-only in scalar code. Encode E4M3FN in integer space so the
// preprocessing path can write the byte representation without constructing an FP8 scalar.
__aicore__ inline uint8_t EncodeFp8E4M3Fn(float value)
{
    const bool negative = value < 0.0F;
    float magnitude = negative ? -value : value;
    uint32_t payload = 0U;
    if (magnitude >= FP8_MAX_VALUE) {
        payload = (15U << 3U) | 6U;
    } else if (magnitude < 0.015625F) { // 2^-6: subnormal step is 2^-9.
        uint32_t mantissa = static_cast<uint32_t>(magnitude * 512.0F + 0.5F);
        payload = mantissa > 7U ? (1U << 3U) : mantissa;
    } else {
        int32_t exponent = 0;
        float normalized = magnitude;
        while (normalized >= 2.0F) {
            normalized *= 0.5F;
            ++exponent;
        }
        while (normalized < 1.0F) {
            normalized *= 2.0F;
            --exponent;
        }
        uint32_t biasedExponent = static_cast<uint32_t>(exponent + 7);
        uint32_t mantissa = static_cast<uint32_t>((normalized - 1.0F) * 8.0F + 0.5F);
        if (mantissa == 8U) {
            mantissa = 0U;
            ++biasedExponent;
        }
        if (biasedExponent >= 15U && mantissa > 6U) {
            biasedExponent = 15U;
            mantissa = 6U;
        }
        payload = (biasedExponent << 3U) | mantissa;
    }
    return static_cast<uint8_t>((negative ? 0x80U : 0U) | payload);
}

template <typename T>
__aicore__ inline uint64_t SourceOffset(
    Layout layout, uint32_t b, uint32_t h, uint32_t s, uint32_t d, uint32_t heads, uint32_t seq, uint32_t dim)
{
    if (layout == Layout::HND) {
        return ((static_cast<uint64_t>(b) * heads + h) * seq + s) * dim + d;
    }
    return ((static_cast<uint64_t>(b) * seq + s) * heads + h) * dim + d;
}

__aicore__ inline uint64_t BsNdOffset(
    uint32_t b, uint32_t h, uint32_t s, uint32_t d, uint32_t heads, uint32_t seq, uint32_t dim)
{
    return ((static_cast<uint64_t>(b) * seq + s) * heads + h) * dim + d;
}

__aicore__ inline uint32_t QScaleGroup(uint32_t rowInBlock, QuantGran gran)
{
    if (gran == QuantGran::PER_WARP) {
        return rowInBlock / WARP_ROWS;
    }
    return (rowInBlock / WARP_ROWS) * THREAD_ROWS + rowInBlock % THREAD_ROWS;
}

__aicore__ inline uint32_t KScaleGroup(uint32_t rowInBlock, QuantGran gran)
{
    return gran == QuantGran::PER_WARP ? 0U : (rowInBlock % THREAD_ROWS) / 2U;
}

} // namespace SageAttention2Kernel
#endif // SAGE_ATTENTION2_COMMON_H_
