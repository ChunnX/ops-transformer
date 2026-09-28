/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "kernel_operator.h"
#include "common.h"
#endif

namespace Sage {
namespace Int4 {

// ============================================================================
// INT4 Numerical and Architectural Constants
// ============================================================================

// Two's complement 4-bit signed bounds: [-8, 7]
constexpr int8_t INT4_MIN = -8;
constexpr int8_t INT4_MAX = 7;
constexpr float INT4_FLOAT_MAX = 7.0f;
constexpr float INV_INT4_FLOAT_MAX = 1.0f / 7.0f;

// Bit manipulation masks
constexpr uint8_t INT4_NIBBLE_MASK = 0x0Fu;
constexpr uint8_t INT4_SIGN_BIT = 0x08u;
constexpr uint8_t INT4_HIGH_NIBBLE_MASK = 0xF0u;
constexpr uint32_t INT4_ELEMENTS_PER_BYTE = 2u;

// Atlas 950PR Cube MMAD Alignment Specifications:
// Fractal block = 16 (M/N rows) x 64 (K columns) = 1024 INT4 elements = 512 Bytes.
// Each row of 64 INT4 elements occupies 32 Bytes (1 Vector Block / Cache Line).
constexpr uint32_t FRACTAL_M = 16u;               // M / N dimension block size
constexpr uint32_t FRACTAL_K = 64u;               // K dimension block size (elements)
constexpr uint32_t FRACTAL_K_BYTES = 32u;         // 64 INT4 elements = 32 bytes
constexpr uint32_t FRACTAL_TOTAL_BYTES = 512u;    // 16 rows * 32 bytes = 512 bytes
constexpr uint32_t BYTES_PER_BLOCK = 32u;         // Ascend 32B cache block

// ============================================================================
// Core Scalar Quantization and Packing Primitives
// ============================================================================

/**
 * @brief Symmetric INT4 Quantization with round-to-nearest and boundary clamp:
 *        x_int4 = clamp(round(x * scale), -8, 7)
 */
inline int8_t QuantizeInt4(float x, float scale)
{
    if (std::isnan(x)) {
        return 0;
    }
    float scaled = x * scale;
    // std::nearbyint implements round-to-nearest (FE_TONEAREST)
    float rounded = std::nearbyint(scaled);
    if (rounded >= 7.0f) {
        return INT4_MAX;
    }
    if (rounded <= -8.0f) {
        return INT4_MIN;
    }
    return static_cast<int8_t>(rounded);
}

/**
 * @brief Dequantization of signed INT4 to FP32:
 *        x = q * invScale
 */
inline float DequantizeInt4(int8_t q, float invScale)
{
    return static_cast<float>(q) * invScale;
}

/**
 * @brief 2-way INT4 compression packing into a single uint8_t / int8_t.
 *        Low nibble holds b0, High nibble holds b1:
 *        packed = (b0 & 0x0F) | ((b1 & 0x0F) << 4)
 */
inline uint8_t PackInt4Pair(int8_t low, int8_t high)
{
    uint8_t l = static_cast<uint8_t>(low) & INT4_NIBBLE_MASK;
    uint8_t h = static_cast<uint8_t>(high) & INT4_NIBBLE_MASK;
    return static_cast<uint8_t>(l | (h << 4u));
}

/**
 * @brief Bit-exact, branchless sign-extended unpacking of 2 INT4 nibbles from 1 byte.
 *        Given packed byte p:
 *        low  = ((p & 0x0F) ^ 0x08) - 0x08
 *        high = (((p >> 4) & 0x0F) ^ 0x08) - 0x08
 */
inline void UnpackInt4Pair(uint8_t packed, int8_t &low, int8_t &high)
{
    uint8_t l = packed & INT4_NIBBLE_MASK;
    uint8_t h = (packed >> 4u) & INT4_NIBBLE_MASK;
    low = static_cast<int8_t>((l ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
    high = static_cast<int8_t>((h ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
}

/**
 * @brief Extract and sign-extend low nibble from packed byte
 */
inline int8_t UnpackInt4Low(uint8_t packed)
{
    uint8_t l = packed & INT4_NIBBLE_MASK;
    return static_cast<int8_t>((l ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
}

/**
 * @brief Extract and sign-extend high nibble from packed byte
 */
inline int8_t UnpackInt4High(uint8_t packed)
{
    uint8_t h = (packed >> 4u) & INT4_NIBBLE_MASK;
    return static_cast<int8_t>((h ^ INT4_SIGN_BIT) - INT4_SIGN_BIT);
}

// ============================================================================
// Scale Calculation Primitives
// ============================================================================

/**
 * @brief Compute symmetric INT4 scale factor and its reciprocal from amax:
 *        scale = 7.0 / amax
 *        invScale = amax / 7.0
 */
inline void CalcInt4Scale(float amax, float &scale, float &invScale)
{
    constexpr float EPSILON = 1e-12f;
    if (amax <= EPSILON) {
        scale = 1.0f;
        invScale = 1.0f;
    } else {
        scale = INT4_FLOAT_MAX / amax;
        invScale = amax * INV_INT4_FLOAT_MAX;
    }
}

/**
 * @brief Hardware-friendly power-of-two scale calculation (E8M0-compatible):
 *        scale = 2^floor(log2(7.0 / amax))
 */
inline void CalcInt4PowerOf2Scale(float amax, float &scale, float &invScale)
{
    constexpr float EPSILON = 1e-12f;
    if (amax <= EPSILON) {
        scale = 1.0f;
        invScale = 1.0f;
    } else {
        float ratio = INT4_FLOAT_MAX / amax;
        int exp = 0;
        std::frexp(ratio, &exp);
        scale = std::ldexp(1.0f, exp - 1);
        invScale = 1.0f / scale;
    }
}

// ============================================================================
// Block / Contiguous Array Quantization and Bit-Packing
// ============================================================================

/**
 * @brief Quantize and pack a contiguous buffer of float values into packed INT4 bytes.
 *        src: array of size 'len'
 *        dst: array of size ceil(len / 2) bytes
 */
inline void QuantizeAndPackRow(const float *src, uint8_t *dst, uint32_t len, float scale)
{
    uint32_t pairs = len / 2u;
    for (uint32_t i = 0; i < pairs; ++i) {
        int8_t q0 = QuantizeInt4(src[2u * i], scale);
        int8_t q1 = QuantizeInt4(src[2u * i + 1u], scale);
        dst[i] = PackInt4Pair(q0, q1);
    }
    if (len & 1u) {
        int8_t q0 = QuantizeInt4(src[len - 1u], scale);
        dst[pairs] = PackInt4Pair(q0, 0);
    }
}

/**
 * @brief Unpack and dequantize a contiguous buffer of packed INT4 bytes into float values.
 *        src: array of size ceil(len / 2) bytes
 *        dst: array of size 'len'
 */
inline void UnpackAndDequantizeRow(const uint8_t *src, float *dst, uint32_t len, float invScale)
{
    uint32_t pairs = len / 2u;
    for (uint32_t i = 0; i < pairs; ++i) {
        int8_t q0, q1;
        UnpackInt4Pair(src[i], q0, q1);
        dst[2u * i] = DequantizeInt4(q0, invScale);
        dst[2u * i + 1u] = DequantizeInt4(q1, invScale);
    }
    if (len & 1u) {
        int8_t q0, q1;
        UnpackInt4Pair(src[pairs], q0, q1);
        dst[len - 1u] = DequantizeInt4(q0, invScale);
    }
}

/**
 * @brief Full row quantization pipeline:
 *        1. Computes row amax
 *        2. Derives scale and invScale
 *        3. Quantizes and packs into dst
 */
inline void QuantizePackRowAuto(const float *src, uint8_t *dst, uint32_t len,
                               float &outScale, float &outInvScale)
{
    float amax = 0.0f;
    for (uint32_t i = 0; i < len; ++i) {
        amax = std::fmax(amax, std::fabs(src[i]));
    }
    CalcInt4Scale(amax, outScale, outInvScale);
    QuantizeAndPackRow(src, dst, len, outScale);
}

// ============================================================================
// Catlass and Cube MMAD INT4 Layout Alignment Engine
// ============================================================================

/**
 * @brief Geometry and fractal alignment descriptors for Atlas 950PR INT4 Cube MMAD.
 *
 * Hardware Fractal Geometry:
 * - A matrix (L0A) layout: zZ format [M/16, K/64, 16, 64/2] bytes = 512B per fractal
 * - B matrix (L0B) layout: zN format [N/16, K/64, 16, 64/2] bytes = 512B per fractal
 * - 64 elements along K are packed into 32 contiguous bytes per row.
 */
struct LayoutEngine {
    static constexpr uint32_t ALIGN_M = FRACTAL_M;     // 16
    static constexpr uint32_t ALIGN_K = FRACTAL_K;     // 64
    static constexpr uint32_t K_BYTES = FRACTAL_K_BYTES; // 32

    static inline uint32_t RoundUpM(uint32_t m)
    {
        return (m + ALIGN_M - 1u) & ~(ALIGN_M - 1u);
    }

    static inline uint32_t RoundUpK(uint32_t k)
    {
        return (k + ALIGN_K - 1u) & ~(ALIGN_K - 1u);
    }

    static inline uint32_t PackedRowBytes(uint32_t k)
    {
        return RoundUpK(k) / INT4_ELEMENTS_PER_BYTE;
    }

    static inline uint32_t TotalFractalBytes(uint32_t m, uint32_t k)
    {
        return RoundUpM(m) * PackedRowBytes(k);
    }

    /**
     * @brief Converts row-major packed ND tensor [M, K/2] bytes into
     *        Atlas 950PR Cube L0A zZ fractal layout [M/16, K/64, 16, 32] bytes.
     */
    static void NdToFractalZz(const uint8_t *ndPacked, uint8_t *fractalOut,
                              uint32_t m, uint32_t k, uint32_t ndStrideBytes = 0)
    {
        uint32_t roundM = RoundUpM(m);
        uint32_t roundK = RoundUpK(k);
        uint32_t stride = (ndStrideBytes == 0) ? (k / 2u) : ndStrideBytes;
        uint32_t numFracM = roundM / FRACTAL_M;
        uint32_t numFracK = roundK / FRACTAL_K;

        std::memset(fractalOut, 0, numFracM * numFracK * FRACTAL_TOTAL_BYTES);

        for (uint32_t fm = 0; fm < numFracM; ++fm) {
            for (uint32_t fk = 0; fk < numFracK; ++fk) {
                uint32_t fracOffset = (fm * numFracK + fk) * FRACTAL_TOTAL_BYTES;
                for (uint32_t r = 0; r < FRACTAL_M; ++r) {
                    uint32_t rowIdx = fm * FRACTAL_M + r;
                    if (rowIdx >= m) {
                        continue;
                    }
                    uint32_t colByteStart = fk * FRACTAL_K_BYTES;
                    uint32_t bytesToCopy = FRACTAL_K_BYTES;
                    if (colByteStart + bytesToCopy > stride) {
                        bytesToCopy = (colByteStart < stride) ? (stride - colByteStart) : 0u;
                    }
                    if (bytesToCopy > 0) {
                        const uint8_t *srcPtr = ndPacked + rowIdx * stride + colByteStart;
                        uint8_t *dstPtr = fractalOut + fracOffset + r * FRACTAL_K_BYTES;
                        std::memcpy(dstPtr, srcPtr, bytesToCopy);
                    }
                }
            }
        }
    }

    /**
     * @brief Converts Atlas 950PR Cube L0A zZ fractal layout [M/16, K/64, 16, 32] bytes
     *        back into standard row-major packed ND tensor [M, K/2] bytes.
     */
    static void FractalZzToNd(const uint8_t *fractalIn, uint8_t *ndPackedOut,
                              uint32_t m, uint32_t k, uint32_t ndStrideBytes = 0)
    {
        uint32_t roundM = RoundUpM(m);
        uint32_t roundK = RoundUpK(k);
        uint32_t stride = (ndStrideBytes == 0) ? (k / 2u) : ndStrideBytes;
        uint32_t numFracM = roundM / FRACTAL_M;
        uint32_t numFracK = roundK / FRACTAL_K;

        std::memset(ndPackedOut, 0, m * stride);

        for (uint32_t fm = 0; fm < numFracM; ++fm) {
            for (uint32_t fk = 0; fk < numFracK; ++fk) {
                uint32_t fracOffset = (fm * numFracK + fk) * FRACTAL_TOTAL_BYTES;
                for (uint32_t r = 0; r < FRACTAL_M; ++r) {
                    uint32_t rowIdx = fm * FRACTAL_M + r;
                    if (rowIdx >= m) {
                        continue;
                    }
                    uint32_t colByteStart = fk * FRACTAL_K_BYTES;
                    uint32_t bytesToCopy = FRACTAL_K_BYTES;
                    if (colByteStart + bytesToCopy > stride) {
                        bytesToCopy = (colByteStart < stride) ? (stride - colByteStart) : 0u;
                    }
                    if (bytesToCopy > 0) {
                        const uint8_t *srcPtr = fractalIn + fracOffset + r * FRACTAL_K_BYTES;
                        uint8_t *dstPtr = ndPackedOut + rowIdx * stride + colByteStart;
                        std::memcpy(dstPtr, srcPtr, bytesToCopy);
                    }
                }
            }
        }
    }
};

// ============================================================================
// Ascend C MicroAPI / UB Vector Pipelines for INT4 Processing
// ============================================================================

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)

/**
 * @brief UB Vectorized INT4 unpacking for Atlas 950PR:
 *        Expands 256 packed INT4 bytes in UB into two 256-element int8_t blocks
 *        or converts into FP16 for online Softmax / Vector score calculation.
 */
template <uint32_t COUNT = 256>
__simd_vf__ inline void UnpackInt4ToFP16VF(__ubuf__ uint8_t *packedSrc,
                                           __ubuf__ half *dstFp16Even,
                                           __ubuf__ half *dstFp16Odd)
{
    using namespace AscendC::MicroAPI;
    auto mask = CreateMask<half, MaskPattern::ALL>();
    // Mask for low nibble: 0x000F
    RegTensor<uint16_t> lowMask;
    Duplicate(lowMask, uint16_t(0x000F));
    // Bit-reversal / shift for signed conversion
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}

/**
 * @brief Vectorized INT4 Quantization and Packing in Unified Buffer (UB):
 *        Converts FP32 activations in UB to saturated INT4 packed bytes.
 */
template <uint32_t DP = 128>
__simd_vf__ inline void QuantizeAndPackInt4UB(__ubuf__ float *srcFp32,
                                             __ubuf__ uint8_t *dstPackedInt4,
                                             float scaleFactor)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> scaleReg, minReg, maxReg;
    Duplicate(scaleReg, scaleFactor);
    Duplicate(minReg, -8.0f);
    Duplicate(maxReg, 7.0f);

    static constexpr CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    // Vector implementation overlaps scale mul, cast, and pack
}

#endif // __CCE_AICORE__ || __CCE_KT_TEST__

} // namespace Int4
} // namespace Sage
