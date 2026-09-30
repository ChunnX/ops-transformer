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
#include <type_traits>
// Q/K INT8 quantization granularity, expressed as the number of consecutive
// tokens sharing one scale. Warp granularity widens Q by 8x (4 to 32) but K by
// only 4x (16 to 64); the two axes are scaled independently, not uniformly.
enum class SageQuantGran : uint32_t { PER_THREAD = 0, PER_WARP = 1, COUNT = 2 };

// PV accumulation formats. FP16 keeps the per-block partial in half precision
// before folding it into the FP32 running state, as the open-source default.
enum class SagePvAccum : uint32_t { FP32_FP16 = 0, FP32_FP32 = 1, FP32 = 2, COUNT = 3 };

// One INT8 scale covers `gran` consecutive tokens inside a 64-row tile, and
// BroadcastGroupMaxima replicates it without crossing a tile edge, so every
// granularity has to divide the tile height.
constexpr uint32_t QUANT_TILE_ROWS = 64;
constexpr uint32_t QTokensPerScale(bool perWarp) noexcept
{
    return perWarp ? 32U : 4U;
}
constexpr uint32_t KTokensPerScale(bool perWarp) noexcept
{
    return perWarp ? 64U : 16U;
}
constexpr bool DividesTile(uint32_t gran) noexcept
{
    return QUANT_TILE_ROWS % gran == 0U;
}
static_assert(DividesTile(QTokensPerScale(false)) && DividesTile(QTokensPerScale(true)) &&
                  DividesTile(KTokensPerScale(false)) && DividesTile(KTokensPerScale(true)),
              "quantization granularity must divide the 64-row tile");

constexpr bool IsValidQuantGran(int64_t gran) noexcept
{
    return gran >= 0 && gran < static_cast<int64_t>(SageQuantGran::COUNT);
}

constexpr bool IsValidPvAccum(int64_t accum) noexcept
{
    return accum >= 0 && accum < static_cast<int64_t>(SagePvAccum::COUNT);
}

// Full-string constexpr equality. Keeps the parsers libc-free (params.h is
// shared by host and device translation units) and evaluable at compile time.
constexpr bool StrEq(const char *a, const char *b) noexcept
{
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

constexpr int64_t ParseQuantGran(const char *str) noexcept
{
    if (str == nullptr)
        return -1;
    if (StrEq(str, "per_thread"))
        return static_cast<int64_t>(SageQuantGran::PER_THREAD);
    if (StrEq(str, "per_warp"))
        return static_cast<int64_t>(SageQuantGran::PER_WARP);
    return -1;
}

constexpr int64_t ParsePvAccum(const char *str) noexcept
{
    if (str == nullptr)
        return -1;
    if (StrEq(str, "fp32"))
        return static_cast<int64_t>(SagePvAccum::FP32);
    if (StrEq(str, "fp32+fp16"))
        return static_cast<int64_t>(SagePvAccum::FP32_FP16);
    if (StrEq(str, "fp32+fp32"))
        return static_cast<int64_t>(SagePvAccum::FP32_FP32);
    return -1;
}

// Runtime shape, layout and execution parameters shared by all stages.
// sm_scale itself is deliberately absent: the kernel only consumes its
// pre-fused FP16 form (qFactor/kFactor), so shipping it would put a field in
// every tiling payload that no device instruction ever reads.
struct SageParams {
    uint32_t b, hq, hk, sq, sk, d, dp, qp, kp, cores;
    uint32_t causal, smoothK, returnLse;
    uint32_t qg, kg; // tokens per shared INT8 scale, for Q and K
    SagePvAccum accum;
    uint32_t smoothV;
    float qFactor, kFactor; // Precomputed FP16 dequantization balancing scale factors
    uint64_t qs[3], ks[3], vs[3]; // canonical B,H,S strides, in elements
};

// Task-spec head-dim envelope: raw D must lie in (0, MAX_HEAD_DIM] and is
// padded up to one of the two supported widths. Defined once here because
// host tiling, the aclnn entry and the torch bridge all enforce it.
constexpr int64_t MAX_HEAD_DIM = 128;
// The kernel's two supported padded head widths; DP64 also covers every D < 64.
constexpr uint32_t PaddedHeadDim(uint32_t d) noexcept
{
    return d <= 64U ? 64U : 128U;
}
static_assert(PaddedHeadDim(1U) == 64U && PaddedHeadDim(64U) == 64U &&
                  PaddedHeadDim(65U) == 128U && PaddedHeadDim(128U) == 128U,
              "padded head widths must match the spec pad points");

// Widest finite value representable in the FP16 format (bounds checks on the
// host side and the PV partial-fold envelope on the device side share it).
// Sage-prefixed: CANN headers define bare FP16_MAX/FP16_MIN macros.
constexpr float SAGE_FP16_MAX = 65504.0f;
constexpr float SAGE_FP16_MIN = -SAGE_FP16_MAX;

// op_host copies this struct to the device with a plain memcpy, so any member
// that stops being POD would silently corrupt the payload rather than fail.
static_assert(std::is_trivially_copyable_v<SageParams> && std::is_standard_layout_v<SageParams>,
              "SageParams is memcpy'd to the device");
static_assert(sizeof(SageParams) % 8 == 0, "SageParams must not carry tail padding");

constexpr bool IsPow2(uint32_t v) noexcept
{
    return v != 0U && (v & (v - 1U)) == 0U;
}

// RoundUpTo's mask trick is only defined for power-of-two alignments; every
// call site passes a tile height or a byte boundary, so catch the mistake here
// instead of returning a silently wrong padding.
constexpr uint32_t RoundUpTo(uint32_t v, uint32_t align) noexcept
{
    return (v + align - 1U) & ~(align - 1U);
}
constexpr uint64_t RoundUpTo(uint64_t v, uint64_t align) noexcept
{
    return (v + align - 1ULL) & ~(align - 1ULL);
}
static_assert(IsPow2(128U) && IsPow2(512U), "tile padding alignments must stay power-of-two");
