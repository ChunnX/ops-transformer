/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

// Clean-room core type layer (todo 模块一): the host/device-shared parameter
// and validation vocabulary, consumed through the op_kernel/params.h
// compatibility shims. Deliberately host-compilable (no AscendC headers):
// device intrinsics live in op_kernel/common.h and friends. Anything defined
// here must agree bit-for-bit with the verified kernel constants; parallel
// copies of kernel internals belong in the kernel headers only.

namespace sage {

// Q/K INT8 quantization granularity.
enum class SageQuantGran : uint32_t { PER_THREAD = 0, PER_WARP = 1, COUNT = 2 };

// PV accumulation formats: FP16 rounds the per-block partial before folding
// it into the FP32 running state, as the open-source default does.
enum class SagePvAccum : uint32_t { FP32_FP16 = 0, FP32_FP32 = 1, FP32 = 2, COUNT = 3 };

constexpr uint32_t QUANT_TILE_ROWS = 64;
constexpr uint32_t QTokensPerScale(bool perWarp) noexcept { return perWarp ? 32U : 4U; }
constexpr uint32_t KTokensPerScale(bool perWarp) noexcept { return perWarp ? 64U : 16U; }
constexpr bool DividesTile(uint32_t gran) noexcept { return QUANT_TILE_ROWS % gran == 0U; }

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

// Full-string constexpr equality keeps the parsers libc-free and evaluable at
// compile time (params.h is shared by host and device translation units).
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
    if (str == nullptr) return -1;
    if (StrEq(str, "per_thread")) return static_cast<int64_t>(SageQuantGran::PER_THREAD);
    if (StrEq(str, "per_warp")) return static_cast<int64_t>(SageQuantGran::PER_WARP);
    return -1;
}

constexpr int64_t ParsePvAccum(const char *str) noexcept
{
    if (str == nullptr) return -1;
    if (StrEq(str, "fp32")) return static_cast<int64_t>(SagePvAccum::FP32);
    if (StrEq(str, "fp32+fp16")) return static_cast<int64_t>(SagePvAccum::FP32_FP16);
    if (StrEq(str, "fp32+fp32")) return static_cast<int64_t>(SagePvAccum::FP32_FP32);
    return -1;
}

// Runtime shape, layout and execution parameters shared by all stages. The
// struct is memcpy'd host->device, so it must stay trivially copyable without
// tail padding; op_kernel/params.h carries the layout assertions.
struct SageParams {
    uint32_t b, hq, hk, sq, sk, d, dp, qp, kp, cores;
    uint32_t causal, smoothK, returnLse;
    uint32_t qg, kg; // tokens per shared INT8 scale, for Q and K
    SagePvAccum accum;
    uint32_t smoothV;
    float qFactor, kFactor; // precomputed FP16 dequantization balancing factors
    uint64_t qs[3], ks[3], vs[3]; // canonical B,H,S strides, in elements
};

constexpr float SAGE_FP16_MAX = 65504.0f;
constexpr float SAGE_FP16_MIN = -SAGE_FP16_MAX;

constexpr bool IsPowerOfTwo(uint32_t v) noexcept
{
    return v != 0U && (v & (v - 1U)) == 0U;
}

// Mask trick requires power-of-two alignment; callers pass tile heights and
// byte boundaries only (params.h asserts representative alignments).
constexpr uint32_t RoundUpTo(uint32_t value, uint32_t align) noexcept
{
    return (value + align - 1U) & ~(align - 1U);
}
constexpr uint64_t RoundUpTo(uint64_t value, uint64_t align) noexcept
{
    return (value + align - 1ULL) & ~(align - 1ULL);
}

// Task-spec head-dim envelope: raw D in (0, 128], padded to one of the two
// supported widths (op_kernel/params.h owns the shared MAX_HEAD_DIM literal).
constexpr uint32_t PaddedHeadDim(uint32_t d) noexcept
{
    return d <= 64U ? 64U : 128U;
}

} // namespace sage
