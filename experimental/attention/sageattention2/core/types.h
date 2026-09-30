/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

// Clean-room core type layer (todo 模块一): the host/device-shared parameter
// and validation vocabulary. Deliberately host-compilable (no AscendC headers).
// Anything defined here agrees bit-for-bit with the verified microkernel constants.

#if defined(__CCE__) || defined(__AICORE__) || defined(ASCENDC_CPU_DEBUG)
#define SAGE_DEVICE_FUNC __aicore__
#else
#define SAGE_DEVICE_FUNC
#endif

namespace sage {

// Q/K INT8 quantization granularity.
enum class SageQuantGran : uint32_t { PER_THREAD = 0, PER_WARP = 1, COUNT = 2 };

// PV accumulation formats: FP16 rounds the per-block partial before folding
// it into the FP32 running state, as the open-source default does.
enum class SagePvAccum : uint32_t { FP32_FP16 = 0, FP32_FP32 = 1, FP32 = 2, COUNT = 3 };

constexpr uint32_t QUANT_TILE_ROWS = 64;
SAGE_DEVICE_FUNC constexpr uint32_t QTokensPerScale(bool perWarp) noexcept { return perWarp ? 32U : 4U; }
SAGE_DEVICE_FUNC constexpr uint32_t KTokensPerScale(bool perWarp) noexcept { return perWarp ? 64U : 16U; }
SAGE_DEVICE_FUNC constexpr bool DividesTile(uint32_t gran) noexcept { return QUANT_TILE_ROWS % gran == 0U; }

static_assert(DividesTile(QTokensPerScale(false)) && DividesTile(QTokensPerScale(true)) &&
                  DividesTile(KTokensPerScale(false)) && DividesTile(KTokensPerScale(true)),
              "quantization granularity must divide the 64-row tile");

SAGE_DEVICE_FUNC constexpr bool IsValidQuantGran(int64_t gran) noexcept
{
    return gran >= 0 && gran < static_cast<int64_t>(SageQuantGran::COUNT);
}

SAGE_DEVICE_FUNC constexpr bool IsValidPvAccum(int64_t accum) noexcept
{
    return accum >= 0 && accum < static_cast<int64_t>(SagePvAccum::COUNT);
}

// Full-string constexpr equality keeps the parsers libc-free and evaluable at compile time.
// String parsing is host-only (aclnn attr validation): bisheng places device
// string literals in the __gm__ address space, which const char* cannot bind,
// and the kernel never parses strings anyway.
#ifndef __CCE__
SAGE_DEVICE_FUNC constexpr bool StrEq(const char *a, const char *b) noexcept
{
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

SAGE_DEVICE_FUNC constexpr int64_t ParseQuantGran(const char *str) noexcept
{
    if (str == nullptr) return -1;
    if (StrEq(str, "per_thread")) return static_cast<int64_t>(SageQuantGran::PER_THREAD);
    if (StrEq(str, "per_warp")) return static_cast<int64_t>(SageQuantGran::PER_WARP);
    return -1;
}

SAGE_DEVICE_FUNC constexpr int64_t ParsePvAccum(const char *str) noexcept
{
    if (str == nullptr) return -1;
    if (StrEq(str, "fp32")) return static_cast<int64_t>(SagePvAccum::FP32);
    if (StrEq(str, "fp32+fp16")) return static_cast<int64_t>(SagePvAccum::FP32_FP16);
    if (StrEq(str, "fp32+fp32")) return static_cast<int64_t>(SagePvAccum::FP32_FP32);
    return -1;
}
#endif // __CCE__ host-only parsers

// Runtime shape, layout and execution parameters shared by all stages. The
// struct is memcpy'd host->device, so it must stay trivially copyable without tail padding.
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

SAGE_DEVICE_FUNC constexpr bool IsPowerOfTwo(uint32_t v) noexcept
{
    return v != 0U && (v & (v - 1U)) == 0U;
}

// Mask trick requires power-of-two alignment; callers pass tile heights and byte boundaries only.
SAGE_DEVICE_FUNC constexpr uint32_t RoundUpTo(uint32_t value, uint32_t align) noexcept
{
    return (value + align - 1U) & ~(align - 1U);
}
SAGE_DEVICE_FUNC constexpr uint64_t RoundUpTo(uint64_t value, uint64_t align) noexcept
{
    return (value + align - 1ULL) & ~(align - 1ULL);
}

// Task-spec head-dim envelope: raw D in (0, 128], padded to one of the two supported widths.
SAGE_DEVICE_FUNC constexpr uint32_t PaddedHeadDim(uint32_t d) noexcept
{
    return d <= 64U ? 64U : 128U;
}

// Tile geometry and die-affinity scheduling helpers for the sage:: layer (SnakeScheduler).
constexpr uint32_t M = 128, N = 256, HALF = 64;

SAGE_DEVICE_FUNC constexpr uint32_t CalcKh(uint32_t h, uint32_t hq, uint32_t hk) noexcept
{
    if (hq == hk) return h;
    const uint32_t ratio = hq / hk;
    return (ratio == 4) ? (h >> 2) :
           (ratio == 8) ? (h >> 3) :
           (ratio == 2) ? (h >> 1) :
           (ratio == 16) ? (h >> 4) : (h / ratio);
}

SAGE_DEVICE_FUNC constexpr bool UseDieGroups(const SageParams &p) noexcept
{
    return p.cores > 0 && p.cores % 2 == 0 && (uint64_t(p.b) * p.hk) % 2 == 0 &&
           p.sq % 128 == 0;
}

SAGE_DEVICE_FUNC constexpr uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) noexcept
{
    const uint32_t half = cores / 2;
    return (task % cores / half) * (total / 2) + (task / cores) * half + task % half;
}

SAGE_DEVICE_FUNC constexpr uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) noexcept
{
    if (!UseDieGroups(p)) return total;
    const uint32_t half = p.cores / 2, rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = local > rank ? (local - rank + half - 1) / half : 0;
    return core + count * p.cores;
}

constexpr int64_t MAX_HEAD_DIM = 128;

// Mathematical, quantization, and microarchitecture constants
constexpr float LOG2_E = 1.4426950408889634f;
constexpr float LN_448 = 6.104793232414985f;
constexpr uint64_t DEQ_SCALAR_DEQF16 = 0x3a000000ULL; // 2^-11
constexpr float INT8_MAX_FLOAT = 127.0f;
constexpr float INV_INT8_MAX = 1.0f / INT8_MAX_FLOAT;
constexpr float MX_FP8_MAX = 448.0f;
constexpr float INV_MX_FP8_MAX = 1.0f / MX_FP8_MAX;
constexpr uint32_t PV_PARTIAL_SHIFT = 8;
constexpr uint16_t PV_PARTIAL_E8M0 = 127U - PV_PARTIAL_SHIFT;
constexpr uint16_t PV_PARTIAL_E8M0_PAIR = static_cast<uint16_t>(PV_PARTIAL_E8M0 * 0x0101U);
constexpr float PV_PARTIAL_UNDO = static_cast<float>(1U << PV_PARTIAL_SHIFT);
static_assert(PV_PARTIAL_E8M0_PAIR == 0x7777U, "both P-scale bytes must carry 2^-shift");

constexpr uint32_t FAST_RECIPROCAL_MAGIC = 0x7EEF370B;
constexpr uint32_t EXPONENT_ROUND_BIAS = 0x007fffff;
constexpr uint32_t EXPONENT_MASK = 0x7f800000;
constexpr uint32_t RECIPROCAL_EXP_BIAS = 0x7f000000;
constexpr uint16_t FP16_ABS_MASK = 0x7fff;
constexpr float MIN_POSITIVE_FLOAT = 1.0e-30f;
constexpr uint32_t UB_HIGH_WATER = 248 * 1024;

// Pipeline slot depths and static buffer offsets
constexpr uint32_t PIPE_SLOTS = 3;
constexpr uint32_t LOOKAHEAD = PIPE_SLOTS - 1;
constexpr uint32_t ALPHA_SLOTS = 4;
constexpr uint32_t PV_SLOTS = 2;
constexpr uint32_t P_NZ_PAD_ROWS = 1;
constexpr uint32_t P_SLOT_BYTES = (HALF + P_NZ_PAD_ROWS) * N;
constexpr uint32_t P_L1 = 65536;
constexpr uint32_t B_L1 = P_L1 + PIPE_SLOTS * M * N;
constexpr uint32_t WIDE_B_L1 = P_L1 + 4 * M * N;
constexpr uint32_t PV_SLOT_BYTES = HALF * 128 * 4;
constexpr uint32_t SCORE_UB = 0;
constexpr uint32_t PV_UB = PIPE_SLOTS * HALF * N * 2;
constexpr uint32_t ACCUM_UB = PV_UB + PV_SLOTS * PV_SLOT_BYTES;
constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4;
constexpr uint32_t KS_UB = P_UB + PIPE_SLOTS * P_SLOT_BYTES;
constexpr uint32_t QS_UB = KS_UB + 2 * N * 2;
constexpr uint32_t GROUP_STAT_BYTES = N * 2;
constexpr uint32_t MAX_UB = QS_UB + GROUP_STAT_BYTES;
constexpr uint32_t DEN_UB = MAX_UB + GROUP_STAT_BYTES;
constexpr uint32_t ALPHA_UB = DEN_UB + GROUP_STAT_BYTES;
constexpr uint32_t SUM_UB = ALPHA_UB + ALPHA_SLOTS * GROUP_STAT_BYTES;
constexpr uint32_t VM_UB = SUM_UB + HALF * 4;
constexpr uint32_t UB_END = VM_UB + 128 * sizeof(float);

static_assert(UB_END <= UB_HIGH_WATER, "Static UB plan must stay within 248KB budget");

SAGE_DEVICE_FUNC constexpr uint64_t Offset(const uint64_t *s, uint64_t b, uint64_t h, uint64_t t) noexcept
{
    return b * s[0] + h * s[1] + t * s[2];
}

// Explicit on-chip communication protocol state machine (Task 1.3)
struct EventHandshake {
    static constexpr uint32_t QK_READY = 0;
    static constexpr uint32_t P_READY = 1;
    static constexpr uint32_t PV_READY = 2;
    static constexpr uint32_t PV_RELEASE = 3;
    static constexpr uint32_t Q_READY = 4;
    static constexpr uint32_t AIV_FLAG_OFFSET = 16;
    static constexpr uint32_t QK_EVENT = 2;
    static constexpr uint32_t PV_EVENT = 3;
    static constexpr uint32_t SCALE_EVENT_BASE = 4;
    static constexpr uint32_t QK_SECOND_FIX_EVENT = 6;
};

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

// Byte offsets relative to GetUserWorkspace(), aligned to 512 bytes.
struct SageAttention2TilingData {
    SageParams params;
    uint64_t mean, vMean, vMxScale, q8, k8, v8, qScale, kScale;
};

} // namespace sage

