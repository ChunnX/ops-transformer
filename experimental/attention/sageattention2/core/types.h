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

#ifndef __aicore__
#define __aicore__
#endif

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/gemm.hpp"
#endif

namespace sage {

// Q/K INT8 quantization granularity
enum class SageQuantGran : uint32_t {
    PER_THREAD = 0,
    PER_WARP = 1,
    COUNT = 2
};

// PV accumulation precision formats
enum class SagePvAccum : uint32_t {
    FP32_FP16 = 0,
    FP32_FP32 = 1,
    FP32 = 2,
    COUNT = 3
};

constexpr uint32_t QUANT_TILE_ROWS = 64;
constexpr uint32_t QTokensPerScale(bool perWarp) noexcept { return perWarp ? 32U : 4U; }
constexpr uint32_t KTokensPerScale(bool perWarp) noexcept { return perWarp ? 64U : 16U; }
constexpr bool DividesTile(uint32_t gran) noexcept { return QUANT_TILE_ROWS % gran == 0U; }

static_assert(DividesTile(QTokensPerScale(false)) && DividesTile(QTokensPerScale(true)) &&
              DividesTile(KTokensPerScale(false)) && DividesTile(KTokensPerScale(true)),
              "quantization granularity must divide the 64-row tile");

constexpr bool IsValidQuantGran(int64_t gran) noexcept {
    return gran >= 0 && gran < static_cast<int64_t>(SageQuantGran::COUNT);
}

constexpr bool IsValidPvAccum(int64_t accum) noexcept {
    return accum >= 0 && accum < static_cast<int64_t>(SagePvAccum::COUNT);
}

constexpr bool StrEq(const char *a, const char *b) noexcept {
    while (*a != '\0' && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}

constexpr int64_t ParseQuantGran(const char *str) noexcept {
    if (str == nullptr) return -1;
    if (StrEq(str, "per_thread")) return static_cast<int64_t>(SageQuantGran::PER_THREAD);
    if (StrEq(str, "per_warp")) return static_cast<int64_t>(SageQuantGran::PER_WARP);
    return -1;
}

constexpr int64_t ParsePvAccum(const char *str) noexcept {
    if (str == nullptr) return -1;
    if (StrEq(str, "fp32")) return static_cast<int64_t>(SagePvAccum::FP32);
    if (StrEq(str, "fp32+fp16")) return static_cast<int64_t>(SagePvAccum::FP32_FP16);
    if (StrEq(str, "fp32+fp32")) return static_cast<int64_t>(SagePvAccum::FP32_FP32);
    return -1;
}

// Runtime kernel and shape parameters
struct SageParams {
    uint32_t b, hq, hk, sq, sk, d, dp, qp, kp, cores;
    uint32_t causal, smoothK, returnLse;
    uint32_t qg, kg;
    SagePvAccum accum;
    uint32_t smoothV;
    float qFactor, kFactor;
    uint64_t qs[3], ks[3], vs[3];
};

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
using ::half;
using ::bfloat16_t;
using ::float8_e4m3_t;
using ::float8_e8m0_t;

template <typename T>
struct is_supported_input_type : std::false_type {};
template <>
struct is_supported_input_type<half> : std::true_type {};
template <>
struct is_supported_input_type<bfloat16_t> : std::true_type {};

template <typename T>
inline constexpr bool is_supported_input_type_v = is_supported_input_type<T>::value;
#endif

constexpr bool IsPowerOfTwo(uint32_t v) noexcept {
    return v != 0U && (v & (v - 1U)) == 0U;
}

constexpr uint32_t RoundUpTo(uint32_t value, uint32_t multiple) noexcept {
    return (value + multiple - 1U) / multiple * multiple;
}

constexpr uint32_t PaddedHeadDim(uint32_t d) noexcept {
    return d <= 64U ? 64U : 128U;
}

__aicore__ inline constexpr uint32_t CalcKh(uint32_t h, uint32_t hq, uint32_t hk) noexcept {
    const uint32_t ratio = hq / hk;
    return (ratio == 1) ? h :
           (ratio == 4) ? (h >> 2) :
           (ratio == 8) ? (h >> 3) :
           (ratio == 2) ? (h >> 1) :
           (ratio == 16) ? (h >> 4) : (h / ratio);
}

__aicore__ inline constexpr bool UseDieGroups(const SageParams &p) noexcept {
    return p.cores > 0 && p.cores % 2 == 0 && (uint64_t(p.b) * p.hk) % 2 == 0 &&
           p.sq % 128 == 0;
}

__aicore__ inline constexpr uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) noexcept {
    const uint32_t half = cores / 2;
    return (task % cores / half) * (total / 2) + (task / cores) * half + task % half;
}

__aicore__ inline constexpr uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) noexcept {
    if (!UseDieGroups(p)) return total;
    const uint32_t half = p.cores / 2, rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = local > rank ? (local - rank + half - 1) / half : 0;
    return core + count * p.cores;
}

__aicore__ inline constexpr uint64_t Offset(uint64_t const *strides, uint32_t b, uint32_t h, uint32_t s) noexcept {
    return b * strides[0] + h * strides[1] + s * strides[2];
}

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
// Atlas 950PR Architecture Constants
using Arch = Catlass::Arch::Ascend950;
#endif
constexpr uint32_t M = 128, N = 256, HALF = 64;
constexpr uint32_t UB_HIGH_WATER = 224 * 1024; // ThunderKittens 218KB-224KB safe UB ceiling
constexpr uint32_t P_SLOT_BYTES = HALF * N;

constexpr uint32_t FAST_RECIPROCAL_MAGIC = 0x7EF311C2U;
constexpr uint32_t PV_PARTIAL_SHIFT = 8;
constexpr float PV_PARTIAL_UNDO = 256.0f;
constexpr uint16_t PV_PARTIAL_E8M0_PAIR = 0x7777;
constexpr float LN_448 = 6.10479324f;
constexpr float SAGE_FP16_MIN = -65504.0f;
constexpr float SAGE_FP16_MAX = 65504.0f;
constexpr float INV_INT8_MAX = 1.0f / 127.0f;
constexpr float INV_MX_FP8_MAX = 1.0f / 448.0f;
constexpr float DEQ_SCALAR_DEQF16 = 1.0f / 2048.0f;

constexpr uint16_t FP16_ABS_MASK = 0x7FFF;
constexpr uint32_t EXPONENT_MASK = 0x7F800000;
constexpr uint32_t EXPONENT_ROUND_BIAS = 0x007FFFFF;
constexpr uint32_t RECIPROCAL_EXP_BIAS = 0x7F000000;
constexpr float MIN_POSITIVE_FLOAT = 1.17549435e-38f;

// Communication & Synchronization Event Machine
struct EventHandshake {
    static constexpr uint32_t QK_READY = 0;
    static constexpr uint32_t PV_READY = 1;
    static constexpr uint32_t PV_RELEASE = 2;
    static constexpr uint32_t Q_READY = 4;
    static constexpr uint32_t AIV_FLAG_OFFSET = 8;
    static constexpr uint32_t SCALE_EVENT_BASE = 2;
    static constexpr uint32_t QK_EVENT = 0;
    static constexpr uint32_t PV_EVENT = 1;
    static constexpr uint32_t QK_SECOND_FIX_EVENT = 6;
};

#if defined(__CCE__) || defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__) || defined(__DAV_C350_VEC__)
// Memory Buffering Resource
template <TPosition POSITION, uint32_t BYTES>
struct FixedBuffer {
    template <class T>
    __aicore__ inline LocalTensor<T> GetBufferByByte(uint32_t offset) const {
        return LocalTensor<T>(POSITION, offset, (BYTES - offset) / sizeof(T));
    }
};

struct FixedResource {
    FixedBuffer<TPosition::A1, Arch::L1_SIZE> l1Buf;
    FixedBuffer<TPosition::A2, Arch::L0A_SIZE> l0ABuf;
    FixedBuffer<TPosition::B2, Arch::L0B_SIZE> l0BBuf;
    FixedBuffer<TPosition::CO1, Arch::L0C_SIZE> l0CBuf;
    FixedBuffer<TPosition::VECCALC, Arch::UB_SIZE> ubBuf;
    __aicore__ inline FixedResource() { PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline ~FixedResource() { PipeBarrier<PIPE_ALL>(); }
};

template <HardEvent E>
__aicore__ inline void Sync() {
    SetFlag<E>(0);
    WaitFlag<E>(0);
}

// Vector Fast Reciprocal via 2-step Newton-Raphson iteration (zero hardware divide)
__simd_callee__ inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                          AscendC::MicroAPI::RegTensor<float> &src,
                                          AscendC::MicroAPI::MaskReg &mask) {
    using namespace AscendC::MicroAPI;
    RegTensor<uint32_t> magic, ix, iy;
    RegTensor<float> y0, term, two;
    Duplicate(two, 2.0f);
    Duplicate(magic, FAST_RECIPROCAL_MAGIC);
    ix = (RegTensor<uint32_t> &)src;
    Sub(iy, magic, ix, mask);
    y0 = (RegTensor<float> &)iy;

    // Iteration 1: y1 = y0 * (2.0f - src * y0)
    Mul(term, src, y0, mask);
    Sub(term, two, term, mask);
    Mul(dst, y0, term, mask);

    // Iteration 2: y2 = y1 * (2.0f - src * y1)
    Mul(term, src, dst, mask);
    Sub(term, two, term, mask);
    Mul(dst, dst, term, mask);
}
#endif

} // namespace sage
