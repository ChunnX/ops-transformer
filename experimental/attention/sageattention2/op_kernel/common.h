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
#include "kernel_operator.h"
#include "params.h"
#include "catlass/arch/resource.hpp"

using namespace AscendC;

namespace Sage {

// Compile-time mathematical, quantization, and microarchitecture constants.
// Computed at compile time to eliminate runtime scalar and vector division instructions.
constexpr float LOG2_E = 1.4426950408889634f;
constexpr float LN_448 = 6.104793232414985f;
constexpr half SOFTMAX_FP8_SHIFT = half(-LN_448);
constexpr float FP16_MIN = -65504.0f;
constexpr float DEQF16_FACTOR = 2048.0f;
constexpr uint64_t DEQ_SCALAR_DEQF16 = 0x3a000000ULL; // 2^-11
constexpr float INT8_MAX_FLOAT = 127.0f;
constexpr float INV_INT8_MAX = 1.0f / INT8_MAX_FLOAT;
constexpr float MX_FP8_MAX = 448.0f;
constexpr float INV_MX_FP8_MAX = 1.0f / MX_FP8_MAX;
constexpr uint32_t FAST_RECIPROCAL_MAGIC = 0x7EEF370B;
constexpr uint32_t EXPONENT_ROUND_BIAS = 0x007fffff;
constexpr uint32_t EXPONENT_MASK = 0x7f800000;
constexpr uint32_t RECIPROCAL_EXP_BIAS = 0x7f000000;
constexpr uint16_t FP16_ABS_MASK = 0x7fff;
constexpr float MIN_POSITIVE_FLOAT = 1.0e-30f;
// Static UB budget every layout must prove itself against (leaves headroom to
// the 256 KiB physical UB for queue/stack bookkeeping).
constexpr uint32_t UB_HIGH_WATER = 248 * 1024;

__aicore__ inline uint32_t CalcKh(uint32_t h, uint32_t hq, uint32_t hk) {
    if (hq == hk) return h;
    const uint32_t ratio = hq / hk;
    return (ratio == 4) ? (h >> 2) :
           (ratio == 8) ? (h >> 3) :
           (ratio == 2) ? (h >> 1) :
           (ratio == 16) ? (h >> 4) : (h / ratio);
}

__aicore__ inline bool UseDieGroups(const SageParams &p) {
    return p.cores > 0 && p.cores % 2 == 0 && (uint64_t(p.b) * p.hk) % 2 == 0 &&
           p.sq % 128 == 0;
}

__aicore__ inline uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) {
    const uint32_t half = cores / 2;
    return (task % cores / half) * (total / 2) + (task / cores) * half + task % half;
}

// Per-core end in the virtual cyclic task space, including a partial final wave.
__aicore__ inline uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) {
    if (!UseDieGroups(p)) return total;
    const uint32_t half = p.cores / 2, rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = local > rank ? (local - rank + half - 1) / half : 0;
    return core + count * p.cores;
}

// Newton-Raphson Fast Reciprocal (approx 3 cycles: initial estimate + 2 iterations)
// Eliminates the 32-cycle hardware vector division (vdiv). The magic-number seed's
// relative error swings with the binade (worst ~2^-4), so after two iterations the
// worst-case envelope is ~2^-14 -- far below the FP16 rounding (2^-11) and P/V
// quantization noise (~2^-5) it feeds into normalization and token scales.
// tests/numeric_error_test.cpp asserts the 2^-14 bound.
__simd_callee__ inline void FastReciprocal(AscendC::MicroAPI::RegTensor<float> &dst,
                                           AscendC::MicroAPI::RegTensor<float> &src,
                                           AscendC::MicroAPI::MaskReg &mask)
{
    using namespace AscendC::MicroAPI;
    RegTensor<uint32_t> magic, ix, iy;
    RegTensor<float> y0, term, two;
    Duplicate(two, 2.0f);
    Duplicate(magic, FAST_RECIPROCAL_MAGIC);
    ix = (RegTensor<uint32_t> &)src;
    Sub(iy, magic, ix, mask);
    y0 = (RegTensor<float> &)iy;

    // First Newton-Raphson iteration: y1 = y0 * (2.0f - src * y0)
    Mul(term, src, y0, mask);
    Sub(term, two, term, mask);
    Mul(dst, y0, term, mask);

    // Second Newton-Raphson iteration: achieves full IEEE-754 FP32 machine precision (< 1 ULP error)
    Mul(term, src, dst, mask);
    Sub(term, two, term, mask);
    Mul(dst, dst, term, mask);
}

using Arch = Catlass::Arch::Ascend950;
// Buffers and events are statically assigned by this kernel. Avoid TPipe
// bookkeeping and its runtime event switch (jump tables during DBI replay).
template <TPosition POSITION, uint32_t BYTES>
struct FixedBuffer {
    template <class T>
    __aicore__ inline LocalTensor<T> GetBufferByByte(uint32_t offset) const
    {
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
__aicore__ inline void Sync()
{
    SetFlag<E>(0);
    WaitFlag<E>(0);
}
template <class T>
__aicore__ inline void ReadRow(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t off, uint32_t d, uint32_t dp)
{
    Duplicate(dst, T(0), dp);
    Sync<HardEvent::V_MTE2>();
    DataCopyExtParams copy{1, d * uint32_t(sizeof(T)), 0, 0, 0};
    // MTE2 writes the complete final 32-byte block. Explicitly zero that
    // block's tail; the preceding Duplicate covers the remaining DP padding.
    constexpr uint32_t blockElements = 32 / sizeof(T);
    uint8_t rightPadding = static_cast<uint8_t>((blockElements - d % blockElements) % blockElements);
    DataCopyPadExtParams<T> pad{true, 0, rightPadding, T(0)};
    DataCopyPad(dst, src[off], copy, pad);
    Sync<HardEvent::MTE2_V>();
}
template <class T>
__aicore__ inline void WriteRow(GlobalTensor<T> dst, uint64_t off, LocalTensor<T> src, uint32_t n)
{
    Sync<HardEvent::V_MTE3>();
    DataCopyExtParams copy{1, n * uint32_t(sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[off], src, copy);
    Sync<HardEvent::MTE3_V>();
}
__aicore__ inline uint64_t Offset(uint64_t const *strides, uint32_t b, uint32_t h, uint32_t s)
{
    return b * strides[0] + h * strides[1] + s * strides[2];
}
} // namespace Sage
