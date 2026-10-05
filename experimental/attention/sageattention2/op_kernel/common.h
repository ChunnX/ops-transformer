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

namespace Sage {
__aicore__ inline bool UseDieGroups(const SageParams &p) {
    return p.cores > 0 && p.cores % 2 == 0 && (uint64_t(p.b) * p.hk) % 2 == 0 &&
           p.sq % 128 == 0;
}
// Valid only when UseDieGroups(p) is true: cores is positive and even.
// The tiler requires hq % hk == 0, so even b*hk implies even
// total = b*hq*queryTiles. Both die groups must have exactly total/2 tasks.
// Keep this invariant if the grouping predicate or task space changes.
__aicore__ inline uint64_t DieTask(uint64_t task, uint64_t total, uint32_t cores) {
    const uint32_t half = cores / 2;
    return (task % cores / half) * (total / 2) + (task / cores) * half + task % half;
}
}

namespace Sage {
// Per-core end in the virtual cyclic task space, including a partial final wave.
// The grouped branch requires the same even-total invariant as DieTask.
__aicore__ inline uint64_t DieTaskEnd(const SageParams &p, uint64_t total, uint32_t core) {
    if (!UseDieGroups(p)) return total;
    const uint32_t half = p.cores / 2, rank = core % half;
    const uint64_t local = total / 2;
    const uint64_t count = local > rank ? (local - rank + half - 1) / half : 0;
    return core + count * p.cores;
}
}

#include "catlass/arch/resource.hpp"
using namespace AscendC;
namespace Sage {
using Arch = Catlass::Arch::Ascend950;
// Buffers and events are statically assigned by this kernel. Avoid TPipe
// bookkeeping and its runtime event switch (jump tables during DBI replay).
template <TPosition POSITION, uint32_t BYTES>
struct FixedBuffer {
    template <class T = half>
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
__aicore__ inline void ReadRows(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t off,
                               uint32_t rows, uint64_t rowStride, uint32_t dim, uint32_t paddedDim)
{
    Duplicate(dst, T(0), rows * paddedDim);
    Sync<HardEvent::V_MTE2>();
    constexpr uint32_t blockElements = 32 / sizeof(T);
    uint32_t padded = (dim + blockElements - 1) / blockElements * blockElements;
    DataCopyExtParams copy{static_cast<uint16_t>(rows), dim * uint32_t(sizeof(T)),
                           (int64_t(rowStride) - dim) * int64_t(sizeof(T)),
                           (paddedDim - padded) / blockElements, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(padded - dim), T(0)};
    DataCopyPad(dst, src[off], copy, pad);
    Sync<HardEvent::MTE2_V>();
}
template <class T>
__aicore__ inline void ReadRow(LocalTensor<T> dst, GlobalTensor<T> src, uint64_t off, uint32_t d, uint32_t dp)
{
    ReadRows(dst, src, off, 1, d, d, dp);
}
template <class T>
__aicore__ inline void WriteRow(GlobalTensor<T> dst, uint64_t off, LocalTensor<T> src, uint32_t n)
{
    Sync<HardEvent::V_MTE3>();
    DataCopyExtParams copy{1, n * uint32_t(sizeof(T)), 0, 0, 0};
    DataCopyPad(dst[off], src, copy);
    Sync<HardEvent::MTE3_V>();
}
template <class T>
__aicore__ inline void WriteRows(GlobalTensor<T> dst, uint64_t off, LocalTensor<T> src,
                                uint32_t rows, uint32_t dim, uint32_t paddedDim)
{
    if (dim == paddedDim) {
        WriteRow(dst, off, src, rows * dim);
        return;
    }
    constexpr uint32_t blockElements = 32 / sizeof(T);
    uint32_t padded = (dim + blockElements - 1) / blockElements * blockElements;
    DataCopyExtParams copy{static_cast<uint16_t>(rows), dim * uint32_t(sizeof(T)),
                           (paddedDim - padded) / blockElements, 0, 0};
    Sync<HardEvent::V_MTE3>();
    DataCopyPad(dst[off], src, copy);
    Sync<HardEvent::MTE3_V>();
}
template <class T>
__simd_vf__ inline void Pack8(__ubuf__ T *out, __ubuf__ float *in, uint32_t count)
{
    using namespace AscendC::MicroAPI;
    constexpr static CastTrait trait{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto mask = CreateMask<float, MaskPattern::ALL>();
    for (uint32_t i = 0; i < count; i += 64) {
        RegTensor<float> x;
        RegTensor<T> y;
        LoadAlign(x, in + i);
        if constexpr (AscendC::IsSameType<T, int8_t>::value) {
            RegTensor<int32_t> integers;
            Cast<int32_t, float, trait>(integers, x, mask);
            y = (RegTensor<T> &)integers;
        } else {
            Cast<T, float, trait>(y, x, mask);
        }
        StoreAlign<T, StoreDist::DIST_PACK4_B32>(out + i, y, mask);
    }
}
__aicore__ inline float Scalar(LocalTensor<float> x)
{
    Sync<HardEvent::V_S>();
    return x.GetValue(0);
}
__aicore__ inline uint64_t Offset(uint64_t const *strides, uint32_t b, uint32_t h, uint32_t s)
{
    return b * strides[0] + h * strides[1] + s * strides[2];
}
} // namespace Sage
