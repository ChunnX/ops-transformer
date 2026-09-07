/*
 * Copyright (c) 2026 G_W_E. All rights reserved.
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
using Arch = Catlass::Arch::Ascend950;
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
