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
#include "common.h"
namespace Sage {

// Constants depend only on sm_scale, never on Q/K values or other cores.
// Q carries the sign; the product compensates fixed DEQF16(2^-11).
__simd_vf__ inline void FixedScaleFactorsVF(__ubuf__ float *factors, float scale)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> s, magnitude, factor, negative, zero;
    Duplicate(s, scale);
    Abs(magnitude, s, full);
    Muls(magnitude, magnitude, DEQF16_FACTOR, full);
    Sqrt(factor, magnitude, full);
    // Overlap sign comparison setup during Sqrt pipeline latency
    Duplicate(zero, 0.f);
    MaskReg sign;
    Compare<float, CMPMODE::LT>(sign, s, zero, full);
    StoreAlign(factors + 64, factor, full);
    Muls(negative, factor, -1.f, full);
    Select(factor, negative, factor, sign);
    StoreAlign(factors, factor, full);
}

__simd_callee__ inline void PackFixedScaleBlock(__ubuf__ float *input, __ubuf__ half *output, __ubuf__ float *factor,
                                                uint32_t valid)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    auto validMask = UpdateMask<float>(valid);
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> x, f, zero;
    RegTensor<half> h;
    Duplicate(zero, 0.f);
    LoadAlign(x, input);
    LoadAlign(f, factor);
    Mul(x, x, f, full);
    Select(x, x, zero, validMask);
    Cast<half, float, cast>(h, x, full);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(output, h, full);
}

__simd_vf__ inline void PackFixedScaleVF(__ubuf__ float *input, __ubuf__ half *output, __ubuf__ float *factor,
                                         uint32_t valid)
{
    PackFixedScaleBlock(input, output, factor, valid);
}

} // namespace Sage
