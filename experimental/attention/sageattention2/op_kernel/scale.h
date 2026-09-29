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

// Host tiling precomputes the FP16 balancing factors (qh = qs·sign(sm_scale)·
// sqrt(|sm_scale|·2048), kh = ks·sqrt(|sm_scale|·2048)); the 2048 factor folds
// the fixed DEQF16(2^-11) fixpipe dequantization into the quantization scales.
// Publishing them via SageParams removes the per-launch Sqrt and sign-select
// vector instructions from the AIV.
__simd_vf__ inline void FixedScaleFactorsVF(__ubuf__ float *factors, float qFactor, float kFactor)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> qVec, kVec;
    Duplicate(qVec, qFactor);
    Duplicate(kVec, kFactor);
    StoreAlign(factors, qVec, full);
    StoreAlign(factors + 64, kVec, full);
}

// Fast path: factor is directly a scalar float (precomputed on Host).
// Broadcasts directly into a vector register via Duplicate without round-tripping through UB memory.
__simd_callee__ inline void PackFixedScaleBlock(__ubuf__ float *input, __ubuf__ half *output, float factor,
                                                uint32_t valid)
{
    using namespace AscendC::MicroAPI;
    auto full = CreateMask<float, MaskPattern::ALL>();
    auto validMask = UpdateMask<float>(valid);
    static constexpr CastTrait cast{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> x, f, zero;
    RegTensor<half> h;
    Duplicate(zero, 0.f);
    Duplicate(f, factor);
    LoadAlign(x, input);
    Mul(x, x, f, full);
    Select(x, x, zero, validMask);
    Cast<half, float, cast>(h, x, full);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(output, h, full);
}

// Backward-compatible overload for UB pointer
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

} // namespace Sage
