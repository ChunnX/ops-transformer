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
// Q/K INT8 quantization granularity, expressed as the number of consecutive
// tokens sharing one scale. Matches the open-source CUDA thread/warp mapping:
// one thread owns 4 Q rows and 16 K rows, one warp owns 8 times that.
enum class SageQuantGran : uint32_t { PER_THREAD = 0, PER_WARP = 1, COUNT = 2 };

// PV accumulation formats. FP16 keeps the per-block partial in half precision
// before folding it into the FP32 running state, as the open-source default.
enum class SagePvAccum : uint32_t { FP32_FP16 = 0, FP32_FP32 = 1, FP32 = 2, COUNT = 3 };

// Runtime shape, layout and execution parameters shared by all stages.
struct SageParams {
    uint32_t b, hq, hk, sq, sk, d, dp, qp, kp, cores;
    uint32_t causal, smoothK, returnLse;
    uint32_t qg, kg; // tokens per shared INT8 scale, for Q and K
    uint32_t accum, smoothV;
    float smScale;
    uint64_t qs[3], ks[3], vs[3]; // canonical B,H,S strides, in elements
};

constexpr uint32_t RoundUpTo(uint32_t v, uint32_t align)
{
    return (v + align - 1U) & ~(align - 1U);
}
constexpr uint64_t RoundUpTo(uint64_t v, uint64_t align)
{
    return (v + align - 1ULL) & ~(align - 1ULL);
}
