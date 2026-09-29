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
// tokens sharing one scale: one thread owns 4 Q rows and 16 K rows, one warp
// owns 8 times that.
enum class SageQuantGran : uint32_t { PER_THREAD = 0, PER_WARP = 1, COUNT = 2 };

// PV accumulation formats. FP16 keeps the per-block partial in half precision
// before folding it into the FP32 running state, as the open-source default.
enum class SagePvAccum : uint32_t { FP32_FP16 = 0, FP32_FP32 = 1, FP32 = 2, COUNT = 3 };

constexpr bool IsValidQuantGran(int64_t gran) noexcept
{
    return gran >= 0 && gran < static_cast<int64_t>(SageQuantGran::COUNT);
}

constexpr bool IsValidPvAccum(int64_t accum) noexcept
{
    return accum >= 0 && accum < static_cast<int64_t>(SagePvAccum::COUNT);
}

inline int64_t ParseQuantGran(const char *str) noexcept
{
    if (!str) return -1;
    if (str[0] == 'p' && str[1] == 'e' && str[2] == 'r' && str[3] == '_') {
        if (str[4] == 't' && str[5] == 'h' && str[6] == 'r' && str[7] == 'e' && str[8] == 'a' && str[9] == 'd' && str[10] == '\0')
            return static_cast<int64_t>(SageQuantGran::PER_THREAD);
        if (str[4] == 'w' && str[5] == 'a' && str[6] == 'r' && str[7] == 'p' && str[8] == '\0')
            return static_cast<int64_t>(SageQuantGran::PER_WARP);
    }
    return -1;
}

inline int64_t ParsePvAccum(const char *str) noexcept
{
    if (!str) return -1;
    if (str[0] == 'f' && str[1] == 'p' && str[2] == '3' && str[3] == '2') {
        if (str[4] == '\0') return static_cast<int64_t>(SagePvAccum::FP32);
        if (str[4] == '+') {
            if (str[5] == 'f' && str[6] == 'p' && str[7] == '1' && str[8] == '6' && str[9] == '\0')
                return static_cast<int64_t>(SagePvAccum::FP32_FP16);
            if (str[5] == 'f' && str[6] == 'p' && str[7] == '3' && str[8] == '2' && str[9] == '\0')
                return static_cast<int64_t>(SagePvAccum::FP32_FP32);
        }
    }
    return -1;
}

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
