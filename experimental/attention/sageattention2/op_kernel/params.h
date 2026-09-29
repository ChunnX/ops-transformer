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
// Runtime shape, layout and execution parameters shared by all stages.
struct SageParams {
    uint32_t b, hq, hk, sq, sk, d, dp, qp, kp, cores;
    uint32_t causal, smoothK, returnLse;
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
