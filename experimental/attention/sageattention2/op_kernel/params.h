/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../core/types.h"

// Backward-compatible type aliases mapping to clean-room core::types
using SageQuantGran = sage::SageQuantGran;
using SagePvAccum = sage::SagePvAccum;
using SageParams = sage::SageParams;

constexpr uint32_t QUANT_TILE_ROWS = sage::QUANT_TILE_ROWS;
constexpr int64_t MAX_HEAD_DIM = 128;
constexpr float SAGE_FP16_MAX = sage::SAGE_FP16_MAX;
constexpr float SAGE_FP16_MIN = sage::SAGE_FP16_MIN;

constexpr uint32_t QTokensPerScale(bool perWarp) noexcept
{
    return sage::QTokensPerScale(perWarp);
}

constexpr uint32_t KTokensPerScale(bool perWarp) noexcept
{
    return sage::KTokensPerScale(perWarp);
}

constexpr bool DividesTile(uint32_t gran) noexcept
{
    return QUANT_TILE_ROWS % gran == 0U;
}

static_assert(DividesTile(QTokensPerScale(false)) && DividesTile(QTokensPerScale(true)) &&
              DividesTile(KTokensPerScale(false)) && DividesTile(KTokensPerScale(true)),
              "quantization granularity must divide the 64-row tile");

constexpr bool IsValidQuantGran(int64_t gran) noexcept
{
    return sage::IsValidQuantGran(gran);
}

constexpr bool IsValidPvAccum(int64_t accum) noexcept
{
    return sage::IsValidPvAccum(accum);
}

constexpr bool StrEq(const char *a, const char *b) noexcept
{
    return sage::StrEq(a, b);
}

constexpr int64_t ParseQuantGran(const char *str) noexcept
{
    return sage::ParseQuantGran(str);
}

constexpr int64_t ParsePvAccum(const char *str) noexcept
{
    return sage::ParsePvAccum(str);
}

constexpr uint32_t PaddedHeadDim(uint32_t d) noexcept
{
    return sage::PaddedHeadDim(d);
}

static_assert(PaddedHeadDim(1U) == 64U && PaddedHeadDim(64U) == 64U &&
              PaddedHeadDim(65U) == 128U && PaddedHeadDim(128U) == 128U,
              "padded head widths must match spec pad points");

constexpr bool IsPow2(uint32_t v) noexcept
{
    return sage::IsPowerOfTwo(v);
}

template <typename T>
constexpr T RoundUpTo(T v, T align) noexcept
{
    return sage::RoundUpTo(v, align);
}

static_assert(IsPow2(128U) && IsPow2(512U), "tile padding alignments must stay power-of-two");
