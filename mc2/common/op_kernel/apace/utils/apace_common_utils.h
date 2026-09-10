/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apace_common_utils.h
 * \brief Host-side argument parsing, shape helpers, and error utilities for matmul examples.
 */

#pragma once
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace mm {
// Quantized MX element layouts for non-type template parameters.
enum class DataType {
    DT_FLOAT4_E2M1,
    DT_FLOAT8_E4M3FN,
    DT_FLOAT8_E5M2,
    DT_HIFLOAT4, // HiFloat4: S1P2 element, packed 2/byte, group size 64
};

// Bias element dtype for the SWAT tiling L1 budget; width mapping mirrors the
// ops-nn v4 SWAT solver's GetBiasDataSize (fp32 = 4B, fp16/bf16 = 2B).
enum class BiasDataType {
    DT_FLOAT,
    DT_FLOAT16,
    DT_BFLOAT16,
};

template <BiasDataType biasDtype>
constexpr uint64_t GetBiasDataSize()
{
    // apace_common_utils.h is included ahead of apace_constant.h, so keep the
    // named widths local instead of pulling in that dependency.
    constexpr uint64_t FP32_BYTES = 4UL;
    constexpr uint64_t FP16_BYTES = 2UL;
    if constexpr (biasDtype == BiasDataType::DT_FLOAT) {
        return FP32_BYTES;
    } else {
        return FP16_BYTES;
    }
}
} // namespace mm

#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR]  " fmt "\n", ##args)
#define CHECK_COND(cond, msg) \
    do { \
        if (!(cond)) { \
            throw std::runtime_error(std::string("Error: ") + msg + "\nFile: " + __FILE__ + \
                                     "\nLine: " + std::to_string(__LINE__)); \
        } \
    } while (0)

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b + static_cast<T>(a % b != 0);
}

template <typename T>
inline T Align(T a, T b)
{
    return CeilDiv(a, b) * b;
}

template <typename T>
inline T FloorAlign(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

enum class DataType {
    FP4,
    FP8
};

template <mm::DataType dataType, typename T>
constexpr T GetShapeWithDataType(T size)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1 || dataType == mm::DataType::DT_HIFLOAT4) {
        constexpr T shift = static_cast<T>(1);
        return size << shift;
    } else {
        return size;
    }
}

template <mm::DataType dataType, typename T>
constexpr T GetSizeWithDataType(T shape)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1 || dataType == mm::DataType::DT_HIFLOAT4) {
        constexpr T shift = static_cast<T>(1);
        return (shape + static_cast<T>(1)) >> shift;
    } else {
        return shape;
    }
}

// Per-group element count of the quantization format along the K axis:
// MX formats carry one scale unit per 32 elements, HiF4 per 64 elements.
// The canonical names live in apace_constant.h (MX_GROUP_SIZE / HIF4_GROUP_SIZE);
// this header may be included ahead of apace_constant.h, so keep the widths
// local instead of pulling in that dependency (mirrors GetBiasDataSize above).
template <mm::DataType dataType>
constexpr uint64_t GetPerGroupSize()
{
    if constexpr (dataType == mm::DataType::DT_HIFLOAT4) {
        return 64UL; // HIF4_GROUP_SIZE
    } else {
        return 32UL; // MX_GROUP_SIZE
    }
}

// Scale bytes carried per group: MX emits one E8M0 byte per 32-element group,
// HiF4 emits one 4-byte hif4_scale element (E6M2 + E1_8 + E1_16) per
// 64-element group, i.e. scale shape (M, K/group, 1) with 4-byte elements.
template <mm::DataType dataType>
constexpr uint64_t GetScaleUnitPerGroup()
{
    if constexpr (dataType == mm::DataType::DT_HIFLOAT4) {
        return 4UL; // HIF4_SCALE_UNIT (sizeof(hif4_scale))
    } else {
        return 1UL;
    }
}

// Scale-group alignment (in scale units) required by the L1 layout framing:
// MX frames E8M0 scale units in pairs, HiF4 needs no extra alignment.
template <mm::DataType dataType>
constexpr uint64_t GetAlignSize()
{
    if constexpr (dataType == mm::DataType::DT_HIFLOAT4) {
        return 1UL; // TILING_HIF4_MULTI_BASE_SIZE
    } else {
        return 2UL; // TILING_MXFP_MULTI_BASE_SIZE
    }
}
