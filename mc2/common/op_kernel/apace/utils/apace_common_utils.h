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
 * CeilDiv 为 host/device 双态方法, 可同时被宿主侧与 kernel 侧 (bisheng -xasc) 包含。
 */

#pragma once
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string>

// 双态标注: 纯宿主编译态将 __aicore__ 展开为空, 保证本头文件可被纯 host TU 解析
// (bisheng 混合 TU (-xasc) 中 host(__host__)/device(__aicore__) 函数禁止互调, 故 CeilDiv 双版本共存)
#ifndef __CCE_AICORE__
#ifndef __aicore__
#define __aicore__
#endif
#endif

namespace mm {
// Quantized MX element layouts for non-type template parameters.
enum class DataType {
    DT_FLOAT4_E2M1,
    DT_FLOAT8_E4M3FN,
    DT_FLOAT8_E5M2,
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

// CeilDiv host 侧模板: tiling/ST 等宿主函数调用
// 先除后加形式: (a+b-1)/b 在 a+b-1 超过类型上限时回绕, 本写法对任意值域安全
template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return a;
    }
    return a / b + (a % b != 0 ? 1 : 0);
}

// CeilDiv 设备侧版本: kernel 内 __aicore__ 函数调用
// bisheng 禁止 __host__↔__aicore__ 互调; 独立命名空间隔离, 避免与 host 模板及
// Blaze::Gemm::CeilDiv 产生跨侧重载干扰
namespace ApaceKernel {
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return a / b + (a % b != 0 ? 1U : 0U);
}
} // namespace ApaceKernel

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
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        constexpr T shift = static_cast<T>(1);
        return size << shift;
    } else {
        return size;
    }
}

template <mm::DataType dataType, typename T>
constexpr T GetSizeWithDataType(T shape)
{
    if constexpr (dataType == mm::DataType::DT_FLOAT4_E2M1) {
        constexpr T shift = static_cast<T>(1);
        return (shape + static_cast<T>(1)) >> shift;
    } else {
        return shape;
    }
}

// 量化数据类型位宽（bit）
constexpr uint64_t MXFP4_BITS = 4UL;
constexpr uint64_t MXFP8_BITS = 8UL;

// win 区 buffer 复用深度决策常量
// HCCL_BUFFSIZE 默认 200MB，框架层翻倍分配为 400MB
constexpr uint64_t WIN_REUSE_BUDGET_CAP = 400UL * 1024UL * 1024UL;
constexpr uint64_t WIN_REUSE_DEPTH_CAP = 128UL; // flag 同步安全上限（FLAG_ID_MODULO=11，64/11≈6 轮复用）
constexpr uint64_t WIN_REUSE_DEPTH_MIN = 2UL;
