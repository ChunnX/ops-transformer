/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ascendc_utils.h
 * \brief Common pure-math and generic utility functions for AscendC kernels.
 */
#ifndef ASCENDC_UTILS_H
#define ASCENDC_UTILS_H

#include <limits>
#include "kernel_operator.h"

using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::TPosition;

// Common constants extracted from tool.h variants
static constexpr uint32_t FP32_BLOCK_SIZE = 8;
static constexpr uint32_t FP16_BLOCK_SIZE = 16;
static constexpr uint64_t FLAG_ID_MAX = 16;
static constexpr uint64_t MX_GROUPSIZE = 32;

// Max / Min (converged from grouped_matmul_utils.h and tool.h)
template <typename T>
__aicore__ inline T Max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

// AlignUp / AlignDown (from grouped_matmul_utils.h)
template <uint32_t base, typename T = uint32_t>
__aicore__ inline T AlignUp(T a)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignUp(T a, T base)
{
    return (a + base - 1) / base * base;
}

template <typename T>
__aicore__ inline T AlignDown(T a, T base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

template <>
__aicore__ inline uint32_t AlignUp<4, uint32_t>(uint32_t a)
{
    return (a + 3) & ~3;
}

template <>
__aicore__ inline uint32_t AlignUp<8, uint32_t>(uint32_t a)
{
    return (a + 7) & ~7;
}

template <>
__aicore__ inline uint32_t AlignUp<16, uint32_t>(uint32_t a)
{
    return (a + 15) & ~15;
}

template <>
__aicore__ inline uint32_t AlignUp<32, uint32_t>(uint32_t a)
{
    return (a + 31) & ~31;
}

// CeilAlign — alias of AlignUp with assert (from tool.h)
template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b)
{
    ASCENDC_ASSERT(a <= (std::numeric_limits<uint32_t>::max() - b),
                   { KERNEL_LOG(KERNEL_ERROR, "CeilAlign uint32 over limit."); });
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

// CeilDiv / CeilDivide (from tool.h)
template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T CeilDivide(T a, T b)
{
    return CeilDiv(a, b);
}

// FloorDiv (from mc2 tool.h)
template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return a / b;
}

// GCD / LCM (from grouped_matmul_utils.h)
template <typename T>
__aicore__ inline T GreatestCommonDivisor(T a, T b)
{
    T c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename T>
__aicore__ inline T LeastCommonMultiple(T a, T b)
{
    return a * b / GreatestCommonDivisor(a, b);
}

// GetTensorAddr (from grouped_matmul_utils.h)
template <typename T>
__aicore__ inline __gm__ T *GetTensorAddr(uint16_t index, GM_ADDR tensorPtr)
{
    __gm__ uint64_t *dataAddr = reinterpret_cast<__gm__ uint64_t *>(tensorPtr);
    uint64_t tensorPtrOffset = *dataAddr;
    __gm__ uint64_t *retPtr = dataAddr + (tensorPtrOffset >> 3);
    return reinterpret_cast<__gm__ T *>(*(retPtr + index));
}

// GetTypeBits (from grouped_matmul_utils.h)
template <typename T>
__aicore__ inline constexpr uint32_t GetTypeBits()
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return 4;
    }
    return sizeof(T) * 8;
}

// GetKBUnit — merged from weight_quant_basic_block/tool.h and a16w4_msd/tool.h
template <typename T>
__aicore__ constexpr uint32_t GetKBUnit()
{
    if constexpr (IsSameType<T, int4b_t>::value
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                  || IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, fp4x2_e1m2_t>::value
#endif
                  ) {
        return 2048;
    }
    return 1024 / sizeof(T);
}

// DataCopyPad2D — merged from all tool.h variants (mc2 version is the most comprehensive)
template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t fullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (fullDim0 - dim0) * sizeof(T);
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    if (dim0 % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(dim0, static_cast<uint32_t>(32 / sizeof(T))) - dim0;
        padParams.paddingValue = 0;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t blockCount,
                                     uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(dstInnerLength >= blockLen, {
        KERNEL_LOG(KERNEL_ERROR, "dstInnerLength[%d] should be larger than blockLen[%d].", dstInnerLength, blockLen);
    });
#endif
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }
    if constexpr (IsSameType<T, int4b_t>::value
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                  || IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, fp4x2_e1m2_t>::value
#endif
                  ) {
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = 0;
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                  || IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, fp4x2_e1m2_t>::value
#endif
                  ) {
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                  || IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, fp4x2_e1m2_t>::value
#endif
                  ) {
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
    DataCopyPad(dst, src, params);
}

#endif // ASCENDC_UTILS_H
