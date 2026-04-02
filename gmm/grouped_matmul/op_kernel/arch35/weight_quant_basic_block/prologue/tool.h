/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tool.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H
#define GROUPED_MATMUL_WEIGHT_QUANT_TOOL_H

#include "kernel_operator.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::fp8_e8m0_t;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::TPosition;
using AscendC::VECTOR_REG_WIDTH;

namespace WeightQuantBatchMatmulV2::Arch35 {
// buffer相关定义
static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
static constexpr int64_t L1_SIZE = 512;
static constexpr int64_t L1_SIZE_BYTE = L1_SIZE * 1024;
static constexpr int64_t BIAS_L1_SIZE = 4;
static constexpr uint64_t A_L1_MAX_SIZE_WITH_BIAS_QUANT = 240UL * 1024UL;
static constexpr uint64_t MX_BIAS_SINGLE_VECTOR_SIZE = 128;

// 参数约束定义
static constexpr uint64_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(half);
static constexpr uint32_t FP32_BLOCK_SIZE = 8;
static constexpr uint32_t FP16_BLOCK_SIZE = 16;
static constexpr int32_t C0_SIZE_B8 = 32;
static constexpr uint32_t SCALE_FACTOR_B_BIT = 8;

// 同步定义
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t FLAG_ID_MAX = 16;

// 函数定义
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

template <typename T>
__aicore__ inline T CeilDivide(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t blockCount,
                                     uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
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

    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        // 4bit场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDivide((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ constexpr uint32_t GetKBUnit()
{
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        return 2048; // 2048个int4是1kb
    }
    return 1024 / sizeof(T); // 1kb总共大小为1024
}

}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif
