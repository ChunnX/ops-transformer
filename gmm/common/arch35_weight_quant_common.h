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
 * \file gmm/common/arch35_weight_quant_common.h
 * \brief Common definitions for Arch35 weight quant GMM kernels.
 */
#ifndef GMM_COMMON_ARCH35_WEIGHT_QUANT_COMMON_H
#define GMM_COMMON_ARCH35_WEIGHT_QUANT_COMMON_H

#include "kernel_operator.h"
#include "gmm/common/gmm_kernel_utils.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::fp8_e8m0_t;
using AscendC::GetUserWorkspace;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::TPosition;
using AscendC::VECTOR_REG_WIDTH;
using matmul::MatmulCallBackFunc;
using matmul::MatmulImpl;
using matmul::MatmulType;
using matmul::MatmulTypeWithScale;

#define SHORT_MIX_LOG(format, ...)

namespace WeightQuantBatchMatmulV2::Arch35 {

enum class QuantType {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4,
};

// buffer相关定义
static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;
static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
static constexpr int32_t SINGLE_BUFFER_NUM = 1;
static constexpr int64_t L1_SIZE = 512;
static constexpr int64_t L1_SIZE_BYTE = L1_SIZE * 1024;
static constexpr int64_t L1_HALF_SIZE = L1_SIZE / 2;
static constexpr int64_t L1_SIZE_WITH_QUANTSCALE = 504;
static constexpr int64_t L1_SIZE_WITH_QUANTSCALE_BYTE = L1_SIZE_WITH_QUANTSCALE * 1024;
static constexpr int64_t BIAS_L1_SIZE = 4;
static constexpr uint64_t A_L1_MAX_SIZE_WITH_BIAS_QUANT = 240UL * 1024UL;
static constexpr uint64_t MX_BIAS_SINGLE_VECTOR_SIZE = 128;
static constexpr uint64_t MX_SCALE_K_L1_SIZE = 4096;

// 控制参数定义
static constexpr int32_t BASIC_BLOCK_PROCESS_NUM = 2;
static constexpr uint64_t SCALE_COPY_GROUP_SIZE = 2;
static constexpr int32_t SCALE_COPY_DEFAULT_STRIDE = 0;
static constexpr int32_t SCALE_COPY_DEFAULT_N_STRIDE = 1;

// 参数约束定义
static constexpr uint64_t MX_GROUPSIZE = 32;
static constexpr uint64_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(half);
static constexpr uint64_t VEC_MAX_ELEM_B32 = VECTOR_REG_WIDTH / sizeof(float);
static constexpr uint32_t FP32_BLOCK_SIZE = 8;
static constexpr uint32_t FP16_BLOCK_SIZE = 16;
static constexpr int32_t C0_SIZE_B8 = 32;
static constexpr uint32_t SCALE_FACTOR_B_BIT = 8;

// 同步定义
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t FLAG_ID_MAX = 16;

// 引入通用工具函数到本 namespace
using GmmOpKernel::CeilAlign;
using GmmOpKernel::CeilDivide;
using GmmOpKernel::DataCopyPad2D;
using GmmOpKernel::GetKBUnit;
using GmmOpKernel::Min;

template <typename xType, QuantType antiQuantType>
__aicore__ constexpr bool IsMxA8W4()
{
    return antiQuantType == QuantType::MX && IsSameType<xType, fp8_e4m3fn_t>::value;
}

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

} // namespace WeightQuantBatchMatmulV2::Arch35
#endif
