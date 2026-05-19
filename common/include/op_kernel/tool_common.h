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
 * \file tool_common.h
 * \brief Common tool functions and constants shared across modules.
 */
#ifndef TOOL_COMMON_H
#define TOOL_COMMON_H

#include <limits>
#include "kernel_operator.h"

namespace CommonTool {

static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;
static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
static constexpr int32_t SINGLE_BUFFER_NUM = 1;
static constexpr uint32_t FP32_BLOCK_SIZE = 8;
static constexpr uint32_t FP16_BLOCK_SIZE = 16;
static constexpr int32_t C0_SIZE_B8 = 32;
static constexpr uint64_t MX_GROUPSIZE = 32;
static constexpr uint64_t FLAG_ID_MAX = 16;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(half);

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b)
{
    ASCENDC_ASSERT(
        a <= (std::numeric_limits<uint32_t>::max() - b), { KERNEL_LOG(KERNEL_ERROR, "CeilAlign uint32 over limit."); });
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

} // namespace CommonTool

#endif // TOOL_COMMON_H
