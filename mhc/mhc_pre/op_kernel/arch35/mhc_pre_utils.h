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
 * \file mhc_pre_utils.h
 * \brief Generic alignment and cross-core synchronization utilities
 */
#ifndef ASCENDC_MHC_PRE_UTILS_H
#define ASCENDC_MHC_PRE_UTILS_H

namespace MhcPreUtils {

// UB data-movement alignment.
constexpr uint32_t UB_ALIGN_SIZE = 32U;

// BS and M-K AIV/AIC synchronization protocol. X_READY/X_CONSUMED protect
// operand staging; MM_READY is used only by BS for the L0C-to-UB result.
// The second AIV subblock uses the same flags at SUBBLOCK_FLAG_OFFSET.
constexpr uint8_t CROSS_CORE_SYNC_MODE = 4U;
constexpr uint16_t SUBBLOCK_FLAG_OFFSET = 16U;
constexpr uint16_t X_READY_FLAG = 8U;
constexpr uint16_t X_CONSUMED_FLAG = 9U;
constexpr uint16_t MM_READY_FLAG = 10U;

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b)
{
    return CeilDiv(a, b) * b;
}
/**
 * Get the size of vector registers in bytes
 */
__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}
} // namespace MhcPreUtils

#endif // ASCENDC_MHC_PRE_UTILS_H
