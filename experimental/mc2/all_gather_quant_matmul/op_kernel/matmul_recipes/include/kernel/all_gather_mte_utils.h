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
 * \file all_gather_mte_utils.h
 * \brief Device-side math utilities (CeilDiv, CeilAlignU64, BlockAlignMod) for AG data movement.
 */

#ifndef ALL_GATHER_MTE_UTILS_H
#define ALL_GATHER_MTE_UTILS_H

namespace AscendC {
// 向上取整除法：计算a除以b的向上取整结果
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
};

// 向上对齐：将a向上对齐到b的倍数
__aicore__ inline uint64_t CeilAlignU64(uint64_t a, uint32_t b)
{
    uint64_t bTemp = static_cast<uint64_t>(b);
    return (bTemp == 0) ? a : CeilDiv(a, bTemp) * bTemp;
};

// 尾块模运算：计算a对b取模，模为0时返回b（用于数据块尾块的计算）
__aicore__ inline uint64_t BlockAlignMod(uint64_t a, uint32_t b)
{
    if (b == 0) {
        return 0;
    }
    
    uint64_t c = a % b;
    return c ? c : b;
}
}  // namespace AscendC

#endif  // ALL_GATHER_MTE_UTILS_H