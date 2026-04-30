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
 * \file common_def.h
 * \brief
 */

 #ifndef COMMON_DEF_H
 #define COMMON_DEF_H

namespace MatmulReduceScatterV2Impl {
using A_DTYPE = DTYPE_X1;                       // A矩阵数据类型（与x1输入一致）
using B_DTYPE = DTYPE_X1;                       // B矩阵数据类型（与x1输入一致）
using C_DTYPE = DTYPE_Y;                        // C矩阵（输出）数据类型
using BIAS_DTYPE = DTYPE_Y;                     // Bias数据类型（与输出一致）
constexpr uint8_t MAX_HANDLE = 64;              // 最大handle数（通信句柄上限）
constexpr uint8_t MC2_DEBUG_ONLY_CUBE = 1;      // 调试模式：只计算Matmul不通信
constexpr uint8_t MC2_DEBUG_ONLY_AICPU = 4;     // 调试模式：只通信不计算Matmul
constexpr uint64_t BLOCK_SIZE = 128;            // 块大小常量，用于对齐计算
template <class T>
struct BiasType {
    using type = float;                         // 默认bias类型为float
};
template <>
struct BiasType<half> {
    using type = half;                          // half类型的bias特化
};
__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y -1) / y;
}

}
#endif // COMMON_DEF_H