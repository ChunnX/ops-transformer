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
 * \file chunk_gated_delta_rule_utils.h
 * \brief Common constants and alignment helpers for chunk_gated_delta_rule kernels
 */

#pragma once

#include "basic_api/kernel_basic_intf.h"

namespace ChunkGatedDeltaRule {
// Cube/NZ 布局按 16 个元素分块；行、列对齐单位分别保留名称，供搬运和地址计算使用。
constexpr uint32_t MM_BLOCK_CUBE = 16;
constexpr uint32_t MM_NZ_FRACTAL = 16;

// 向上对齐：ceil(value / align) * align，align > 0；参数使用相同单位（元素或字节）。
__aicore__ inline uint32_t CeilAlign(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

__aicore__ inline uint32_t GetZnOffset(uint32_t row, uint32_t col, uint32_t fullRows)
{
    // compact zN/NZ 按 16 列成块：offset=(col/16)*align(fullRows,16)*16+row*16+col%16。
    return (row % MM_BLOCK_CUBE) * MM_NZ_FRACTAL + (row / MM_BLOCK_CUBE) * MM_BLOCK_CUBE * MM_NZ_FRACTAL +
           (col % MM_NZ_FRACTAL) + (col / MM_NZ_FRACTAL) * CeilAlign(fullRows, MM_BLOCK_CUBE) * MM_NZ_FRACTAL;
}

// 同步信号
constexpr uint64_t V_MTE3_EVENT = 0;
constexpr uint64_t V_S_EVENT = 1;
constexpr uint64_t MTE2_V_EVENT = 2;
constexpr uint64_t S_V_EVENT = 3;
constexpr uint64_t MTE3_MTE2_EVENT = 4;
constexpr uint64_t MTE3_S_EVENT = 5;
constexpr uint64_t FIX_MTE2_EVENT = 6;
constexpr uint64_t S_MTE3_EVENT = 6;
constexpr uint64_t V_MTE2_EVENT = 7;

constexpr uint64_t NUM_ONE = 1;
constexpr uint64_t BUFFER_NUM_ONE = 1;
constexpr uint64_t BUFFER_NUM_TWO = 2;
constexpr uint64_t TQUE_DEPTH_TWO = 2;
constexpr uint64_t AIC_AIV_1_1 = 2;
constexpr uint64_t BROADCAST_AXIS = 2;
constexpr uint64_t TASK_RATIO = 2;
constexpr uint64_t STAGE3_BUFFER_COUNT = 4;
constexpr uint32_t MAX_L0_SIZE = 64 * 1024; // 64KB
constexpr uint32_t BLOCK_SIZE = 32;         // copypad对齐块大小
constexpr uint32_t BLOCK_FLOAT_NUM = 8;
constexpr uint32_t BLOCK_BF16_NUM = 16;
constexpr uint32_t TILE_LEN = 1024; // 1024 = 1kb，经测试1kb和10kb性能差异很小
} // namespace ChunkGatedDeltaRule
