/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file mhc_pre_split_post_res_const.h
 * \brief MhcPreSplitPostRes Host 侧共享规格常量（tiling/aclnn/def 统一引用，单一事实来源）
 */

#ifndef MHC_PRE_SPLIT_POST_RES_CONST_H_
#define MHC_PRE_SPLIT_POST_RES_CONST_H_

#include <cstddef>
#include <stdint.h>

namespace optiling {

// 固定规格：n 固定为 4；dimTile 上限为 UB 单次能处理的最大 D 大小
constexpr int64_t MHC_N_VALUE = 4;
constexpr int64_t MHC_DIM_TILE_MAX = 2560;

// 入参限制：T 无上限（B·S 折叠后任意规模）
constexpr int64_t MHC_MIN_T = 1;

// D 白名单：仅支持 4 种典型 shape（2560/5120/6144/7168，均为 128 对齐）
constexpr int64_t MHC_D_2560 = 2560;
constexpr int64_t MHC_D_5120 = 5120;
constexpr int64_t MHC_D_6144 = 6144;
constexpr int64_t MHC_D_7168 = 7168;

// norm_eps 默认值：def.cpp schema 为权威源，tiling 校验回退值与此同源；
// kernel 侧 base.h 成员默认值因编译环境隔离保留字面量，注释锚定本常量
constexpr float MHC_DEFAULT_NORM_EPS = 1e-6f;

// shape 维度数（秩）
constexpr size_t MHC_DIM_NUM_1 = 1;
constexpr size_t MHC_DIM_NUM_2 = 2;
constexpr size_t MHC_DIM_NUM_3 = 3;
constexpr size_t MHC_DIM_NUM_4 = 4;

// shape 维度下标
constexpr int64_t MHC_DIM_ZERO = 0;
constexpr int64_t MHC_DIM_ONE = 1;
constexpr int64_t MHC_DIM_TWO = 2;
constexpr int64_t MHC_DIM_THREE = 3;

// FP32 向量操作最小对齐单位（32B / 4B = 8 个 float）
constexpr int64_t MIN_FP32_ALIGN_ELEM_SIZE = 8;

}  // namespace optiling

#endif  // MHC_PRE_SPLIT_POST_RES_CONST_H_
