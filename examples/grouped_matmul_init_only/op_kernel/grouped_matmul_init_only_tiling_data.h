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
 * \file grouped_matmul_init_only_tiling_data.h
 * \brief tiling data struct（较大结构体，模拟 TCubeTiling，体现 GMM 类"重" Init）
 */

#ifndef _GROUPED_MATMUL_INIT_ONLY_TILING_DATA_H_
#define _GROUPED_MATMUL_INIT_ONLY_TILING_DATA_H_

#include <cstdint>

// 最大 group 数（覆盖 DESIGN §3.2 最大 G=16）
constexpr int32_t GMM_INIT_ONLY_MAX_G = 64;

// 模拟 GroupedMatmul 类算子的 host tiling：既包含 group 相关的 group_list/偏移数组，
// 又包含一整套 matmul 结构标量（模拟 TCubeTiling 的大结构），使 host 侧 tiling 计算量与
// 结构体拷贝开销显著大于 Add，用于对照 Init 复杂度对头开销的影响。
struct GroupedMatmulInitOnlyTilingData {
    // 基本 shape 信息
    int64_t M;
    int64_t K;
    int64_t N;
    int64_t G;

    // group 相关：group_list 原值 + per-group 累积 m 偏移
    int64_t groupList[GMM_INIT_ONLY_MAX_G];
    int64_t groupMOffset[GMM_INIT_ONLY_MAX_G];

    // 模拟 TCubeTiling 的 matmul 标量字段（host 计算并填充）
    int64_t baseM;
    int64_t baseN;
    int64_t baseK;
    int64_t singleCoreM;
    int64_t singleCoreN;
    int64_t singleCoreK;
    int64_t stepM;
    int64_t stepN;
    int64_t stepKa;
    int64_t stepKb;
    int64_t depthA1;
    int64_t depthB1;
    int64_t iterateOrder;
    int64_t usedCoreNum;
    int64_t coreNum;
    int64_t ubSize;
    int64_t l1Size;
    int64_t l0cSize;
    int64_t l0aSize;
    int64_t l0bSize;
    int64_t mTiles;
    int64_t nTiles;
    int64_t kTiles;
};
#endif
