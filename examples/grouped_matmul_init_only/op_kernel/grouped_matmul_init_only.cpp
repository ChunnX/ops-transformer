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
 * \file grouped_matmul_init_only.cpp
 * \brief
 */

#include "grouped_matmul_init_only.h"

// ============================================================================
// init-only 算子（Issue #36 第二个算子）：kernel 入口在 op.Init() 完成后直接 return，
// 跳过任何 Iterate / mma（不进入 matmul 计算阶段），用于测量 launch+Init 的固定头开销。
// Init() 内构造显著重于 Add 的资源初始化（4 路 GM + group_list 解析 + 多块 L1/UB +
// 大 tilingData 访问），用于对照 Init 复杂度对头开销的影响。
// 单一 dtype fp16，schMode/tilingKey = 0。
// ============================================================================
template <uint32_t schMode>
__global__ __aicore__ void grouped_matmul_init_only(
    GM_ADDR x, GM_ADDR weight, GM_ADDR group_list, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupedMatmulInitOnlyTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupedMatmulInitOnlyTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(GMM_INIT_ONLY_TPL_SCH_MODE_0)) {
        NsGmmInitOnly::GroupedMatmulInitOnly<half> op; // 算子 kernel 实例获取
        op.Init(x, weight, group_list, y, &tilingData); // 算子 kernel 实例初始化（重 Init）
        return;                                         // init-only：Init 完成即返回
    }
}
