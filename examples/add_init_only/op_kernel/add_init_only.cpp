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
 * \file add_init_only.cpp
 * \brief
 */

#include "add_init_only.h"

enum class AddInitOnlyTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

// ============================================================================
// init-only 算子（Issue #36）：kernel 入口在 op.Init() 完成后直接 return，
// 跳过 op.Process()（不进入计算阶段），用于测量 launch+Init 的固定头开销。
// Init() 内保留真实的资源初始化（GM 绑定 + UB queue 分配）以反映真实 Init 开销。
// ============================================================================
template <uint32_t schMode>
__global__ __aicore__ void add_init_only(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddInitOnlyTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddInitOnlyTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(AddInitOnlyTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        NsAddInitOnly::AddInitOnly<float> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);        // 算子kernel实例初始化
        return;                               // init-only：Init 完成即返回，跳过 Process()
    }
    if constexpr (schMode == static_cast<uint32_t>(AddInitOnlyTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        NsAddInitOnly::AddInitOnly<int32_t> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);          // 算子kernel实例初始化
        return;                                 // init-only：Init 完成即返回，跳过 Process()
    }
}
