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
 * \file moe_tutel_dispatch_tiling.h
 * \brief
 */

#ifndef MOE_TUTEL_DISPATCH_TILING_H
#define MOE_TUTEL_DISPATCH_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MoeTutelDispatchTilingData)
// 全部 int64：B * capacity * hidden 可以轻易超过 2^31
TILING_DATA_FIELD_DEF(int64_t, samples);   // S = x.dim(0)
TILING_DATA_FIELD_DEF(int64_t, hidden);    // H = x.dim(1)
TILING_DATA_FIELD_DEF(int64_t, expertNum); // B = gates.dim(0) = y.dim(0)
TILING_DATA_FIELD_DEF(int64_t, capacity);  // C = attr capacity = y.dim(1)
TILING_DATA_FIELD_DEF(int64_t, routeNum);  // B * S
TILING_DATA_FIELD_DEF(int64_t, yElemCnt);  // B * C * H
// 不落 usedCoreNum 字段：它已经等于 blockDim，kernel 侧每个核只需要自己的
// initPerCore / routePerCore 区间并对总量取 min。**也不允许**用它做
// `if (blockIdx >= usedCoreNum) return;` 的早退——那会让部分核跳过 SyncAll 造成死锁。
TILING_DATA_FIELD_DEF(int64_t, initPerCore);  // 阶段 A 每核清零元素数
TILING_DATA_FIELD_DEF(int64_t, routePerCore); // 阶段 B 每核路由数
TILING_DATA_FIELD_DEF(int64_t, tileRoutes);   // 阶段 B 每批路由条数
TILING_DATA_FIELD_DEF(int64_t, hChunkElems);  // SPLIT_H 每块元素数；FULL_ROW 时等于 hidden
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeTutelDispatch, MoeTutelDispatchTilingData);

// TilingKey：不含 dtype，dtype 由 kernel 侧 DTYPE_X 宏模板化
constexpr uint64_t MOE_TUTEL_DISPATCH_TILING_KEY_FULL_ROW = 0UL; // 整行驻留 UB
constexpr uint64_t MOE_TUTEL_DISPATCH_TILING_KEY_SPLIT_H = 1UL;  // 单行切 H
constexpr uint64_t MOE_TUTEL_DISPATCH_TILING_KEY_EMPTY = 2UL;    // y 为空张量，空跑

struct MoeTutelDispatchCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};
} // namespace optiling

#endif // MOE_TUTEL_DISPATCH_TILING_H
