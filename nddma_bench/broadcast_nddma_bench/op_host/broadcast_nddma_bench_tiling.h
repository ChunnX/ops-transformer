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
 * \file broadcast_nddma_bench_tiling.h
 * \brief BroadcastNddmaBench TilingData 定义（Host/Kernel 共享）
 *
 * TilingData 承接原直调 attr 结构 BroadcastNDDMAAttr 中的 shape/控制字段：
 *   dim0   : 分核维（blockNum = min(dim0, coreNum)）
 *   dim1   : 唯一核内 for 循环维
 *   dim2   : NDDMA 最外轴（单条内遍历，非 for）
 *   dim3   : NDDMA 中轴
 *   dim4   : NDDMA 最内连续轴
 *   isMte3 : 1 = MTE3 搬出到 GM（精度对比）；0 = 去掉 MTE3（纯 NDDMA 带宽）
 *   scene  : 0=AB / 1=BA / 2=BAB / 3=ABA（广播场景，DESIGN §9.0）
 * dtype 不进 TilingData，改由 TilingKey 区分（fp32=0 / fp16=1 / int8=2）。
 *
 * ⚠ 广播算子输入 shape 的广播轴被折叠为 1，无法从 input shape 恢复完整输出 dim2/dim3/dim4，
 *   故 dim2/dim3/dim4/scene/is_mte3 全部由 OpDef Attr 传入；dim0/dim1 从 input shape 读取。
 */
#ifndef ASCEND_OPS_BROADCAST_NDDMA_BENCH_TILING_H
#define ASCEND_OPS_BROADCAST_NDDMA_BENCH_TILING_H
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BroadcastNddmaBenchTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dim0);    // 分核维
TILING_DATA_FIELD_DEF(uint32_t, dim1);    // 唯一核内循环维
TILING_DATA_FIELD_DEF(uint32_t, dim2);    // NDDMA 最外轴
TILING_DATA_FIELD_DEF(uint32_t, dim3);    // NDDMA 中轴
TILING_DATA_FIELD_DEF(uint32_t, dim4);    // NDDMA 最内连续轴
TILING_DATA_FIELD_DEF(uint32_t, isMte3);  // 是否 MTE3 搬出
TILING_DATA_FIELD_DEF(uint32_t, scene);   // 广播场景枚举
END_TILING_DATA_DEF;

// 3 个 dtype 分支（fp32=0 / fp16=1 / int8=2）共用同一 TilingData 布局，
// 故按 optype 注册一次默认结构即可；dtype 差异由运行期 TilingKey 区分，与结构布局无关。
REGISTER_TILING_DATA_CLASS(BroadcastNddmaBench, BroadcastNddmaBenchTilingData)
} // namespace optiling
#endif // ASCEND_OPS_BROADCAST_NDDMA_BENCH_TILING_H
