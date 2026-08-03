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
 * \file nlast_transpose_nddma_bench_tiling.h
 * \brief NlastTransposeNddmaBench TilingData 定义（Host/Kernel 共享）
 *
 * TilingData 承接原直调 attr 结构 NLastTransposeNDDMAAttr 中的 shape/控制字段：
 *   dim0   : 分核维（blockNum = dim0）
 *   dim1   : 核内 for 循环维
 *   dim2   : 被交换的中间轴之一（转置前 axis 2）
 *   dim3   : 被交换的中间轴之一（转置前 axis 3）
 *   dim4   : 最内连续轴（保留不动，随块搬运）
 *   isMte3 : 1 = MTE3 搬出到 GM（精度对比）；0 = 去掉 MTE3（纯 NDDMA 带宽）
 * dtype 不进 TilingData，改由 TilingKey 区分（fp32=0 / fp16=1 / int8=2）。
 */
#ifndef ASCEND_OPS_NLAST_TRANSPOSE_NDDMA_BENCH_TILING_H
#define ASCEND_OPS_NLAST_TRANSPOSE_NDDMA_BENCH_TILING_H
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(NlastTransposeNddmaBenchTilingData)
TILING_DATA_FIELD_DEF(uint32_t, dim0);    // 分核维
TILING_DATA_FIELD_DEF(uint32_t, dim1);    // 核内循环维
TILING_DATA_FIELD_DEF(uint32_t, dim2);    // 中间轴之一
TILING_DATA_FIELD_DEF(uint32_t, dim3);    // 中间轴之一
TILING_DATA_FIELD_DEF(uint32_t, dim4);    // 最内连续轴
TILING_DATA_FIELD_DEF(uint32_t, isMte3);  // 是否 MTE3 搬出
END_TILING_DATA_DEF;

// 3 个 dtype 分支（fp32=0 / fp16=1 / int8=2）共用同一 TilingData 布局，
// 故按 optype 注册一次默认结构即可；dtype 差异由运行期 TilingKey 区分，与结构布局无关。
REGISTER_TILING_DATA_CLASS(NlastTransposeNddmaBench, NlastTransposeNddmaBenchTilingData)
} // namespace optiling
#endif // ASCEND_OPS_NLAST_TRANSPOSE_NDDMA_BENCH_TILING_H
