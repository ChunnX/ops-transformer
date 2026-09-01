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
 * \file quant_matmul_reduce_scatter_urma_tiling_data.h
 * \brief quantMatmulReduceScatter tiling data — 计算在前 + AllToAll PUT + reduceSum
 *
 * 与 all_to_all_quant_matmul 的关键区别：
 *   - 通信数据类型为 CType (bfloat16)，不是 AType (fp8)
 *   - 无 ScaleA 通信（计算在前，通信的是输出 C）
 *   - 无 localMatmul 字段（无 LOCAL/REMOTE 双阶段）
 *
 * Tiling 结构:
 *   - mmTilingData: 全量 matmul tiling (m=完整M), 用于 T=1 退化路径
 *   - subMmTilingData: 子问题 tiling (m=headMSize), 用于 T>1 per-tile head tile 路径 (baseM 减半优化)
 *   - tailMmTilingData: tail tile tiling (m=rankSize*tailMSize), 用于 T>1 tail tile 路径
 *   - blockDim 固定使用 mmTilingData.usedCoreNum（不随子问题 tiling 变化）
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "apace/tiling/quant_matmul_tiling_data.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/core/aiv_comm/collective_comm_context.h"

namespace Apace {
namespace AivComm {
// Convenience wrapper so the kernel receives one pointer instead of two.
struct CommContext {
    CommUdmaContext udmaCtx;
    CommUbmemContext ubmemCtx;
};
} // namespace AivComm
} // namespace Apace

using Apace::AivComm::CommContext;

struct QuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;          // C 矩阵 AlltoAll 通信切分（M 轴）
    QuantMatmulTilingData mmTilingData;     // 量化 MatMul tiling (全量, T=1 退化路径)
    QuantMatmulTilingData subMmTilingData;  // 子问题 tiling (m=headMSize, T>1 per-tile head tile 路径)
    QuantMatmulTilingData tailMmTilingData; // tail tile tiling (m=rankSize*tailMSize, T>1 tail tile 路径)
};
