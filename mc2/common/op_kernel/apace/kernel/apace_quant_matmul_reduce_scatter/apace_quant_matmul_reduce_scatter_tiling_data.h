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
 * \file quant_matmul_reduce_scatter_tiling_data.h
 * \brief quantMatmulReduceScatter tiling data — 计算在前 + AlltoAll + reduceSum
 *
 * 与 all_to_all_quant_matmul 的关键区别：
 *   - 通信数据类型为 CType (bfloat16)，不是 AType (fp8)
 *   - 无 ScaleA 通信（计算在前，通信的是输出 C）
 *   - 无 localMatmul 字段（无 LOCAL/REMOTE 双阶段）
 *
 * Round 4 优化：
 *   - 新增 subMmTilingData：子问题 tiling (m=headMSize)，用于 T>1 per-tile 流水路径
 *   - mmTilingData 仍用于 T=1 退化路径和 tail tile
 *   - blockDim 固定使用 mmTilingData.usedCoreNum（不随子问题 tiling 变化）
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#include "../../tiling/quant_matmul_tiling_data.h"
#include "../../tiling/comm_tiling_data.h"
#include "../../block/aiv_comm/collective_comm_context.h"
// CommContext 定义来自 all_to_all_matmul_tiling_data.h（复用，不重定义）

struct CommContext {
    Apace::AivComm::CommUdmaContext udmaCtx;
    Apace::AivComm::CommUbmemContext ubmemCtx;
};
struct ApaceQuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;          // C 矩阵 AlltoAll 通信切分（M 轴）
    QuantMatmulTilingData mmTilingData;     // 量化 MatMul tiling (全量, T=1 退化路径 + tail tile)
    QuantMatmulTilingData subMmTilingData;  // 子问题 tiling (m=headMSize, T>1 per-tile head tile 路径)
};
