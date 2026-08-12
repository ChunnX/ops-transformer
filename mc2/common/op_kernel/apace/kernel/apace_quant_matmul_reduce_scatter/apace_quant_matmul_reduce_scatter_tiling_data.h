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
 * \file apace_quant_matmul_reduce_scatter_tiling_data.h
 * \brief v2 ReduceScatter (AllToAll PUT + workspace 流水线) MX 量化 Matmul tiling data。
 *
 * v2 架构：3 级流水 mm→aivComm→reduceSum，PIPELINE_NUM=4 多 buffer workspace 轮转。
 * tiling 结构体从 host launcher 传到 kernel。
 * ReduceScatter 语义：每 rank 输出 C[M/rankSize, N]，M 轴切分。
 */

#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "../../tiling/quant_matmul_tiling_data.h"
#include "../../tiling/comm_tiling_data.h"
#include "../../block/aiv_comm/collective_comm_context.h"

// v2 流水线常量（DESIGN.md §2.7）
// 注意：这些常量在 quant_matmul_mx_kernel.h 中有前向定义（#ifndef 保护），
// 此处定义仅在 kernel.h 未被 include 时生效
#ifndef SYNC_MODE2_DEFINED
#define SYNC_MODE2_DEFINED
constexpr uint32_t PIPELINE_NUM = 4;        // workspace 多 buffer 轮转深度
constexpr uint64_t SYNC_AIV_AIC_FLAG = 3;   // 回压通道 flagId（AIV→AIC，workspace 可复用）
constexpr uint64_t SYNC_MM_DONE_FLAG = 5;   // mm 完成通道 flagId（AIC→AIV，mm 输出已写 workspace）
constexpr uint64_t SYNC_MODE2 = 2;           // CrossCore flag 模式 2（950PR 模板参数生效）
#endif

// v2 ReduceScatter tiling 结构体（UDMA 模式，3 级流水线）
// 与 v1 差异：新增 pipelineNum/coreNum/ubCalSize/totalMmTiles/workspaceSize
struct ApaceQuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;              // data 通道通信切分
    CommTilingData scaleCommTilingData;         // scale 通道通信切分
    QuantMatmulTilingData tileQbmmTilingData;   // 计算 tiling (m=M/rankSize)
    uint32_t localMatmul{0};                     // 0:默认不启用(REMOTE-only); 1:DEFERRED_SYNC(可选优化)
    uint32_t pipelineNum{PIPELINE_NUM};       // PIPELINE_NUM=4（workspace 多 buffer 轮转深度）
    uint32_t coreNum{0};                         // AIC 核数（=usedCoreNum）
    uint32_t ubCalSize{0};                       // reduceSum UB 单 tile 大小（baseM*baseN）
    // DEPRECATED（DESIGN.md §2.16 Issue 2）：host 填 0，kernel 不消费。
    // AIV reduceSum 循环由 bsAiv.GetTileIdx() 隐式终止，与 AIC SetFlag(5) 次数构造一致。
    uint32_t totalMmTiles{0};
    uint64_t workspaceSize{0};                   // workspace 总大小（字节）
};

struct CommContext {
    Apace::AivComm::CommUdmaContext udmaCtx;
    Apace::AivComm::CommUbmemContext ubmemCtx;
};