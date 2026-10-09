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
 * Tiling 结构 (scope 级 tiling — 每个 scope 的 tiling m 必须等于该 scope 实际逻辑 M):
 *   各 scope 的 fragment 数不同 (REMOTE: rankSize-1 个, LOCAL: 1 个), 调度器 tail 参数
 *   (mTailMain/mBaseTailSplitCnt 等) 按设计 m 编码绝对位置, m 不匹配时 tile 形状非法
 *   → L0A/L1 越界。故按 scope 分别计算:
 *   - remoteHeadMmTilingData: REMOTE head 片段 (m=(R-1)*headMSize)
 *   - remoteTailMmTilingData: REMOTE tail 片段 (m=(R-1)*tailMSize, 无 tail 时置空)
 *   - localHeadMmTilingData:  LOCAL head 片段 (m=headMSize; T=1 时 = chunkM)
 *   - localTailMmTilingData:  LOCAL tail 片段 (m=tailMSize, 无 tail 时置空)
 *   - mmTilingData: 全量 tiling, 仅取 usedCoreNum 作 blockDim/AIV 角色派生
 *   - blockDim 固定使用 mmTilingData.usedCoreNum（不随 scope 级 tiling 变化）
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
    CommTilingData commTilingData;                // C 矩阵 AlltoAll 通信切分（M 轴）
    QuantMatmulTilingData mmTilingData;           // 全量 tiling (仅取 usedCoreNum)
    QuantMatmulTilingData remoteHeadMmTilingData; // REMOTE head 段 (m=(R-1)*headMSize)
    QuantMatmulTilingData remoteTailMmTilingData; // REMOTE tail 段 (m=(R-1)*tailMSize)
    QuantMatmulTilingData localHeadMmTilingData;  // LOCAL head tile (m=headMSize)
    QuantMatmulTilingData localTailMmTilingData;  // LOCAL tail tile (m=tailMSize)
    // scope 执行顺序: 0 = LOCAL 先 (默认), 1 = LOCAL 后置。
    // 单 flag 语义下二者时序等价 (flag = 双 scope 全部完成), 策略开关。
    uint8_t localScopeLast{0};
};

// 通信 tiles (splitAxisTileCnt + splitAxisTailCnt) 上限 — host/kernel 同源常量。
// tile 通知 flagid 直接使用 tid, 每 id 严格一次 Set ↔ 一次 Wait;
// dav_3510 flagid 为 4bit (0-15): id 14 保留给 SyncAll<true> (SYNC_AIV_ONLY_ALL),
// id 15 保留未用, tiles 可用 {0..13} ⇒ T ≤ 14。
constexpr uint32_t QMRS_MAX_COMM_TILES = 14;
