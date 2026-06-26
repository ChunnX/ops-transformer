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
 * \file dispatch_ccu_common.h
 * \brief dispatch-ccu host / device 共用：编排入参、teardown tiling、Per-Rank HBM GM 布局常量
 *        设计文档：docs/CCU_DISPATCH_COMBINE_IMPL_DESIGN.md §9 §10
 */
#ifndef DISPATCH_CCU_COMMON_H
#define DISPATCH_CCU_COMMON_H

#include <cstdint>

// ── 编译期规模常量 ──
constexpr uint32_t DISPATCH_CCU_MAX_RANK         = 64U;  // fullmesh 支持的最大卡数
constexpr uint32_t DISPATCH_CCU_READY_FLAG_BYTES = 4U;   // 每个 READY_FLAG 占 4B
constexpr uint64_t DISPATCH_CCU_META_PER_RANK    = 64U;  // METADATA 区每卡字节数（token_count / expert_offset，预留）

// ── host 侧编排入参 ──
struct DispatchCcuParam {
    uint32_t worldSize;   // EP 域卡数（fullmesh rank 数）
    uint32_t myRank;      // 本卡 rank id
    uint32_t blockDim;    // teardown AICore 核函数 block 数
    uint32_t topk;        // 每 token 路由到的 expert 数
    uint32_t expertNum;   // expert 总数
    uint64_t tokenNum;    // 本卡 token 数 (BS)
    uint64_t hidden;      // 隐藏维 H
};

// ── teardown(AICore 核函数) 读取的 tiling ──
// host 在 MoeDistributeDispatchCcu 内构造、拷贝到 device，teardown 核函数解析。
struct DispatchCcuTilingData {
    uint64_t worldSize;
    uint64_t myRank;
    uint64_t tokenNum;
    uint64_t hidden;
    uint64_t topk;
    uint64_t expertNum;
    uint64_t maxTokenPerRank;   // DATA 区每卡 token 槽位上限
    uint64_t readyFlagsOffset;  // 见下方 GM 布局
    uint64_t metadataOffset;
    uint64_t dataOffset;
};

// ── Per-Rank HBM GM Buffer 布局（设计文档 §10）──
//  +readyFlagsOffset  READY_FLAGS  (worldSize × 4B)            peer r 写完置位；teardown 在 [0] 写 1 做 ack
//  +metadataOffset    METADATA     (worldSize × META_PER_RANK) token_count / expert_offset 等元数据
//  +dataOffset        DATA         (worldSize × maxTok × H)    peer r 推来的 token 原文
//  数据 + 元数据同在 HBM GM（Q11）；三段偏移由 host 算好写进 tiling，host / device / 对端三方共用。

#endif // DISPATCH_CCU_COMMON_H
