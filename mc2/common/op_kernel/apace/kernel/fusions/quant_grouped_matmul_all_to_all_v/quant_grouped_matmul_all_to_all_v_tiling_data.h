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
 * \file quant_grouped_matmul_all_to_all_v_tiling_data.h
 * \brief QuantGroupedMatMul + AllToAllV 融合算子 tiling 数据定义（Hcomm/CCU 路径）
 *
 * 该结构体同时被 host 侧 tiling（原算子 op_host/op_tiling/arch35）与 device 侧
 * apace 融合 kernel（mx_quant_grouped_matmul_all_to_all_v_hcomm_impl.h）使用，
 * 字段顺序属于 host-device 契约，不可随意调整。
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

namespace Apace {

// e * epWorldSize 上限，与原算子约束一致（groupList / cGroupOffsetTable 表长）
constexpr uint32_t GMM_A2AV_MAX_EXPERT_NUM = 256U;
// AIV 侧 AllToAllV send/recv cnt/offset 数组按 rank 数开表的上限（epWorldSize 超限时 tiling 侧回退原通路）
constexpr uint32_t GMM_A2AV_MAX_RANK_NUM = 128U;
// 分批通算掩盖流水：expertNumPerBatch = CeilDiv(e, 4)（与原通路 TARGET_COMM_ROUNDS=4 口径一致），
// totalLoopCnt = CeilDiv(e, expertNumPerBatch) ≤ 4（e ≤ 32）；AIV 侧 handle 数组按此上限留余量
constexpr uint32_t GMM_A2AV_MAX_COMM_LOOP_CNT = 8U;

// GroupedMatmulMxKernel 的 groupList 存储格式（与 blaze GroupedMatmulMxKernel 定义一致）
constexpr uint8_t GMM_A2AV_GROUP_LIST_TYPE_LENGTH = 1U; // 每组存 token 数
constexpr uint8_t GMM_A2AV_GROUP_LIST_TYPE_SPARSE = 2U; // [专家id, token数] 成对存储

/*!
 * \brief Hcomm/CCU 路径 tiling 结构体（apace 自有，不依赖原算子 QuantGmmA2avTilingData 布局）
 *
 * 数据布局契约（专家分批通算掩盖流水，最终语义与原算子一致）：
 *   - gmmX 按 [expert0: rank0..rankN-1 tokens][expert1: ...] 专家主序排列；
 *     kernel 侧 groupList 采用 SPARSE 格式按 (expertIdx * ep + rankIdx) 主序生成，
 *     A/scaleA 按批推进基址后走 GroupedMatmulMxKernel 默认累计 offset，B/scaleB 按 sparse 专家 id 寻址；
 *   - GMM 输出（AllToAllV 发送缓冲）按全局 rank-first 布局，由 cGroupOffsetTable 提供；
 *     分批通信读取其中本批的不相交连续 chunk（sendOff = rank 全局基 + 批内专家前缀）；
 *   - y 输出按 rank-first 接收（recvCounts 按源 rank 聚合后顺序拼接）。
 */
struct hcommQuantGmmAlltoAllvTilingData {
    // ============ HCCL tiling（AllToAllV） ============
    Mc2InitTiling mc2InitTiling; // HCCL 初始化参数
    Mc2CcTiling mc2CcTiling;     // HCCL CC task 配置

    // ============ Matmul tiling（apace SWAT 引擎产出） ============
    QuantMatmulTilingData gmmTile;      // 路由专家 GMM：m=maxM, n=N1, k=H1
    QuantMatmulTilingData sharedMmTile; // 共享专家 MM：m=BS, n=N2, k=H2（hasSharedMm=1 时有效）

    // ============ 问题维度 ============
    uint64_t h1;     // 路由专家 hidden size（GMM 的 K）
    uint64_t n1;     // 路由专家 FFN 中间维（GMM 的 N / 通信行宽）
    uint64_t h2;     // 共享专家 hidden size
    uint64_t n2;     // 共享专家 FFN 中间维
    uint64_t bs;     // 共享专家 batch sequence size
    uint64_t aTotal; // 本卡发送 token 总数（sendCounts 之和）
    uint64_t bsk;    // 本卡接收 token 总数（recvCounts 之和）
    uint64_t maxM;   // 单组最大 token 数（max over expert of Σ_rank sendCnt）

    // ============ user workspace 内部布局（字节偏移） ============
    // [0, wsGroupListOffset)              GMM 输出 / AllToAllV 发送缓冲（rank-first，A*N1 个 CType）
    // [wsGroupListOffset, wsCTableOffset) groupList（SPARSE：e*ep 对 int64）
    // [wsCTableOffset, wsSharedGroupListOffset) cGroupOffsetTable（e*ep 个 uint64）
    // [wsSharedGroupListOffset, +8)       共享专家单组 groupList（int64 [BS]）
    uint64_t wsGroupListOffset;
    uint64_t wsCTableOffset;
    uint64_t wsSharedGroupListOffset;

    // ============ 通信计数（sendCounts/recvCounts，按 [expertId + rank * e] 展开） ============
    int32_t sendCnt[GMM_A2AV_MAX_EXPERT_NUM];
    int32_t recvCnt[GMM_A2AV_MAX_EXPERT_NUM];

    // ============ 专家并行参数 ============
    uint32_t e;           // 单卡专家数
    uint32_t epWorldSize; // EP 域大小
    uint32_t hasSharedMm; // 是否携带共享专家 MM 计算
};

} // namespace Apace
