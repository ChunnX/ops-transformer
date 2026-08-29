/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_CONSTANTS_H
#define MEGA_MOE_CONSTANTS_H

#include <cstdint>

namespace MegaMoeImpl {

enum class MegaMoeActMode : uint8_t {
    SWIGLU = 0U,
    SITU = 1U,
    SWIGLU_STEP = 2U,
    SWIGLU_OAI = 3U,
};

enum class MegaMoeActSubMode : uint8_t {
    DEFAULT = 0U,
    LINEAR = 1U,
};

enum class MegaMoeGmmMode : uint8_t {
    A8W8_ND = 0U,
    A8W4_NZ = 1U,
    A8W8_NZ = 2U,
    A4W4_ND = 3U,
    A4W4_NZ = 4U,
};

constexpr uint8_t GMM_MODE_A8W8_ND = static_cast<uint8_t>(MegaMoeGmmMode::A8W8_ND);
constexpr uint8_t GMM_MODE_A8W4_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A8W4_NZ);
constexpr uint8_t GMM_MODE_A8W8_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A8W8_NZ);
constexpr uint8_t GMM_MODE_A4W4_ND = static_cast<uint8_t>(MegaMoeGmmMode::A4W4_ND);
constexpr uint8_t GMM_MODE_A4W4_NZ = static_cast<uint8_t>(MegaMoeGmmMode::A4W4_NZ);

constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint64_t IDX_GMM1_OFFSET = 12UL;
constexpr uint64_t IDX_GMM2_OFFSET = 13UL;

constexpr int32_t INT_CACHELINE = 16;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr uint32_t BITS_PER_BYTE = 8U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t TOPK_INDEX = 2U;
constexpr uint32_t WEIGHT_INDEX = 3U;
constexpr uint32_t SYNC_EVENT_ID0 = 0;
constexpr uint32_t SYNC_EVENT_ID1 = 1;
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr uint32_t SYNC_EVENT_ID3 = 3;
constexpr uint32_t SYNC_EVENT_ID4 = 4;
constexpr uint32_t SYNC_EVENT_ID5 = 5;
constexpr int64_t GM_FLAG_POLL_BACKOFF_CYCLES = 200;
// W8/W4 权重 L2 预取（仅 W1，fire-and-forget）。双窗口独立游标、独立预算（按 firstExpert 专家
// 边界分割数据）：窗口 0 由 AIV0 平面在量化段后倾泻（AIV1 平面独占 SendMask，指令流分离）、
// SyncAll 前 PipeBarrier 排空；窗口 1 在 count 等待自旋处交付首份额（MTE2_V 事件排空收尾）+
// wave 循环尾滑动补发。两窗口共用预算公式与分档上限，各自开关独立控制；均由 AIV0 持有，后续
// 可按专家边界合并为连续覆盖。当前配置：窗口 0 开启，窗口 1 关闭。
constexpr uint32_t WEIGHT_L2_PREFETCH_MTE2_V_EVENT_ID = SYNC_EVENT_ID5;
// 窗口开关：false 时对应构造点游标保持 nullptr（窗口 1 的 count 等待退回原版等待；窗口 0 不发射）。
// constexpr 编译期裁剪，无运行时开销。
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_WINDOW0 = true;
constexpr bool WEIGHT_L2_PREFETCH_ENABLE_WINDOW1 = false;
// 预取 scratch 偏移的兜底默认值：正常运行时 scratch 位于平台 UB 顶部 [ubSize−16K, ubSize)（host 按
// ubSize 动态下发 weightPrefetchUbOffset，SendMask 布局同值钳位），本常量仅作字段未设置时的回落。
constexpr uint32_t WEIGHT_L2_PREFETCH_UB_OFFSET = 192U * 1024U;
constexpr uint32_t WEIGHT_L2_PREFETCH_CHUNK_BYTES = 16U * 1024U;
// 自适应总量（单窗口预算）：min(min(expertCount - firstExpert, MAX_EXPERTS) * W1 单专家量,
// 拓扑分档上限)。上限按拓扑分档：跨机（>8P）8MB、单机（<=8P）2MB——单机 RTT 小、GMM1 消费早，
// 预取过多早期倾泻白发率高，上限进一步收紧。
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_TOTAL_BYTES_CROSS_NODE = 8U * 1024U * 1024U;
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_TOTAL_BYTES_SINGLE_NODE = 2U * 1024U * 1024U;
constexpr uint32_t WEIGHT_L2_PREFETCH_MAX_EXPERTS = 2U;
// 窗口 1 滑动补发配额（chunk/wave）：count 自旋仅 μs 级（CrossRankSync 已保证远端 mask 就绪，
// 自旋≈AIV1 本地算表时间），首份额必然欠交；剩余预算按 wave 边界小额补发，使交付量不依赖自旋
// 时长、驻留收敛到 O(1 wave)。a8w8/a8w4/a4w4 三条 wave 路径的循环尾消费。16KB chunk 下配额 4
// 即 64KB/wave；配额取小值以抑制
// 对共享 MTE2 引擎（块内 1AIC+2AIV 共用）上 dispatch/combine 流量的干扰。
constexpr uint32_t WEIGHT_L2_PREFETCH_WAVE_QUOTA_CHUNKS = 4U;
// 权重预取跨机门控阈值（实测）：worldSize > 8 才使能。作用于窗口 0 与窗口 1 的构造门控
// （W4/W8 统一：跨机才开，单机全关）与 gmm1 权重读的 L2 hint 联动（窗口 1 已由开关整体关闭）。
constexpr uint32_t WEIGHT_PREFETCH_MIN_WORLD_SIZE = 8U;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
constexpr uint32_t META_INFO_SIZE = 8U;
constexpr uint8_t GROUPED_MATMUL_MODE_GENERAL = 0U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W4 = 1U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W8_NZ = 2U;
// a4w4 混合场景：GMM1 走 generic a4w4，GMM2 走 A8W4。
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4 = 3U;
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4_NZ = 4U;

constexpr uint32_t L1_TILE_M_256 = 256U;
constexpr uint32_t L1_TILE_M_128 = 128U;
constexpr uint32_t L1_TILE_N = 256U;
constexpr uint32_t L0_TILE_K = 128U;
constexpr uint32_t ACTIVATION_N_HALF = 2U;

// Hcomm 批量句柄中每条普通读写描述符占 64B。
// Layered 的数据批次仍按最多 256 个 token 组织，64 KiB 可容纳同批 token 的附加 WQE。
constexpr uint32_t HCOMM_BATCH_WQE_BYTES = 64U;
constexpr uint32_t LAYERED_USABLE_UB_BYTES = 248U * 1024U;
constexpr uint32_t LAYERED_HCOMM_BATCH_UB_BYTES = 64U * 1024U;
constexpr uint32_t LAYERED_HCOMM_BATCH_UB_OFFSET = LAYERED_USABLE_UB_BYTES - LAYERED_HCOMM_BATCH_UB_BYTES;
constexpr uint32_t LAYERED_HCOMM_BATCH_WQE_CAPACITY = LAYERED_HCOMM_BATCH_UB_BYTES / HCOMM_BATCH_WQE_BYTES;
} // namespace MegaMoeImpl

#endif // MEGA_MOE_CONSTANTS_H
