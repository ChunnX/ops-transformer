/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file ffn_wb_probe_tiling.h
 * \brief 迭代二 穿刺2 (probe_iter2_boundary_maxshape) 的 Tiling 常量 + 结构体
 *        (纯 C/C++，kernel 与 host 共用)
 *
 * 穿刺目标: 边界上界 shape (A=1024 / topK+1=64 语义) + UB 收敛不溢出验证
 *           (issue_20260708_ub-size-host-kernel-mismatch_01 的 NPU 实跑验证载体)
 * 目标平台: Ascend950DT_9584 (arch35 / DAV_3510)
 *
 * 关键设计点 (issue #01 收敛):
 *   主线 tiling 公式: sortLoopMaxElement = ubSize / (sizeof(int32)*2*4) / 32 * 32 = ubSize/32
 *   多核 sort 每核每 loop 分配 4 个 buffer (in/out/temp/sorted)，每个 = sortNum*sizeof(int32)*2 字节。
 *   当 per-loop tile == sortLoopMaxElement 时:
 *       4 * (ubSize/32) * 8 = ubSize  ← 恰好占满派生 UB。
 *   故 UB 取值口径决定生死:
 *       kernel 侧真实可用 UB (ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE) = 221184 (216KB) → SLM=6912 → footprint 216KB 恰好装下
 *       host 侧 (GetCoreMemSize)                                = 253952 (248KB) → SLM=7936 → footprint 248KB 溢出物理 216KB 32KB
 *   本穿刺让 4 个 sort 核每核 tile == 由 kernel 侧真实 UB 派生的 SLM，使 4-buffer footprint 打到物理 UB 上界 —
 *   这是比标称边界 shape (Y=262144 → 64 核 × 4096 = 128KB 亚上界) 更严的 UB 极限测试。
 *
 * 3 次 kernel 启动，用 aclrtSynchronizeStream 做核间栅栏 (对齐 probe2):
 *   A) 多核 sort   : SORT_CORE_NUM 核，各 Sort<float,true> 一个 tile(=SLM) -> proposal 落 workspace
 *   B) MrgSort 归并: 1 核 4 路 MrgSort<float,true> -> sorted_expert_ids + gather_idx
 *   C) 核间分工     : GATHER 核 gather(y + 打包 id 索引 gather) / GROUP 核 group_listing(逐专家计数)
 */

#ifndef FFN_WB_PROBE_TILING_H
#define FFN_WB_PROBE_TILING_H

#include <cstdint>

// ---- Sort / MrgSort 相关常量 (对齐 ffn_wb_common.h / moe_v3_common.h) ----
constexpr int64_t ONE_REPEAT_SORT_NUM = 32;   // 每个 Sort repeat 处理 32 个元素
constexpr int64_t MAX_MRGSORT_LIST = 4;       // MrgSort 最多 4 路 (禁用 MrgSort4 API)
constexpr int64_t MERGE_LIST_ONE = 1;
constexpr int64_t MERGE_LIST_TWO = 2;
constexpr int64_t MERGE_LIST_THREE = 3;
constexpr int64_t MERGE_LIST_FOUR = 4;
constexpr int64_t MERGE_LIST_IDX_TWO = 2;
constexpr int64_t MERGE_LIST_IDX_THREE = 3;
constexpr int64_t MGR_SORT_MAX_ELEMENT = 1024; // MrgSort 单 loop 每路最大元素 (对齐 ffn_wb_common.h)

// ---- 打包 id：每行 8 个 int32 (32B 对齐)，col0=session col1=micro col2=token，其余 pad ----
constexpr int64_t ID_PACK_WIDTH = 8;
constexpr int64_t ID_SESSION_COL = 0;
constexpr int64_t ID_MICRO_COL = 1;
constexpr int64_t ID_TOKEN_COL = 2;

// 穿刺工程 Tiling (host 侧运行时查询 UB 后派生填充)
struct ProbeTilingData {
    int32_t totalLength;           // Y = SORT_CORE_NUM * perCoreElements (全部 token 数)
    int32_t perCoreElements;       // 每个 sort 核处理的 tile = sortLoopMaxElement (打到 UB 上界)
    int32_t sortCoreNum;           // 参与 sort 的核数 (= 4，触发多核 + 4 路归并)
    int32_t expertNum;             // 专家数
    int32_t hiddenSize;            // H, 每 token hidden states 元素数 (fp16)
    int32_t oneLoopMaxElementsMrg; // MrgSort 单 loop 每路最大元素
    int32_t gatherCoreNum;         // gather 核数
    int32_t groupCoreNum;          // group_listing 核数
    int32_t sortNum;               // ceil(perCoreElements,32)*32
    // ---- issue #01 UB 记录字段 (运行时查询，禁写死 arch 常量) ----
    int64_t ubKernel;              // ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE (kernel 侧真实可用)
    int64_t ubHost;                // GetCoreMemSize(UB) (host 侧平台接口)
    int64_t sortLoopMaxElement;    // = ubKernel/32 (由 kernel 侧真实 UB 派生，保守口径)
};

#endif // FFN_WB_PROBE_TILING_H
