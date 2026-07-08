/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file ffn_wb_probe_tiling.h
 * \brief 迭代一 穿刺2 (probe2_norm_multicore) 的 Tiling 常量 + 结构体
 *        (纯 C/C++，kernel 与 host 共用)
 *
 * 穿刺目标: 多核 sort -> MrgSort 4 路归并 + 核间 gather/group_listing 分工
 * 目标平台: Ascend950DT_9584 (arch35 / DAV_3510)
 *
 * 数据流 (3 次 kernel 启动，用 aclrtSynchronizeStream 做栅栏，替代脆弱的核间 SyncAll):
 *   A) 多核 sort   : SORT_CORE_NUM 个核，各 Sort<float,true> 一个 tile -> proposal 落 workspace
 *   B) MrgSort 归并: 1 个核 4 路 MrgSort<float,true> -> sorted_expert_ids + gather_idx
 *   C) 核间分工     : GATHER 核 gather(y + id 索引 gather) / GROUP 核 group_listing(逐专家计数)
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

// 穿刺工程 Tiling (固定 shape，host 侧填充)
struct ProbeTilingData {
    int32_t totalLength;           // Y = 全部 token 数
    int32_t perCoreElements;       // 每个 sort 核处理的 tile 大小 (必须 32 对齐)
    int32_t sortCoreNum;           // 参与 sort 的核数 (= 4，触发多核 + 4 路归并)
    int32_t expertNum;             // 专家数
    int32_t hiddenSize;            // H, 每 token hidden states 元素数 (fp16)
    int32_t oneLoopMaxElementsMrg; // MrgSort 单 loop 每路最大元素
    int32_t gatherCoreNum;         // gather 核数
    int32_t groupCoreNum;          // group_listing 核数
    int32_t sortNum;               // ceil(perCoreElements,32)*32
};

#endif // FFN_WB_PROBE_TILING_H
