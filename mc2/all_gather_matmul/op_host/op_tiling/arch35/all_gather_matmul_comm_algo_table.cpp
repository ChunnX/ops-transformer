/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_matmul_comm_algo_table.h"

namespace Mc2Tiling {
namespace {
// ======================= 示例模板（当前未生效） =======================
// 正式算法名与适用条件由维护者确认后，取消注释并按需增删条目。
// 字段顺序: commEngine, topoType, minLayers, maxLayers, minBytes, maxBytes,
//           minRankSize, maxRankSize, priority, algoName
// 过滤顺序: commEngine -> rankSize -> layers -> topoType -> commDataBytes，全被滤掉时回退默认算法。
// 使用 COMM_TOPO_CUSTOM 等拓扑枚举时需取消注释: #include "hccl/hccl_rank_graph.h"
//
// constexpr uint8_t ENGINE_AICPU = 2;      // 与 Mc2Comm::ENGINE_AICPU 一致
// constexpr uint8_t ENGINE_CCU_SCHED = 6;  // 与 Mc2Comm::ENGINE_CCU_SCHED 一致
// constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
// constexpr uint32_t MAX_RANK_SIZE = ~0U;
//
// static constexpr Mc2Hcom::CommAlgoEntry ALLGATHER_COMM_ALGO_TABLE[] = {
//     // engine            topoType          minLayers maxLayers minBytes maxBytes      minRank maxRank  prio algoName
//     {ENGINE_AICPU,     COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 2, "AicpuAllGatherSoleMeshUBX"},
//     {ENGINE_AICPU,     COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 1, "AicpuAllGatherSoleMesh"},
//     {ENGINE_CCU_SCHED, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 3,
//     "CcuSchedAllGatherSoleMeshConcurrent"},
// };
// ====================================================================
} // namespace

const Mc2Hcom::CommAlgoEntry *GetAllGatherMatmulCommAlgoTable(uint32_t &count)
{
    // 当前为空表：entryCount==0 时选择器直接返回默认算法 ALLGATHER_MM_DEFAULT_ALGO_NAME，与现网行为一致。
    // 正式启用时取消上方示例注释，并将本函数改为：
    //   count = sizeof(ALLGATHER_COMM_ALGO_TABLE) / sizeof(ALLGATHER_COMM_ALGO_TABLE[0]);
    //   return ALLGATHER_COMM_ALGO_TABLE;
    count = 0;
    return nullptr;
}
} // namespace Mc2Tiling
