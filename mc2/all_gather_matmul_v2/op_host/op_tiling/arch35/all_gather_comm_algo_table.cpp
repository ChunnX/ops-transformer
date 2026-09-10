/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_comm_algo_table.h"
// 启用下方算法条目（使用 COMM_TOPO_CUSTOM 等拓扑枚举）时取消注释: #include "hccl/hccl_rank_graph.h"

namespace Mc2Tiling {

// ======================= 算法条目（当前清空，未生效） =======================
// 空表语义：entryCount==0 时选择器直接返回默认算法 ALLGATHER_DEFAULT_ALGO_NAME
// （"AllGather=level0:fullmesh"），与现网行为一致。
// 恢复算法选择时取消注释并按需增删条目。
// constexpr uint8_t ENGINE_AICPU = 2;      // 与 Mc2Comm::ENGINE_AICPU 一致
// constexpr uint8_t ENGINE_CCU_SCHED = 6;  // 与 Mc2Comm::ENGINE_CCU_SCHED 一致
// constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
// constexpr uint32_t MAX_RANK_SIZE = ~0U;
//
// 字段顺序: commEngine, topoType, minLayers, maxLayers, minBytes, maxBytes,
//           minRankSize, maxRankSize, priority, algoName
// 过滤顺序: commEngine -> rankSize -> layers -> topoType -> commDataBytes，全被滤掉时回退默认算法；
// 存活候选中取 priority 最高者，同优先级取靠前条目。
// static constexpr Mc2Hcom::CommAlgoEntry ALLGATHER_COMM_ALGO_TABLE[] = {
//     // AICPU + CUSTOM + 单层 + 卡数[2,MAX] → AicpuAllGatherSoleMesh
//     {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 2, "AicpuAllGatherSoleMesh"},
//     // AICPU + CUSTOM + 单层 + 卡数[2,MAX] → AicpuAllGatherSoleNHR
//     {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 1, "AicpuAllGatherSoleNHR"},
//     // CCU_SCHED + CUSTOM + 单层 + 卡数[2,MAX] → CcuSchedAllGatherSoleMesh
//     {ENGINE_CCU_SCHED, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 1,
//     "CcuSchedAllGatherSoleMesh"},
//     // AICPU + CUSTOM + 单层 + 卡数[4,4] → AicpuAllGatherConcurMeshNHR
//     {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 1, 0, MAX_DATA_BYTES, 4, 4, 2, "AicpuAllGatherConcurMeshNHR"},
//     // AICPU + CUSTOM + 两层 + 卡数[2,8] → AicpuAllGatherParallelMeshNHRPcie
//     {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, 8, 1, "AicpuAllGatherParallelMeshNHRPcie"},
//     // AICPU + CUSTOM + 两层 + 卡数[2,8] → AicpuAllGatherPipeLinePcie
//     {ENGINE_AICPU, COMM_TOPO_CUSTOM, 1, 2, 0, MAX_DATA_BYTES, 2, 8, 1, "AicpuAllGatherPipeLinePcie"},
// };
// ==========================================================================

const Mc2Hcom::CommAlgoEntry *GetAllGatherCommAlgoTable(uint32_t &count)
{
    // 当前为空表：entryCount==0 时选择器直接返回默认算法 ALLGATHER_DEFAULT_ALGO_NAME，与现网行为一致。
    // 恢复算法选择时取消上方注释，并将本函数改为：
    //   count = sizeof(ALLGATHER_COMM_ALGO_TABLE) / sizeof(ALLGATHER_COMM_ALGO_TABLE[0]);
    //   return ALLGATHER_COMM_ALGO_TABLE;
    count = 0;
    return nullptr;
}
} // namespace Mc2Tiling
