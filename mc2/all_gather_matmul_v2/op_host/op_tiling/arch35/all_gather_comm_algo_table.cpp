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
#include "mc2_tiling_utils.h"
#include "hccl/hccl_rank_graph.h"

namespace Mc2Tiling {

constexpr uint32_t MIN_LAYERS = 1;
constexpr uint32_t MAX_LAYERS_UBX = 1;
constexpr uint32_t MAX_LAYERS_SERVER = 2;
constexpr uint32_t MIN_DATA_BYTES_ZERO = 0;
// 数据量分档边界（半开区间，边界值归高一级档位）
constexpr uint64_t DATA_BYTES_2M = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t DATA_BYTES_4M = 4ULL * 1024ULL * 1024ULL;
constexpr uint64_t DATA_BYTES_8M = 8ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_DATA_BYTES_UNLIMITED = ~0ULL;
constexpr uint32_t RANK_SIZE_2 = 2;
constexpr uint32_t RANK_SIZE_4 = 4;
constexpr uint32_t RANK_SIZE_8 = 8;
constexpr uint32_t RANK_SIZE_16 = 16;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

enum CommAlgoPriority : int32_t {
    DEFAULT = 0,
    MEDIUM = 2,
};

// UBX 场景 AllGather 通信算法表（算法名为 HCCL 新版算法规格串）
// 未选中，回退默认算法"AllGather=level0:fullmesh"
static constexpr Mc2Hcom::CommAlgoEntry ALLGATHER_COMM_ALGO_TABLE[] = {
    // AICPU通信算法
    // rank4 小数据量 → sole[mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO,
     DATA_BYTES_2M - 1, RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "sole[mesh]"},
    // rank4 大数据量 → concur[mesh,nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, DATA_BYTES_2M,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "concur[mesh,nhr]"},
    // rank8~16 全数据量 → sole[nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO,
     MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, RANK_SIZE_16, MEDIUM, "sole[nhr]"},
    // CCU_SCHED通信算法
    // rank4 小数据量 → sole[mesh]
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_8M - 1,
     RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "sole[mesh]"},
    // rank4 大数据量 → concur[mesh,nhr.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, DATA_BYTES_8M, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_4, RANK_SIZE_4, MEDIUM, "concur[mesh,nhr.multi_channel]"},
    // rank8 小数据量 → sole[mesh]
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_4M - 1,
     RANK_SIZE_8, RANK_SIZE_8, MEDIUM, "sole[mesh]"},
    // rank8 大数据量 → parallel[mesh,nhr.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, DATA_BYTES_4M, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_8, RANK_SIZE_8, MEDIUM, "parallel[mesh,nhr.multi_channel]"},
    // rank16 小数据量 → sole[mesh]
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_8M - 1,
     RANK_SIZE_16, RANK_SIZE_16, MEDIUM, "sole[mesh]"},
    // rank16 大数据量 → parallel[mesh,nhr.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, COMM_TOPO_CUSTOM, MIN_LAYERS, MAX_LAYERS_UBX, DATA_BYTES_8M, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_16, RANK_SIZE_16, MEDIUM, "parallel[mesh,nhr.multi_channel]"},
};

const Mc2Hcom::CommAlgoEntry *GetAllGatherCommAlgoTable(uint32_t &count)
{
    count = sizeof(ALLGATHER_COMM_ALGO_TABLE) / sizeof(ALLGATHER_COMM_ALGO_TABLE[0]);
    return ALLGATHER_COMM_ALGO_TABLE;
}
} // namespace Mc2Tiling
