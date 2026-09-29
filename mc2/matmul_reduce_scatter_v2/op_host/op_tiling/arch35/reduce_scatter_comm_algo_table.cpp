/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_comm_algo_table.h"
#include "mc2_tiling_utils.h"

namespace Mc2Tiling {

using Mc2Hcom::TOPO_FLAG_NONE;
using Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL;
using Mc2Hcom::TOPO_FLAG_UBX;
// 通信数据量
constexpr uint32_t MIN_DATA_BYTES_ZERO = 0ULL;
constexpr uint32_t DATA_BYTES_1M = 1ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_4M = 4ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_8M = 8ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_16M = 16ULL * 1024ULL * 1024ULL;
constexpr uint32_t DATA_BYTES_256M = 256ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_DATA_BYTES_UNLIMITED = ~0ULL;
// 卡数
constexpr uint32_t MIN_RANK_SIZE = 0;
constexpr uint32_t RANK_SIZE_2 = 2;
constexpr uint32_t RANK_SIZE_4 = 4;
constexpr uint32_t RANK_SIZE_8 = 8;
constexpr uint32_t RANK_SIZE_16 = 16;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

enum CommAlgoPriority : int32_t {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
};

// ReduceScatter 通信算法表；选择器为硬过滤（任一维度清空候选集即无匹配，回退调用方传入的默认算法）。
// 通信算法统一由本表管理(组网/卡数/数据量/优先级)，调用方直接使用选出的算法名：
// - UBX具体表项(mesh/nhr类规格串)通过TOPO_FLAG_UBX|TOPO_FLAG_SINGLE_LEVEL(单pod UBX)约束，
//   置于同优先级非UBX表项之前(同级优先级按表序取先)，UBX机型优先命中；
// - 16p非UBX(跨机)为通信api benchmark优选算法，不限组网；
// - CCU的4p/8p走All2All+VecReduce通路(AllToAll通信)，兜底为AlltoAll算法；
//   其余卡数(AICPU全部/CCU 2p与16p及以上)走ReduceScatter通路，兜底为ReduceScatter算法
static constexpr Mc2Hcom::CommAlgoEntry REDUCE_SCATTER_COMM_ALGO_TABLE[] = {
    // AICPU通信算法(ReduceScatter通路)
    // 兜底算法:ReduceScatter=level0:fullmesh
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE,
     MAX_RANK_SIZE, LOW, "ReduceScatter=level0:fullmesh"},
    // 单pod UBX + 4p/1M数据量 → sole[mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_1M,
     MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "sole[mesh]"},
    // 单pod UBX + 4p/1M-1024M数据量 → concur[mesh, nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, DATA_BYTES_1M, MAX_DATA_BYTES_UNLIMITED,
     MIN_RANK_SIZE, RANK_SIZE_4, MEDIUM, "concur[mesh,nhr]"},
    // 单pod UBX + 8p/4M数据量 → sole[nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_4M,
     RANK_SIZE_4, RANK_SIZE_8, MEDIUM, "sole[nhr]"},
    // 单pod UBX + 8p/4M-1024M数据量 → parallel[mesh, nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, DATA_BYTES_4M, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_4, RANK_SIZE_8, MEDIUM, "parallel[mesh,nhr]"},
    // 单pod UBX + 16p/8M数据量 → sole[nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_8M,
     RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "sole[nhr]"},
    // 单pod UBX + 16p/8M-1024M数据量 → parallel[mesh, nhr]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, DATA_BYTES_8M, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_8, MAX_RANK_SIZE, MEDIUM, "parallel[mesh,nhr]"},
    // 16p非UBX(跨机)：通信api benchmark优选(置于UBX表项之后，UBX机型优先命中上方具体表项)
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_16,
     RANK_SIZE_16, MEDIUM, "InsReduceScatterParallelMesh1DNHR"},

    // CCU通信算法
    // 2p兜底算法:ReduceScatter=level0:fullmesh
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, MIN_RANK_SIZE,
     RANK_SIZE_2, LOW, "ReduceScatter=level0:fullmesh"},
    // 4p/8p走All2All+VecReduce通路，兜底算法:AlltoAll=level0:fullmesh
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_4, RANK_SIZE_4,
     LOW, "AlltoAll=level0:fullmesh"},
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_8, RANK_SIZE_8,
     LOW, "AlltoAll=level0:fullmesh"},
    // 16p及以上ReduceScatter通路兜底算法:ReduceScatter=level0:fullmesh
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED, RANK_SIZE_16,
     MAX_RANK_SIZE, LOW, "ReduceScatter=level0:fullmesh"},
    // 单pod UBX + 4p/4M数据量 → sole[mesh.muti_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, MIN_DATA_BYTES_ZERO, DATA_BYTES_4M, RANK_SIZE_2,
     RANK_SIZE_4, MEDIUM, "sole[mesh.multi_channel]"},
    // 单pod UBX + 4p/4M-1024M数据量 → concur[mesh, mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, DATA_BYTES_4M, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_2, RANK_SIZE_4, MEDIUM, "concur[mesh,mesh]"},
    // 单pod UBX + 8p/16p → sole[mesh.muti_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_UBX, MIN_DATA_BYTES_ZERO, MAX_DATA_BYTES_UNLIMITED,
     RANK_SIZE_4, MAX_RANK_SIZE, MEDIUM, "sole[mesh.multi_channel]"},
};

const Mc2Hcom::CommAlgoEntry* GetReduceScatterCommAlgoTable(uint32_t& count)
{
    count = sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE) / sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE[0]);
    return REDUCE_SCATTER_COMM_ALGO_TABLE;
}

} // namespace Mc2Tiling
