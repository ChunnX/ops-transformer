/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "allto_all_comm_algo_table.h"
#include "mc2_log_compat.h"
#include "mc2_tiling_utils.h"

namespace Mc2Tiling {

using Mc2Hcom::TOPO_FLAG_NONE;
using Mc2Hcom::TOPO_FLAG_SINGLE_LEVEL;
using Mc2Hcom::TOPO_FLAG_L0_MESH_1D_CLOS;

constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

// 选择器为硬过滤（任一维度清空候选集即无匹配，回退默认算法）；
// 各引擎配一条全区间低优先级兜底表项，承接非Mesh+CLOS混联形态/多层组网/未覆盖卡数等场景回退默认算法；
// 具体算法表项仅Mesh+CLOS混联形态(MESH_1D_CLOS)支持，通过TOPO_FLAG_L0_MESH_1D_CLOS|TOPO_FLAG_SINGLE_LEVEL(单层Mesh+CLOS混联)约束
static constexpr Mc2Hcom::CommAlgoEntry ALLTOALL_COMM_ALGO_TABLE[] = {
    // AICPU + 单层Mesh+CLOS混联 + 卡数[8,16] → sole[mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 8, 16, 1,
     "sole[mesh]"},
    // AICPU + 单层Mesh+CLOS混联 + 卡数[2,4] → concur[mesh,mesh]
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 2, 4, 1,
     "concur[mesh,mesh]"},
    // CCU + 单层Mesh+CLOS混联 + 卡数[2,4] → concur[mesh,mesh]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 2, 4, 1,
     "concur[mesh,mesh]"},
    // CCU + 单层Mesh+CLOS混联 + 卡数[8,16] → sole[mesh.multi_channel]
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_SINGLE_LEVEL | TOPO_FLAG_L0_MESH_1D_CLOS, 0, MAX_DATA_BYTES, 8, 16, 1,
     "sole[mesh.multi_channel]"},
    // AICPU 兜底：非Mesh+CLOS混联形态/未覆盖场景低优先级回退默认算法
    {mc2tiling::A5_AICPU_TS_ENGINE, TOPO_FLAG_NONE, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 0, ALLTOALL_DEFAULT_ALGO_NAME},
    // CCU 兜底：非Mesh+CLOS混联形态/未覆盖场景低优先级回退默认算法
    {mc2tiling::A5_CCU_ENGINE, TOPO_FLAG_NONE, 0, MAX_DATA_BYTES, 2, MAX_RANK_SIZE, 0, ALLTOALL_DEFAULT_ALGO_NAME},
};

const Mc2Hcom::CommAlgoEntry* GetAllToAllCommAlgoTable(uint32_t& count)
{
    count = sizeof(ALLTOALL_COMM_ALGO_TABLE) / sizeof(ALLTOALL_COMM_ALGO_TABLE[0]);
    return ALLTOALL_COMM_ALGO_TABLE;
}

std::string SelectAllToAllAlgoName(const std::string& opName, const std::string& group, uint8_t commEngine,
                                   uint32_t tileM, uint64_t dimSize, uint64_t dtypeSize, uint32_t rankDim)
{
    uint64_t commDataBytes = Mc2Hcom::CalcCommDataBytes(opName.c_str(), tileM, dimSize, dtypeSize);
    OP_LOGI(opName.c_str(), "[SetHcclTiling] commDataBytes=%llu, tileM=%u, dimSize=%llu, rankDim=%u, engine=%u",
            commDataBytes, tileM, dimSize, rankDim, static_cast<uint32_t>(commEngine));
    uint32_t algoCount = 0;
    const Mc2Hcom::CommAlgoEntry* algoEntries = GetAllToAllCommAlgoTable(algoCount);
    std::string algoName = Mc2Hcom::Mc2CommAlgoSelector::SelectAlgoName(
        opName, group.c_str(), commEngine, commDataBytes, rankDim, algoEntries, algoCount, ALLTOALL_DEFAULT_ALGO_NAME);
    OP_LOGI(opName.c_str(), "[SetHcclTiling] selected algoName=%s, group=%s", algoName.c_str(), group.c_str());
    return algoName;
}

} // namespace Mc2Tiling
