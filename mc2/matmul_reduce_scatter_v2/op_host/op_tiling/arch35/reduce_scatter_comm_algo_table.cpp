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

constexpr uint32_t MIN_LAYERS = 1;
constexpr uint32_t MAX_LAYERS_UBX = 1;
constexpr uint32_t MAX_LAYERS_SERVER = 2;
constexpr uint32_t MIN_DATA_BYTES = 0;
constexpr uint64_t MAX_DATA_BYTES = ~0ULL;
constexpr uint32_t MIN_RANK_SIZE = 2;
constexpr uint32_t MAX_RANK_SIZE = ~0U;

enum class CommAlgoPriority : uint32_t {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3,
};

static constexpr Mc2Hcom::CommAlgoEntry REDUCE_SCATTER_COMM_ALGO_TABLE[] = {
    // AICPU通信算法
    // Server/pod → AicpuReduceScatterSoleMesh
    {mc2tiling::A5_AICPU_TS_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES,
     MAX_DATA_BYTES, MIN_RANK_SIZE, MAX_RANK_SIZE, CommAlgoPriority::MEDIUM,
     "AicpuReduceScatterSoleMesh"},
    // UBX/8p fullmesh → AicpuReduceScatterSoleMesh
    {mc2tiling::A5_AICPU_TS_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES,
     MAX_DATA_BYTES, MIN_RANK_SIZE, MAX_RANK_SIZE, CommAlgoPriority::MEDIUM,
     "AicpuReduceScatterSoleNHR"},
    // UBX/4p fullmesh → AicpuReduceScatterConcurMeshNHR
    {mc2tiling::A5_AICPU_TS_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES,
     MAX_DATA_BYTES, MIN_RANK_SIZE, MAX_RANK_SIZE, CommAlgoPriority::LOW,
     "AicpuReduceScatterSoleMesh"},
    // CCU通信算法
    // Server/pod → CcuSchedReduceScatterSoleMesh
    {mc2tiling::A5_CCU_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_SERVER, MIN_DATA_BYTES, MAX_DATA_BYTES,
     MIN_RANK_SIZE, MAX_RANK_SIZE, CommAlgoPriority::MEDIUM,
     "CcuSchedReduceScatterSoleMesh"},
    // UBX/8p fullmesh → CcuSchedReduceScatterConcurMeshNHRMultiLink
    {mc2tiling::A5_CCU_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES, MAX_DATA_BYTES,
     MIN_RANK_SIZE, MAX_RANK_SIZE, CommAlgoPriority::HIGH,
     "CcuSchedReduceScatterSoleNHRMultiLink"},
    // UBX/4p fullmesh → CcuSchedReduceScatterConcurMeshNHRMultiLink
    {mc2tiling::A5_CCU_ENGINE, Mc2Hcom::WILDCARD_TOPO_TYPE, MIN_LAYERS, MAX_LAYERS_UBX, MIN_DATA_BYTES, MAX_DATA_BYTES,
     MIN_RANK_SIZE, MAX_RANK_SIZE, CommAlgoPriority::HIGH,
     "CcuSchedReduceScatterConcurMeshNHRMultiLink"},
};

const Mc2Hcom::CommAlgoEntry *GetReduceScatterCommAlgoTable(uint32_t &count)
{
    count = sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE) / sizeof(REDUCE_SCATTER_COMM_ALGO_TABLE[0]);
    return REDUCE_SCATTER_COMM_ALGO_TABLE;
}

} // namespace Mc2Tiling