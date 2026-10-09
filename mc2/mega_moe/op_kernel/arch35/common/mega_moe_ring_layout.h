/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_RING_LAYOUT_H
#define MEGA_MOE_RING_LAYOUT_H

#include "mega_moe_peermem.h"

namespace MegaMoeImpl {

// Each routed stage owns a separate ring. Three wave budgets preserve dispatch lookahead and GMM2 lag.
constexpr uint32_t MTE_RING_WAVE_COUNT = 3U;

HOST_DEVICE bool IsMteRingBufferEnabled(const MegaMoeTilingData *tilingData)
{
    return tilingData != nullptr && tilingData->topoType == TOPO_TYPE_MTE && tilingData->dedupMode == 0 &&
           tilingData->combineQuantMode == COMBINE_NO_QUANT && tilingData->mGroupsPerWave > 0U;
}

HOST_DEVICE int64_t CalcMteRingRowCapacity(const MegaMoeTilingData *tilingData)
{
    if (tilingData == nullptr) {
        return 0;
    }
    if (!IsMteRingBufferEnabled(tilingData)) {
        return static_cast<int64_t>(tilingData->maxOutputSize);
    }
    return static_cast<int64_t>(MTE_RING_WAVE_COUNT) * tilingData->mGroupsPerWave * L1_TILE_M_256;
}

HOST_DEVICE uint64_t CalcMteRingPhysicalRow(uint64_t globalPaddedRow, uint64_t ringRows)
{
    return ringRows == 0U ? globalPaddedRow : globalPaddedRow % ringRows;
}

// Even a partial expert/group owns all 256 physical rows; logical token rows remain packed in metadata.
HOST_DEVICE uint64_t CalcMtePaddedRowCount(uint64_t validRows)
{
    return Ops::Base::CeilAlign(validRows, static_cast<uint64_t>(L1_TILE_M_256));
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_RING_LAYOUT_H
