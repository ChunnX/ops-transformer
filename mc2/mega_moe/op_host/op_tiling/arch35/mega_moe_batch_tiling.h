/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_BATCH_TILING_H
#define MEGA_MOE_BATCH_TILING_H

#include <algorithm>
#include <cstdint>

#include "../../../op_kernel/arch35/mega_moe_tiling.h"

namespace optiling {
namespace MegaMoeBatchTiling {
constexpr uint32_t GMM1_TILE_M = 256U;
constexpr uint32_t GMM2_GM_TILE_M = 256U;
constexpr uint32_t GMM_TILE_N = 256U;

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + static_cast<uint64_t>(value % divisor != 0U);
}

inline uint32_t CalcExpertsPerBatch(uint32_t bs, uint32_t topK, uint32_t expertPerRank, uint32_t hiddenDim,
                                    uint32_t h, uint32_t aicNum)
{
    if (expertPerRank == 0U || aicNum == 0U) {
        return 1U;
    }

    uint64_t expectedTokens = CeilDiv(static_cast<uint64_t>(bs) * static_cast<uint64_t>(topK), expertPerRank);
    uint64_t expectedMWaves = std::max<uint64_t>(1U, CeilDiv(expectedTokens, GMM1_TILE_M));
    // One GMM1 scheduler N step covers gate+up, hence 2 * GMM_TILE_N output columns.
    uint64_t gmm1TilesPerExpert = expectedMWaves * CeilDiv(hiddenDim, GMM_TILE_N * 2U);
    uint64_t gmm2TilesPerExpert = CeilDiv(expectedTokens, GMM2_GM_TILE_M) * CeilDiv(h, GMM_TILE_N);
    uint64_t limitingTilesPerExpert = std::max<uint64_t>(
        1U, std::min(gmm1TilesPerExpert, gmm2TilesPerExpert));
    uint64_t expertsToFillAic = CeilDiv(aicNum, limitingTilesPerExpert);
    return static_cast<uint32_t>(std::max<uint64_t>(
        1U, std::min<uint64_t>(MEGA_MOE_MAX_EXPERTS_PER_BATCH,
                               std::min<uint64_t>(expertPerRank, expertsToFillAic))));
}
} // namespace MegaMoeBatchTiling
} // namespace optiling

#endif
