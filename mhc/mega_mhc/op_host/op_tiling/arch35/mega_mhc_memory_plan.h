/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEGA_MHC_MEMORY_PLAN_H
#define MEGA_MHC_MEMORY_PLAN_H
#include <cstdint>
#include "../../../op_kernel/arch35/execution_policy.h"

namespace optiling::mega_mhc {
/** Bound the separate Pre/RMS1 epilogue batch to leave room for complete rows and coefficients. */
inline uint32_t ReductionRows(uint32_t rowLimit)
{
    return rowLimit < 48U ? rowLimit : 48U;
}

/** Round each live UB allocation to 256B: DMA alignment plus one full FP32 register for small arrays. */
inline uint64_t UbAllocation(uint64_t bytes)
{
    return (bytes + 255U) / 256U * 256U;
}

/** Count the live full-row producer buffers, including double input and coefficient slots.
 * Mid aliases the residual/output row and adds gamma2; wave production additionally owns epoch storage.
 */
inline uint64_t ProducerUbBytes(uint32_t d, bool hasMid, bool wavePipeline = false, uint32_t gammaElements = 0U)
{
    const uint64_t k = 4ULL * d;
    return UbAllocation(4ULL * d) + UbAllocation(4ULL * k) + UbAllocation(2ULL * d) + UbAllocation(4ULL * d) +
           UbAllocation(256U * 4U) + UbAllocation(8U) +
           (hasMid ? UbAllocation(4ULL * (gammaElements == 0U ? k : gammaElements)) :
                     (wavePipeline ? UbAllocation(32U) : UbAllocation(2ULL * k)));
}

/** Materialized-row projection owns row statistics plus packed input and FP32 transfer half-tiles. */
inline uint64_t ProjectionUbBytes(uint32_t /* k */, uint32_t kL1, uint32_t projectionRows, uint32_t stageSlots = 1U,
                                  uint32_t stageRowStride = 0U)
{
    const uint64_t stageElements = static_cast<uint64_t>(projectionRows) * kL1;
    const uint64_t transferElements =
        static_cast<uint64_t>(stageRowStride == 0U ? projectionRows : stageRowStride) * kL1;
    return UbAllocation(projectionRows * 4U) + UbAllocation(transferElements * 4U * stageSlots) +
           UbAllocation(stageElements * 2U) + UbAllocation(stageElements * 2U);
}

/** Choose the fewest gamma loads whose aligned, equal-sized chunks fit: G=align64(ceil(4D/count)). */
inline uint32_t MidGammaTile(uint32_t d, uint64_t availableUb)
{
    const uint32_t k = 4U * d;
    for (uint32_t count = 1U; count <= k / 64U; ++count) {
        const uint32_t tile = ((k + count - 1U) / count + 63U) / 64U * 64U;
        if (ProducerUbBytes(d, true, false, tile) <= availableUb) {
            return tile;
        }
    }
    return 0U;
}

/** Epilogue owns one x2/hIn row plus a host-sized batch of projection partials. */
inline uint64_t EpilogueUbBytes(uint32_t d, uint32_t partialGroups, uint32_t partialStride, uint32_t reduceRows = 1U,
                                uint32_t kGrid = 1U)
{
    return UbAllocation(4ULL * d * 2U) + 2U * UbAllocation(d * 2U) + UbAllocation(d * 4U) + 3U * 256U +
           UbAllocation(static_cast<uint64_t>((reduceRows + 7U) / 8U * 8U) * 24U * 4U) +
           UbAllocation(static_cast<uint64_t>((reduceRows + 7U) / 8U * 8U) * kGrid * 4U) +
           UbAllocation((static_cast<uint64_t>(partialGroups) * reduceRows * partialStride + 64U) * 4U);
}
/** Select the largest eight-row batch fitting the exact epilogue live allocation; small M keeps its tail. */
inline uint32_t FitReductionRows(uint32_t rowLimit, uint32_t d, uint32_t groups, uint32_t stride, uint32_t kGrid,
                                 uint64_t availableUb)
{
    uint32_t rows = ReductionRows(rowLimit);
    while (rows > 0U && EpilogueUbBytes(d, groups, stride, rows, kGrid) > availableUb) {
        rows = rows > 8U ? (rows - 1U) / 8U * 8U : rows - 1U;
    }
    return rows;
}

/** Balance R per-core rows over B=ceil(R/capacity) batches without increasing the capacity bound.
 * Batch rows=min(capacity,align8(ceil(R/B))); a zero capacity or workload has no valid batch.
 */
inline uint32_t BalanceReductionRows(uint32_t rowsPerCore, uint32_t capacityRows)
{
    if (capacityRows == 0U || rowsPerCore == 0U) {
        return 0U;
    }
    const uint64_t batches = (static_cast<uint64_t>(rowsPerCore) + capacityRows - 1U) / capacityRows;
    const uint32_t balancedRows = static_cast<uint32_t>(((rowsPerCore + batches - 1U) / batches + 7U) / 8U * 8U);
    return balancedRows < capacityRows ? balancedRows : capacityRows;
}

} // namespace optiling::mega_mhc
#endif
