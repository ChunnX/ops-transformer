/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef MEGA_MHC_MEMORY_PLAN_H
#define MEGA_MHC_MEMORY_PLAN_H
#include <cstdint>
#include "../../../op_kernel/arch35/execution_policy.h"

namespace optiling::mega_mhc {
/** Bound partial rows by their stage-local storage: four-op aliases the projection tile;
 * five-op caps the independent batch at 48 rows to preserve room for x2/gamma/coefficients at D=5120.
 */
inline uint32_t ReductionRows(uint32_t projectionRows, uint32_t kL1, uint32_t groups,
                              uint32_t stride, bool hasMid)
{
    if (hasMid) {
        return projectionRows < 48U ? projectionRows : 48U;
    }
    const uint64_t capacity = static_cast<uint64_t>(projectionRows) * kL1 /
                              (static_cast<uint64_t>(groups) * stride);
    return capacity < projectionRows ? static_cast<uint32_t>(capacity) : projectionRows;
}

/** Round each live UB allocation to 256B: DMA alignment plus one full FP32 register for small arrays. */
inline uint64_t UbAllocation(uint64_t bytes)
{
    return (bytes + 255U) / 256U * 256U;
}

/** Sum all simultaneously live producer buffers, in the same order as the scheduler's allocations. */
inline uint64_t ProducerUbBytes(uint32_t rows, uint32_t d, uint32_t kL1, bool hasMid,
                                uint32_t projectionRows, bool prefetchInput = false)
{
    const uint64_t r = rows;
    const uint64_t k = 4ULL * d;
    const uint64_t projectionPrefix = hasMid ? 0U : UbAllocation(projectionRows * 4U) +
        UbAllocation(static_cast<uint64_t>(projectionRows) * kL1 * 4U) +
        UbAllocation(static_cast<uint64_t>(projectionRows) * kL1 * 2U) +
        (AiInfraMegaMhc::kOverlapFourEpilogue ? 2U * UbAllocation(128U) : 0U);
    const uint64_t producerPrefetch = prefetchInput ? UbAllocation(r * (d + k) * 2U) : 0U;
    return projectionPrefix + producerPrefetch + 2U * UbAllocation(r * d * 2U) + 2U * UbAllocation(r * k * 2U) +
        UbAllocation(d * 4U) + (hasMid ? UbAllocation(k * 4U) : 0U) +
        UbAllocation(!hasMid && AiInfraMegaMhc::kOverlapFourEpilogue ?
            static_cast<uint64_t>((projectionRows + 7U) / 8U * 8U) * 24U * 4U : r * 4U * 4U) +
        UbAllocation(!hasMid && AiInfraMegaMhc::kOverlapFourEpilogue ?
            ((projectionRows > r * 16U ? projectionRows : r * 16U) * 4U) : r * 16U * 4U) + UbAllocation(hasMid ? r * 8U : 8U);
}

/** Five-op projection owns row statistics plus packed input and FP32 transfer half-tiles. */
inline uint64_t ProjectionUbBytes(uint32_t k, uint32_t kL1, uint32_t projectionRows)
{
    const uint64_t stageElements = static_cast<uint64_t>(projectionRows) * kL1;
    const uint64_t tileElements = k > stageElements ? k : stageElements;
    return UbAllocation(projectionRows * 4U) + UbAllocation(stageElements * 4U) +
        UbAllocation(tileElements * 2U) + UbAllocation(stageElements * 2U);
}

/** Epilogue owns one x2/hIn row plus a host-sized batch of projection partials. */
inline uint64_t EpilogueUbBytes(uint32_t d, uint32_t partialGroups, uint32_t partialStride, uint32_t reduceRows = 1U, uint32_t kGrid = 1U)
{
    return UbAllocation(4ULL * d * 2U) + 2U * UbAllocation(d * 2U) + UbAllocation(d * 4U) +
        3U * 256U + UbAllocation(static_cast<uint64_t>((reduceRows + 7U) / 8U * 8U) * 24U * 4U) +
        UbAllocation(static_cast<uint64_t>((reduceRows + 7U) / 8U * 8U) * kGrid * 4U) + UbAllocation((static_cast<uint64_t>(partialGroups) * reduceRows * partialStride + 64U) * 4U);
}
} // namespace optiling::mega_mhc
#endif
