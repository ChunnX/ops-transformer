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
inline uint32_t ReductionRows(uint32_t projectionRows)
{
    return projectionRows < 48U ? projectionRows : 48U;
}

/** Round each live UB allocation to 256B: DMA alignment plus one full FP32 register for small arrays. */
inline uint64_t UbAllocation(uint64_t bytes)
{
    return (bytes + 255U) / 256U * 256U;
}

/** Count the live full-row producer buffers, including double input and coefficient slots.
 * Mid aliases the residual/output row and adds gamma2; wave production additionally owns epoch storage.
 */
inline uint64_t ProducerUbBytes(uint32_t d, bool hasMid, bool wavePipeline = false)
{
    const uint64_t k = 4ULL * d;
    return UbAllocation(4ULL * d) + UbAllocation(4ULL * k) + UbAllocation(2ULL * d) + UbAllocation(4ULL * d) +
           UbAllocation(256U * 4U) + UbAllocation(8U) +
           (hasMid ? UbAllocation(4ULL * k) : (wavePipeline ? UbAllocation(32U) : UbAllocation(2ULL * k)));
}

/** Materialized-row projection owns row statistics plus packed input and FP32 transfer half-tiles. */
inline uint64_t ProjectionUbBytes(uint32_t k, uint32_t kL1, uint32_t projectionRows)
{
    const uint64_t stageElements = static_cast<uint64_t>(projectionRows) * kL1;
    const uint64_t tileElements = k > stageElements ? k : stageElements;
    return UbAllocation(projectionRows * 4U) + UbAllocation(stageElements * 4U) + UbAllocation(tileElements * 2U) +
           UbAllocation(stageElements * 2U);
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
} // namespace optiling::mega_mhc
#endif
