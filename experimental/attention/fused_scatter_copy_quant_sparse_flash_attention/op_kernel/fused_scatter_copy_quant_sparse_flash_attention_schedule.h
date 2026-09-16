/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_scatter_copy_quant_sparse_flash_attention_schedule.h
 * \brief
 */

#ifndef FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_SCHEDULE_H
#define FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_SCHEDULE_H
#include <cstdint>
#ifdef __CCE_AICORE__
#define FSCQSFA_SCHEDULE_INLINE __aicore__ inline
#else
#define FSCQSFA_SCHEDULE_INLINE constexpr
#endif
namespace fused_scatter_copy_qsfa {
constexpr int32_t UNION_COPY_THRESHOLD = 512;
// Request-level union count, not per-query misses or the sum over the batch.
FSCQSFA_SCHEDULE_INLINE bool NeedsUnionCopy(int32_t unionCount)
{
    return unionCount > UNION_COPY_THRESHOLD;
}
struct MissWritebackPlan {
    uint16_t blockCount;
    uint32_t dstStride;
};
// Two adjacent UB rows are 672B apart, but each packed KV payload is only 656B.
// UB->GM DataCopyPad rounds its UB block step up to 32B (srcStride=0), while
// dstStride is the byte gap after the 656B payload. Never reorder the UB rows.
FSCQSFA_SCHEDULE_INLINE MissWritebackPlan PlanMissWritebackPair(int64_t destination, int64_t nextDestination)
{
    if (destination < 0 || nextDestination <= destination) return {1, 0};
    const uint64_t distance = static_cast<uint64_t>(nextDestination) - static_cast<uint64_t>(destination);
    // Keep a uint32-representable gap for both existing and newer CANN headers.
    // Larger physical addresses themselves remain int64; only the gap is bounded.
    if (distance < 656 || distance - 656 > UINT32_MAX) return {1, 0};
    return {2, static_cast<uint32_t>(distance - 656)};
} // namespace fused_scatter_copy_qsfa
FSCQSFA_SCHEDULE_INLINE uint32_t GatherIndex(uint32_t batch, uint32_t index)
{
    if (index >= 2048) return index;
    return ((3 - batch % 3 + 4 - index / 512) % 4) * 512 + index % 512;
}
struct GatherBlockPlan {
    uint32_t indexBase;
    bool hasMiss;
};
struct OwnerScanPlan {
    uint32_t begin;
    bool reset;
};
// Reuse earlier-query ownership only within the same, forward-moving request.
FSCQSFA_SCHEDULE_INLINE OwnerScanPlan PlanOwnerScan(int32_t previousBatch, int32_t previousQuery,
    uint32_t batch, uint32_t queryBegin, uint32_t query)
{
    const bool reset = previousBatch != static_cast<int32_t>(batch) ||
        previousQuery < static_cast<int32_t>(queryBegin) || previousQuery > static_cast<int32_t>(query);
    return {reset ? queryBegin : static_cast<uint32_t>(previousQuery), reset};
}
// A 128-token S2 tile never crosses a 512-token permutation boundary.
// Its AIV slice [start,start+count) is contiguous after mapping as well.
FSCQSFA_SCHEDULE_INLINE GatherBlockPlan PlanGatherBlock(uint32_t batch, uint32_t tileBase,
    uint32_t start, uint32_t count, uint32_t misses)
{
    const uint32_t base = misses == 0 ? tileBase : GatherIndex(batch, tileBase);
    return {base, count != 0 && base + start < misses};
}
FSCQSFA_SCHEDULE_INLINE bool OwnsMissIndex(uint32_t index, uint32_t aiv, bool splitHeads)
{
    const uint32_t width = splitHeads ? 32 : 64;
    return (index % 128) / width == aiv % (splitHeads ? 4 : 2);
}
} // namespace fused_scatter_copy_qsfa
#undef FSCQSFA_SCHEDULE_INLINE
#endif
