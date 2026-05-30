/**
 * \file profiler.h
 * \brief Per-operator profiler header for quant_reduce_scatter & quant_all_reduce.
 *
 * Event IDs shared between both operators (they never run in the same kernel).
 * All events are AIV-side (both operators are AIV-only).
 */

#pragma once

#include "profiler_core.h"

enum ProfEventId : uint32_t {
    EVT_KERNEL_BEGIN       = 0x0001,
    EVT_KERNEL_END         = 0x00FF,

    // Stage 1: Copy data from GM to Win area (all2all / allgather)
    EVT_COPY_TO_WIN_BEGIN  = 0x6001,
    EVT_COPY_TO_WIN_END    = 0x6002,

    // Stage 2: Write completion status to all ranks
    EVT_WRITE_STATUS_BEGIN = 0x6003,
    EVT_WRITE_STATUS_END   = 0x6004,

    // Stage 3: Read status + busy-wait (sync barrier)
    EVT_READ_STATUS_BEGIN  = 0x6005,
    EVT_READ_STATUS_END    = 0x6006,

    // Stage 4: Main compute loop (read + dequant + reduce + output)
    EVT_EXECUTE_BEGIN      = 0x6007,
    EVT_EXECUTE_END        = 0x6008,

    // Stage 4 sub-events (inside EXECUTE)
    EVT_BLOCK_BEGIN        = 0x6009,  // per-block iteration (payload = block index)
    EVT_BLOCK_END          = 0x600A,
    EVT_RANK_READ_BEGIN    = 0x600B,  // (unused in round 2 — kept for ID stability)
    EVT_RANK_READ_END      = 0x600C,
    EVT_COPY_OUT_BEGIN     = 0x600D,  // result copy to GM (payload = block index)
    EVT_COPY_OUT_END       = 0x600E,

    // Round 2: fine-grained sub-events
    EVT_CLEAR_SUM_BEGIN    = 0x600F,  // ClearSumTensor (Duplicate)
    EVT_CLEAR_SUM_END      = 0x6010,
    EVT_RANK_LOOP_BEGIN    = 0x6011,  // entire rank for-loop (all ranks)
    EVT_RANK_LOOP_END      = 0x6012,
    EVT_DEQUANT_SUM_BEGIN  = 0x6013,  // DequantReduceSum (CastToFloat+Mul+Add)
    EVT_DEQUANT_SUM_END    = 0x6014,
};
