/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_FAI_COMMON_H
#define SLIDING_TILE_ATTENTION_FAI_COMMON_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <cstdint>
#include <type_traits>

#include "adv_api/matmul_intf.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue_online_softmax_no_mask.hpp"
#include "catlass/epilogue/block/block_epilogue_rescale_o_no_split_row.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_mmad_fai_pv_normal.hpp"
#include "catlass/gemm/block/block_mmad_fai_qk_normal.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"

#include "catlass/gemm/block/block_mmad_fai_pv_dynamic.hpp"
#include "catlass/gemm/block/block_mmad_fai_pv_grouped_d72.hpp"
#include "catlass/gemm/block/block_mmad_fai_qk_grouped.hpp"
#include "catlass/gemm/block/block_mmad_fai_qk_sliced_accum.hpp"
#include "resource_ledger.hpp"
#include "sliding_tile_attention_layout_rules.h"
#include "sliding_tile_attention_tiling_data.h"
#include "sliding_tile_attention_tiling_key.h"

namespace {

using namespace AscendC;
using namespace Catlass;

constexpr uint32_t QK_READY_FLAG_ID = 1;
constexpr uint32_t SOFTMAX_READY_FLAG_ID = 2;
constexpr uint32_t PV_READY_FLAG_ID = 3;
constexpr uint32_t INPUT_Q_READY_FLAG_ID = 4;
constexpr uint32_t DYNAMIC_TASK_DONE_FLAG_ID = 5;
constexpr uint32_t INPUT_KV_READY_FLAG_ID = 6;
constexpr uint32_t SOFTMAX_LOW_READY_FLAG_ID = 7;
constexpr uint32_t PV_LOW_READY_FLAG_ID = 8;
constexpr uint32_t Q_STAGE_FREE_FLAG_ID = 9;
constexpr uint32_t KV_STAGE_FREE_FLAG_ID = 10;
constexpr uint32_t FFTS_AIV_PAIR_MODE = 2;
static_assert(FFTS_AIV_PAIR_MODE <= 0x3, "FFTS mode must fit the two-bit message field");
constexpr uint32_t FAST_PATH_HEAD_COUNT = 8;
constexpr uint32_t FAST_PATH_HEAD_INDEX_MASK = FAST_PATH_HEAD_COUNT - 1;
constexpr uint32_t FAST_PATH_HEAD_INDEX_SHIFT = 3;
constexpr uint32_t WINDOW_VALUES_PER_HEAD = 3;
constexpr uint32_t WINDOW_TIME_OFFSET = 0;
constexpr uint32_t WINDOW_HEIGHT_OFFSET = 1;
constexpr uint32_t WINDOW_WIDTH_OFFSET = 2;
constexpr uint32_t FAI_QUERY_ROWS = 128;
constexpr uint32_t FAI_KV_BASE_ROWS = 128;
constexpr uint32_t FAI_QK_SLICE_DIM = 128;
constexpr uint32_t TILE_ELEMENTS = 384;
constexpr uint32_t QUERY_CHUNKS_PER_TILE = TILE_ELEMENTS / FAI_QUERY_ROWS;
constexpr uint32_t D16_PAIR_QUERY_COUNT = 6;
constexpr uint32_t GRID_HEIGHT_TILES = 6;
constexpr uint32_t UB_REGION_BYTES = 16 * 1024;
constexpr uint32_t UB_SUBREGION_BYTES = 1024;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t VECTOR_BLOCK_ELEMENTS = 16;
constexpr uint32_t FLOAT_BLOCK_ELEMENTS = 8;
constexpr uint32_t FLOAT_VECTOR_ELEMENTS = 64;
constexpr uint32_t BF16_CAST_SOURCE_REPEAT_STRIDE = 4;
constexpr uint32_t FAI_STACK_BLOCK_COUNT = 4;
constexpr uint32_t FAI_PRELAUNCH_DEPTH = 4;
constexpr uint32_t FAI_RING_STAGE_COUNT = FAI_PRELAUNCH_DEPTH + 1;
constexpr uint32_t D16_PAIR_PRELAUNCH_DEPTH = static_cast<uint32_t>(
    opforge::ascend_c::resource_ledger::protocol::kD16PairPrelaunchDepth);
static_assert(opforge::ascend_c::resource_ledger::protocol::ValidateCrossCorePrelaunch(
    opforge::ascend_c::resource_ledger::protocol::kD16PairQueryCount, D16_PAIR_PRELAUNCH_DEPTH,
    opforge::ascend_c::resource_ledger::protocol::kD16PairOwnedTokenBudget));
constexpr std::size_t D256_PACKED_BLOCK_BYTES = FAI_KV_BASE_ROWS * 256 * sizeof(half);
constexpr std::size_t D256_PACKED_OWNER_BYTES = FAI_STACK_BLOCK_COUNT * D256_PACKED_BLOCK_BYTES;
static_assert(opforge::ascend_c::resource_ledger::gm::Span(FAI_KV_BASE_ROWS, 256, 256, sizeof(half)) ==
                  D256_PACKED_BLOCK_BYTES,
              "D256 packed K/V block span must match one physical 128-row owner");
static_assert(opforge::ascend_c::resource_ledger::gm::Fits((FAI_STACK_BLOCK_COUNT - 1) * D256_PACKED_BLOCK_BYTES,
                                                           D256_PACKED_BLOCK_BYTES, D256_PACKED_OWNER_BYTES),
              "D256 packed four-block address envelope exceeds its synthetic GM owner");
constexpr std::size_t FAI_BLOCK_TABLE_BYTES_PER_CORE = FAI_RING_STAGE_COUNT * FAI_STACK_BLOCK_COUNT * sizeof(int32_t);
constexpr opforge::ascend_c::resource_ledger::Region FAI_BLOCK_TABLE_REGIONS[] = {
    {1, 0, FAI_BLOCK_TABLE_BYTES_PER_CORE, alignof(int32_t), 0, 2, FAI_BLOCK_TABLE_BYTES_PER_CORE},
};
constexpr opforge::ascend_c::resource_ledger::Access FAI_BLOCK_TABLE_ACCESSES[] = {
    {1, (FAI_RING_STAGE_COUNT - 1) * FAI_STACK_BLOCK_COUNT * sizeof(int32_t), FAI_STACK_BLOCK_COUNT * sizeof(int32_t),
     FAI_STACK_BLOCK_COUNT * sizeof(int32_t), FAI_BLOCK_TABLE_BYTES_PER_CORE, alignof(int32_t), 0},
};
static_assert(opforge::ascend_c::resource_ledger::ValidateAccesses(FAI_BLOCK_TABLE_REGIONS, FAI_BLOCK_TABLE_ACCESSES),
              "STA paged block-table ring exceeds its per-core GM owner");

constexpr uint32_t FAI_AIV_SUBBLOCK_COUNT = 2;
constexpr uint32_t RESCALE_TILE_ELEMENTS = 4096;
constexpr uint32_t RESCALE_MIN_COLUMNS = 8;
constexpr uint32_t RESCALE_GO_UB_OFFSET = 8 * UB_REGION_BYTES;
constexpr uint32_t RESCALE_LOCAL_UB_OFFSET = 6 * UB_REGION_BYTES;
constexpr uint32_t RESCALE_SCALE_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t RESCALE_ROW_SCALE_ELEMENTS = FAI_QUERY_ROWS / FAI_AIV_SUBBLOCK_COUNT;
constexpr uint32_t GROUPED_STATE_UB_OFFSET = 11 * UB_REGION_BYTES;
constexpr uint32_t GROUPED_STATE_CHUNK_ELEMENTS = RESCALE_ROW_SCALE_ELEMENTS;
constexpr uint32_t GROUPED_STATE_MAX_ELEMENTS = D16_PAIR_QUERY_COUNT * GROUPED_STATE_CHUNK_ELEMENTS;
constexpr uint32_t GROUPED_STATE_SUM_ELEMENTS = D16_PAIR_QUERY_COUNT * GROUPED_STATE_CHUNK_ELEMENTS;
constexpr uint32_t GROUPED_STATE_DM_ELEMENTS = D16_PAIR_QUERY_COUNT * FAI_RING_STAGE_COUNT *
                                               GROUPED_STATE_CHUNK_ELEMENTS;
constexpr uint32_t GROUPED_STATE_TOTAL_BYTES = (GROUPED_STATE_MAX_ELEMENTS + GROUPED_STATE_SUM_ELEMENTS +
                                                GROUPED_STATE_DM_ELEMENTS) *
                                               sizeof(float);
static_assert(GROUPED_STATE_TOTAL_BYTES <= UB_REGION_BYTES, "grouped state must fit the unused no-mask UB region");
static_assert(opforge::ascend_c::resource_ledger::protocol::ValidateUbResidentState(
    D16_PAIR_QUERY_COUNT, FAI_RING_STAGE_COUNT, GROUPED_STATE_CHUNK_ELEMENTS, sizeof(float), UB_REGION_BYTES));
// D16 narrow-score bridge: the staged fp16 row tile lives in the free 16 KiB region
// between the rescale `go` owner and the softmax TV owner. It carries no new UB
// capacity - the authored live set still ends at UB_SIZE - and the staged owner is
// released by its own V->MTE2 slot so the next iteration's copy cannot overwrite a
// tile the vector upcast is still reading.
constexpr uint32_t NARROW_SCORE_STAGE_UB_OFFSET = 9 * UB_REGION_BYTES;
constexpr uint32_t NARROW_SCORE_BLOCK_ELEMENTS = DATA_BLOCK_BYTES / sizeof(half);
constexpr uint32_t NARROW_SCORE_STAGE_EVENT =
    opforge::ascend_c::resource_ledger::kNarrowScoreStageVMte2Slot;
static_assert(opforge::ascend_c::resource_ledger::NarrowScoreStageSlotIsIsolated(NARROW_SCORE_STAGE_EVENT),
              "the D16 narrow-score stage edge must own a dedicated V_MTE2 slot");
constexpr uint32_t BF16_CONVERT_CHUNK_ELEMENTS = 4096;
constexpr uint32_t BF16_CONVERT_VECTOR_ELEMENTS = 2048;
constexpr uint32_t QK_STAGE_TILE_ELEMENTS = BF16_CONVERT_CHUNK_ELEMENTS;
constexpr uint32_t BF16_CONVERT_SOURCE_UB_OFFSET = 11 * UB_REGION_BYTES;
constexpr uint32_t BF16_CONVERT_DESTINATION_UB_OFFSET = BF16_CONVERT_SOURCE_UB_OFFSET +
                                                        BF16_CONVERT_CHUNK_ELEMENTS * sizeof(bfloat16_t);
// StageKv drains this bridge before online softmax reuses CATLASS TV scratch.
constexpr uint32_t BF16_CONVERT_FLOAT_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_CENTER_UB_OFFSET = 6 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_RUNTIME_SCALE_BLOCK_UB_OFFSET = SOFTMAX_CENTER_UB_OFFSET;
constexpr uint32_t SOFTMAX_RUNTIME_SCALE_VECTOR_UB_OFFSET = SOFTMAX_RUNTIME_SCALE_BLOCK_UB_OFFSET +
                                                            FLOAT_BLOCK_ELEMENTS * sizeof(float);
constexpr uint32_t SOFTMAX_TV_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_GM_UB_OFFSET = 10 * UB_REGION_BYTES + 10 * UB_SUBREGION_BYTES;
constexpr uint32_t SOFTMAX_LL_UB_OFFSET = 10 * UB_REGION_BYTES + 11 * UB_SUBREGION_BYTES;
constexpr uint32_t SOFTMAX_GL_UB_OFFSET = 10 * UB_REGION_BYTES + 12 * UB_SUBREGION_BYTES;
// The D32 packed route's online softmax and rescale epilogue only claim the
// sub-regions 10*UB_REGION_BYTES + {0,8,9,10,11,12,13}*UB_SUBREGION_BYTES, so
// +5*UB_SUBREGION_BYTES is an unused 512-byte parking slot for the continuous
// stream's running row sum.
constexpr uint32_t STREAM_PARK_UB_OFFSET = 10 * UB_REGION_BYTES + 5 * UB_SUBREGION_BYTES;
// Per-stream-position task row-sum snapshots for the continuous stream. The snapshot
// must be indexed by the stream position, not by the ring slot: a task whose last
// stack lands exactly FAI_PRELAUNCH_DEPTH positions after the previous task's last
// stack would otherwise save its own row sum over the slot the delayed swap is about
// to read. STREAM_SNAPSHOT_COUNT is therefore larger than FAI_PRELAUNCH_DEPTH so the
// save and the swap at the same stream position never touch the same slot. The
// snapshot is UB-resident: the D32 leaf's online softmax and rescale epilogue claim
// only 10*UB_REGION_BYTES + {0,8,9,10,11,12,13}*UB_SUBREGION_BYTES, so the +5..+7
// sub-region block is unused and holds the park slot plus the snapshot array.
constexpr uint32_t STREAM_SNAPSHOT_UB_OFFSET = STREAM_PARK_UB_OFFSET + UB_SUBREGION_BYTES;
constexpr uint32_t STREAM_SNAPSHOT_ROW_ELEMENTS = FAI_QUERY_ROWS / FAI_AIV_SUBBLOCK_COUNT;
constexpr uint32_t STREAM_SNAPSHOT_COUNT = 8;

// The UB->UB row-sum handoff must use a V-annotated primitive. The installed
// `DataCopy(const LocalTensor<T>&, const LocalTensor<T>&, const DataCopyParams&)`
// overload carries no caller-side `__inout_pipe__` annotation, so the compiler's
// pipe model cannot order it against `PipeBarrier<PIPE_V>`; `AscendC::Copy` is
// `__inout_pipe__(V)` and lowers to `vcopy`.
//
// `Copy`'s single-`uint64_t` overload is the *count* form: the header documents
// it as `@param [in] mask[]/mask mask array/count`, and
// `AscendCUtils::SetMask<T>(int32_t len)` builds the hardware mask from that
// element count. Passing a bitmask (e.g. `~0ULL`) here is wrong: `len` becomes
// -1 and the generated mask is garbage.
//
// `CopyRepeatParams` is `(dstStride, srcStride, dstRepeatSize, srcRepeatSize)`.
// The authoritative contiguous-copy defaults are the CANN C-API ones
// (`asc/impl/c_api/instr_impl/npu_arch_2201/utils_impl/utils_impl.h`:
// `ASC_C_API_DEFAULT_BLOCK_STRIDE = 1`, `ASC_C_API_DEFAULT_REPEAT_STRIDE = 8`),
// and the CATLASS reference uses the same pair
// (`catlass/epilogue/tile/tile_broadcast_inplace_by_row.hpp`). `dstStride` is
// the intra-repeat block stride, so `0` collapses every block onto one address
// and corrupts the copy; the repeat size is the repeat span in datablocks
// (`BLK_NUM_PER_VECTOR_FRACTAL = 8`).

// OPFORGE_DEVICE_INLINE_LAYOUT_FORMULA: resource-ledger constexpr helpers are
// host-only, so the repeat span is re-expressed as a device-inline formula.
__aicore__ inline uint32_t StreamHandoffCopyRepeatSize(uint32_t rowCount)
{
    return rowCount / FLOAT_BLOCK_ELEMENTS;
}

constexpr uint16_t kStreamHandoffBlockStride =
    opforge::ascend_c::resource_ledger::UbCopyContiguousBlockStride();

static_assert(opforge::ascend_c::resource_ledger::ValidateUbHandoffCopyContract(
                  opforge::ascend_c::resource_ledger::kStreamHandoffCopy),
              "the D32 stream row-sum handoff must use a V-annotated copy under PipeBarrier<PIPE_V>");

constexpr opforge::ascend_c::resource_ledger::UbOverwriteContract kAccumUbVectorToMte2Reuse = {
    opforge::ascend_c::resource_ledger::UbAccessRole::VectorReader,
    opforge::ascend_c::resource_ledger::UbAccessRole::Mte2Writer,
    opforge::ascend_c::resource_ledger::UbReuseEvent::V_MTE2, opforge::ascend_c::resource_ledger::UbReuseEvent::V_MTE2,
    true};
constexpr opforge::ascend_c::resource_ledger::UbOverwriteContract kPartialUbVectorToMte2Reuse = {
    opforge::ascend_c::resource_ledger::UbAccessRole::VectorReader,
    opforge::ascend_c::resource_ledger::UbAccessRole::Mte2Writer,
    opforge::ascend_c::resource_ledger::UbReuseEvent::V_MTE2, opforge::ascend_c::resource_ledger::UbReuseEvent::V_MTE2,
    true};
constexpr opforge::ascend_c::resource_ledger::UbOverwriteContract kSrcUbVectorToMte2Reuse = {
    opforge::ascend_c::resource_ledger::UbAccessRole::VectorReader,
    opforge::ascend_c::resource_ledger::UbAccessRole::Mte2Writer,
    opforge::ascend_c::resource_ledger::UbReuseEvent::V_MTE2, opforge::ascend_c::resource_ledger::UbReuseEvent::V_MTE2,
    true};
constexpr float PROBABILITY_RESIDUAL_SCALE = 4096.0f;
constexpr float PROBABILITY_RESIDUAL_ADD_BACK = 0.5f / PROBABILITY_RESIDUAL_SCALE;
constexpr opforge::ascend_c::resource_ledger::protocol::DualProbabilityNumericContract kD72ResidualContract = {
    true, true, true, true, true, true, true, PROBABILITY_RESIDUAL_SCALE, PROBABILITY_RESIDUAL_ADD_BACK};
static_assert(
    opforge::ascend_c::resource_ledger::protocol::ValidateDualProbabilityNumericContract(kD72ResidualContract),
    "D72 compact residual probability must close its FP32-to-main/residual-to-PV-to-addback chain");
constexpr opforge::ascend_c::resource_ledger::L0cReuseContract kD72MainResidualL0cContract = {
    opforge::ascend_c::resource_ledger::L0cCompletionProtocol::UnitFlag,
    opforge::ascend_c::resource_ledger::L0cCompletionProtocol::UnitFlag, true, true, true};
static_assert(opforge::ascend_c::resource_ledger::ValidateL0cReuseContract(kD72MainResidualL0cContract),
              "D72 main and residual PV must share one unit-flag L0C completion protocol");
constexpr uint32_t SUM_V_TILE_ROWS = 32;
constexpr uint32_t SUM_V_ACCUMULATOR_COUNT = 4;
constexpr uint32_t DYNAMIC_CENTER_COLUMN_TILE = 64;
constexpr uint32_t SUM_V_SOURCE_UB_OFFSET = 0;
constexpr uint32_t SUM_V_FLOAT_UB_OFFSET = 2 * UB_REGION_BYTES;
constexpr uint32_t SUM_V_ACCUM_UB_OFFSET = 3 * UB_REGION_BYTES;
// The dual-probability experiment is compile-time disabled. Keep its fixed scratch geometry isolated from the
// production runtime-D path until that experiment is deleted in final cleanup.
constexpr uint32_t SUM_V_VECTOR_ELEMENTS = 64;
constexpr uint32_t SUM_V_ACCUM_STRIDE_BYTES = SUM_V_VECTOR_ELEMENTS * sizeof(float);
constexpr uint32_t CORRECTION_GO_STATE_OFFSET_ELEMENTS = FAI_QUERY_ROWS + 2 * SUM_V_VECTOR_ELEMENTS;
constexpr uint32_t CORRECTION_BROADCAST_UB_OFFSET = 0;
constexpr uint32_t CORRECTION_SUM_V_UB_OFFSET = UB_REGION_BYTES;
constexpr uint32_t CORRECTION_VALUE_UB_OFFSET = 2 * UB_REGION_BYTES;
constexpr uint32_t CORRECTION_CENTER_UB_OFFSET = 3 * UB_REGION_BYTES;
constexpr uint32_t STA_RESCALE_LO_UB_OFFSET = 6 * UB_REGION_BYTES;
constexpr uint32_t STA_RESCALE_GO_UB_OFFSET = 8 * UB_REGION_BYTES;
constexpr uint32_t STA_RESCALE_TV_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t STA_RESCALE_GL_UB_OFFSET = 10 * UB_REGION_BYTES + 12 * UB_SUBREGION_BYTES;
constexpr uint32_t STA_RESCALE_DM_UB_OFFSET = 10 * UB_REGION_BYTES + 13 * UB_SUBREGION_BYTES;
constexpr uint32_t FASTVIDEO_GRID_TIME_TILES = 5;
constexpr uint32_t FASTVIDEO_GRID_WIDTH_TILES = 10;
constexpr uint32_t FASTVIDEO_QUERY_BLOCKS_PER_HEAD = 901;
constexpr uint32_t FASTVIDEO_IMAGE_CHUNKS_PER_HEAD = 900;
constexpr uint32_t STEPVIDEO_GRID_TIME_TILES = 6;
constexpr uint32_t STEPVIDEO_GRID_WIDTH_TILES = 6;
constexpr uint32_t STEPVIDEO_QUERY_BLOCKS_PER_HEAD = 648;
constexpr uint32_t WAN_GRID_TIME_TILES = 3;
constexpr uint32_t WAN_GRID_WIDTH_TILES = 10;
constexpr uint32_t WAN_QUERY_BLOCKS_PER_HEAD = 540;
constexpr uint32_t HEIGHT_SPAN_SINGLE_TILE = 1;
constexpr uint32_t HEIGHT_SPAN_THREE_TILES = 3;
constexpr uint32_t HEIGHT_SPAN_FIVE_TILES = 5;
constexpr uint32_t HEIGHT_SPAN_FULL_GRID = GRID_HEIGHT_TILES;
constexpr uint32_t SEGMENT_ONE_WIDTH_ELEMENTS = TILE_ELEMENTS;
constexpr uint32_t SEGMENT_THREE_WIDTH_ELEMENTS = 3 * TILE_ELEMENTS;
constexpr uint32_t SEGMENT_FIVE_WIDTH_ELEMENTS = 5 * TILE_ELEMENTS;
constexpr uint32_t SEGMENT_SEVEN_WIDTH_ELEMENTS = 7 * TILE_ELEMENTS;
constexpr uint32_t STACK_COUNT_ONE_WIDTH = 1;
constexpr uint32_t STACK_COUNT_THREE_WIDTH = 3;
constexpr uint32_t STACK_COUNT_FIVE_WIDTH = 4;
constexpr uint32_t STACK_COUNT_SEVEN_WIDTH = 6;
constexpr float D64_SOFTMAX_SCALE = 0.125f;

static_assert(BF16_CONVERT_FLOAT_UB_OFFSET + BF16_CONVERT_VECTOR_ELEMENTS * sizeof(float) <= Arch::AtlasA2::UB_SIZE,
              "bf16 conversion scratch exceeds AtlasA2 UB");
static_assert(BF16_CONVERT_DESTINATION_UB_OFFSET + BF16_CONVERT_CHUNK_ELEMENTS * sizeof(half) <= Arch::AtlasA2::UB_SIZE,
              "bf16 conversion destination exceeds AtlasA2 UB");
template <typename T>
__aicore__ inline T StaMin(T a, T b)
{
    return a < b ? a : b;
}

__aicore__ inline uint32_t StaCeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t StaAlignUp(uint32_t a, uint32_t b) { return StaCeilDiv(a, b) * b; }

__aicore__ inline void LoadSlidingTileAttentionTilingData(GM_ADDR tiling, SlidingTileAttentionTilingData& out)
{
    auto* dst64 = reinterpret_cast<uint64_t*>(&out);
    const __gm__ auto* src64 = reinterpret_cast<const __gm__ uint64_t*>(tiling);
    constexpr int64_t TILING_WORD_COUNT = static_cast<int64_t>(sizeof(SlidingTileAttentionTilingData) /
                                                               sizeof(uint64_t));
    for (int64_t i = 0; i < TILING_WORD_COUNT; ++i) {
        dst64[i] = src64[i];
    }
    constexpr int64_t TILING_BYTE_START = TILING_WORD_COUNT * static_cast<int64_t>(sizeof(uint64_t));
    auto* dst8 = reinterpret_cast<uint8_t*>(&out);
    const __gm__ auto* src8 = reinterpret_cast<const __gm__ uint8_t*>(tiling);
    for (int64_t i = TILING_BYTE_START; i < static_cast<int64_t>(sizeof(SlidingTileAttentionTilingData)); ++i) {
        dst8[i] = src8[i];
    }
}

struct StaFaiWorkspace {
    GM_ADDR s = nullptr;
    GM_ADDR p = nullptr;
    GM_ADDR p_lo = nullptr;
    GM_ADDR correction = nullptr;
    GM_ADDR o_temp = nullptr;
    GM_ADDR o_temp_lo = nullptr;
    GM_ADDR o_update = nullptr;
    GM_ADDR softmax_max = nullptr;
    GM_ADDR softmax_sum = nullptr;
    GM_ADDR softmax_dm = nullptr;
    GM_ADDR block_table = nullptr;
    GM_ADDR q_stage = nullptr;
    GM_ADDR k_stage = nullptr;
    GM_ADDR v_stage = nullptr;
    int64_t blocks = 0;
    int64_t block_elems = 0;
    int64_t staged_elems = 0;
};

template <typename T>
__aicore__ inline StaFaiWorkspace MakeFaiWorkspace(GM_ADDR workspace, int64_t stagedElems, int64_t blocks,
                                                   int64_t blockElems, int64_t dim, bool withQkStaging,
                                                   bool withVStaging, bool withDualProbability,
                                                   int64_t groupedQueryChunks)
{
    StaFaiWorkspace out;
    __gm__ uint8_t* base = (__gm__ uint8_t*)workspace;
    int64_t offset = 0;
    int64_t qkDim = ((dim + VECTOR_BLOCK_ELEMENTS - 1) / VECTOR_BLOCK_ELEMENTS) * VECTOR_BLOCK_ELEMENTS;
    int64_t qkSliceCount = (qkDim + FAI_QK_SLICE_DIM - 1) / FAI_QK_SLICE_DIM;
    int64_t qkStageDim = qkSliceCount > 1 ? qkSliceCount * FAI_QK_SLICE_DIM : qkDim;
    const bool needsGroupedSoftmaxState = dim <= FAI_QK_SLICE_DIM;
    out.s = base + offset;
    offset += stagedElems * qkSliceCount * static_cast<int64_t>(sizeof(float));
    out.p = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(T));
    if (withDualProbability) {
        out.p_lo = base + offset;
        offset += stagedElems * static_cast<int64_t>(sizeof(T));
    }
    out.o_temp = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(float));
    if (withDualProbability) {
        out.o_temp_lo = base + offset;
        offset += stagedElems * static_cast<int64_t>(sizeof(float));
        out.correction = out.o_temp_lo;
    }
    out.o_update = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(float));
    if (needsGroupedSoftmaxState) {
        out.softmax_max = base + offset;
        offset += blocks * groupedQueryChunks * FAI_QUERY_ROWS * static_cast<int64_t>(sizeof(float));
        out.softmax_sum = base + offset;
        offset += blocks * groupedQueryChunks * FAI_QUERY_ROWS * static_cast<int64_t>(sizeof(float));
        out.softmax_dm = base + offset;
        offset += blocks * groupedQueryChunks * FAI_RING_STAGE_COUNT * FAI_QUERY_ROWS *
                  static_cast<int64_t>(sizeof(float));
    }
    out.block_table = base + offset;
    offset += blocks * FAI_RING_STAGE_COUNT * FAI_STACK_BLOCK_COUNT * static_cast<int64_t>(sizeof(int32_t));
    if (withQkStaging) {
        out.q_stage = base + offset;
        int64_t queryStageSlots = dim == 8 ? QUERY_CHUNKS_PER_TILE : 1;
        offset += blocks * queryStageSlots * FAI_QUERY_ROWS * qkStageDim * static_cast<int64_t>(sizeof(T));
        out.k_stage = base + offset;
        offset += blocks * FAI_RING_STAGE_COUNT * FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS * qkStageDim *
                  static_cast<int64_t>(sizeof(T));
    }
    if (withVStaging) {
        out.v_stage = base + offset;
    }
    out.blocks = blocks;
    out.block_elems = blockElems;
    out.staged_elems = stagedElems;
    return out;
}


} // namespace

#endif
