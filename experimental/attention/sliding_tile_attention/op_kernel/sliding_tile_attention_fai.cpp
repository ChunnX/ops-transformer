/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/block/block_epilogue_online_softmax_no_mask.hpp"
#include "catlass/epilogue/block/block_epilogue_rescale_o_no_split_row.hpp"
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
constexpr uint32_t D16_PAIR_PRELAUNCH_DEPTH =
    static_cast<uint32_t>(opforge::ascend_c::resource_ledger::protocol::kD16PairPrelaunchDepth);
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
constexpr uint32_t GROUPED_STATE_DM_ELEMENTS =
    D16_PAIR_QUERY_COUNT * FAI_RING_STAGE_COUNT * GROUPED_STATE_CHUNK_ELEMENTS;
constexpr uint32_t GROUPED_STATE_TOTAL_BYTES =
    (GROUPED_STATE_MAX_ELEMENTS + GROUPED_STATE_SUM_ELEMENTS + GROUPED_STATE_DM_ELEMENTS) * sizeof(float);
static_assert(GROUPED_STATE_TOTAL_BYTES <= UB_REGION_BYTES, "grouped state must fit the unused no-mask UB region");
static_assert(opforge::ascend_c::resource_ledger::protocol::ValidateUbResidentState(
    D16_PAIR_QUERY_COUNT, FAI_RING_STAGE_COUNT, GROUPED_STATE_CHUNK_ELEMENTS, sizeof(float), UB_REGION_BYTES));
constexpr uint32_t BF16_CONVERT_CHUNK_ELEMENTS = 4096;
constexpr uint32_t BF16_CONVERT_VECTOR_ELEMENTS = 2048;
constexpr uint32_t QK_STAGE_TILE_ELEMENTS = BF16_CONVERT_CHUNK_ELEMENTS;
constexpr uint32_t BF16_CONVERT_SOURCE_UB_OFFSET = 11 * UB_REGION_BYTES;
constexpr uint32_t BF16_CONVERT_DESTINATION_UB_OFFSET =
    BF16_CONVERT_SOURCE_UB_OFFSET + BF16_CONVERT_CHUNK_ELEMENTS * sizeof(bfloat16_t);
// StageKv drains this bridge before online softmax reuses CATLASS TV scratch.
constexpr uint32_t BF16_CONVERT_FLOAT_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_CENTER_UB_OFFSET = 6 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_RUNTIME_SCALE_BLOCK_UB_OFFSET = SOFTMAX_CENTER_UB_OFFSET;
constexpr uint32_t SOFTMAX_RUNTIME_SCALE_VECTOR_UB_OFFSET =
    SOFTMAX_RUNTIME_SCALE_BLOCK_UB_OFFSET + FLOAT_BLOCK_ELEMENTS * sizeof(float);
constexpr uint32_t SOFTMAX_TV_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_GM_UB_OFFSET = 10 * UB_REGION_BYTES + 10 * UB_SUBREGION_BYTES;
constexpr uint32_t SOFTMAX_LL_UB_OFFSET = 10 * UB_REGION_BYTES + 11 * UB_SUBREGION_BYTES;
constexpr uint32_t SOFTMAX_GL_UB_OFFSET = 10 * UB_REGION_BYTES + 12 * UB_SUBREGION_BYTES;
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

__aicore__ inline uint32_t StaCeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline uint32_t StaAlignUp(uint32_t a, uint32_t b)
{
    return StaCeilDiv(a, b) * b;
}

__aicore__ inline void LoadSlidingTileAttentionTilingData(GM_ADDR tiling, SlidingTileAttentionTilingData &out)
{
    auto *dst64 = reinterpret_cast<uint64_t *>(&out);
    const __gm__ auto *src64 = reinterpret_cast<const __gm__ uint64_t *>(tiling);
    constexpr int64_t TILING_WORD_COUNT =
        static_cast<int64_t>(sizeof(SlidingTileAttentionTilingData) / sizeof(uint64_t));
    for (int64_t i = 0; i < TILING_WORD_COUNT; ++i) {
        dst64[i] = src64[i];
    }
    constexpr int64_t TILING_BYTE_START = TILING_WORD_COUNT * static_cast<int64_t>(sizeof(uint64_t));
    auto *dst8 = reinterpret_cast<uint8_t *>(&out);
    const __gm__ auto *src8 = reinterpret_cast<const __gm__ uint8_t *>(tiling);
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
    __gm__ uint8_t *base = (__gm__ uint8_t *)workspace;
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
        offset +=
            blocks * groupedQueryChunks * FAI_RING_STAGE_COUNT * FAI_QUERY_ROWS * static_cast<int64_t>(sizeof(float));
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

template <typename InputT, typename OutputT, bool PACKED_D128 = false, bool PACKED_D256 = false,
          bool PACKED_D72_FULL = false, bool PACKED_D3SLICE = false, bool PACKED_D256_TEXT = false>
class SlidingTileAttentionFaiTextOp {
public:
    using ArchTag = Arch::AtlasA2;
    using ElementInput = InputT;
    using ElementQ = ElementInput;
    using LayoutQ = layout::RowMajor;
    using ElementK = ElementInput;
    using LayoutK = layout::ColumnMajor;
    // Q/K stay native bf16. Only V is converted once to half GM workspace inside this same kernel because the
    // installed CATLASS PV primitive does not provide the required mixed bf16-V lowering for this path.
    using ElementV = half;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementCompactS = float;
    using ElementP = half;
    using LayoutP = layout::RowMajor;
    using ElementO = OutputT;
    using LayoutO = layout::RowMajor;
    using ElementMask = ElementP;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    using L1TileShape = GemmShape<FAI_QUERY_ROWS, FAI_KV_BASE_ROWS, FAI_KV_BASE_ROWS>;
    using L0TileShape = L1TileShape;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<false, true>;
    using DispatchPolicyQKPaged = Gemm::MmadAtlasA2FAIQK<true, true>;
    using DispatchPolicyQKGrouped = Gemm::MmadAtlasA2FAIQK<false, true>;
    using DispatchPolicyQKSliced = Gemm::MmadAtlasA2FAIQK<false, false>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<false, true>;
    using DispatchPolicyPVPaged = Gemm::MmadAtlasA2FAIPV<true, true>;
    using DispatchPolicyPVDynamicSingleBlock = Gemm::MmadAtlasA2FAIPV<false, true>;
    using DispatchPolicyPVDynamicMultiBlock = Gemm::MmadAtlasA2FAIPV<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using CompactSType = Gemm::GemmType<ElementCompactS, LayoutS>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using UpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;
    using BlockMmadQKPaged =
        Gemm::Block::BlockMmad<DispatchPolicyQKPaged, L1TileShape, L0TileShape, QType, KType, SType>;
    using BlockMmadQKCompact =
        Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, CompactSType>;
    using BlockMmadQKGrouped = SlidingTileAttentionCatlass::BlockMmadFaiQkGrouped<DispatchPolicyQKGrouped, L1TileShape,
                                                                                  L0TileShape, QType, KType, SType>;
    using BlockMmadQKGroupedPairD16 =
        SlidingTileAttentionCatlass::BlockMmadFaiQkGrouped<DispatchPolicyQKGrouped, L1TileShape, L0TileShape, QType,
                                                           KType, SType, 6, 2>;
    using BlockMmadQKSlicedAccum =
        SlidingTileAttentionCatlass::BlockMmadFaiQkSlicedAccum<DispatchPolicyQKSliced, L1TileShape, L0TileShape, QType,
                                                               KType, SType>;
    using BlockMmadPVNormal =
        Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;
    using BlockMmadPVPaged =
        Gemm::Block::BlockMmad<DispatchPolicyPVPaged, L1TileShape, L0TileShape, PType, VType, OTmpType>;
    using BlockMmadPVDynamicSingleBlock =
        SlidingTileAttentionCatlass::BlockMmadFaiPvDynamic<DispatchPolicyPVDynamicSingleBlock, L1TileShape, L0TileShape,
                                                           PType, VType, OTmpType>;
    using BlockMmadPVDynamicMultiBlock =
        SlidingTileAttentionCatlass::BlockMmadFaiPvDynamic<DispatchPolicyPVDynamicMultiBlock, L1TileShape, L0TileShape,
                                                           PType, VType, OTmpType>;
    using BlockMmadPVGroupedD72 =
        SlidingTileAttentionCatlass::BlockMmadFaiPvGroupedD72<DispatchPolicyPVDynamicSingleBlock, L1TileShape,
                                                              L0TileShape, PType, VType, OTmpType>;
    using BlockMmadPVGroupedPairD16 =
        SlidingTileAttentionCatlass::BlockMmadFaiPvGroupedD72<DispatchPolicyPVDynamicSingleBlock, L1TileShape,
                                                              L0TileShape, PType, VType, OTmpType, 6, 16>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2OnlineSoftmax, PType, SType, MaskType>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2RescaleO, OType, OTmpType, UpdateType>;
    static constexpr bool NEEDS_INPUT_STAGING = !std::is_same_v<ElementInput, half>;
    static constexpr bool USE_DUAL_PROBABILITY = false;

    static_assert((std::is_same_v<ElementInput, half> && std::is_same_v<ElementO, half>) ||
                      (std::is_same_v<ElementInput, bfloat16_t> && std::is_same_v<ElementO, bfloat16_t>),
                  "STA FAI supports matched fp16 or bf16 public input/output types");

    CATLASS_DEVICE SlidingTileAttentionFaiTextOp() {}

    CATLASS_DEVICE void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, uint32_t softmaxScaleBitsIn,
                             SlidingTileAttentionMeta metaIn, int64_t totalJobsIn, float softmaxScaleIn,
                             StaFaiWorkspace workspaceIn)
    {
        meta = metaIn;
        totalJobs = totalJobsIn;
        softmaxScale = softmaxScaleIn;
        workspace = workspaceIn;
        qInputGm.SetGlobalBuffer((__gm__ ElementInput *)q);
        kInputGm.SetGlobalBuffer((__gm__ ElementInput *)k);
        vInputGm.SetGlobalBuffer((__gm__ ElementInput *)v);
        if (NeedsQkStaging()) {
            qGm.SetGlobalBuffer((__gm__ ElementQ *)workspace.q_stage);
            kGm.SetGlobalBuffer((__gm__ ElementK *)(UsesOwnedStridedKLoader() ? k : workspace.k_stage));
        } else {
            qGm.SetGlobalBuffer((__gm__ ElementQ *)q);
            kGm.SetGlobalBuffer((__gm__ ElementK *)k);
        }
        if constexpr (NEEDS_INPUT_STAGING) {
            vGm.SetGlobalBuffer((__gm__ ElementV *)workspace.v_stage);
        } else {
            vGm.SetGlobalBuffer((__gm__ ElementV *)v);
        }
        outGm.SetGlobalBuffer((__gm__ ElementO *)out);
        dummyBlockTable.SetGlobalBuffer((__gm__ int32_t *)workspace.o_update);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)workspace.block_table);
        sGm.SetGlobalBuffer((__gm__ ElementS *)workspace.s);
        compactSGm.SetGlobalBuffer((__gm__ ElementCompactS *)workspace.s);
        pGm.SetGlobalBuffer((__gm__ ElementP *)workspace.p);
        oTmpGm.SetGlobalBuffer((__gm__ ElementOTmp *)workspace.o_temp);
        if (UsesResidualProbability()) {
            pLoGm.SetGlobalBuffer((__gm__ ElementP *)workspace.p_lo);
            oTmpLoGm.SetGlobalBuffer((__gm__ ElementOTmp *)workspace.o_temp_lo);
        }
        oUpdateGm.SetGlobalBuffer((__gm__ ElementUpdate *)workspace.o_update);
        softmaxMaxGm.SetGlobalBuffer((__gm__ float *)workspace.softmax_max);
        softmaxSumGm.SetGlobalBuffer((__gm__ float *)workspace.softmax_sum);
        softmaxDmGm.SetGlobalBuffer((__gm__ float *)workspace.softmax_dm);
        softmaxScaleBits = softmaxScaleBitsIn;
        if constexpr (USE_DUAL_PROBABILITY) {
            correctionGm.SetGlobalBuffer((__gm__ float *)workspace.correction);
        }
        if (UsesResidualProbability()) {
            correctionGm.SetGlobalBuffer((__gm__ float *)workspace.correction);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void Process();

    template <>
    CATLASS_DEVICE void Process<AscendC::AIC>()
    {
        InitAicEvents();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        if (AscendC::GetBlockIdx() == 0) {
            AscendC::printf("STA_LIFECYCLE AIC_ENTER dim=%u jobs=%u\\n", static_cast<uint32_t>(meta.dim),
                            static_cast<uint32_t>(totalJobs));
        }
#endif
        if constexpr (NEEDS_INPUT_STAGING) {
            AscendC::SyncAll<false>();
        }
        if constexpr (!(PACKED_D256 || PACKED_D3SLICE)) {
            if (NeedsQkStaging() && meta.dim > FAI_QK_SLICE_DIM) {
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qStageFree);
                if (NeedsKvQkStaging()) {
                    Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(kvStageFree);
                }
            }
        }
        static constexpr uint32_t L1_QK_GROUPED_BYTES =
            QUERY_CHUNKS_PER_TILE * BlockMmadQKGrouped::L1_A_BYTES + 2 * BlockMmadQKGrouped::L1_B_BYTES;
        static constexpr uint32_t L1_QK_D128_EARLY_PUBLISH_BYTES =
            QUERY_CHUNKS_PER_TILE * BlockMmadQKGrouped::L1_A_BYTES +
            BlockMmadQKGrouped::MAX_STACK_BLOCKS * BlockMmadQKGrouped::L1_B_BYTES;
        static constexpr uint32_t L1_PV_PACKED_D128_BYTES =
            BlockMmadPVGroupedD72::L1_A_BYTES +
            BlockMmadPVGroupedD72::MAX_STACK_BLOCKS * BlockMmadPVGroupedD72::L1_B_BLOCK_BYTES;
        static_assert(L1_QK_D128_EARLY_PUBLISH_BYTES + L1_PV_PACKED_D128_BYTES <= ArchTag::L1_SIZE,
                      "packed D128 grouped QK plus sequential PV exceeds L1 capacity");
        static constexpr uint32_t L1_PV_DYNAMIC_BYTES =
            BlockMmadPVDynamicMultiBlock::L1A_SIZE * BlockMmadPVDynamicMultiBlock::STAGES *
                BlockMmadPVDynamicMultiBlock::STAGES +
            BlockMmadPVDynamicMultiBlock::L1B_SIZE * BlockMmadPVDynamicMultiBlock::STAGES;
        static_assert(L1_QK_D128_EARLY_PUBLISH_BYTES + L1_PV_DYNAMIC_BYTES <= ArchTag::L1_SIZE,
                      "D128 query-major QK plus dynamic PV exceeds L1 capacity");
        static constexpr uint32_t L1_QK_NORMAL_BYTES =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
        static_assert(2 * BlockMmadQKSlicedAccum::L1_A_SIZE + BlockMmadQKSlicedAccum::L1_B_SIZE <= L1_QK_NORMAL_BYTES,
                      "cached D256 Q slices plus one K slice exceed the existing QK L1 owner");
        static constexpr uint32_t L1_QK_D3SLICE_BYTES =
            3 * BlockMmadQKSlicedAccum::L1_A_SIZE + BlockMmadQKSlicedAccum::L1_B_SIZE;
        static_assert(L1_QK_D3SLICE_BYTES + L1_PV_DYNAMIC_BYTES <= ArchTag::L1_SIZE,
                      "three-slice packed QK plus dynamic PV exceeds L1 capacity");
        const uint32_t l1QkBytes =
            PACKED_D3SLICE ?
                L1_QK_D3SLICE_BYTES :
            (PACKED_D72_FULL || UsesD128GroupedEarlyPublishPath()) ?
                L1_QK_D128_EARLY_PUBLISH_BYTES :
                ((UsesGroupedQkPath() || UsesCompactFullWindowPath()) ? L1_QK_GROUPED_BYTES : L1_QK_NORMAL_BYTES);
        if constexpr (PACKED_D72_FULL) {
            BlockMmadQKGrouped blockMmadQKGrouped(resource);
            BlockMmadPVNormal blockMmadPVNormal(resource, l1QkBytes);
            BlockMmadPVDynamicSingleBlock blockMmadPVDynamicSingleBlock(resource, l1QkBytes);
            BlockMmadPVDynamicMultiBlock blockMmadPVDynamicMultiBlock(resource, l1QkBytes);
            BlockMmadPVGroupedD72 blockMmadPVGroupedD72(resource, l1QkBytes);
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                ProcessAicGroupedTask(taskIdx, blockMmadQKGrouped, blockMmadPVNormal, blockMmadPVDynamicSingleBlock,
                                      blockMmadPVDynamicMultiBlock, blockMmadPVGroupedD72);
            }
            WaitAicEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if (UsesCompactFullWindowPath()) {
            BlockMmadQKCompact blockMmadQKCompact(resource);
            BlockMmadQKPaged blockMmadQKPaged(resource);
            BlockMmadQKSlicedAccum blockMmadQKSlicedAccum(resource);
            BlockMmadPVNormal blockMmadPVNormal(resource, l1QkBytes);
            BlockMmadPVPaged blockMmadPVPaged(resource, l1QkBytes);
            BlockMmadPVDynamicSingleBlock blockMmadPVDynamicSingleBlock(resource, l1QkBytes);
            BlockMmadPVDynamicMultiBlock blockMmadPVDynamicMultiBlock(resource, l1QkBytes);
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    ProcessAicTask(GroupedChunkTaskIndex(taskIdx, chunk), blockMmadQKCompact, blockMmadQKPaged,
                                   compactSGm, true, blockMmadQKSlicedAccum, blockMmadPVNormal, blockMmadPVPaged,
                                   blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                }
            }
            WaitAicEvents();
            // Compact and sliced paths share mixed AIC/AIV resources. Close the
            // block-wide ownership boundary before handing them to a later launch.
            AscendC::SyncAll<false>();
            return;
        }
        if (meta.pair_axis >= 0 && UsesGroupedD16QkPath()) {
            static_assert(BlockMmadQKGroupedPairD16::QUERY_COUNT ==
                          opforge::ascend_c::resource_ledger::protocol::kD16PairQueryCount);
            static constexpr uint32_t L1_QK_PAIR_D16_BYTES =
                BlockMmadQKGroupedPairD16::QUERY_COUNT * BlockMmadQKGroupedPairD16::L1_A_BYTES +
                BlockMmadQKGroupedPairD16::RESIDENT_K_STAGES * BlockMmadQKGroupedPairD16::L1_B_BYTES;
            static constexpr uint32_t L1_PV_PAIR_D16_BYTES =
                BlockMmadPVGroupedPairD16::L1_A_BYTES +
                BlockMmadPVGroupedPairD16::MAX_STACK_BLOCKS * BlockMmadPVGroupedPairD16::L1_B_BLOCK_BYTES;
            static_assert(L1_QK_PAIR_D16_BYTES + L1_PV_PAIR_D16_BYTES <= ArchTag::L1_SIZE,
                          "D16 pair QK/PV owners exceed L1 capacity");
            BlockMmadQKGroupedPairD16 blockMmadQKPair(resource);
            BlockMmadPVGroupedPairD16 blockMmadPVPair(resource, L1_QK_PAIR_D16_BYTES);
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                if (IsGroupedImageJob(taskIdx)) {
                    ProcessAicGroupedPairD16Task(taskIdx, blockMmadQKPair, blockMmadPVPair);
                }
            }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIC phase=image_done\n");
            }
#endif
            AscendC::SyncAll<false>();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIC phase=sync1_done\n");
            }
#endif
            BlockMmadQK blockMmadQK(resource);
            BlockMmadQKPaged blockMmadQKPaged(resource);
            BlockMmadQKSlicedAccum blockMmadQKSlicedAccum(resource);
            BlockMmadPVNormal blockMmadPVNormal(resource, L1_QK_NORMAL_BYTES);
            BlockMmadPVPaged blockMmadPVPaged(resource, L1_QK_NORMAL_BYTES);
            BlockMmadPVDynamicSingleBlock blockMmadPVDynamicSingleBlock(resource, L1_QK_NORMAL_BYTES);
            BlockMmadPVDynamicMultiBlock blockMmadPVDynamicMultiBlock(resource, L1_QK_NORMAL_BYTES);
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                if (!IsGroupedImageJob(taskIdx)) {
                    if (UsesGroupedTextD16Path()) {
                        if (IsFirstGroupedTextJob(taskIdx)) {
                            ProcessAicGroupedTextD16Task(taskIdx, blockMmadQKPair, blockMmadPVPair);
                        }
                    } else {
                        ProcessAicTask(MixedTextTaskIndex(taskIdx), blockMmadQK, blockMmadQKPaged, sGm, false,
                                       blockMmadQKSlicedAccum, blockMmadPVNormal, blockMmadPVPaged,
                                       blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                    }
                }
            }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIC phase=text_done\n");
            }
#endif
            WaitAicEvents();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIC phase=events_done\n");
            }
#endif
            AscendC::SyncAll<false>();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIC phase=sync2_done\n");
            }
#endif
            return;
        }
        BlockMmadQK blockMmadQK(resource);
        BlockMmadQKPaged blockMmadQKPaged(resource);
        BlockMmadQKGrouped blockMmadQKGrouped(resource);
        BlockMmadQKSlicedAccum blockMmadQKSlicedAccum(resource);
        BlockMmadPVNormal blockMmadPVNormal(resource, l1QkBytes);
        BlockMmadPVPaged blockMmadPVPaged(resource, l1QkBytes);
        BlockMmadPVDynamicSingleBlock blockMmadPVDynamicSingleBlock(resource, l1QkBytes);
        BlockMmadPVDynamicMultiBlock blockMmadPVDynamicMultiBlock(resource, l1QkBytes);
        BlockMmadPVGroupedD72 blockMmadPVGroupedD72(resource, l1QkBytes);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        if constexpr (PACKED_D128) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                ProcessAicPackedD128Task(taskIdx, blockMmadQKGrouped, blockMmadPVGroupedD72);
            }
            WaitAicEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (PACKED_D256) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    ProcessAicPackedD256Task(GroupedChunkTaskIndex(taskIdx, chunk), blockMmadQKSlicedAccum,
                                             blockMmadPVNormal, blockMmadPVDynamicSingleBlock,
                                             blockMmadPVDynamicMultiBlock);
                }
            }
            WaitAicEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (PACKED_D3SLICE) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    ProcessAicPackedD3SliceTask(GroupedChunkTaskIndex(taskIdx, chunk), blockMmadQKSlicedAccum,
                                                blockMmadPVNormal, blockMmadPVDynamicSingleBlock,
                                                blockMmadPVDynamicMultiBlock);
                }
            }
            WaitAicEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (PACKED_D256_TEXT) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                if (IsGroupedImageJob(taskIdx)) {
                    for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                        ProcessAicPackedD256Task<true>(GroupedChunkTaskIndex(taskIdx, chunk), blockMmadQKSlicedAccum,
                                                       blockMmadPVNormal, blockMmadPVDynamicSingleBlock,
                                                       blockMmadPVDynamicMultiBlock);
                    }
                } else {
                    ProcessAicTask(MixedTextTaskIndex(taskIdx), blockMmadQK, blockMmadQKPaged, sGm, false,
                                   blockMmadQKSlicedAccum, blockMmadPVNormal, blockMmadPVPaged,
                                   blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                    AscendC::CrossCoreWaitFlag(DYNAMIC_TASK_DONE_FLAG_ID);
                }
            }
            WaitAicEvents();
            AscendC::SyncAll<false>();
            return;
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
            if (UsesGroupedQkPath()) {
                if (IsGroupedImageJob(taskIdx)) {
                    ProcessAicGroupedTask(taskIdx, blockMmadQKGrouped, blockMmadPVNormal, blockMmadPVDynamicSingleBlock,
                                          blockMmadPVDynamicMultiBlock, blockMmadPVGroupedD72);
                } else {
                    ProcessAicTask(MixedTextTaskIndex(taskIdx), blockMmadQK, blockMmadQKPaged, sGm, false,
                                   blockMmadQKSlicedAccum, blockMmadPVNormal, blockMmadPVPaged,
                                   blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                }
                continue;
            }
            uint32_t chunkCount = UsesGroupedTileJobs() ? QUERY_CHUNKS_PER_TILE : 1;
            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
                uint32_t chunkTaskIdx = UsesGroupedTileJobs() ? GroupedChunkTaskIndex(taskIdx, chunk) : taskIdx;
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
                const bool debugFirstD256Task =
                    meta.dim > FAI_QK_SLICE_DIM && chunkTaskIdx < coreNum * QUERY_CHUNKS_PER_TILE;
#endif
                ProcessAicTask(chunkTaskIdx, blockMmadQK, blockMmadQKPaged, sGm, false, blockMmadQKSlicedAccum,
                               blockMmadPVNormal, blockMmadPVPaged, blockMmadPVDynamicSingleBlock,
                               blockMmadPVDynamicMultiBlock);
                if (meta.dim > FAI_QK_SLICE_DIM) {
                    AscendC::CrossCoreWaitFlag(DYNAMIC_TASK_DONE_FLAG_ID);
                }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
                if (debugFirstD256Task) {
                    AscendC::printf("STA_AIC_STAGE phase=dynamic_done dim=%u task=%u block=%u\\n",
                                    static_cast<uint32_t>(meta.dim), chunkTaskIdx, AscendC::GetBlockIdx());
                }
#endif
            }
        }
        WaitAicEvents();
        AscendC::SyncAll<false>();
    }

    template <>
    CATLASS_DEVICE void Process<AscendC::AIV>()
    {
        const float activeSoftmaxScale = meta.dim == 64 ? D64_SOFTMAX_SCALE : softmaxScale;
        InitAivEvents();
        if constexpr (PACKED_D72_FULL) {
            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, activeSoftmaxScale);
            EpilogueRescaleO epilogueRescaleO(resource);
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            uint32_t coreNum = AscendC::GetBlockNum();
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                ProcessAivGroupedTask(taskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
            }
            WaitAivEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if (UsesCompactFullWindowPath()) {
            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, activeSoftmaxScale);
            EpilogueRescaleO epilogueRescaleO(resource);
            uint32_t subBlockNum = AscendC::GetSubBlockNum();
            uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            uint32_t coreNum = AscendC::GetBlockNum();
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    ProcessAivCompactTask(GroupedChunkTaskIndex(taskIdx, chunk), epilogueOnlineSoftmax,
                                          epilogueRescaleO);
                }
            }
            WaitAivEvents();
            // Compact path must match the non-compact epilogue and drain all local
            // producers before the next launch can reuse L0/L1 resources.
            AscendC::SyncAll<false>();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::printf("STA_LIFECYCLE AIV_DONE dim=%u jobs=%u\\n", static_cast<uint32_t>(meta.dim),
                                static_cast<uint32_t>(totalJobs));
            }
#endif
            return;
        }
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, activeSoftmaxScale);
        EpilogueRescaleO epilogueRescaleO(resource);

        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t coreNum = AscendC::GetBlockNum();
        if (meta.pair_axis >= 0 && UsesGroupedD16QkPath()) {
            if constexpr (NEEDS_INPUT_STAGING) {
                // BF16 pair jobs consume the same immutable staged-V tensor as the
                // generic path. Publish it before entering the pair event lifecycle
                // so this collective matches the AIC entrance collective.
                StageAllV();
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SyncAll<false>();
            }
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                if (IsGroupedImageJob(taskIdx)) {
                    ProcessAivGroupedPairD16Task(taskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
                }
            }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIV phase=image_done\n");
            }
#endif
            AscendC::SyncAll<false>();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIV phase=sync1_done\n");
            }
#endif
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                if (!IsGroupedImageJob(taskIdx)) {
                    if (UsesGroupedTextD16Path()) {
                        if (IsFirstGroupedTextJob(taskIdx)) {
                            ProcessAivGroupedTextD16Task(taskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
                        }
                    } else {
                        ProcessAivTask(MixedTextTaskIndex(taskIdx), epilogueOnlineSoftmax, epilogueRescaleO);
                    }
                }
            }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIV phase=text_done\n");
            }
#endif
            WaitAivEvents();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIV phase=events_done\n");
            }
#endif
            AscendC::SyncAll<false>();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
            if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
                AscendC::printf("STA_PAIR_STAGE engine=AIV phase=sync2_done\n");
            }
#endif
            return;
        }
        if constexpr (PACKED_D128) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                ProcessAivPackedD128Task(taskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
            }
            WaitAivEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (PACKED_D256) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    ProcessAivPackedD256Task(GroupedChunkTaskIndex(taskIdx, chunk), epilogueOnlineSoftmax,
                                             epilogueRescaleO);
                }
            }
            WaitAivEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (PACKED_D3SLICE) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    ProcessAivPackedD3SliceTask(GroupedChunkTaskIndex(taskIdx, chunk), epilogueOnlineSoftmax,
                                                epilogueRescaleO);
                }
            }
            WaitAivEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (PACKED_D256_TEXT) {
            for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
                if (IsGroupedImageJob(taskIdx)) {
                    for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                        ProcessAivPackedD256Task<true>(GroupedChunkTaskIndex(taskIdx, chunk), epilogueOnlineSoftmax,
                                                       epilogueRescaleO);
                    }
                } else {
                    ProcessAivTask(MixedTextTaskIndex(taskIdx), epilogueOnlineSoftmax, epilogueRescaleO);
                    AscendC::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(DYNAMIC_TASK_DONE_FLAG_ID);
                }
            }
            WaitAivEvents();
            AscendC::SyncAll<false>();
            return;
        }
        if constexpr (NEEDS_INPUT_STAGING) {
            StageAllV();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
            if (UsesGroupedQkPath()) {
                if (IsGroupedImageJob(taskIdx)) {
                    ProcessAivGroupedTask(taskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
                } else {
                    ProcessAivTask(MixedTextTaskIndex(taskIdx), epilogueOnlineSoftmax, epilogueRescaleO);
                }
                continue;
            }
            uint32_t chunkCount = UsesGroupedTileJobs() ? QUERY_CHUNKS_PER_TILE : 1;
            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
                uint32_t chunkTaskIdx = UsesGroupedTileJobs() ? GroupedChunkTaskIndex(taskIdx, chunk) : taskIdx;
                ProcessAivTask(chunkTaskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
                if (meta.dim > FAI_QK_SLICE_DIM) {
                    AscendC::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(DYNAMIC_TASK_DONE_FLAG_ID);
                }
            }
        }
        WaitAivEvents();
        AscendC::SyncAll<false>();
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 0) {
            AscendC::printf("STA_LIFECYCLE AIV_DONE dim=%u jobs=%u\\n", static_cast<uint32_t>(meta.dim),
                            static_cast<uint32_t>(totalJobs));
        }
#endif
    }

private:
    CATLASS_DEVICE bool UsesGroupedQkPath() const
    {
        if constexpr (PACKED_D72_FULL) {
            return true;
        }
        if (UsesGroupedD16QkPath()) {
            return true;
        }
        if constexpr (!std::is_same_v<ElementInput, half> || !std::is_same_v<ElementO, half>) {
            return false;
        }
        return UsesGroupedD8QkPath() || (meta.has_text == 0 && meta.tile_numel == TILE_ELEMENTS && meta.dim > 64 &&
                                         meta.dim <= FAI_QK_SLICE_DIM && !UsesCompactFullWindowPath());
    }

    CATLASS_DEVICE bool UsesGroupedD16QkPath() const
    {
        return meta.tile_numel == TILE_ELEMENTS && meta.dim == 16;
    }

    CATLASS_DEVICE bool UsesD128GroupedEarlyPublishPath() const
    {
        if constexpr (!std::is_same_v<ElementInput, half> || !std::is_same_v<ElementO, half>) {
            return false;
        }
        return meta.has_text == 0 && meta.tile_numel == TILE_ELEMENTS && meta.dim == 128 &&
               !UsesCompactFullWindowPath();
    }

    CATLASS_DEVICE bool UsesGroupedD8QkPath() const
    {
        if constexpr (!std::is_same_v<ElementInput, half> || !std::is_same_v<ElementO, half>) {
            return false;
        }
        return meta.has_text == 0 && meta.tile_numel == TILE_ELEMENTS && meta.dim == 8;
    }

    CATLASS_DEVICE bool UsesGroupedD16PvPath() const
    {
        return meta.tile_numel == TILE_ELEMENTS && meta.dim == 16 && !UsesCompactFullWindowPath();
    }

    CATLASS_DEVICE bool UsesGroupedD72PvPath() const
    {
        if constexpr (PACKED_D72_FULL) {
            return true;
        }
        // Other D72 windows retain the independently validated dynamic-PV path.
        return false;
    }

    CATLASS_DEVICE bool UsesPackedD72ImageStacks() const
    {
        // D16 image segments contain an integral number of 128-row blocks, so
        // adjacent segments may share a four-block stack without changing the
        // lexicographic attention order or introducing padded rows.
        return UsesGroupedD72PvPath() || UsesGroupedD16PvPath();
    }

    CATLASS_DEVICE bool UsesPackedD128N1ImageStacks() const
    {
        if constexpr (!PACKED_D128) {
            return false;
        }
        if constexpr (!std::is_same_v<ElementInput, half> || !std::is_same_v<ElementO, half>) {
            return false;
        }
        return meta.batch > 0 && meta.heads == 1 && meta.has_text == 0 && meta.text_len == 0 &&
               meta.tile_numel == TILE_ELEMENTS && meta.dim == 128;
    }

    CATLASS_DEVICE bool UsesCompactFullWindowPath() const
    {
        if constexpr (!std::is_same_v<ElementInput, half> || !std::is_same_v<ElementO, half>) {
            return false;
        }
        if (meta.has_text != 0 || meta.tile_numel != TILE_ELEMENTS || meta.dim <= 64 || meta.dim > FAI_QK_SLICE_DIM) {
            return false;
        }
        for (uint32_t head = 0; head < static_cast<uint32_t>(meta.heads); ++head) {
            uint32_t base = head * WINDOW_VALUES_PER_HEAD;
            if (meta.windows[base + WINDOW_TIME_OFFSET] < meta.grid_t ||
                meta.windows[base + WINDOW_HEIGHT_OFFSET] < meta.grid_h ||
                meta.windows[base + WINDOW_WIDTH_OFFSET] < meta.grid_w) {
                return false;
            }
        }
        return true;
    }

    CATLASS_DEVICE bool UsesGroupedTileJobs() const
    {
        if constexpr (PACKED_D256_TEXT) {
            return true;
        }
        if (UsesGroupedD16QkPath()) {
            return true;
        }
        // Host tiling uses grouped jobs for every FP16 no-text tile family.
        // Other BF16 paths retain one task per 128-row query chunk.
        if constexpr (!std::is_same_v<ElementInput, half> || !std::is_same_v<ElementO, half>) {
            return false;
        }
        return meta.tile_numel == TILE_ELEMENTS && meta.has_text == 0;
    }

    CATLASS_DEVICE uint32_t GroupedChunkTaskIndex(uint32_t groupedTaskIdx, uint32_t chunk) const
    {
        uint32_t tilesPerHead = static_cast<uint32_t>(meta.grid_t * meta.grid_h * meta.grid_w);
        uint32_t jobsPerHead = GroupedJobsPerHead();
        uint32_t localJob = groupedTaskIdx % jobsPerHead;
        uint32_t batchHead = groupedTaskIdx / jobsPerHead;
        uint32_t genericChunksPerHead = tilesPerHead * QUERY_CHUNKS_PER_TILE + TextChunksPerHead();
        if (meta.pair_axis < 0 || localJob >= PairImageJobsPerHead()) {
            if (UsesGroupedTextD16Path()) {
                uint32_t textGroup = localJob - PairImageJobsPerHead();
                return batchHead * genericChunksPerHead + tilesPerHead * QUERY_CHUNKS_PER_TILE +
                       textGroup * D16_PAIR_QUERY_COUNT + chunk;
            }
            return batchHead * genericChunksPerHead + localJob * QUERY_CHUNKS_PER_TILE + chunk;
        }
        uint32_t tile = PairTileId(localJob, chunk / QUERY_CHUNKS_PER_TILE);
        return batchHead * genericChunksPerHead + tile * QUERY_CHUNKS_PER_TILE + chunk % QUERY_CHUNKS_PER_TILE;
    }

    CATLASS_DEVICE uint32_t TextChunksPerHead() const
    {
        return meta.has_text != 0 ? StaCeilDiv(static_cast<uint32_t>(meta.seq - meta.image_seq), FAI_QUERY_ROWS) : 0;
    }

    CATLASS_DEVICE bool UsesGroupedTextD16Path() const
    {
        // Correctness-clean but rejected: eval_20260829_231103 regressed active time.
        return false;
    }

    CATLASS_DEVICE uint32_t TextJobsPerHead() const
    {
        return UsesGroupedTextD16Path() ? 1U : TextChunksPerHead();
    }

    CATLASS_DEVICE uint32_t GroupedJobsPerHead() const
    {
        return PairImageJobsPerHead() + TextJobsPerHead();
    }

    CATLASS_DEVICE uint32_t PairImageJobsPerHead() const
    {
        uint32_t tiles = static_cast<uint32_t>(meta.grid_t * meta.grid_h * meta.grid_w);
        if (meta.pair_axis < 0) {
            return tiles;
        }
        uint32_t axisExtent = meta.pair_axis == 0 ? static_cast<uint32_t>(meta.grid_t) :
                                                    (meta.pair_axis == 1 ? static_cast<uint32_t>(meta.grid_h) :
                                                                           static_cast<uint32_t>(meta.grid_w));
        return tiles / axisExtent * static_cast<uint32_t>(meta.pair_group_count);
    }

    CATLASS_DEVICE uint32_t PairTileId(uint32_t localJob, uint32_t member) const
    {
        uint32_t groupCount = static_cast<uint32_t>(meta.pair_group_count);
        uint32_t group = localJob % groupCount;
        uint32_t outer = localJob / groupCount;
        uint32_t axisCoord = static_cast<uint32_t>(meta.pair_group_start[group]) + member;
        uint32_t t = 0;
        uint32_t h = 0;
        uint32_t w = 0;
        if (meta.pair_axis == 2) {
            h = outer % static_cast<uint32_t>(meta.grid_h);
            t = outer / static_cast<uint32_t>(meta.grid_h);
            w = axisCoord;
        } else if (meta.pair_axis == 1) {
            w = outer % static_cast<uint32_t>(meta.grid_w);
            t = outer / static_cast<uint32_t>(meta.grid_w);
            h = axisCoord;
        } else {
            w = outer % static_cast<uint32_t>(meta.grid_w);
            h = outer / static_cast<uint32_t>(meta.grid_w);
            t = axisCoord;
        }
        return (t * static_cast<uint32_t>(meta.grid_h) + h) * static_cast<uint32_t>(meta.grid_w) + w;
    }

    CATLASS_DEVICE uint32_t GroupedQueryCount(uint32_t groupedTaskIdx) const
    {
        uint32_t localJob = groupedTaskIdx % GroupedJobsPerHead();
        if (meta.pair_axis < 0 || localJob >= PairImageJobsPerHead()) {
            return QUERY_CHUNKS_PER_TILE;
        }
        uint32_t group = localJob % static_cast<uint32_t>(meta.pair_group_count);
        return static_cast<uint32_t>(meta.pair_group_span[group]) * QUERY_CHUNKS_PER_TILE;
    }

    CATLASS_DEVICE bool IsGroupedImageJob(uint32_t groupedTaskIdx) const
    {
        uint32_t localJob = groupedTaskIdx % GroupedJobsPerHead();
        return localJob < PairImageJobsPerHead();
    }

    CATLASS_DEVICE bool IsFirstGroupedTextJob(uint32_t groupedTaskIdx) const
    {
        uint32_t localJob = groupedTaskIdx % GroupedJobsPerHead();
        return UsesGroupedTextD16Path() && localJob == PairImageJobsPerHead();
    }

    CATLASS_DEVICE uint32_t MixedTextTaskIndex(uint32_t groupedTaskIdx) const
    {
        uint32_t tilesPerHead = static_cast<uint32_t>(meta.grid_t * meta.grid_h * meta.grid_w);
        uint32_t jobsPerHead = GroupedJobsPerHead();
        uint32_t localJob = groupedTaskIdx % jobsPerHead;
        uint32_t batchHead = groupedTaskIdx / jobsPerHead;
        uint32_t genericChunksPerHead = tilesPerHead * QUERY_CHUNKS_PER_TILE + TextChunksPerHead();
        uint32_t textChunk = localJob - PairImageJobsPerHead();
        return batchHead * genericChunksPerHead + tilesPerHead * QUERY_CHUNKS_PER_TILE + textChunk;
    }

    struct TaskInfo {
        uint32_t batch = 0;
        uint32_t head = 0;
        uint32_t qStart = 0;
        uint32_t qRows = 0;
        uint32_t qBlocks = 0;
        uint32_t qTileId = 0;
        uint32_t isText = 0;
    };

    struct SegmentInfo {
        uint32_t start = 0;
        uint32_t len = 0;
    };

    struct WindowRange {
        uint32_t tStart = 0;
        uint32_t tEnd = 0;
        uint32_t hStart = 0;
        uint32_t hEnd = 0;
        uint32_t wStart = 0;
        uint32_t wEnd = 0;
    };

    struct StackInfo {
        uint64_t kvOffset = 0;
        uint64_t computeKvOffset = 0;
        uint64_t computeVOffset = 0;
        uint32_t segLen = 0;
        uint32_t kvLoopNoMask = 0;
        uint32_t kvIdx = 0;
        uint32_t computeSegLen = 0;
        uint32_t computeKvLoopNoMask = 0;
        uint32_t computeKvIdx = 0;
        uint32_t stackSeqTile = 0;
        uint32_t stackSeqCount = 0;
        uint64_t packedBlockOffsets[FAI_STACK_BLOCK_COUNT] = {};
        uint32_t packedBlockIds[FAI_STACK_BLOCK_COUNT] = {};
        uint32_t packedBlockCount = 0;
        bool usesPagedTable = false;
    };

    CATLASS_DEVICE TaskInfo DecodeTask(uint32_t taskIdx) const
    {
        if (meta.tile_numel == TILE_ELEMENTS && meta.heads == FAST_PATH_HEAD_COUNT && meta.dim == 64 &&
            meta.grid_h == GRID_HEIGHT_TILES) {
            if (meta.grid_t == FASTVIDEO_GRID_TIME_TILES && meta.grid_w == FASTVIDEO_GRID_WIDTH_TILES &&
                meta.has_text != 0 && meta.text_len == FAI_QUERY_ROWS) {
                return DecodeTaskFixed384<FASTVIDEO_QUERY_BLOCKS_PER_HEAD, FASTVIDEO_IMAGE_CHUNKS_PER_HEAD>(taskIdx);
            }
            if (meta.grid_t == STEPVIDEO_GRID_TIME_TILES && meta.grid_w == STEPVIDEO_GRID_WIDTH_TILES &&
                meta.has_text == 0) {
                return DecodeTaskFixed384<STEPVIDEO_QUERY_BLOCKS_PER_HEAD, STEPVIDEO_QUERY_BLOCKS_PER_HEAD>(taskIdx);
            }
            if (meta.grid_t == WAN_GRID_TIME_TILES && meta.grid_w == WAN_GRID_WIDTH_TILES && meta.has_text == 0) {
                return DecodeTaskFixed384<WAN_QUERY_BLOCKS_PER_HEAD, WAN_QUERY_BLOCKS_PER_HEAD>(taskIdx);
            }
        }
        TaskInfo info;
        uint32_t chunksPerTile = StaCeilDiv(static_cast<uint32_t>(meta.tile_numel), FAI_QUERY_ROWS);
        uint32_t numTiles = static_cast<uint32_t>(meta.grid_t * meta.grid_h * meta.grid_w);
        uint32_t imageChunksPerHead = numTiles * chunksPerTile;
        uint32_t textChunksPerHead =
            meta.has_text != 0 ? StaCeilDiv(static_cast<uint32_t>(meta.seq - meta.image_seq), FAI_QUERY_ROWS) : 0;
        info.qBlocks = imageChunksPerHead + textChunksPerHead;
        uint32_t localChunk = taskIdx % info.qBlocks;
        uint32_t tmp = taskIdx / info.qBlocks;
        info.head = tmp % static_cast<uint32_t>(meta.heads);
        info.batch = tmp / static_cast<uint32_t>(meta.heads);
        if (localChunk < imageChunksPerHead) {
            uint32_t tileId = localChunk / chunksPerTile;
            uint32_t chunkInTile = localChunk - tileId * chunksPerTile;
            uint32_t tileOffset = chunkInTile * FAI_QUERY_ROWS;
            info.qTileId = tileId;
            info.qStart = tileId * static_cast<uint32_t>(meta.tile_numel) + tileOffset;
            info.qRows = StaMin(FAI_QUERY_ROWS, static_cast<uint32_t>(meta.tile_numel) - tileOffset);
            info.isText = 0;
        } else {
            uint32_t textChunk = localChunk - imageChunksPerHead;
            uint32_t textOffset = textChunk * FAI_QUERY_ROWS;
            info.qStart = static_cast<uint32_t>(meta.image_seq) + textOffset;
            info.qRows = StaMin(FAI_QUERY_ROWS, static_cast<uint32_t>(meta.seq - meta.image_seq) - textOffset);
            info.isText = 1;
        }
        return info;
    }

    template <uint32_t QBlocksPerHead, uint32_t ImageChunksPerHead>
    CATLASS_DEVICE TaskInfo DecodeTaskFixed384(uint32_t taskIdx) const
    {
        TaskInfo info;
        info.qBlocks = QBlocksPerHead;
        uint32_t localChunk = taskIdx % QBlocksPerHead;
        uint32_t tmp = taskIdx / QBlocksPerHead;
        info.head = tmp & FAST_PATH_HEAD_INDEX_MASK;
        info.batch = tmp >> FAST_PATH_HEAD_INDEX_SHIFT;
        if (localChunk < ImageChunksPerHead) {
            uint32_t tileId = localChunk / QUERY_CHUNKS_PER_TILE;
            uint32_t chunkInTile = localChunk - tileId * QUERY_CHUNKS_PER_TILE;
            info.qTileId = tileId;
            info.qStart = tileId * TILE_ELEMENTS + chunkInTile * FAI_QUERY_ROWS;
            info.qRows = FAI_QUERY_ROWS;
            info.isText = 0;
        } else {
            info.qStart = static_cast<uint32_t>(meta.image_seq);
            info.qRows = FAI_QUERY_ROWS;
            info.isText = 1;
        }
        return info;
    }

    CATLASS_DEVICE uint32_t ClampCenterU32(uint32_t coord, uint32_t window, uint32_t extent) const
    {
        uint32_t radius = window / 2;
        uint32_t maxCenter = extent - 1 - radius;
        if (coord < radius) {
            return radius;
        }
        return coord > maxCenter ? maxCenter : coord;
    }

    CATLASS_DEVICE WindowRange GetWindowRange(const TaskInfo &info) const
    {
        WindowRange range;
        uint32_t gridT = static_cast<uint32_t>(meta.grid_t);
        uint32_t gridH = static_cast<uint32_t>(meta.grid_h);
        uint32_t gridW = static_cast<uint32_t>(meta.grid_w);
        uint32_t qt = info.qTileId / (gridH * gridW);
        uint32_t rem = info.qTileId - qt * gridH * gridW;
        uint32_t qh = rem / gridW;
        uint32_t qw = rem - qh * gridW;
        const uint32_t windowBase = info.head * WINDOW_VALUES_PER_HEAD;
        uint32_t winT = WindowExtent(meta.windows[windowBase + WINDOW_TIME_OFFSET], gridT);
        uint32_t winH = WindowExtent(meta.windows[windowBase + WINDOW_HEIGHT_OFFSET], gridH);
        uint32_t winW = WindowExtent(meta.windows[windowBase + WINDOW_WIDTH_OFFSET], gridW);
        if (meta.tile_numel == TILE_ELEMENTS && gridH == GRID_HEIGHT_TILES && ((winT & 1U) != 0) &&
            ((winH & 1U) != 0) && ((winW & 1U) != 0)) {
            range.tStart = OddWindowStart(qt, winT, gridT);
            range.hStart = OddWindowStart(qh, winH, gridH);
            range.wStart = OddWindowStart(qw, winW, gridW);
            range.tEnd = range.tStart + StaMin(winT, gridT);
            range.hEnd = range.hStart + StaMin(winH, gridH);
            range.wEnd = range.wStart + StaMin(winW, gridW);
            return range;
        }
        uint32_t ct = ClampCenterU32(qt, winT, gridT);
        uint32_t ch = ClampCenterU32(qh, winH, gridH);
        uint32_t cw = ClampCenterU32(qw, winW, gridW);
        uint32_t rt = winT / 2;
        uint32_t rh = winH / 2;
        uint32_t rw = winW / 2;
        range.tStart = ct > rt ? ct - rt : 0;
        range.hStart = ch > rh ? ch - rh : 0;
        range.wStart = cw > rw ? cw - rw : 0;
        range.tEnd = StaMin(ct + rt + 1, gridT);
        range.hEnd = StaMin(ch + rh + 1, gridH);
        range.wEnd = StaMin(cw + rw + 1, gridW);
        return range;
    }

    CATLASS_DEVICE uint32_t WindowExtent(int64_t window, uint32_t extent) const
    {
        return window >= static_cast<int64_t>(extent) ? extent : static_cast<uint32_t>(window);
    }

    CATLASS_DEVICE uint32_t OddWindowStart(uint32_t coord, uint32_t window, uint32_t extent) const
    {
        if (window >= extent) {
            return 0;
        }
        uint32_t radius = window / 2;
        if (coord <= radius) {
            return 0;
        }
        uint32_t start = coord - radius;
        uint32_t maxStart = extent - window;
        return start > maxStart ? maxStart : start;
    }

    CATLASS_DEVICE uint32_t SegmentCount(const TaskInfo &info) const
    {
        WindowRange range;
        if (info.isText == 0) {
            range = GetWindowRange(info);
        }
        return SegmentCountFromRange(info, range);
    }

    CATLASS_DEVICE SegmentInfo GetSegment(const TaskInfo &info, uint32_t segmentIdx) const
    {
        WindowRange range;
        if (info.isText == 0) {
            range = GetWindowRange(info);
        }
        return GetSegmentFromRange(info, range, segmentIdx);
    }

    CATLASS_DEVICE uint32_t SegmentCountFromRange(const TaskInfo &info, const WindowRange &range) const
    {
        if (info.isText != 0) {
            return 1;
        }
        uint32_t count = (range.tEnd - range.tStart) * (range.hEnd - range.hStart);
        if (meta.has_text != 0 && meta.text_len > 0) {
            ++count;
        }
        return count;
    }

    CATLASS_DEVICE SegmentInfo GetSegmentFromRange(const TaskInfo &info, const WindowRange &range,
                                                   uint32_t segmentIdx) const
    {
        SegmentInfo seg;
        if (info.isText != 0) {
            seg.start = 0;
            seg.len = static_cast<uint32_t>(meta.image_seq + meta.text_len);
            return seg;
        }

        uint32_t gridH = static_cast<uint32_t>(meta.grid_h);
        uint32_t gridW = static_cast<uint32_t>(meta.grid_w);
        uint32_t tileNumel = static_cast<uint32_t>(meta.tile_numel);
        uint32_t hSpan = range.hEnd - range.hStart;
        uint32_t imageSegments = (range.tEnd - range.tStart) * hSpan;
        if (segmentIdx >= imageSegments) {
            seg.start = static_cast<uint32_t>(meta.image_seq);
            seg.len = static_cast<uint32_t>(meta.text_len);
            return seg;
        }
        uint32_t relT = DivBySmallSpan(segmentIdx, hSpan);
        uint32_t relH = segmentIdx - relT * hSpan;
        uint32_t t = range.tStart + relT;
        uint32_t h = range.hStart + relH;
        uint32_t tileId = (t * gridH + h) * gridW + range.wStart;
        seg.start = tileId * tileNumel;
        seg.len = (range.wEnd - range.wStart) * tileNumel;
        return seg;
    }

    CATLASS_DEVICE uint32_t DivBySmallSpan(uint32_t value, uint32_t span) const
    {
        if (span == HEIGHT_SPAN_SINGLE_TILE) {
            return value;
        }
        if (span == HEIGHT_SPAN_THREE_TILES) {
            return value / HEIGHT_SPAN_THREE_TILES;
        }
        if (span == HEIGHT_SPAN_FIVE_TILES) {
            return value / HEIGHT_SPAN_FIVE_TILES;
        }
        if (span == HEIGHT_SPAN_FULL_GRID) {
            return value / HEIGHT_SPAN_FULL_GRID;
        }
        return value / span;
    }

    CATLASS_DEVICE uint32_t TotalStackSeqNum(const TaskInfo &info) const
    {
        WindowRange range;
        if (info.isText == 0) {
            range = GetWindowRange(info);
        }
        return TotalStackSeqNumFromRange(info, range);
    }

    CATLASS_DEVICE uint32_t TotalStackSeqNumFromRange(const TaskInfo &info, const WindowRange &range) const
    {
        uint32_t stackTile = FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS;
        if (info.isText != 0) {
            return StaCeilDiv(static_cast<uint32_t>(meta.image_seq + meta.text_len), stackTile);
        }
        if (UsesPagedD64ImageStacks()) {
            uint32_t imageStacks = StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
            uint32_t textStacks = StaCeilDiv(static_cast<uint32_t>(meta.text_len), stackTile);
            return imageStacks + textStacks;
        }
        if (UsesPackedD72ImageStacks()) {
            uint32_t imageStacks = StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
            uint32_t textStacks = StaCeilDiv(static_cast<uint32_t>(meta.text_len), stackTile);
            return imageStacks + textStacks;
        }
        uint32_t imageSegments = (range.tEnd - range.tStart) * (range.hEnd - range.hStart);
        uint32_t imageSegLen = (range.wEnd - range.wStart) * static_cast<uint32_t>(meta.tile_numel);
        uint32_t total = imageSegments * StackCountForSegment(imageSegLen, stackTile);
        if (meta.has_text != 0 && meta.text_len > 0) {
            total += StaCeilDiv(static_cast<uint32_t>(meta.text_len), stackTile);
        }
        return total;
    }

    CATLASS_DEVICE uint32_t PackedImageBlockCount(const WindowRange &range) const
    {
        uint32_t segmentCount = (range.tEnd - range.tStart) * (range.hEnd - range.hStart);
        uint32_t blocksPerSegment =
            (range.wEnd - range.wStart) * static_cast<uint32_t>(meta.tile_numel) / FAI_KV_BASE_ROWS;
        return segmentCount * blocksPerSegment;
    }

    CATLASS_DEVICE StackInfo MakePackedImageStack(const TaskInfo &info, const WindowRange &range,
                                                  uint32_t stackSeqCount, uint64_t headBase, uint32_t embed) const
    {
        StackInfo stack;
        uint32_t blocksPerSegment =
            (range.wEnd - range.wStart) * static_cast<uint32_t>(meta.tile_numel) / FAI_KV_BASE_ROWS;
        uint32_t totalBlocks = PackedImageBlockCount(range);
        uint32_t firstBlock = stackSeqCount * FAI_STACK_BLOCK_COUNT;
        stack.packedBlockCount = StaMin(FAI_STACK_BLOCK_COUNT, totalBlocks - firstBlock);
        uint32_t segmentIdx = firstBlock / blocksPerSegment;
        uint32_t blockInSegment = firstBlock - segmentIdx * blocksPerSegment;
        SegmentInfo segment = GetSegmentFromRange(info, range, segmentIdx);
        for (uint32_t block = 0; block < stack.packedBlockCount; ++block) {
            uint32_t blockStart = segment.start + blockInSegment * FAI_KV_BASE_ROWS;
            stack.packedBlockOffsets[block] = headBase + static_cast<uint64_t>(blockStart) * embed;
            stack.packedBlockIds[block] = blockStart / FAI_KV_BASE_ROWS;
            ++blockInSegment;
            if (block + 1 < stack.packedBlockCount && blockInSegment == blocksPerSegment) {
                blockInSegment = 0;
                segment = GetSegmentFromRange(info, range, ++segmentIdx);
            }
        }
        stack.segLen = stack.packedBlockCount * FAI_KV_BASE_ROWS;
        stack.computeSegLen = stack.segLen;
        stack.kvLoopNoMask = stack.packedBlockCount;
        stack.computeKvLoopNoMask = stack.packedBlockCount;
        stack.kvIdx = 0;
        stack.computeKvIdx = 0;
        stack.stackSeqTile = stack.segLen;
        stack.stackSeqCount = stackSeqCount;
        stack.kvOffset = headBase;
        stack.computeKvOffset = headBase;
        stack.computeVOffset = headBase;
        stack.usesPagedTable = UsesPagedD64ImageStacks();
        return stack;
    }

    CATLASS_DEVICE uint32_t FullTextStackCount() const
    {
        uint32_t validRows = static_cast<uint32_t>(meta.image_seq + meta.text_len);
        return StaCeilDiv(validRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
    }

    CATLASS_DEVICE StackInfo MakeFullTextStack(uint32_t stackSeqCount, uint64_t headBase, uint32_t embed) const
    {
        StackInfo stack;
        uint32_t firstRow = stackSeqCount * FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS;
        uint32_t validRows = static_cast<uint32_t>(meta.image_seq + meta.text_len);
        stack.stackSeqTile = StaMin(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, validRows - firstRow);
        stack.packedBlockCount = StaCeilDiv(stack.stackSeqTile, FAI_KV_BASE_ROWS);
        for (uint32_t block = 0; block < stack.packedBlockCount; ++block) {
            stack.packedBlockOffsets[block] =
                headBase + static_cast<uint64_t>(firstRow + block * FAI_KV_BASE_ROWS) * embed;
            stack.packedBlockIds[block] = (firstRow + block * FAI_KV_BASE_ROWS) / FAI_KV_BASE_ROWS;
        }
        stack.segLen = stack.stackSeqTile;
        stack.computeSegLen = stack.stackSeqTile;
        stack.kvLoopNoMask = stack.packedBlockCount;
        stack.computeKvLoopNoMask = stack.packedBlockCount;
        stack.stackSeqCount = stackSeqCount;
        stack.kvOffset = headBase + static_cast<uint64_t>(firstRow) * embed;
        stack.computeKvOffset = stack.kvOffset;
        stack.computeVOffset = stack.kvOffset;
        return stack;
    }

    CATLASS_DEVICE uint32_t PackedImageTextStackCount(const WindowRange &range) const
    {
        uint32_t imageStacks = StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
        uint32_t textStacks =
            StaCeilDiv(static_cast<uint32_t>(meta.text_len), FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
        return imageStacks + textStacks;
    }

    CATLASS_DEVICE StackInfo MakePackedImageOrTextStack(const TaskInfo &info, const WindowRange &range,
                                                        uint32_t stackSeqCount, uint64_t headBase, uint32_t embed) const
    {
        uint32_t imageStacks = StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
        if (stackSeqCount < imageStacks) {
            return MakePackedImageStack(info, range, stackSeqCount, headBase, embed);
        }
        StackInfo stack;
        uint32_t textStackIdx = stackSeqCount - imageStacks;
        uint32_t firstTextBlock = textStackIdx * FAI_STACK_BLOCK_COUNT;
        uint32_t totalTextBlocks = StaCeilDiv(static_cast<uint32_t>(meta.text_len), FAI_KV_BASE_ROWS);
        stack.packedBlockCount = StaMin(FAI_STACK_BLOCK_COUNT, totalTextBlocks - firstTextBlock);
        uint32_t firstTextRow = firstTextBlock * FAI_KV_BASE_ROWS;
        stack.stackSeqTile =
            StaMin(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, static_cast<uint32_t>(meta.text_len) - firstTextRow);
        for (uint32_t block = 0; block < stack.packedBlockCount; ++block) {
            uint32_t blockStart = static_cast<uint32_t>(meta.image_seq) + (firstTextBlock + block) * FAI_KV_BASE_ROWS;
            stack.packedBlockOffsets[block] = headBase + static_cast<uint64_t>(blockStart) * embed;
            stack.packedBlockIds[block] = blockStart / FAI_KV_BASE_ROWS;
        }
        stack.segLen = stack.stackSeqTile;
        stack.computeSegLen = stack.stackSeqTile;
        stack.kvLoopNoMask = stack.packedBlockCount;
        stack.computeKvLoopNoMask = stack.packedBlockCount;
        stack.stackSeqCount = stackSeqCount;
        stack.kvOffset = headBase + static_cast<uint64_t>(meta.image_seq) * embed;
        stack.computeKvOffset = stack.kvOffset;
        stack.computeVOffset = stack.kvOffset;
        return stack;
    }

    CATLASS_DEVICE uint32_t PackedCombinedImageTextStackCount(const WindowRange &range) const
    {
        uint32_t imageBlocks = PackedImageBlockCount(range);
        uint32_t textBlocks = StaCeilDiv(static_cast<uint32_t>(meta.text_len), FAI_KV_BASE_ROWS);
        return StaCeilDiv(imageBlocks + textBlocks, FAI_STACK_BLOCK_COUNT);
    }

    CATLASS_DEVICE StackInfo MakePackedCombinedImageTextStack(const TaskInfo &info, const WindowRange &range,
                                                              uint32_t stackSeqCount, uint64_t headBase,
                                                              uint32_t embed) const
    {
        uint32_t imageBlocks = PackedImageBlockCount(range);
        uint32_t totalTextBlocks = StaCeilDiv(static_cast<uint32_t>(meta.text_len), FAI_KV_BASE_ROWS);
        uint32_t firstBlock = stackSeqCount * FAI_STACK_BLOCK_COUNT;
        StackInfo stack;
        uint32_t firstTextBlock = 0;
        if (firstBlock < imageBlocks) {
            stack = MakePackedImageStack(info, range, stackSeqCount, headBase, embed);
            if (stack.packedBlockCount == FAI_STACK_BLOCK_COUNT || totalTextBlocks == 0) {
                return stack;
            }
        } else {
            firstTextBlock = firstBlock - imageBlocks;
        }

        uint32_t imageBlockCount = stack.packedBlockCount;
        uint32_t textBlockCapacity = FAI_STACK_BLOCK_COUNT - imageBlockCount;
        uint32_t textBlockCount = StaMin(textBlockCapacity, totalTextBlocks - firstTextBlock);
        uint32_t firstTextRow = firstTextBlock * FAI_KV_BASE_ROWS;
        uint32_t textRows =
            StaMin(textBlockCount * FAI_KV_BASE_ROWS, static_cast<uint32_t>(meta.text_len) - firstTextRow);
        for (uint32_t block = 0; block < textBlockCount; ++block) {
            uint32_t packedBlock = imageBlockCount + block;
            uint32_t blockStart = static_cast<uint32_t>(meta.image_seq) + (firstTextBlock + block) * FAI_KV_BASE_ROWS;
            stack.packedBlockOffsets[packedBlock] = headBase + static_cast<uint64_t>(blockStart) * embed;
            stack.packedBlockIds[packedBlock] = blockStart / FAI_KV_BASE_ROWS;
        }
        stack.packedBlockCount = imageBlockCount + textBlockCount;
        stack.stackSeqTile = imageBlockCount * FAI_KV_BASE_ROWS + textRows;
        stack.segLen = stack.stackSeqTile;
        stack.computeSegLen = stack.stackSeqTile;
        stack.kvLoopNoMask = stack.packedBlockCount;
        stack.computeKvLoopNoMask = stack.packedBlockCount;
        stack.kvIdx = 0;
        stack.computeKvIdx = 0;
        stack.stackSeqCount = stackSeqCount;
        stack.kvOffset = imageBlockCount == 0 ? headBase + static_cast<uint64_t>(meta.image_seq) * embed : headBase;
        stack.computeKvOffset = stack.kvOffset;
        stack.computeVOffset = stack.kvOffset;
        return stack;
    }

    CATLASS_DEVICE StackInfo MakePackedWindow3D256Stack(const WindowRange &range, uint32_t stackSeqCount,
                                                        uint64_t headBase) const
    {
        constexpr uint32_t BLOCKS_PER_SEGMENT = 9;
        constexpr uint32_t SEGMENTS_PER_TIME = 3;
        constexpr uint32_t TOTAL_BLOCKS = 81;
        constexpr uint32_t EMBED = 256;
        StackInfo stack;
        uint32_t firstBlock = stackSeqCount * FAI_STACK_BLOCK_COUNT;
        stack.packedBlockCount = StaMin(FAI_STACK_BLOCK_COUNT, TOTAL_BLOCKS - firstBlock);
        for (uint32_t block = 0; block < stack.packedBlockCount; ++block) {
            uint32_t ordinal = firstBlock + block;
            uint32_t segmentIdx = ordinal / BLOCKS_PER_SEGMENT;
            uint32_t blockInSegment = ordinal - segmentIdx * BLOCKS_PER_SEGMENT;
            uint32_t relT = segmentIdx / SEGMENTS_PER_TIME;
            uint32_t relH = segmentIdx - relT * SEGMENTS_PER_TIME;
            uint32_t tileId = ((range.tStart + relT) * static_cast<uint32_t>(meta.grid_h) + range.hStart + relH) *
                                  static_cast<uint32_t>(meta.grid_w) +
                              range.wStart;
            uint32_t blockStart = tileId * TILE_ELEMENTS + blockInSegment * FAI_KV_BASE_ROWS;
            stack.packedBlockOffsets[block] = headBase + static_cast<uint64_t>(blockStart) * EMBED;
            stack.packedBlockIds[block] = blockStart / FAI_KV_BASE_ROWS;
        }
        stack.segLen = stack.packedBlockCount * FAI_KV_BASE_ROWS;
        stack.computeSegLen = stack.segLen;
        stack.kvLoopNoMask = stack.packedBlockCount;
        stack.computeKvLoopNoMask = stack.packedBlockCount;
        stack.stackSeqTile = stack.segLen;
        stack.stackSeqCount = stackSeqCount;
        stack.kvOffset = headBase;
        stack.computeKvOffset = headBase;
        stack.computeVOffset = headBase;
        return stack;
    }

    CATLASS_DEVICE StackInfo MakePackedWindow3D3SliceStack(const WindowRange &range, uint32_t stackSeqCount,
                                                           uint64_t headBase, uint32_t embed) const
    {
        constexpr uint32_t BLOCKS_PER_SEGMENT = 9;
        constexpr uint32_t SEGMENTS_PER_TIME = 3;
        constexpr uint32_t TOTAL_BLOCKS = 81;
        StackInfo stack;
        uint32_t firstBlock = stackSeqCount * FAI_STACK_BLOCK_COUNT;
        stack.packedBlockCount = StaMin(FAI_STACK_BLOCK_COUNT, TOTAL_BLOCKS - firstBlock);
        for (uint32_t block = 0; block < stack.packedBlockCount; ++block) {
            uint32_t ordinal = firstBlock + block;
            uint32_t segmentIdx = ordinal / BLOCKS_PER_SEGMENT;
            uint32_t blockInSegment = ordinal - segmentIdx * BLOCKS_PER_SEGMENT;
            uint32_t relT = segmentIdx / SEGMENTS_PER_TIME;
            uint32_t relH = segmentIdx - relT * SEGMENTS_PER_TIME;
            uint32_t tileId = ((range.tStart + relT) * static_cast<uint32_t>(meta.grid_h) + range.hStart + relH) *
                                  static_cast<uint32_t>(meta.grid_w) +
                              range.wStart;
            uint32_t blockStart = tileId * TILE_ELEMENTS + blockInSegment * FAI_KV_BASE_ROWS;
            stack.packedBlockOffsets[block] = headBase + static_cast<uint64_t>(blockStart) * embed;
            stack.packedBlockIds[block] = blockStart / FAI_KV_BASE_ROWS;
        }
        stack.segLen = stack.packedBlockCount * FAI_KV_BASE_ROWS;
        stack.computeSegLen = stack.segLen;
        stack.kvLoopNoMask = stack.packedBlockCount;
        stack.computeKvLoopNoMask = stack.packedBlockCount;
        stack.stackSeqTile = stack.segLen;
        stack.stackSeqCount = stackSeqCount;
        stack.kvOffset = headBase;
        stack.computeKvOffset = headBase;
        stack.computeVOffset = headBase;
        return stack;
    }

    CATLASS_DEVICE bool UsesPagedD64ImageStacks() const
    {
        return meta.dim == 64 && meta.tile_numel == TILE_ELEMENTS;
    }

    CATLASS_DEVICE uint32_t PagedImageStackCount(const WindowRange &range) const
    {
        return StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
    }

    CATLASS_DEVICE StackInfo MakePagedD64TextStack(const TaskInfo &info, const WindowRange &range,
                                                   uint32_t stackSeqCount, uint64_t headBase, uint32_t embed) const
    {
        uint32_t imageStackCount = PagedImageStackCount(range);
        if (stackSeqCount < imageStackCount) {
            return MakePackedImageStack(info, range, stackSeqCount, headBase, embed);
        }
        StackInfo stack;
        uint32_t textStack = stackSeqCount - imageStackCount;
        uint32_t textKvIdx = textStack * FAI_STACK_BLOCK_COUNT;
        uint32_t textBlocks = StaCeilDiv(static_cast<uint32_t>(meta.text_len), FAI_KV_BASE_ROWS);
        stack.kvOffset = headBase + static_cast<uint64_t>(meta.image_seq) * embed;
        stack.segLen = static_cast<uint32_t>(meta.text_len);
        stack.kvLoopNoMask = textBlocks;
        stack.kvIdx = textKvIdx;
        stack.stackSeqTile = StackSeqTile(stack.segLen, textBlocks, textKvIdx);
        stack.stackSeqCount = stackSeqCount;
        return stack;
    }

    CATLASS_DEVICE uint32_t StackCountForSegment(uint32_t imageSegLen, uint32_t stackTile) const
    {
        if (meta.tile_numel == TILE_ELEMENTS && stackTile == FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS) {
            if (imageSegLen == SEGMENT_ONE_WIDTH_ELEMENTS) {
                return STACK_COUNT_ONE_WIDTH;
            }
            if (imageSegLen == SEGMENT_THREE_WIDTH_ELEMENTS) {
                return STACK_COUNT_THREE_WIDTH;
            }
            if (imageSegLen == SEGMENT_FIVE_WIDTH_ELEMENTS) {
                return STACK_COUNT_FIVE_WIDTH;
            }
            if (imageSegLen == SEGMENT_SEVEN_WIDTH_ELEMENTS) {
                return STACK_COUNT_SEVEN_WIDTH;
            }
        }
        return StaCeilDiv(imageSegLen, stackTile);
    }

    CATLASS_DEVICE uint64_t HeadBase(uint32_t batch, uint32_t head) const
    {
        return (static_cast<uint64_t>(batch) * static_cast<uint64_t>(meta.heads) + head) *
               static_cast<uint64_t>(meta.seq) * static_cast<uint64_t>(meta.dim);
    }

    CATLASS_DEVICE uint64_t WorkspaceOffset(uint32_t coreIdx, uint32_t pingpong, uint32_t chunk = 0) const
    {
        uint32_t chunkCount = UsesGroupedQkPath() ? static_cast<uint32_t>(meta.grouped_query_chunks) : 1;
        return static_cast<uint64_t>(coreIdx) * static_cast<uint64_t>(workspace.block_elems) * FAI_RING_STAGE_COUNT *
                   chunkCount +
               static_cast<uint64_t>(chunk) * static_cast<uint64_t>(workspace.block_elems) * FAI_RING_STAGE_COUNT +
               static_cast<uint64_t>(pingpong) * static_cast<uint64_t>(workspace.block_elems);
    }

    CATLASS_DEVICE uint64_t BlockTableOffset(uint32_t coreIdx, uint32_t slot) const
    {
        return (static_cast<uint64_t>(coreIdx) * FAI_RING_STAGE_COUNT + slot) * FAI_STACK_BLOCK_COUNT;
    }

    CATLASS_DEVICE GlobalTensor<int32_t> StackBlockTable(uint32_t coreIdx, uint32_t slot)
    {
        return blockTableGm[BlockTableOffset(coreIdx, slot)];
    }

    CATLASS_DEVICE void StoreStackBlockTable(const StackInfo &stack, uint32_t coreIdx, uint32_t slot)
    {
        GlobalTensor<int32_t> table = StackBlockTable(coreIdx, slot);
        for (uint32_t block = 0; block < stack.packedBlockCount; ++block) {
            table.SetValue(block, static_cast<int32_t>(stack.packedBlockIds[block]));
        }
    }

    CATLASS_DEVICE uint64_t CompactScoreOffset(uint32_t coreIdx, uint32_t pingpong) const
    {
        return WorkspaceOffset(coreIdx, pingpong) * (sizeof(ElementS) / sizeof(ElementCompactS));
    }

    CATLASS_DEVICE uint64_t SoftmaxStateOffset(uint32_t coreIdx, uint32_t chunk) const
    {
        return (static_cast<uint64_t>(coreIdx) * static_cast<uint64_t>(meta.grouped_query_chunks) + chunk) *
               FAI_QUERY_ROWS;
    }

    CATLASS_DEVICE uint64_t SoftmaxDmOffset(uint32_t coreIdx, uint32_t chunk, uint32_t slot) const
    {
        return ((static_cast<uint64_t>(coreIdx) * static_cast<uint64_t>(meta.grouped_query_chunks) + chunk) *
                    FAI_RING_STAGE_COUNT +
                slot) *
               FAI_QUERY_ROWS;
    }

    CATLASS_DEVICE void RestoreSoftmaxState(uint32_t coreIdx, uint32_t chunk, uint32_t qRows, bool restoreMax)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        uint64_t stateOffset = SoftmaxStateOffset(coreIdx, chunk) + rowOffset;
        LocalTensor<float> gmUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GM_UB_OFFSET);
        LocalTensor<float> glUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GL_UB_OFFSET);
        if (restoreMax) {
            AscendC::DataCopy(gmUb, softmaxMaxGm[stateOffset], rowCount);
        }
        AscendC::DataCopy(glUb, softmaxSumGm[stateOffset], rowCount);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void SaveSoftmaxState(uint32_t coreIdx, uint32_t chunk, uint32_t qRows)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        uint64_t stateOffset = SoftmaxStateOffset(coreIdx, chunk) + rowOffset;
        LocalTensor<float> gmUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GM_UB_OFFSET);
        LocalTensor<float> glUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GL_UB_OFFSET);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(softmaxMaxGm[stateOffset], gmUb, rowCount);
        AscendC::DataCopy(softmaxSumGm[stateOffset], glUb, rowCount);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void RestoreSoftmaxDm(uint32_t coreIdx, uint32_t chunk, uint32_t slot, uint32_t qRows)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        uint64_t stateOffset = SoftmaxDmOffset(coreIdx, chunk, slot) + rowOffset;
        LocalTensor<float> dmUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_DM_UB_OFFSET);
        AscendC::DataCopy(dmUb[slot * EpilogueRescaleO::MAX_ROW_NUM_SUB_CORE], softmaxDmGm[stateOffset], rowCount);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void SaveSoftmaxDm(uint32_t coreIdx, uint32_t chunk, uint32_t slot, uint32_t qRows)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        uint64_t stateOffset = SoftmaxDmOffset(coreIdx, chunk, slot) + rowOffset;
        LocalTensor<float> dmUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_DM_UB_OFFSET);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(softmaxDmGm[stateOffset], dmUb[slot * EpilogueRescaleO::MAX_ROW_NUM_SUB_CORE], rowCount);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void RestoreGroupedSoftmaxStateUb(uint32_t chunk, bool restoreMax)
    {
        LocalTensor<float> stateUb = resource.ubBuf.template GetBufferByByte<float>(GROUPED_STATE_UB_OFFSET);
        LocalTensor<float> gmUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GM_UB_OFFSET);
        LocalTensor<float> glUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GL_UB_OFFSET);
        uint32_t maxOffset = chunk * GROUPED_STATE_CHUNK_ELEMENTS;
        uint32_t sumOffset = GROUPED_STATE_MAX_ELEMENTS + chunk * GROUPED_STATE_CHUNK_ELEMENTS;
        AscendC::PipeBarrier<PIPE_V>();
        if (restoreMax) {
            AscendC::DataCopy(gmUb, stateUb[maxOffset],
                              AscendC::DataCopyParams(1, GROUPED_STATE_CHUNK_ELEMENTS / FLOAT_BLOCK_ELEMENTS, 0, 0));
        }
        AscendC::DataCopy(glUb, stateUb[sumOffset],
                          AscendC::DataCopyParams(1, GROUPED_STATE_CHUNK_ELEMENTS / FLOAT_BLOCK_ELEMENTS, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void SaveGroupedSoftmaxStateUb(uint32_t chunk)
    {
        LocalTensor<float> stateUb = resource.ubBuf.template GetBufferByByte<float>(GROUPED_STATE_UB_OFFSET);
        LocalTensor<float> gmUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GM_UB_OFFSET);
        LocalTensor<float> glUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GL_UB_OFFSET);
        uint32_t maxOffset = chunk * GROUPED_STATE_CHUNK_ELEMENTS;
        uint32_t sumOffset = GROUPED_STATE_MAX_ELEMENTS + chunk * GROUPED_STATE_CHUNK_ELEMENTS;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(stateUb[maxOffset], gmUb,
                          AscendC::DataCopyParams(1, GROUPED_STATE_CHUNK_ELEMENTS / FLOAT_BLOCK_ELEMENTS, 0, 0));
        AscendC::DataCopy(stateUb[sumOffset], glUb,
                          AscendC::DataCopyParams(1, GROUPED_STATE_CHUNK_ELEMENTS / FLOAT_BLOCK_ELEMENTS, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RestoreGroupedSoftmaxDmUb(uint32_t chunk, uint32_t slot)
    {
        LocalTensor<float> stateUb = resource.ubBuf.template GetBufferByByte<float>(GROUPED_STATE_UB_OFFSET);
        LocalTensor<float> dmUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_DM_UB_OFFSET);
        uint32_t stateOffset = GROUPED_STATE_MAX_ELEMENTS + GROUPED_STATE_SUM_ELEMENTS +
                               (chunk * FAI_RING_STAGE_COUNT + slot) * GROUPED_STATE_CHUNK_ELEMENTS;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(dmUb[slot * EpilogueRescaleO::MAX_ROW_NUM_SUB_CORE], stateUb[stateOffset],
                          AscendC::DataCopyParams(1, GROUPED_STATE_CHUNK_ELEMENTS / FLOAT_BLOCK_ELEMENTS, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void SaveGroupedSoftmaxDmUb(uint32_t chunk, uint32_t slot)
    {
        LocalTensor<float> stateUb = resource.ubBuf.template GetBufferByByte<float>(GROUPED_STATE_UB_OFFSET);
        LocalTensor<float> dmUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_DM_UB_OFFSET);
        uint32_t stateOffset = GROUPED_STATE_MAX_ELEMENTS + GROUPED_STATE_SUM_ELEMENTS +
                               (chunk * FAI_RING_STAGE_COUNT + slot) * GROUPED_STATE_CHUNK_ELEMENTS;
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(stateUb[stateOffset], dmUb[slot * EpilogueRescaleO::MAX_ROW_NUM_SUB_CORE],
                          AscendC::DataCopyParams(1, GROUPED_STATE_CHUNK_ELEMENTS / FLOAT_BLOCK_ELEMENTS, 0, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RestoreGroupedGoState(uint32_t coreIdx, uint32_t chunk, uint32_t qRows, uint32_t embedRound)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        uint64_t stateOffset = WorkspaceOffset(coreIdx, 0, chunk) + static_cast<uint64_t>(rowOffset) * embedRound;
        LocalTensor<float> goUb = resource.ubBuf.template GetBufferByByte<float>(RESCALE_GO_UB_OFFSET);
        AscendC::DataCopy(goUb, oUpdateGm[stateOffset], rowCount * embedRound);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void SaveGroupedGoState(uint32_t coreIdx, uint32_t chunk, uint32_t qRows, uint32_t embedRound)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        uint64_t stateOffset = WorkspaceOffset(coreIdx, 0, chunk) + static_cast<uint64_t>(rowOffset) * embedRound;
        LocalTensor<float> goUb = resource.ubBuf.template GetBufferByByte<float>(RESCALE_GO_UB_OFFSET);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(oUpdateGm[stateOffset], goUb, rowCount * embedRound);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE uint32_t StackSlot(uint32_t stackSeqCount) const
    {
        return stackSeqCount % FAI_RING_STAGE_COUNT;
    }

    CATLASS_DEVICE uint64_t QueryStageOffset(uint32_t coreIdx, uint32_t chunk = 0) const
    {
        uint32_t queryStageSlots = meta.dim == 8 ? QUERY_CHUNKS_PER_TILE : 1;
        return (static_cast<uint64_t>(coreIdx) * queryStageSlots + chunk) * FAI_QUERY_ROWS * QkStageStride();
    }

    CATLASS_DEVICE uint64_t KvStageOffset(uint32_t coreIdx, uint32_t slot) const
    {
        return (static_cast<uint64_t>(coreIdx) * FAI_RING_STAGE_COUNT + slot) * FAI_STACK_BLOCK_COUNT *
               FAI_KV_BASE_ROWS * QkStageStride();
    }

    CATLASS_DEVICE uint32_t QkDim() const
    {
        return StaAlignUp(static_cast<uint32_t>(meta.dim), VECTOR_BLOCK_ELEMENTS);
    }

    CATLASS_DEVICE uint32_t QkSliceCount() const
    {
        return StaCeilDiv(QkDim(), FAI_QK_SLICE_DIM);
    }

    CATLASS_DEVICE uint32_t QkStageStride() const
    {
        return QkSliceCount() > 1 ? QkSliceCount() * FAI_QK_SLICE_DIM : QkDim();
    }

    CATLASS_DEVICE uint32_t QkSliceStorageDim() const
    {
        return QkSliceCount() > 1 ? FAI_QK_SLICE_DIM : QkDim();
    }

    CATLASS_DEVICE uint64_t QkSliceStageOffset(uint32_t sliceIdx, uint32_t rowCapacity) const
    {
        return static_cast<uint64_t>(sliceIdx) * rowCapacity * QkSliceStorageDim();
    }

    CATLASS_DEVICE void AccumulateQkSlices(uint32_t taskIdx, uint32_t stackSeqCount, uint64_t sOffset, uint32_t rowNum,
                                           uint32_t columnNum, uint32_t rowStride)
    {
        static_assert(opforge::ascend_c::resource_ledger::ValidateUbOverwriteContract(kAccumUbVectorToMte2Reuse),
                      "accumUb requires Vector-reader to next-MTE2-writer release");
        static_assert(opforge::ascend_c::resource_ledger::ValidateUbOverwriteContract(kPartialUbVectorToMte2Reuse),
                      "partialUb requires Vector-reader to next-MTE2-writer release");
        uint32_t sliceCount = QkSliceCount();
        if (sliceCount <= 1 || rowNum == 0 || columnNum == 0) {
            return;
        }
        constexpr uint32_t TILE_ELEMENTS = 4096;
        LocalTensor<float> accumUb = resource.ubBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> partialUb = resource.ubBuf.template GetBufferByByte<float>(TILE_ELEMENTS * sizeof(float));
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t physicalElements = static_cast<uint32_t>(workspace.block_elems);
        uint32_t elementsPerSubBlock = physicalElements / subBlockNum;
        uint32_t elementStart = subBlockIdx * elementsPerSubBlock;
        uint32_t elementCount = subBlockIdx == subBlockNum - 1 ? physicalElements - elementStart : elementsPerSubBlock;
        if (elementCount == 0) {
            return;
        }

        for (uint32_t base = 0; base < elementCount; base += TILE_ELEMENTS) {
            uint32_t tileElements = StaMin(TILE_ELEMENTS, elementCount - base);
            uint64_t tileOffset = static_cast<uint64_t>(elementStart) + base;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            AscendC::DataCopy(accumUb, sGm[sOffset + tileOffset], tileElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);

            for (uint32_t sliceIdx = 1; sliceIdx < sliceCount; ++sliceIdx) {
                uint64_t sliceOffset = static_cast<uint64_t>(sliceIdx) * workspace.staged_elems;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
                AscendC::DataCopy(partialUb, sGm[sOffset + sliceOffset + tileOffset], tileElements);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
                AscendC::Add(accumUb, accumUb, partialUb, tileElements);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
            AscendC::DataCopy(sGm[sOffset + tileOffset], accumUb, tileElements);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        (void)taskIdx;
        (void)stackSeqCount;
        (void)rowNum;
        (void)columnNum;
        (void)rowStride;
    }

    CATLASS_DEVICE bool NeedsQkStaging() const
    {
        if (QkSliceCount() > 1) {
            return true;
        }
        return meta.dim < 64 && QkDim() != static_cast<uint32_t>(meta.dim);
    }

    CATLASS_DEVICE bool UsesOwnedStridedKLoader() const
    {
        return QkSliceCount() > 1;
    }

    CATLASS_DEVICE bool NeedsKvQkStaging() const
    {
        return NeedsQkStaging() && !UsesOwnedStridedKLoader();
    }

    CATLASS_DEVICE uint32_t QkComputeStride() const
    {
        return NeedsQkStaging() ? QkStageStride() : static_cast<uint32_t>(meta.dim);
    }

    CATLASS_DEVICE bool UsesResidualProbability() const
    {
        return !PACKED_D72_FULL && meta.dim == 72 && UsesCompactFullWindowPath();
    }

    CATLASS_DEVICE void SetComputeStackInfo(StackInfo &stack, uint32_t coreIdx, uint32_t slot) const
    {
        (void)coreIdx;
        (void)slot;
        stack.computeKvOffset = stack.kvOffset;
        stack.computeKvIdx = stack.kvIdx;
        stack.computeKvLoopNoMask = stack.kvLoopNoMask;
        stack.computeSegLen = stack.segLen;
        stack.computeVOffset = stack.kvOffset;
    }

    CATLASS_DEVICE void StageAllV()
    {
        if constexpr (NEEDS_INPUT_STAGING) {
            uint64_t totalElems = static_cast<uint64_t>(meta.batch) * static_cast<uint64_t>(meta.heads) *
                                  static_cast<uint64_t>(meta.seq) * static_cast<uint64_t>(meta.dim);
            uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t pairElems = FAI_AIV_SUBBLOCK_COUNT * BF16_CONVERT_CHUNK_ELEMENTS;
            uint64_t workerStride = static_cast<uint64_t>(coreNum) * pairElems;
            for (uint64_t offset = static_cast<uint64_t>(coreIdx) * pairElems; offset < totalElems;
                 offset += workerStride) {
                uint32_t current = static_cast<uint32_t>(StaMin<uint64_t>(pairElems, totalElems - offset));
                ConvertInputTile(vInputGm[offset], vGm[offset], current, 0, 2, -1, -1);
            }
            DrainInputStageWrite();
        }
    }

    CATLASS_DEVICE void ConvertInputTile(GlobalTensor<ElementInput> src, GlobalTensor<half> dst, uint32_t count,
                                         uint32_t debugTask, uint32_t debugStage, int32_t debugStack, int32_t debugSlot)
    {
        if constexpr (NEEDS_INPUT_STAGING) {
            static_assert(opforge::ascend_c::resource_ledger::ValidateUbOverwriteContract(kSrcUbVectorToMte2Reuse),
                          "srcUb requires Vector-reader to next-MTE2-writer release");
            LocalTensor<ElementInput> srcUb =
                resource.ubBuf.template GetBufferByByte<ElementInput>(BF16_CONVERT_SOURCE_UB_OFFSET);
            LocalTensor<half> dstUb = resource.ubBuf.template GetBufferByByte<half>(BF16_CONVERT_DESTINATION_UB_OFFSET);
            LocalTensor<float> floatUb = resource.ubBuf.template GetBufferByByte<float>(BF16_CONVERT_FLOAT_UB_OFFSET);
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t laneStride = FAI_AIV_SUBBLOCK_COUNT * BF16_CONVERT_CHUNK_ELEMENTS;
            for (uint32_t done = subBlockIdx * BF16_CONVERT_CHUNK_ELEMENTS; done < count; done += laneStride) {
                uint32_t current = StaMin(BF16_CONVERT_CHUNK_ELEMENTS, count - done);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                AscendC::DataCopy(srcUb, src[done], current);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
                for (uint32_t converted = 0; converted < current; converted += BF16_CONVERT_VECTOR_ELEMENTS) {
                    uint32_t vectorCount = StaMin(BF16_CONVERT_VECTOR_ELEMENTS, current - converted);
                    if (converted != 0) {
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                    AscendC::Cast(floatUb, srcUb[converted], AscendC::RoundMode::CAST_NONE, vectorCount);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(dstUb[converted], floatUb, AscendC::RoundMode::CAST_NONE, vectorCount);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                (void)debugTask;
                (void)debugStage;
                (void)debugStack;
                (void)debugSlot;
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                AscendC::DataCopy(dst[done], dstUb, current);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            }
        }
    }

    CATLASS_DEVICE void StageQuery(const TaskInfo &info, uint32_t taskIdx, uint32_t coreIdx, uint64_t qOffset)
    {
        if (NeedsQkStaging()) {
            if (meta.dim > FAI_QK_SLICE_DIM) {
                Arch::CrossCoreWaitFlag(qStageFree);
            }
            StagePaddedRows(qInputGm[qOffset], qGm[QueryStageOffset(coreIdx)], info.qRows, FAI_QUERY_ROWS, true);
            DrainInputStageWrite();
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(inputQReady);
        }
        (void)taskIdx;
    }

    CATLASS_DEVICE void StageGroupedQueries(const TaskInfo (&infos)[QUERY_CHUNKS_PER_TILE], uint32_t coreIdx)
    {
        if (!UsesGroupedD8QkPath()) {
            return;
        }
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            uint64_t headBase = HeadBase(infos[chunk].batch, infos[chunk].head);
            uint64_t qOffset = headBase + static_cast<uint64_t>(infos[chunk].qStart) * meta.dim;
            StagePaddedRows(qInputGm[qOffset], qGm[QueryStageOffset(coreIdx, chunk)], infos[chunk].qRows,
                            FAI_QUERY_ROWS, true);
        }
        DrainInputStageWrite();
        Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(inputQReady);
    }

    CATLASS_DEVICE void StageKv(const StackInfo &stack, uint32_t taskIdx, uint32_t coreIdx, uint32_t slot)
    {
        if (NeedsKvQkStaging()) {
            if (meta.dim > FAI_QK_SLICE_DIM) {
                Arch::CrossCoreWaitFlag(kvStageFree);
            }
            uint64_t inputOffset = stack.kvOffset + static_cast<uint64_t>(stack.kvIdx) * FAI_KV_BASE_ROWS * meta.dim;
            StagePaddedRows(kInputGm[inputOffset], kGm[KvStageOffset(coreIdx, slot)], stack.stackSeqTile,
                            FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, false);
            DrainInputStageWrite();
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(inputKvReady);
        }
        (void)taskIdx;
    }

    CATLASS_DEVICE void StagePaddedRows(GlobalTensor<ElementInput> src, GlobalTensor<ElementInput> dst, uint32_t rows,
                                        uint32_t rowCapacity, bool isQuery)
    {
        (void)isQuery;
        if (rows == 0) {
            return;
        }
        static_assert(opforge::ascend_c::resource_ledger::ValidateUbOverwriteContract(kSrcUbVectorToMte2Reuse),
                      "srcUb requires Vector-reader to next-MTE2-writer release");
        LocalTensor<ElementInput> srcUb =
            resource.ubBuf.template GetBufferByByte<ElementInput>(BF16_CONVERT_SOURCE_UB_OFFSET);
        LocalTensor<ElementInput> packedUb =
            resource.ubBuf.template GetBufferByByte<ElementInput>(BF16_CONVERT_DESTINATION_UB_OFFSET);
        uint32_t dim = static_cast<uint32_t>(meta.dim);
        uint32_t sliceCount = QkSliceCount();
        uint32_t sliceStorageDim = QkSliceStorageDim();
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowsPerTile = QK_STAGE_TILE_ELEMENTS / sliceStorageDim;
        if (dim > FAI_QK_SLICE_DIM) {
            for (uint32_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
                uint32_t column = sliceIdx * FAI_QK_SLICE_DIM;
                uint32_t current = StaMin(FAI_QK_SLICE_DIM, dim - column);
                uint32_t packed = sliceStorageDim;
                uint32_t inputAligned = StaAlignUp(current, VECTOR_BLOCK_ELEMENTS);
                for (uint32_t row = subBlockIdx * rowsPerTile; row < rows; row += subBlockNum * rowsPerTile) {
                    uint32_t currentRows = StaMin(rowsPerTile, rows - row);
                    uint64_t dstOffset =
                        QkSliceStageOffset(sliceIdx, rowCapacity) + static_cast<uint64_t>(row) * sliceStorageDim;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                    AscendC::DataCopyPad(srcUb, src[static_cast<uint64_t>(row) * dim + column],
                                         AscendC::DataCopyExtParams(currentRows, current * sizeof(ElementInput),
                                                                    (dim - current) * sizeof(ElementInput), 0, 0),
                                         AscendC::DataCopyPadExtParams<ElementInput>(true, 0, inputAligned - current,
                                                                                     static_cast<ElementInput>(0.0f)));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    if (current == packed && inputAligned == packed) {
                        AscendC::DataCopy(packedUb, srcUb, currentRows * packed);
                    } else {
                        AscendC::Duplicate(packedUb, static_cast<ElementInput>(0.0f), currentRows * packed);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        for (uint32_t tileRow = 0; tileRow < currentRows; ++tileRow) {
                            AscendC::DataCopy(packedUb[static_cast<uint64_t>(tileRow) * packed],
                                              srcUb[static_cast<uint64_t>(tileRow) * inputAligned], inputAligned);
                        }
                    }
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyPad(
                        dst[dstOffset], packedUb,
                        AscendC::DataCopyExtParams(1, currentRows * packed * sizeof(ElementInput), 0, 0, 0));
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                }
            }
            return;
        }
        for (uint32_t sliceIdx = 0; sliceIdx < sliceCount; ++sliceIdx) {
            uint32_t column = sliceIdx * FAI_QK_SLICE_DIM;
            uint32_t current = StaMin(FAI_QK_SLICE_DIM, dim - column);
            uint32_t packed = sliceStorageDim;
            uint32_t inputAligned = StaAlignUp(current, VECTOR_BLOCK_ELEMENTS);
            for (uint32_t row = subBlockIdx * rowsPerTile; row < rows; row += subBlockNum * rowsPerTile) {
                uint32_t currentRows = StaMin(rowsPerTile, rows - row);
                uint64_t dstOffset =
                    QkSliceStageOffset(sliceIdx, rowCapacity) + static_cast<uint64_t>(row) * sliceStorageDim;
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                AscendC::DataCopyPad(srcUb, src[static_cast<uint64_t>(row) * dim + column],
                                     AscendC::DataCopyExtParams(currentRows, current * sizeof(ElementInput),
                                                                (dim - current) * sizeof(ElementInput), 0, 0),
                                     AscendC::DataCopyPadExtParams<ElementInput>(true, 0, inputAligned - current,
                                                                                 static_cast<ElementInput>(0.0f)));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Duplicate(packedUb, static_cast<ElementInput>(0.0f), currentRows * packed);
                AscendC::PipeBarrier<PIPE_ALL>();
                for (uint32_t tileRow = 0; tileRow < currentRows; ++tileRow) {
                    AscendC::DataCopy(packedUb[static_cast<uint64_t>(tileRow) * packed],
                                      srcUb[static_cast<uint64_t>(tileRow) * inputAligned], inputAligned);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyPad(
                    dst[dstOffset], packedUb,
                    AscendC::DataCopyExtParams(1, currentRows * packed * sizeof(ElementInput), 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            }
        }
    }

    CATLASS_DEVICE void DrainInputStageWrite()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
    }

    template <bool USE_PACKED_OFFSETS = false, bool FORCE_EXPLICIT_SINGLE_BLOCK_PV = false>
    CATLASS_DEVICE void RunAicPvStack(uint32_t taskIdx, const TaskInfo &info, const StackInfo &stack, uint32_t coreIdx,
                                      uint32_t embed, uint32_t strideKV, const LayoutV &layoutV,
                                      BlockMmadPVNormal &blockMmadPVNormal,
                                      BlockMmadPVDynamicSingleBlock &blockMmadPVDynamicSingleBlock,
                                      BlockMmadPVDynamicMultiBlock &blockMmadPVDynamicMultiBlock,
                                      uint32_t workspaceChunk = 0)
    {
        uint32_t pvPingPong = StackSlot(stack.stackSeqCount);
        uint64_t pOffset = WorkspaceOffset(coreIdx, pvPingPong, workspaceChunk);
        LayoutP layoutP(info.qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
        GemmCoord pvShape{info.qRows, embed, stack.stackSeqTile};
        uint64_t computeVOffset = stack.computeVOffset;
        uint32_t computeKvIdx = stack.computeKvIdx;
        uint32_t computeKvLoopNoMask = stack.computeKvLoopNoMask;
        uint32_t computeSegLen = stack.computeSegLen;
        uint32_t blockSize = FAI_KV_BASE_ROWS;
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        const bool debugFirstD256Pv = meta.dim > FAI_QK_SLICE_DIM &&
                                      taskIdx < AscendC::GetBlockNum() * QUERY_CHUNKS_PER_TILE &&
                                      stack.stackSeqCount == 0;
        const bool debugSecondD256Pv = meta.dim > FAI_QK_SLICE_DIM &&
                                       taskIdx < AscendC::GetBlockNum() * QUERY_CHUNKS_PER_TILE &&
                                       stack.stackSeqCount == 1;
        if (debugFirstD256Pv) {
            AscendC::printf("STA_AIC_STAGE phase=pv_enter dim=%u task=%u block=%u\\n", static_cast<uint32_t>(meta.dim),
                            taskIdx, AscendC::GetBlockIdx());
        }
        if (debugSecondD256Pv) {
            AscendC::printf("STA_AIC_STAGE phase=ring_pv1_enter dim=%u task=%u block=%u\\n",
                            static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
        }
#endif
        if (embed == 64) {
            blockMmadPVNormal(pGm[pOffset], vGm[computeVOffset], oTmpGm[pOffset], dummyBlockTable, layoutP, layoutV,
                              pvShape, computeKvIdx, computeKvLoopNoMask, blockSize, computeSegLen, strideKV,
                              softmaxReady);
        } else if (embed <= FAI_QK_SLICE_DIM) {
            const uint32_t outputStride = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
            if (UsesResidualProbability()) {
                blockMmadPVDynamicSingleBlock(pGm[pOffset], vGm[computeVOffset], oTmpGm[pOffset], layoutP, layoutV,
                                              pvShape, computeKvIdx, computeKvLoopNoMask, computeSegLen, strideKV,
                                              outputStride, softmaxReady);
            } else if (UsesD128GroupedEarlyPublishPath() &&
                       stack.stackSeqTile == FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS) {
                blockMmadPVDynamicSingleBlock.FullD128(pGm[pOffset], vGm[computeVOffset], oTmpGm[pOffset], layoutP,
                                                       layoutV, computeKvIdx, strideKV, outputStride, softmaxReady);
            } else {
                blockMmadPVDynamicSingleBlock(pGm[pOffset], vGm[computeVOffset], oTmpGm[pOffset], layoutP, layoutV,
                                              pvShape, computeKvIdx, computeKvLoopNoMask, computeSegLen, strideKV,
                                              outputStride, softmaxReady);
            }
        } else {
            const uint32_t outputStride = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
            if constexpr (USE_PACKED_OFFSETS) {
                static_assert(!FORCE_EXPLICIT_SINGLE_BLOCK_PV ||
                                  opforge::ascend_c::resource_ledger::ValidateL0cReuseContract(
                                      {opforge::ascend_c::resource_ledger::L0cCompletionProtocol::ExplicitMFixFixM,
                                       opforge::ascend_c::resource_ledger::L0cCompletionProtocol::ExplicitMFixFixM,
                                       true, true, true}),
                              "forced packed single-block PV must retain explicit same-L0C completion");
                if (stack.packedBlockCount == 1 && !FORCE_EXPLICIT_SINGLE_BLOCK_PV) {
                    blockMmadPVDynamicSingleBlock.ComputePackedD256(
                        pGm[pOffset], vGm, oTmpGm[pOffset], layoutP, layoutV, pvShape, 1, stack.stackSeqTile,
                        outputStride, softmaxReady, stack.packedBlockOffsets);
                } else {
                    blockMmadPVDynamicMultiBlock.ComputePackedD256(
                        pGm[pOffset], vGm, oTmpGm[pOffset], layoutP, layoutV, pvShape, stack.packedBlockCount,
                        stack.stackSeqTile, outputStride, softmaxReady, stack.packedBlockOffsets);
                }
            } else {
                blockMmadPVDynamicMultiBlock(pGm[pOffset], vGm[computeVOffset], oTmpGm[pOffset], layoutP, layoutV,
                                             pvShape, computeKvIdx, computeKvLoopNoMask, computeSegLen, strideKV,
                                             outputStride, softmaxReady);
            }
        }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        if (debugFirstD256Pv) {
            AscendC::printf("STA_AIC_STAGE phase=pv_done dim=%u task=%u block=%u\\n", static_cast<uint32_t>(meta.dim),
                            taskIdx, AscendC::GetBlockIdx());
        }
        if (debugSecondD256Pv) {
            AscendC::printf("STA_AIC_STAGE phase=ring_pv1_done dim=%u task=%u block=%u\\n",
                            static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
        }
#endif
        Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
        if (UsesResidualProbability()) {
            const uint32_t outputStride = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
            blockMmadPVDynamicSingleBlock(pLoGm[pOffset], vGm[computeVOffset], oTmpLoGm[pOffset], layoutP, layoutV,
                                          pvShape, computeKvIdx, computeKvLoopNoMask, computeSegLen, strideKV,
                                          outputStride, softmaxLowReady);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvLowReady);
        }
    }

    CATLASS_DEVICE void RunAicPagedPvStack(const TaskInfo &info, const StackInfo &stack, uint32_t coreIdx,
                                           const LayoutV &layoutV, BlockMmadPVPaged &blockMmadPVPaged)
    {
        uint32_t slot = StackSlot(stack.stackSeqCount);
        uint64_t pOffset = WorkspaceOffset(coreIdx, slot);
        LayoutP layoutP(info.qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
        GemmCoord pvShape{info.qRows, static_cast<uint32_t>(meta.dim), stack.stackSeqTile};
        uint32_t pagedKvIdx = 0;
        uint32_t pagedKvLoop = stack.packedBlockCount;
        uint32_t blockSize = FAI_KV_BASE_ROWS;
        blockMmadPVPaged(pGm[pOffset], vGm[stack.computeVOffset], oTmpGm[pOffset], StackBlockTable(coreIdx, slot),
                         layoutP, layoutV, pvShape, pagedKvIdx, pagedKvLoop, blockSize, stack.computeSegLen,
                         static_cast<uint32_t>(meta.dim), softmaxReady);
        Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
    }

    CATLASS_DEVICE void RunAivRescaleStack(uint32_t taskIdx, const StackInfo &stack, uint32_t coreIdx,
                                           uint64_t outOffset, uint32_t qRows, uint32_t embed, uint32_t embedRound,
                                           uint32_t totalStackSeqNum, EpilogueRescaleO &epilogueRescaleO,
                                           uint32_t workspaceChunk = 0, bool waitForPvReady = true)
    {
        uint32_t pvStackMod = StackSlot(stack.stackSeqCount);
        uint32_t recurrenceSlot = pvStackMod;
        uint64_t oTmpOffset = WorkspaceOffset(coreIdx, pvStackMod, workspaceChunk);
        LayoutO layoutO(qRows, embed);
        uint32_t oTmpStride = embed == 64 ? embed : embedRound;
        LayoutOTmp layoutOTmp(qRows, embed, oTmpStride);
        GemmCoord pvShape{qRows, embed, stack.stackSeqTile};
        if (waitForPvReady) {
            Arch::CrossCoreWaitFlag(pvReady);
            if (UsesResidualProbability()) {
                Arch::CrossCoreWaitFlag(pvLowReady);
            }
        }
        bool isFirstStack = stack.stackSeqCount == 0;
        bool isLastStack = stack.stackSeqCount == totalStackSeqNum - 1;
        if constexpr (USE_DUAL_PROBABILITY) {
            RunBf16RecurrenceStack(taskIdx, stack, oTmpOffset, outOffset, qRows, embed, embedRound, isFirstStack,
                                   isLastStack, pvStackMod);
        } else if (embed <= FAI_QK_SLICE_DIM) {
            if (UsesGroupedQkPath() && isLastStack) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            }
            epilogueRescaleO(outGm[outOffset], oTmpGm[oTmpOffset], layoutO, layoutOTmp, pvShape, qRows, 1, isFirstStack,
                             isLastStack, recurrenceSlot);
            if (UsesGroupedQkPath() && isLastStack) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            }
        } else {
            (void)taskIdx;
            (void)layoutO;
            (void)layoutOTmp;
            (void)pvShape;
            (void)epilogueRescaleO;
            RunDynamicRescaleStack(oTmpOffset, outOffset, qRows, embed, embedRound, isFirstStack, isLastStack,
                                   recurrenceSlot, workspaceChunk);
        }
        if constexpr (NEEDS_INPUT_STAGING) {
            if (stack.stackSeqCount == totalStackSeqNum - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            }
        }
    }

    CATLASS_DEVICE void RunDynamicRescaleStack(uint64_t oTmpOffset, uint64_t outOffset, uint32_t qRows, uint32_t embed,
                                               uint32_t oTmpStride, bool isFirstStack, bool isLastStack,
                                               uint32_t pvStackMod, uint32_t workspaceChunk = 0)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        if (rowCount == 0) {
            return;
        }

        LocalTensor<float> goUb = resource.ubBuf.template GetBufferByByte<float>(RESCALE_GO_UB_OFFSET);
        LocalTensor<float> localUb = resource.ubBuf.template GetBufferByByte<float>(RESCALE_LOCAL_UB_OFFSET);
        LocalTensor<float> scaleUb = resource.ubBuf.template GetBufferByByte<float>(RESCALE_SCALE_UB_OFFSET);
        LocalTensor<ElementO> packedUb = resource.ubBuf.template GetBufferByByte<ElementO>(RESCALE_LOCAL_UB_OFFSET);
        LocalTensor<float> glUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_GL_UB_OFFSET);
        LocalTensor<float> dmUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_DM_UB_OFFSET);
        LocalTensor<float> lowUb = resource.ubBuf.template GetBufferByByte<float>(SUM_V_ACCUM_UB_OFFSET);
        uint32_t rowTile = StaMin(rowCount, RESCALE_TILE_ELEMENTS / RESCALE_MIN_COLUMNS);
        uint64_t stateBase = WorkspaceOffset(coreIdx, 0, workspaceChunk);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        for (uint32_t rowBase = 0; rowBase < rowCount; rowBase += rowTile) {
            uint32_t rows = StaMin(rowTile, rowCount - rowBase);
            uint32_t colTile = StaAlignUp(RESCALE_TILE_ELEMENTS / rows, VECTOR_BLOCK_ELEMENTS);
            if (colTile > FLOAT_VECTOR_ELEMENTS) {
                colTile = FLOAT_VECTOR_ELEMENTS;
            }
            if (colTile > embed) {
                colTile = embed;
            }
            for (uint32_t colBase = 0; colBase < embed; colBase += colTile) {
                uint32_t cols = StaMin(colTile, embed - colBase);
                uint32_t colsRound = StaAlignUp(cols, VECTOR_BLOCK_ELEMENTS);
                uint32_t elemCount = rows * colsRound;
                uint64_t stateOffset = stateBase + static_cast<uint64_t>(rowOffset + rowBase) * embed + colBase;
                uint64_t localOffset = oTmpOffset + static_cast<uint64_t>(rowOffset + rowBase) * oTmpStride + colBase;

                if (isFirstStack) {
                    AscendC::DataCopyPad(goUb, oTmpGm[localOffset],
                                         AscendC::DataCopyExtParams(rows, cols * sizeof(float),
                                                                    (oTmpStride - cols) * sizeof(float), 0, 0),
                                         AscendC::DataCopyPadExtParams<float>(true, 0, colsRound - cols, 0.0f));
                    if (UsesResidualProbability()) {
                        AscendC::DataCopyPad(localUb, oTmpLoGm[localOffset],
                                             AscendC::DataCopyExtParams(rows, cols * sizeof(float),
                                                                        (oTmpStride - cols) * sizeof(float), 0, 0),
                                             AscendC::DataCopyPadExtParams<float>(true, 0, colsRound - cols, 0.0f));
                    }
                } else {
                    AscendC::DataCopyPad(
                        goUb, oUpdateGm[stateOffset],
                        AscendC::DataCopyExtParams(rows, cols * sizeof(float), (embed - cols) * sizeof(float), 0, 0),
                        AscendC::DataCopyPadExtParams<float>(true, 0, colsRound - cols, 0.0f));
                    AscendC::DataCopyPad(localUb, oTmpGm[localOffset],
                                         AscendC::DataCopyExtParams(rows, cols * sizeof(float),
                                                                    (oTmpStride - cols) * sizeof(float), 0, 0),
                                         AscendC::DataCopyPadExtParams<float>(true, 0, colsRound - cols, 0.0f));
                    if (UsesResidualProbability()) {
                        AscendC::DataCopyPad(lowUb, oTmpLoGm[localOffset],
                                             AscendC::DataCopyExtParams(rows, cols * sizeof(float),
                                                                        (oTmpStride - cols) * sizeof(float), 0, 0),
                                             AscendC::DataCopyPadExtParams<float>(true, 0, colsRound - cols, 0.0f));
                    }
                }
                const uint32_t rowsRound = StaAlignUp(rows, FLOAT_BLOCK_ELEMENTS);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);

                AscendC::PipeBarrier<PIPE_V>();
                if (UsesResidualProbability()) {
                    if (isFirstStack) {
                        AscendC::Muls(localUb, localUb, PROBABILITY_RESIDUAL_ADD_BACK, elemCount);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(goUb, goUb, localUb, elemCount);
                    } else {
                        AscendC::Muls(lowUb, lowUb, PROBABILITY_RESIDUAL_ADD_BACK, elemCount);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add(localUb, localUb, lowUb, elemCount);
                    }
                }

                if (!isFirstStack) {
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Brcb(scaleUb.template ReinterpretCast<uint32_t>(),
                                  dmUb[pvStackMod * EpilogueRescaleO::MAX_ROW_NUM_SUB_CORE + rowBase]
                                      .template ReinterpretCast<uint32_t>(),
                                  rowsRound / FLOAT_BLOCK_ELEMENTS, AscendC::BrcbRepeatParams(1, FLOAT_BLOCK_ELEMENTS));
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t col = 0; col < cols; col += FLOAT_VECTOR_ELEMENTS) {
                        uint32_t current = StaMin(FLOAT_VECTOR_ELEMENTS, cols - col);
                        if (current != FLOAT_VECTOR_ELEMENTS) {
                            SetFloatMask(current);
                        }
                        AscendC::Mul<float, false>(
                            goUb[col], goUb[col], scaleUb, static_cast<uint64_t>(0), static_cast<uint8_t>(rows),
                            AscendC::BinaryRepeatParams(1, 1, 0, colsRound / FLOAT_BLOCK_ELEMENTS,
                                                        colsRound / FLOAT_BLOCK_ELEMENTS, 1));
                        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(goUb, goUb, localUb, elemCount);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (!isLastStack) {
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                    AscendC::DataCopyPad(oUpdateGm[stateOffset], goUb,
                                         AscendC::DataCopyExtParams(rows, cols * sizeof(float),
                                                                    (colsRound - cols) / FLOAT_BLOCK_ELEMENTS,
                                                                    (embed - cols) * sizeof(float), 0));
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
                    continue;
                }

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Brcb(scaleUb.template ReinterpretCast<uint32_t>(),
                              glUb[rowBase].template ReinterpretCast<uint32_t>(),
                              StaAlignUp(rows, FLOAT_BLOCK_ELEMENTS) / FLOAT_BLOCK_ELEMENTS,
                              AscendC::BrcbRepeatParams(1, FLOAT_BLOCK_ELEMENTS));
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t col = 0; col < cols; col += FLOAT_VECTOR_ELEMENTS) {
                    uint32_t current = StaMin(FLOAT_VECTOR_ELEMENTS, cols - col);
                    if (current != FLOAT_VECTOR_ELEMENTS) {
                        SetFloatMask(current);
                    }
                    AscendC::Div<float, false>(goUb[col], goUb[col], scaleUb, static_cast<uint64_t>(0),
                                               static_cast<uint8_t>(rows),
                                               AscendC::BinaryRepeatParams(1, 1, 0, colsRound / FLOAT_BLOCK_ELEMENTS,
                                                                           colsRound / FLOAT_BLOCK_ELEMENTS, 1));
                    AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast<ElementO, float, false>(
                    packedUb, goUb,
                    std::is_same_v<ElementO, bfloat16_t> ? AscendC::RoundMode::CAST_RINT :
                                                           AscendC::RoundMode::CAST_NONE,
                    static_cast<uint64_t>(0), (elemCount + FLOAT_VECTOR_ELEMENTS - 1) / FLOAT_VECTOR_ELEMENTS,
                    AscendC::UnaryRepeatParams(1, 1, BF16_CAST_SOURCE_REPEAT_STRIDE, FLOAT_BLOCK_ELEMENTS));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
                uint32_t sourceRowGapBytes = (colsRound - cols) * sizeof(ElementO);
                uint64_t outputBase = outOffset + static_cast<uint64_t>(rowOffset + rowBase) * embed + colBase;
                if (sourceRowGapBytes % DATA_BLOCK_BYTES == 0) {
                    AscendC::DataCopyPad(
                        outGm[outputBase], packedUb,
                        AscendC::DataCopyExtParams(rows, cols * sizeof(ElementO), sourceRowGapBytes / DATA_BLOCK_BYTES,
                                                   (embed - cols) * sizeof(ElementO), 0));
                } else {
                    for (uint32_t row = 0; row < rows; ++row) {
                        AscendC::DataCopyPad(outGm[outputBase + static_cast<uint64_t>(row) * embed],
                                             packedUb[static_cast<uint64_t>(row) * colsRound],
                                             AscendC::DataCopyExtParams(1, cols * sizeof(ElementO), 0, 0, 0));
                    }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    }

    CATLASS_DEVICE void SetFloatMask(uint32_t count) const
    {
        uint64_t mask = count == FLOAT_VECTOR_ELEMENTS ? static_cast<uint64_t>(-1) : ((1ULL << count) - 1);
        AscendC::SetVectorMask<int8_t>(0, mask);
    }

    CATLASS_DEVICE void RunBf16RecurrenceStack(uint32_t taskIdx, const StackInfo &stack, uint64_t oTmpOffset,
                                               uint64_t outOffset, uint32_t qRows, uint32_t embed, uint32_t embedRound,
                                               bool isFirstStack, bool isLastStack, uint32_t pvStackMod)
    {
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowOffset = subBlockIdx * rowSplit;
        uint32_t rowCount = subBlockIdx == subBlockNum - 1 ? qRows - rowOffset : rowSplit;
        if (rowCount == 0) {
            return;
        }

        LocalTensor<float> goUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_GO_UB_OFFSET);
        LocalTensor<float> glUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_GL_UB_OFFSET);
        LocalTensor<float> tvUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_TV_UB_OFFSET);
        LocalTensor<float> dmUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_DM_UB_OFFSET);
        LocalTensor<float> loUb = resource.ubBuf.template GetBufferByByte<float>(STA_RESCALE_LO_UB_OFFSET);
        LocalTensor<ElementO> packedUb = resource.ubBuf.template GetBufferByByte<ElementO>(STA_RESCALE_LO_UB_OFFSET);
        LocalTensor<float> sumVUb = resource.ubBuf.template GetBufferByByte<float>(CORRECTION_SUM_V_UB_OFFSET);
        LocalTensor<float> correctionUb = resource.ubBuf.template GetBufferByByte<float>(CORRECTION_VALUE_UB_OFFSET);
        LocalTensor<float> centerUb = resource.ubBuf.template GetBufferByByte<float>(CORRECTION_CENTER_UB_OFFSET);
        LocalTensor<float> centerBroadcastUb =
            resource.ubBuf.template GetBufferByByte<float>(CORRECTION_BROADCAST_UB_OFFSET);
        uint32_t rowCountRound = StaAlignUp(rowCount, EpilogueRescaleO::FLOAT_BLOCK_SIZE);
        uint32_t elemCount = rowCount * embed;
        uint32_t stateElemCount = rowCount * embedRound;
        uint64_t laneOTmpOffset = oTmpOffset + static_cast<uint64_t>(rowOffset) * embedRound;
        uint64_t goStateOffset = WorkspaceOffset(coreIdx, 0) + CORRECTION_GO_STATE_OFFSET_ELEMENTS +
                                 static_cast<uint64_t>(rowOffset) * embedRound;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        if (isFirstStack) {
            AscendC::DataCopy(
                goUb, oTmpGm[laneOTmpOffset],
                AscendC::DataCopyParams(1, rowCount * embedRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE, 0, 0));
        } else {
            AscendC::DataCopy(goUb, correctionGm[goStateOffset],
                              AscendC::DataCopyParams(1, stateElemCount / EpilogueRescaleO::FLOAT_BLOCK_SIZE, 0, 0));
            AscendC::DataCopy(
                loUb, oTmpGm[laneOTmpOffset],
                AscendC::DataCopyParams(1, rowCount * embedRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        if (!isFirstStack) {
            AscendC::Brcb(
                tvUb.template ReinterpretCast<uint32_t>(),
                dmUb[pvStackMod * EpilogueRescaleO::MAX_ROW_NUM_SUB_CORE].template ReinterpretCast<uint32_t>(),
                rowCountRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE, AscendC::BrcbRepeatParams(1, FLOAT_BLOCK_ELEMENTS));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            for (uint32_t column = 0; column < embed / EpilogueRescaleO::FLOAT_VECTOR_SIZE; ++column) {
                AscendC::Mul<float, false>(
                    goUb[column * EpilogueRescaleO::FLOAT_VECTOR_SIZE],
                    goUb[column * EpilogueRescaleO::FLOAT_VECTOR_SIZE], tvUb, static_cast<uint64_t>(0),
                    static_cast<uint8_t>(rowCount),
                    AscendC::BinaryRepeatParams(1, 1, 0, embedRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE,
                                                embedRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE, 1));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add<float, false>(
                goUb, goUb, loUb, static_cast<uint64_t>(0),
                static_cast<uint8_t>((rowCount * embedRound + EpilogueRescaleO::FLOAT_VECTOR_SIZE - 1) /
                                     EpilogueRescaleO::FLOAT_VECTOR_SIZE),
                AscendC::BinaryRepeatParams(1, 1, 1, FLOAT_BLOCK_ELEMENTS, FLOAT_BLOCK_ELEMENTS, FLOAT_BLOCK_ELEMENTS));
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::DataCopy(sumVUb, correctionGm[oTmpOffset + FAI_QUERY_ROWS + subBlockIdx * embed], embed);
        AscendC::DataCopy(centerUb, correctionGm[oTmpOffset + subBlockIdx * (FAI_QUERY_ROWS / FAI_AIV_SUBBLOCK_COUNT)],
                          rowCountRound);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID4);
        AscendC::Brcb(centerBroadcastUb.template ReinterpretCast<uint32_t>(),
                      centerUb.template ReinterpretCast<uint32_t>(), rowCountRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE,
                      AscendC::BrcbRepeatParams(1, FLOAT_BLOCK_ELEMENTS));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        for (uint32_t row = 0; row < rowCount; ++row) {
            AscendC::Mul<float, false>(
                correctionUb[row * embed], sumVUb, centerBroadcastUb[row * EpilogueRescaleO::FLOAT_BLOCK_SIZE],
                static_cast<uint64_t>(0), 1,
                AscendC::BinaryRepeatParams(1, 1, 0, embed / EpilogueRescaleO::FLOAT_BLOCK_SIZE, 0, 0));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(goUb, goUb, correctionUb, elemCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);

        if (!isLastStack) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
            AscendC::DataCopy(correctionGm[goStateOffset], goUb, stateElemCount);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            return;
        }

        AscendC::Brcb(tvUb.template ReinterpretCast<uint32_t>(), glUb.template ReinterpretCast<uint32_t>(),
                      rowCountRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE,
                      AscendC::BrcbRepeatParams(1, FLOAT_BLOCK_ELEMENTS));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        for (uint32_t column = 0; column < embed / EpilogueRescaleO::FLOAT_VECTOR_SIZE; ++column) {
            AscendC::Div<float, false>(
                goUb[column * EpilogueRescaleO::FLOAT_VECTOR_SIZE], goUb[column * EpilogueRescaleO::FLOAT_VECTOR_SIZE],
                tvUb, static_cast<uint64_t>(0), static_cast<uint8_t>(rowCount),
                AscendC::BinaryRepeatParams(1, 1, 0, embedRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE,
                                            embedRound / EpilogueRescaleO::FLOAT_BLOCK_SIZE, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        (void)taskIdx;
        AscendC::Cast<ElementO, float, false>(
            packedUb, goUb, AscendC::RoundMode::CAST_RINT, static_cast<uint64_t>(0),
            (rowCount * embedRound + EpilogueRescaleO::FLOAT_VECTOR_SIZE - 1) / EpilogueRescaleO::FLOAT_VECTOR_SIZE,
            AscendC::UnaryRepeatParams(1, 1, BF16_CAST_SOURCE_REPEAT_STRIDE, FLOAT_BLOCK_ELEMENTS));
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
        AscendC::DataCopyPad(outGm[outOffset + static_cast<uint64_t>(rowOffset) * embed], packedUb,
                             AscendC::DataCopyExtParams(rowCount, embed * sizeof(ElementO),
                                                        (embedRound - embed) * sizeof(ElementO), 0, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
    }

    CATLASS_DEVICE void ComputeSumV(uint32_t taskIdx, const StackInfo &stack, uint64_t correctionOffset,
                                    uint32_t strideKV)
    {
        LocalTensor<ElementV> srcUb = resource.ubBuf.template GetBufferByByte<ElementV>(SUM_V_SOURCE_UB_OFFSET);
        LocalTensor<float> floatUb = resource.ubBuf.template GetBufferByByte<float>(SUM_V_FLOAT_UB_OFFSET);
        LocalTensor<float> accum0 = resource.ubBuf.template GetBufferByByte<float>(SUM_V_ACCUM_UB_OFFSET);
        LocalTensor<float> accum1 =
            resource.ubBuf.template GetBufferByByte<float>(SUM_V_ACCUM_UB_OFFSET + SUM_V_ACCUM_STRIDE_BYTES);
        LocalTensor<float> accum2 =
            resource.ubBuf.template GetBufferByByte<float>(SUM_V_ACCUM_UB_OFFSET + 2 * SUM_V_ACCUM_STRIDE_BYTES);
        LocalTensor<float> accum3 = resource.ubBuf.template GetBufferByByte<float>(
            SUM_V_ACCUM_UB_OFFSET + (SUM_V_ACCUMULATOR_COUNT - 1) * SUM_V_ACCUM_STRIDE_BYTES);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint64_t computeVOffset =
            stack.computeVOffset + static_cast<uint64_t>(stack.computeKvIdx) * FAI_KV_BASE_ROWS * strideKV;
        for (uint32_t colBase = 0; colBase < strideKV; colBase += DYNAMIC_CENTER_COLUMN_TILE) {
            uint32_t cols = StaMin(DYNAMIC_CENTER_COLUMN_TILE, strideKV - colBase);
            uint32_t colsRound = StaAlignUp(cols, FLOAT_BLOCK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::Duplicate(accum0, 0.0f, colsRound);
            AscendC::Duplicate(accum1, 0.0f, colsRound);
            AscendC::Duplicate(accum2, 0.0f, colsRound);
            AscendC::Duplicate(accum3, 0.0f, colsRound);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t rowBase = 0; rowBase < stack.stackSeqTile; rowBase += SUM_V_TILE_ROWS) {
                uint32_t currentRows = StaMin(SUM_V_TILE_ROWS, stack.stackSeqTile - rowBase);
                uint32_t currentElems = currentRows * colsRound;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::DataCopyPad(
                    srcUb, vGm[computeVOffset + static_cast<uint64_t>(rowBase) * strideKV + colBase],
                    AscendC::DataCopyExtParams(currentRows, cols * sizeof(ElementV),
                                               (strideKV - cols) * sizeof(ElementV), 0, 0),
                    AscendC::DataCopyPadExtParams<ElementV>(true, 0, colsRound - cols, static_cast<ElementV>(0.0f)));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::Cast(floatUb, srcUb, AscendC::RoundMode::CAST_NONE, currentElems);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < currentRows; row += SUM_V_ACCUMULATOR_COUNT) {
                    AscendC::Add(accum0, accum0, floatUb[static_cast<uint64_t>(row) * colsRound], colsRound);
                    if (row + 1 < currentRows) {
                        AscendC::Add(accum1, accum1, floatUb[static_cast<uint64_t>(row + 1) * colsRound], colsRound);
                    }
                    if (row + 2 < currentRows) {
                        AscendC::Add(accum2, accum2, floatUb[static_cast<uint64_t>(row + 2) * colsRound], colsRound);
                    }
                    if (row + 3 < currentRows) {
                        AscendC::Add(accum3, accum3, floatUb[static_cast<uint64_t>(row + 3) * colsRound], colsRound);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
            AscendC::Add(accum0, accum0, accum1, colsRound);
            AscendC::Add(accum2, accum2, accum3, colsRound);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accum0, accum0, accum2, colsRound);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopyPad(correctionGm[correctionOffset + FAI_QUERY_ROWS +
                                              static_cast<uint64_t>(subBlockIdx) * strideKV + colBase],
                                 accum0, AscendC::DataCopyExtParams(1, cols * sizeof(float), 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        // Rescale runs after this call in the same AIV stream, so the MTE3 completion is ordered here.
        // EVENT_ID6 also closes the GM publication for the existing dynamic/BF16 staging lifecycle.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        (void)taskIdx;
    }

    CATLASS_DEVICE void RunCompactOnlineSoftmax(EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                uint64_t compactScoreOffset, uint64_t pOffset, const LayoutP &layoutP,
                                                const LayoutS &layoutS, GemmCoord qkShape, bool isFirstStack,
                                                uint32_t qRows, uint32_t curStackMod)
    {
        constexpr uint32_t P_BLOCK_ELEMENTS = 16;
        constexpr uint32_t S_BLOCK_ELEMENTS = 8;
        constexpr uint32_t MAX_UB_S_ELEMENTS = EpilogueOnlineSoftmax::MAX_UB_S_ELEM_NUM;
        uint32_t rowNum = qkShape.m();
        uint32_t columnNum = qkShape.n();
        uint32_t columnNumRound = StaAlignUp(columnNum, P_BLOCK_ELEMENTS);
        uint32_t columnNumPad = layoutS.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowActual = subBlockIdx == subBlockNum - 1 ? rowNum - rowSplit : rowSplit;
        uint32_t rowBase = subBlockIdx * rowSplit;
        uint32_t maxRowsPerLoop = MAX_UB_S_ELEMENTS / columnNumRound;
        uint32_t rowTile = maxRowsPerLoop / P_BLOCK_ELEMENTS * P_BLOCK_ELEMENTS;
        uint32_t rowLoops = StaCeilDiv(rowActual, rowTile);
        LocalTensor<float> pFloatUb = resource.ubBuf.template GetBufferByByte<float>(0);
        LocalTensor<ElementP> pPackedUb = resource.ubBuf.template GetBufferByByte<ElementP>(4 * UB_REGION_BYTES);
        LocalTensor<float> residualUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_CENTER_UB_OFFSET);

        for (uint32_t loop = 0; loop < rowLoops; ++loop) {
            uint32_t pingpong = loop % 2;
            uint32_t rowInLoop = loop * rowTile;
            uint32_t currentRows = loop == rowLoops - 1 ? rowActual - rowInLoop : rowTile;
            uint32_t gmRow = rowBase + rowInLoop;
            int64_t scoreOffset = layoutS.GetOffset(MatrixCoord(gmRow, 0));
            int64_t outputOffset = layoutP.GetOffset(MatrixCoord(gmRow, 0));
            uint32_t sUbOffset = pingpong * MAX_UB_S_ELEMENTS;
            uint32_t rowRound = StaAlignUp(currentRows, EpilogueOnlineSoftmax::FLOAT_BLOCK_SIZE);
            uint32_t dmOffset = curStackMod * EpilogueOnlineSoftmax::MAX_ROW_NUM_SUB_CORE + rowInLoop;
            uint32_t elemCount = currentRows * columnNumRound;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpong);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpong);
            if (columnNumRound == columnNumPad) {
                AscendC::DataCopy(pFloatUb[sUbOffset], compactSGm[compactScoreOffset + scoreOffset],
                                  AscendC::DataCopyParams(1, elemCount / S_BLOCK_ELEMENTS, 0, 0));
            } else {
                AscendC::DataCopy(pFloatUb[sUbOffset], compactSGm[compactScoreOffset + scoreOffset],
                                  AscendC::DataCopyParams(currentRows, columnNumRound / S_BLOCK_ELEMENTS,
                                                          (columnNumPad - columnNumRound) / S_BLOCK_ELEMENTS, 0));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpong);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpong);
            ScaleSFromTiling(sUbOffset, currentRows, columnNumRound);
            epilogueOnlineSoftmax.CalcLocalRowMax(sUbOffset, rowRound, columnNum, columnNumRound, rowInLoop);
            epilogueOnlineSoftmax.UpdateGlobalRowMax(currentRows, rowRound, columnNum, columnNumRound, dmOffset,
                                                     rowInLoop, isFirstStack);
            epilogueOnlineSoftmax.CalcExp(sUbOffset, currentRows, rowRound, columnNum, columnNumRound, rowInLoop);
            epilogueOnlineSoftmax.DownCastP(sUbOffset, currentRows, columnNumRound);
            if (UsesResidualProbability()) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(residualUb, pPackedUb[sUbOffset], AscendC::RoundMode::CAST_NONE, elemCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(residualUb, pFloatUb[sUbOffset], residualUb, elemCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(residualUb, residualUb, PROBABILITY_RESIDUAL_SCALE, elemCount);
                AscendC::PipeBarrier<PIPE_V>();
            }
            epilogueOnlineSoftmax.CalcLocalRowSum(sUbOffset, rowRound, columnNum, columnNumRound, rowInLoop);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpong);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpong);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpong);
            if (columnNumRound == columnNumPad) {
                AscendC::DataCopy(pGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                  AscendC::DataCopyParams(1, elemCount / P_BLOCK_ELEMENTS, 0, 0));
            } else {
                AscendC::DataCopy(pGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                  AscendC::DataCopyParams(currentRows, columnNumRound / P_BLOCK_ELEMENTS, 0,
                                                          (columnNumPad - columnNumRound) / P_BLOCK_ELEMENTS));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpong);
            epilogueOnlineSoftmax.UpdateGlobalRowSum(sUbOffset, currentRows, rowRound, dmOffset, rowInLoop,
                                                     isFirstStack);
            if (UsesResidualProbability()) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpong);
                AscendC::Cast(pPackedUb[sUbOffset], residualUb, AscendC::RoundMode::CAST_RINT, elemCount);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpong);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpong);
                if (columnNumRound == columnNumPad) {
                    AscendC::DataCopy(pLoGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                      AscendC::DataCopyParams(1, elemCount / P_BLOCK_ELEMENTS, 0, 0));
                } else {
                    AscendC::DataCopy(pLoGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                      AscendC::DataCopyParams(currentRows, columnNumRound / P_BLOCK_ELEMENTS, 0,
                                                              (columnNumPad - columnNumRound) / P_BLOCK_ELEMENTS));
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpong);
            }
        }
    }

    CATLASS_DEVICE void RunOnlineSoftmax(EpilogueOnlineSoftmax &epilogueOnlineSoftmax, uint64_t pOffset,
                                         const LayoutP &layoutP, const LayoutS &layoutS, GemmCoord qkShape,
                                         bool isFirstStack, uint32_t qRows, uint32_t curStackMod,
                                         uint32_t recurrenceSlot = UINT32_MAX)
    {
        uint32_t stateSlot = recurrenceSlot == UINT32_MAX ? curStackMod : recurrenceSlot;
        if (meta.dim <= FAI_QK_SLICE_DIM) {
            epilogueOnlineSoftmax(pGm[pOffset], sGm[pOffset], layoutP, layoutS, qkShape, isFirstStack, qRows, 1,
                                  stateSlot);
            return;
        }

        constexpr uint32_t P_BLOCK_ELEMENTS = 16;
        constexpr uint32_t MAX_UB_S_ELEMENTS = EpilogueOnlineSoftmax::MAX_UB_S_ELEM_NUM;
        uint32_t rowNum = qkShape.m();
        uint32_t columnNum = qkShape.n();
        uint32_t columnNumRound = StaAlignUp(columnNum, P_BLOCK_ELEMENTS);
        uint32_t columnNumPad = layoutS.stride(0);
        uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t rowSplit = qRows / subBlockNum;
        uint32_t rowActual = subBlockIdx == subBlockNum - 1 ? rowNum - rowSplit : rowSplit;
        uint32_t rowBase = subBlockIdx * rowSplit;
        uint32_t maxRowsPerLoop = MAX_UB_S_ELEMENTS / columnNumRound;
        uint32_t rowTile = maxRowsPerLoop / P_BLOCK_ELEMENTS * P_BLOCK_ELEMENTS;
        uint32_t rowLoops = StaCeilDiv(rowActual, rowTile);
        LocalTensor<float> pFloatUb = resource.ubBuf.template GetBufferByByte<float>(0);
        LocalTensor<ElementP> pPackedUb = resource.ubBuf.template GetBufferByByte<ElementP>(4 * UB_REGION_BYTES);
        LocalTensor<float> residualUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_CENTER_UB_OFFSET);

        for (uint32_t loop = 0; loop < rowLoops + 1; ++loop) {
            if (loop < rowLoops) {
                uint32_t pingpong = loop % 2;
                uint32_t rowInLoop = loop * rowTile;
                uint32_t currentRows = loop == rowLoops - 1 ? rowActual - rowInLoop : rowTile;
                uint32_t gmRow = rowBase + rowInLoop;
                int64_t sGmOffset = layoutS.GetOffset(MatrixCoord(gmRow, 0));
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(pingpong);
                epilogueOnlineSoftmax.CopySGmToUb(sGm[pOffset + sGmOffset], pingpong * MAX_UB_S_ELEMENTS, currentRows,
                                                  columnNumRound, columnNumPad);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(pingpong);
            }
            if (loop == 0) {
                continue;
            }
            uint32_t delayed = loop - 1;
            uint32_t pingpong = delayed % 2;
            uint32_t rowInLoop = delayed * rowTile;
            uint32_t currentRows = delayed == rowLoops - 1 ? rowActual - rowInLoop : rowTile;
            uint32_t gmRow = rowBase + rowInLoop;
            int64_t outputOffset = layoutP.GetOffset(MatrixCoord(gmRow, 0));
            uint32_t sUbOffset = pingpong * MAX_UB_S_ELEMENTS;
            uint32_t rowRound = StaAlignUp(currentRows, EpilogueOnlineSoftmax::FLOAT_BLOCK_SIZE);
            uint32_t dmOffset = stateSlot * EpilogueOnlineSoftmax::MAX_ROW_NUM_SUB_CORE + rowInLoop;
            uint32_t elemCount = currentRows * columnNumRound;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpong);
            ScaleSFromTiling(sUbOffset, currentRows, columnNumRound);
            epilogueOnlineSoftmax.CalcLocalRowMax(sUbOffset, rowRound, columnNum, columnNumRound, rowInLoop);
            epilogueOnlineSoftmax.UpdateGlobalRowMax(currentRows, rowRound, columnNum, columnNumRound, dmOffset,
                                                     rowInLoop, isFirstStack);
            epilogueOnlineSoftmax.CalcExp(sUbOffset, currentRows, rowRound, columnNum, columnNumRound, rowInLoop);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpong);

            epilogueOnlineSoftmax.DownCastP(sUbOffset, currentRows, columnNumRound);
            if (UsesResidualProbability()) {
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(residualUb, pPackedUb[sUbOffset], AscendC::RoundMode::CAST_NONE, elemCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(residualUb, pFloatUb[sUbOffset], residualUb, elemCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Muls(residualUb, residualUb, PROBABILITY_RESIDUAL_SCALE, elemCount);
                AscendC::PipeBarrier<PIPE_V>();
            }
            epilogueOnlineSoftmax.CalcLocalRowSum(sUbOffset, rowRound, columnNum, columnNumRound, rowInLoop);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pingpong);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpong);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpong);
            if (columnNumRound == columnNumPad) {
                AscendC::DataCopy(pGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                  AscendC::DataCopyParams(1, elemCount / P_BLOCK_ELEMENTS, 0, 0));
            } else {
                AscendC::DataCopy(pGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                  AscendC::DataCopyParams(currentRows, columnNumRound / P_BLOCK_ELEMENTS, 0,
                                                          (columnNumPad - columnNumRound) / P_BLOCK_ELEMENTS));
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpong);
            epilogueOnlineSoftmax.UpdateGlobalRowSum(sUbOffset, currentRows, rowRound, dmOffset, rowInLoop,
                                                     isFirstStack);

            if (UsesResidualProbability()) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpong);
                AscendC::Cast(pPackedUb[sUbOffset], residualUb, AscendC::RoundMode::CAST_RINT, elemCount);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pingpong);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pingpong);
                if (columnNumRound == columnNumPad) {
                    AscendC::DataCopy(pLoGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                      AscendC::DataCopyParams(1, elemCount / P_BLOCK_ELEMENTS, 0, 0));
                } else {
                    AscendC::DataCopy(pLoGm[pOffset + outputOffset], pPackedUb[sUbOffset],
                                      AscendC::DataCopyParams(currentRows, columnNumRound / P_BLOCK_ELEMENTS, 0,
                                                              (columnNumPad - columnNumRound) / P_BLOCK_ELEMENTS));
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpong);
            }
        }
    }

    CATLASS_DEVICE void ScaleSFromTiling(uint32_t sUbOffset, uint32_t rowNum, uint32_t columnNumRound)
    {
        LocalTensor<float> sUb = resource.ubBuf.template GetBufferByByte<float>(0);
        LocalTensor<float> scaleBlockUb =
            resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_RUNTIME_SCALE_BLOCK_UB_OFFSET);
        LocalTensor<float> scaleVectorUb =
            resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_RUNTIME_SCALE_VECTOR_UB_OFFSET);
        LocalTensor<uint32_t> scaleBlockBits = scaleBlockUb.ReinterpretCast<uint32_t>();
        for (uint32_t index = 0; index < FLOAT_BLOCK_ELEMENTS; ++index) {
            scaleBlockBits.SetValue(index, softmaxScaleBits);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(scaleVectorUb.ReinterpretCast<uint32_t>(), scaleBlockUb.ReinterpretCast<uint32_t>(), 1,
                      AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul<float, false>(sUb[sUbOffset], sUb[sUbOffset], scaleVectorUb, static_cast<uint64_t>(0),
                                   (rowNum * columnNumRound + FLOAT_VECTOR_ELEMENTS - 1) / FLOAT_VECTOR_ELEMENTS,
                                   AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0));
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void ProcessAicPackedD128Task(uint32_t groupedTaskIdx, BlockMmadQKGrouped &blockMmadQKGrouped,
                                                 BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        TaskInfo infos[QUERY_CHUNKS_PER_TILE];
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
        }
        uint32_t coreIdx = AscendC::GetBlockIdx();
        constexpr uint32_t embed = 128;
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(infos[0].qStart) * embed;
        LayoutQ layoutQ(FAI_QUERY_ROWS, embed, embed);
        LayoutK layoutK(embed, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        blockMmadQKGrouped.LoadQueries(qGm[qOffset], layoutQ, FAI_QUERY_ROWS, FAI_QUERY_ROWS);
        WindowRange range = GetWindowRange(infos[0]);
        ProcessAicPackedImageStacks<true>(infos, range, coreIdx, embed, headBase, layoutK, layoutV, blockMmadQKGrouped,
                                          blockMmadPVGroupedD72);
    }

    CATLASS_DEVICE void ProcessAivPackedD128Task(uint32_t groupedTaskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                 EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo infos[QUERY_CHUNKS_PER_TILE];
        uint64_t outOffsets[QUERY_CHUNKS_PER_TILE];
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            outOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * 128;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        constexpr uint32_t embed = 128;
        constexpr uint32_t embedRound = 128;
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        WindowRange range = GetWindowRange(infos[0]);
        ProcessAivPackedD72Stacks(groupedTaskIdx, infos, outOffsets, range, coreIdx, embed, embedRound, headBase,
                                  epilogueOnlineSoftmax, epilogueRescaleO);
    }

    CATLASS_DEVICE void ProcessAicPackedD72FullTask(uint32_t groupedTaskIdx, BlockMmadQKGrouped &blockMmadQKGrouped,
                                                    BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        TaskInfo infos[QUERY_CHUNKS_PER_TILE];
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
        }
        uint32_t coreIdx = AscendC::GetBlockIdx();
        constexpr uint32_t embed = 72;
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(infos[0].qStart) * embed;
        LayoutQ layoutQ(FAI_QUERY_ROWS, embed, embed);
        LayoutK layoutK(embed, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        blockMmadQKGrouped.LoadQueries(qGm[qOffset], layoutQ, FAI_QUERY_ROWS, FAI_QUERY_ROWS);
        WindowRange range = GetWindowRange(infos[0]);
        ProcessAicPackedImageStacks<false>(infos, range, coreIdx, embed, headBase, layoutK, layoutV, blockMmadQKGrouped,
                                           blockMmadPVGroupedD72);
    }

    CATLASS_DEVICE void ProcessAivPackedD72FullTask(uint32_t groupedTaskIdx,
                                                    EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                    EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo infos[QUERY_CHUNKS_PER_TILE];
        uint64_t outOffsets[QUERY_CHUNKS_PER_TILE];
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            outOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * 72;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        constexpr uint32_t embed = 72;
        constexpr uint32_t embedRound = 80;
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        WindowRange range = GetWindowRange(infos[0]);
        ProcessAivPackedD72Stacks(groupedTaskIdx, infos, outOffsets, range, coreIdx, embed, embedRound, headBase,
                                  epilogueOnlineSoftmax, epilogueRescaleO);
    }

    template <bool INCLUDE_TEXT = false>
    CATLASS_DEVICE void ProcessAicPackedD256Task(uint32_t taskIdx, BlockMmadQKSlicedAccum &blockMmadQKSlicedAccum,
                                                 BlockMmadPVNormal &blockMmadPVNormal,
                                                 BlockMmadPVDynamicSingleBlock &blockMmadPVDynamicSingleBlock,
                                                 BlockMmadPVDynamicMultiBlock &blockMmadPVDynamicMultiBlock)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        constexpr uint32_t embed = 256;
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(info.qStart) * embed;
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        WindowRange range = GetWindowRange(info);
        blockMmadQKSlicedAccum.LoadPackedD256Queries(qInputGm[qOffset], info.qRows, embed);
        uint32_t stackCount = INCLUDE_TEXT ? PackedCombinedImageTextStackCount(range) :
                                             StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = INCLUDE_TEXT ?
                                MakePackedCombinedImageTextStack(info, range, stackSeqCount, headBase, embed) :
                                MakePackedWindow3D256Stack(range, stackSeqCount, headBase);
            ring[slot] = cur;
            uint64_t sOffset = WorkspaceOffset(coreIdx, slot);
            blockMmadQKSlicedAccum.ComputePackedD256Stack(kGm, sGm[sOffset], info.qRows, cur.stackSeqTile,
                                                          FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS,
                                                          cur.packedBlockOffsets);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qkReady);
            if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                RunAicPvStack<true>(taskIdx, info, delayed, coreIdx, embed, embed, layoutV, blockMmadPVNormal,
                                    blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
            }
        }
        uint32_t drainStart = stackCount > FAI_PRELAUNCH_DEPTH ? stackCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAicPvStack<true>(taskIdx, info, delayed, coreIdx, embed, embed, layoutV, blockMmadPVNormal,
                                blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
        }
    }

    template <bool INCLUDE_TEXT = false>
    CATLASS_DEVICE void ProcessAivPackedD256Task(uint32_t taskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                 EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        constexpr uint32_t embed = 256;
        constexpr uint32_t embedRound = 256;
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(info.qStart) * embed;
        uint64_t outOffset = qOffset;
        WindowRange range = GetWindowRange(info);
        uint32_t stackCount = INCLUDE_TEXT ? PackedCombinedImageTextStackCount(range) :
                                             StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = INCLUDE_TEXT ?
                                MakePackedCombinedImageTextStack(info, range, stackSeqCount, headBase, embed) :
                                MakePackedWindow3D256Stack(range, stackSeqCount, headBase);
            ring[slot] = cur;
            uint64_t sOffset = WorkspaceOffset(coreIdx, slot);
            LayoutP layoutP(info.qRows, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            LayoutS layoutS(info.qRows, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
            Arch::CrossCoreWaitFlag(qkReady);
            RunOnlineSoftmax(epilogueOnlineSoftmax, sOffset, layoutP, layoutS, qkShape, stackSeqCount == 0, info.qRows,
                             slot);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
            if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound, stackCount,
                                   epilogueRescaleO);
            }
        }
        uint32_t drainStart = stackCount > FAI_PRELAUNCH_DEPTH ? stackCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound, stackCount,
                               epilogueRescaleO);
        }
    }

    CATLASS_DEVICE void ProcessAicPackedD3SliceTask(uint32_t taskIdx, BlockMmadQKSlicedAccum &blockMmadQKSlicedAccum,
                                                    BlockMmadPVNormal &blockMmadPVNormal,
                                                    BlockMmadPVDynamicSingleBlock &blockMmadPVDynamicSingleBlock,
                                                    BlockMmadPVDynamicMultiBlock &blockMmadPVDynamicMultiBlock)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(info.qStart) * embed;
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        WindowRange range = GetWindowRange(info);
        blockMmadQKSlicedAccum.LoadPackedD3SliceQueries(qInputGm[qOffset], info.qRows, embed);
        uint32_t stackCount = StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedWindow3D3SliceStack(range, stackSeqCount, headBase, embed);
            ring[slot] = cur;
            uint64_t sOffset = WorkspaceOffset(coreIdx, slot);
            blockMmadQKSlicedAccum.ComputePackedD3SliceStack(kGm, sGm[sOffset], info.qRows, cur.stackSeqTile,
                                                             FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed,
                                                             cur.packedBlockOffsets);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qkReady);
            if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                RunAicPvStack<true, true>(taskIdx, info, delayed, coreIdx, embed, embed, layoutV, blockMmadPVNormal,
                                          blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
            }
        }
        uint32_t drainStart = stackCount > FAI_PRELAUNCH_DEPTH ? stackCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAicPvStack<true, true>(taskIdx, info, delayed, coreIdx, embed, embed, layoutV, blockMmadPVNormal,
                                      blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
        }
    }

    CATLASS_DEVICE void ProcessAivPackedD3SliceTask(uint32_t taskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                    EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t embedRound = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t outOffset = headBase + static_cast<uint64_t>(info.qStart) * embed;
        WindowRange range = GetWindowRange(info);
        uint32_t stackCount = StaCeilDiv(PackedImageBlockCount(range), FAI_STACK_BLOCK_COUNT);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedWindow3D3SliceStack(range, stackSeqCount, headBase, embed);
            ring[slot] = cur;
            uint64_t sOffset = WorkspaceOffset(coreIdx, slot);
            LayoutP layoutP(info.qRows, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            LayoutS layoutS(info.qRows, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
            Arch::CrossCoreWaitFlag(qkReady);
            RunOnlineSoftmax(epilogueOnlineSoftmax, sOffset, layoutP, layoutS, qkShape, stackSeqCount == 0, info.qRows,
                             slot);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
            if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound, stackCount,
                                   epilogueRescaleO);
            }
        }
        uint32_t drainStart = stackCount > FAI_PRELAUNCH_DEPTH ? stackCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound, stackCount,
                               epilogueRescaleO);
        }
    }

    CATLASS_DEVICE void ProcessAicGroupedTask(uint32_t groupedTaskIdx, BlockMmadQKGrouped &blockMmadQKGrouped,
                                              BlockMmadPVNormal &blockMmadPVNormal,
                                              BlockMmadPVDynamicSingleBlock &blockMmadPVDynamicSingleBlock,
                                              BlockMmadPVDynamicMultiBlock &blockMmadPVDynamicMultiBlock,
                                              BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        TaskInfo infos[QUERY_CHUNKS_PER_TILE];
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
        }
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t qkEmbed = UsesGroupedD8QkPath() ? QkDim() : embed;
        uint32_t qkStride = UsesGroupedD8QkPath() ? QkStageStride() : embed;
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(infos[0].qStart) * embed;
        LayoutQ layoutQ(FAI_QUERY_ROWS, qkEmbed, qkStride);
        LayoutK layoutK(qkEmbed, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, qkStride);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        if (UsesGroupedD8QkPath()) {
            AscendC::CrossCoreWaitFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(inputQReady.id);
        }
        uint64_t computeQOffset = UsesGroupedD8QkPath() ? QueryStageOffset(coreIdx) : qOffset;
        blockMmadQKGrouped.LoadQueries(qGm[computeQOffset], layoutQ, FAI_QUERY_ROWS, FAI_QUERY_ROWS);

        WindowRange range = GetWindowRange(infos[0]);
        if (UsesPackedD72ImageStacks()) {
            ProcessAicPackedImageStacks<false>(infos, range, coreIdx, embed, headBase, layoutK, layoutV,
                                               blockMmadQKGrouped, blockMmadPVGroupedD72);
            return;
        }
        uint32_t stackSeqCount = 0;
        StackInfo ring[FAI_RING_STAGE_COUNT];
        uint32_t segCount = SegmentCountFromRange(infos[0], range);
        for (uint32_t segIdx = 0; segIdx < segCount; ++segIdx) {
            SegmentInfo seg = GetSegmentFromRange(infos[0], range, segIdx);
            if (seg.len == 0) {
                continue;
            }
            uint64_t kvOffset = headBase + static_cast<uint64_t>(seg.start) * embed;
            uint32_t kvLoopNoMask = StaCeilDiv(seg.len, FAI_KV_BASE_ROWS);
            for (uint32_t kvIdx = 0; kvIdx < kvLoopNoMask; kvIdx += FAI_STACK_BLOCK_COUNT) {
                uint32_t slot = StackSlot(stackSeqCount);
                StackInfo cur;
                cur.kvOffset = kvOffset;
                cur.computeKvOffset = kvOffset;
                cur.computeVOffset = kvOffset;
                cur.segLen = seg.len;
                cur.computeSegLen = seg.len;
                cur.kvLoopNoMask = kvLoopNoMask;
                cur.computeKvLoopNoMask = kvLoopNoMask;
                cur.kvIdx = kvIdx;
                cur.computeKvIdx = kvIdx;
                cur.stackSeqTile = StackSeqTile(seg.len, kvLoopNoMask, kvIdx);
                cur.stackSeqCount = stackSeqCount;
                ring[slot] = cur;

                uint64_t scoreBase = WorkspaceOffset(coreIdx, slot, 0);
                uint64_t scoreChunkStride = static_cast<uint64_t>(workspace.block_elems) * FAI_RING_STAGE_COUNT;
                if (UsesGroupedD8QkPath()) {
                    AscendC::CrossCoreWaitFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(inputKvReady.id);
                }
                uint64_t computeKvOffset = UsesGroupedD8QkPath() ? KvStageOffset(coreIdx, slot) : kvOffset;
                uint32_t computeKvIdx = UsesGroupedD8QkPath() ? 0 : kvIdx;
                uint32_t computeKvBlocks =
                    UsesGroupedD8QkPath() ? StaCeilDiv(cur.stackSeqTile, FAI_KV_BASE_ROWS) : kvLoopNoMask;
                uint32_t computeSegLen = UsesGroupedD8QkPath() ? cur.stackSeqTile : seg.len;
                if (UsesD128GroupedEarlyPublishPath()) {
                    blockMmadQKGrouped.ComputeStackQueryMajor(
                        kGm[computeKvOffset], sGm[scoreBase], layoutK, FAI_QUERY_ROWS, cur.stackSeqTile, computeKvIdx,
                        computeKvBlocks, computeSegLen, qkStride, scoreChunkStride, qkReady);
                } else {
                    blockMmadQKGrouped.ComputeStack(kGm[computeKvOffset], sGm[scoreBase], layoutK, FAI_QUERY_ROWS,
                                                    cur.stackSeqTile, computeKvIdx, computeKvBlocks, computeSegLen,
                                                    qkStride, scoreChunkStride, qkReady);
                }

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    if (UsesGroupedD72PvPath()) {
                        RunAicGroupedD72PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
                    } else if (UsesGroupedD8QkPath() || UsesGroupedD16PvPath()) {
                        RunAicGroupedD8PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
                    } else {
                        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                            uint32_t chunkTaskIdx = GroupedChunkTaskIndex(groupedTaskIdx, chunk);
                            RunAicPvStack(chunkTaskIdx, infos[chunk], delayed, coreIdx, embed, embed, layoutV,
                                          blockMmadPVNormal, blockMmadPVDynamicSingleBlock,
                                          blockMmadPVDynamicMultiBlock, chunk);
                        }
                    }
                }
                ++stackSeqCount;
            }
        }

        uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            if (UsesGroupedD72PvPath()) {
                RunAicGroupedD72PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
            } else if (UsesGroupedD8QkPath() || UsesGroupedD16PvPath()) {
                RunAicGroupedD8PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
            } else {
                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    uint32_t chunkTaskIdx = GroupedChunkTaskIndex(groupedTaskIdx, chunk);
                    RunAicPvStack(chunkTaskIdx, infos[chunk], delayed, coreIdx, embed, embed, layoutV,
                                  blockMmadPVNormal, blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock,
                                  chunk);
                }
            }
        }
    }

    CATLASS_DEVICE void RunAicGroupedPairD16PvStack(const TaskInfo (&infos)[6], uint32_t queryCount,
                                                    const StackInfo &stack, uint32_t coreIdx, uint32_t embed,
                                                    const LayoutV &layoutV, BlockMmadPVGroupedPairD16 &blockMmadPV)
    {
        uint32_t slot = StackSlot(stack.stackSeqCount);
        blockMmadPV.LoadV(vGm, layoutV, stack.stackSeqTile, 0, stack.packedBlockCount, stack.stackSeqTile, embed,
                          stack.packedBlockOffsets);
        for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
            uint64_t offset = WorkspaceOffset(coreIdx, slot, chunk);
            LayoutP layoutP(infos[chunk].qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            blockMmadPV.ComputeQuery(pGm[offset], oTmpGm[offset], layoutP, infos[chunk].qRows, embed,
                                     stack.stackSeqTile, StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS), softmaxReady, chunk);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
        }
        blockMmadPV.ReleaseV();
    }

    CATLASS_DEVICE void ProcessAicGroupedPairD16Task(uint32_t groupedTaskIdx, BlockMmadQKGroupedPairD16 &blockMmadQK,
                                                     BlockMmadPVGroupedPairD16 &blockMmadPV)
    {
        uint32_t queryCount = GroupedQueryCount(groupedTaskIdx);
        TaskInfo infos[6];
        uint64_t qOffsets[6];
        for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            qOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * meta.dim;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        LayoutQ layoutQ(FAI_QUERY_ROWS, embed, embed);
        LayoutK layoutK(embed, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        blockMmadQK.LoadQueriesOffsets(qGm, layoutQ, FAI_QUERY_ROWS, qOffsets, queryCount);
        WindowRange range = GetWindowRange(infos[0]);
        uint32_t stackCount = PackedImageTextStackCount(range);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedImageOrTextStack(infos[0], range, stackSeqCount, headBase, embed);
            ring[slot] = cur;
            uint64_t scoreBase = WorkspaceOffset(coreIdx, slot, 0);
            uint64_t scoreChunkStride = static_cast<uint64_t>(workspace.block_elems) * FAI_RING_STAGE_COUNT;
            blockMmadQK.ComputeStack(kGm, sGm[scoreBase], layoutK, FAI_QUERY_ROWS, cur.stackSeqTile, 0,
                                     cur.packedBlockCount, cur.stackSeqTile, embed, scoreChunkStride, qkReady,
                                     cur.packedBlockOffsets, queryCount);
            if (stackSeqCount >= D16_PAIR_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - D16_PAIR_PRELAUNCH_DEPTH)];
                RunAicGroupedPairD16PvStack(infos, queryCount, delayed, coreIdx, embed, layoutV, blockMmadPV);
            }
        }
        uint32_t drainStart = stackCount > D16_PAIR_PRELAUNCH_DEPTH ? stackCount - D16_PAIR_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAicGroupedPairD16PvStack(infos, queryCount, delayed, coreIdx, embed, layoutV, blockMmadPV);
        }
    }

    CATLASS_DEVICE void ProcessAicGroupedTextD16Task(uint32_t groupedTaskIdx, BlockMmadQKGroupedPairD16 &blockMmadQK,
                                                     BlockMmadPVGroupedPairD16 &blockMmadPV)
    {
        uint32_t queryCount = TextChunksPerHead();
        TaskInfo infos[6];
        uint64_t qOffsets[6];
        for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
            uint32_t taskIdx = GroupedChunkTaskIndex(groupedTaskIdx, chunk);
            infos[chunk] = DecodeTask(taskIdx);
            if (infos[chunk].isText == 0 || infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            qOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * meta.dim;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        WindowRange range = GetWindowRange(infos[0]);
        LayoutQ layoutQ(FAI_QUERY_ROWS, embed, embed);
        LayoutK layoutK(embed, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed);
        blockMmadQK.LoadQueriesOffsets(qGm, layoutQ, FAI_QUERY_ROWS, qOffsets, queryCount);
        uint32_t stackCount = PackedImageTextStackCount(range);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedImageOrTextStack(infos[0], range, stackSeqCount, headBase, embed);
            ring[slot] = cur;
            uint64_t scoreBase = WorkspaceOffset(coreIdx, slot, 0);
            uint64_t scoreChunkStride = static_cast<uint64_t>(workspace.block_elems) * FAI_RING_STAGE_COUNT;
            blockMmadQK.ComputeStack(kGm, sGm[scoreBase], layoutK, FAI_QUERY_ROWS, cur.stackSeqTile, 0,
                                     cur.packedBlockCount, cur.stackSeqTile, embed, scoreChunkStride, qkReady,
                                     cur.packedBlockOffsets, queryCount);
            if (stackSeqCount >= D16_PAIR_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - D16_PAIR_PRELAUNCH_DEPTH)];
                RunAicGroupedPairD16PvStack(infos, queryCount, delayed, coreIdx, embed, layoutV, blockMmadPV);
            }
        }
        uint32_t drainStart = stackCount > D16_PAIR_PRELAUNCH_DEPTH ? stackCount - D16_PAIR_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAicGroupedPairD16PvStack(infos, queryCount, delayed, coreIdx, embed, layoutV, blockMmadPV);
        }
    }

    template <bool D128>
    CATLASS_DEVICE void ProcessAicPackedImageStacks(const TaskInfo (&infos)[QUERY_CHUNKS_PER_TILE],
                                                    const WindowRange &range, uint32_t coreIdx, uint32_t embed,
                                                    uint64_t headBase, const LayoutK &layoutK, const LayoutV &layoutV,
                                                    BlockMmadQKGrouped &blockMmadQKGrouped,
                                                    BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        uint32_t stackCount = PackedImageTextStackCount(range);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < stackCount; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedImageOrTextStack(infos[0], range, stackSeqCount, headBase, embed);
            ring[slot] = cur;

            uint64_t scoreBase = WorkspaceOffset(coreIdx, slot, 0);
            uint64_t scoreChunkStride = static_cast<uint64_t>(workspace.block_elems) * FAI_RING_STAGE_COUNT;
            blockMmadQKGrouped.ComputeStack(kGm, sGm[scoreBase], layoutK, FAI_QUERY_ROWS, cur.stackSeqTile, 0,
                                            cur.packedBlockCount, cur.stackSeqTile, embed, scoreChunkStride, qkReady,
                                            cur.packedBlockOffsets);

            if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                if constexpr (D128) {
                    RunAicPackedD128PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
                } else {
                    RunAicGroupedD72PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
                }
            }
        }

        uint32_t drainStart = stackCount > FAI_PRELAUNCH_DEPTH ? stackCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            if constexpr (D128) {
                RunAicPackedD128PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
            } else {
                RunAicGroupedD72PvStack(infos, delayed, coreIdx, embed, layoutV, blockMmadPVGroupedD72);
            }
        }
    }

    CATLASS_DEVICE void RunAicPackedD128PvStack(const TaskInfo (&infos)[QUERY_CHUNKS_PER_TILE], const StackInfo &stack,
                                                uint32_t coreIdx, uint32_t embed, const LayoutV &layoutV,
                                                BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        uint32_t slot = StackSlot(stack.stackSeqCount);
        blockMmadPVGroupedD72.LoadV(vGm, layoutV, stack.stackSeqTile, 0, stack.packedBlockCount, stack.stackSeqTile,
                                    embed, stack.packedBlockOffsets);
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            uint64_t offset = WorkspaceOffset(coreIdx, slot, chunk);
            LayoutP layoutP(infos[chunk].qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            blockMmadPVGroupedD72.ComputeQuerySingleD128(pGm[offset], oTmpGm[offset], layoutP, infos[chunk].qRows,
                                                         stack.stackSeqTile, embed, softmaxReady);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
        }
        blockMmadPVGroupedD72.ReleaseV();
    }

    CATLASS_DEVICE void RunAicGroupedD72PvStack(const TaskInfo (&infos)[QUERY_CHUNKS_PER_TILE], const StackInfo &stack,
                                                uint32_t coreIdx, uint32_t embed, const LayoutV &layoutV,
                                                BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        uint32_t slot = StackSlot(stack.stackSeqCount);
        if (stack.packedBlockCount != 0) {
            blockMmadPVGroupedD72.LoadV(vGm, layoutV, stack.stackSeqTile, 0, stack.packedBlockCount, stack.stackSeqTile,
                                        embed, stack.packedBlockOffsets);
        } else {
            blockMmadPVGroupedD72.LoadV(vGm[stack.computeVOffset], layoutV, stack.stackSeqTile, stack.computeKvIdx,
                                        stack.computeKvLoopNoMask, stack.computeSegLen, embed);
        }
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            uint64_t offset = WorkspaceOffset(coreIdx, slot, chunk);
            LayoutP layoutP(infos[chunk].qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            blockMmadPVGroupedD72.ComputeQuery(pGm[offset], oTmpGm[offset], layoutP, infos[chunk].qRows, embed,
                                               stack.stackSeqTile, StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS),
                                               softmaxReady, chunk);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
        }
        blockMmadPVGroupedD72.ReleaseV();
    }

    CATLASS_DEVICE void RunAicGroupedD8PvStack(const TaskInfo (&infos)[QUERY_CHUNKS_PER_TILE], const StackInfo &stack,
                                               uint32_t coreIdx, uint32_t embed, const LayoutV &layoutV,
                                               BlockMmadPVGroupedD72 &blockMmadPVGroupedD72)
    {
        uint32_t slot = StackSlot(stack.stackSeqCount);
        // The grouped helper retains each V block in L1 while all three 128-row queries consume it.
        blockMmadPVGroupedD72.LoadV(vGm[stack.computeVOffset], layoutV, stack.stackSeqTile, stack.computeKvIdx,
                                    stack.computeKvLoopNoMask, stack.computeSegLen, embed);
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            uint64_t offset = WorkspaceOffset(coreIdx, slot, chunk);
            LayoutP layoutP(infos[chunk].qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
            blockMmadPVGroupedD72.ComputeQuery(pGm[offset], oTmpGm[offset], layoutP, infos[chunk].qRows, embed,
                                               stack.stackSeqTile, StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS),
                                               softmaxReady, chunk);
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
        }
        blockMmadPVGroupedD72.ReleaseV();
    }

    CATLASS_DEVICE void RunAivGroupedRescaleStack(uint32_t groupedTaskIdx, const StackInfo &stack, uint32_t coreIdx,
                                                  const uint64_t (&outOffsets)[QUERY_CHUNKS_PER_TILE], uint32_t embed,
                                                  uint32_t embedRound, uint32_t totalStackSeqNum,
                                                  EpilogueRescaleO &epilogueRescaleO)
    {
        if (!UsesGroupedD72PvPath()) {
            for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                RestoreSoftmaxState(coreIdx, chunk, FAI_QUERY_ROWS, false);
                RestoreSoftmaxDm(coreIdx, chunk, StackSlot(stack.stackSeqCount), FAI_QUERY_ROWS);
                if (stack.stackSeqCount != 0) {
                    RestoreGroupedGoState(coreIdx, chunk, FAI_QUERY_ROWS, embedRound);
                }
                RunAivRescaleStack(GroupedChunkTaskIndex(groupedTaskIdx, chunk), stack, coreIdx, outOffsets[chunk],
                                   FAI_QUERY_ROWS, embed, embedRound, totalStackSeqNum, epilogueRescaleO, chunk);
                if (stack.stackSeqCount + 1 < totalStackSeqNum) {
                    SaveGroupedGoState(coreIdx, chunk, FAI_QUERY_ROWS, embedRound);
                }
            }
            return;
        }
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            Arch::CrossCoreWaitFlag(pvReady);
        }
        uint32_t startChunk =
            (QUERY_CHUNKS_PER_TILE - stack.stackSeqCount % QUERY_CHUNKS_PER_TILE) % QUERY_CHUNKS_PER_TILE;
        for (uint32_t orderIdx = 0; orderIdx < QUERY_CHUNKS_PER_TILE; ++orderIdx) {
            uint32_t chunk = (startChunk + orderIdx) % QUERY_CHUNKS_PER_TILE;
            RestoreGroupedSoftmaxStateUb(chunk, false);
            RestoreGroupedSoftmaxDmUb(chunk, StackSlot(stack.stackSeqCount));
            bool consumesResidentGo = stack.stackSeqCount != 0 && orderIdx == 0;
            if (stack.stackSeqCount != 0 && !consumesResidentGo) {
                RestoreGroupedGoState(coreIdx, chunk, FAI_QUERY_ROWS, embedRound);
            }
            RunAivRescaleStack(GroupedChunkTaskIndex(groupedTaskIdx, chunk), stack, coreIdx, outOffsets[chunk],
                               FAI_QUERY_ROWS, embed, embedRound, totalStackSeqNum, epilogueRescaleO, chunk, false);
            bool keepsResidentGo = stack.stackSeqCount + 1 < totalStackSeqNum && orderIdx + 1 == QUERY_CHUNKS_PER_TILE;
            if (stack.stackSeqCount + 1 < totalStackSeqNum && !keepsResidentGo) {
                SaveGroupedGoState(coreIdx, chunk, FAI_QUERY_ROWS, embedRound);
            }
        }
    }

    CATLASS_DEVICE void ProcessAivGroupedTask(uint32_t groupedTaskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                              EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo infos[QUERY_CHUNKS_PER_TILE];
        uint64_t outOffsets[QUERY_CHUNKS_PER_TILE];
        for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            outOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * meta.dim;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t embedRound = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        WindowRange range = GetWindowRange(infos[0]);
        uint32_t totalStackSeqNum = TotalStackSeqNumFromRange(infos[0], range);
        uint32_t stackSeqCount = 0;
        StackInfo ring[FAI_RING_STAGE_COUNT];
        StageGroupedQueries(infos, coreIdx);
        if (UsesPackedD72ImageStacks()) {
            ProcessAivPackedD72Stacks(groupedTaskIdx, infos, outOffsets, range, coreIdx, embed, embedRound, headBase,
                                      epilogueOnlineSoftmax, epilogueRescaleO);
            return;
        }
        uint32_t segCount = SegmentCountFromRange(infos[0], range);
        for (uint32_t segIdx = 0; segIdx < segCount; ++segIdx) {
            SegmentInfo seg = GetSegmentFromRange(infos[0], range, segIdx);
            if (seg.len == 0) {
                continue;
            }
            uint64_t kvOffset = headBase + static_cast<uint64_t>(seg.start) * embed;
            uint32_t kvLoopNoMask = StaCeilDiv(seg.len, FAI_KV_BASE_ROWS);
            for (uint32_t kvIdx = 0; kvIdx < kvLoopNoMask; kvIdx += FAI_STACK_BLOCK_COUNT) {
                uint32_t slot = StackSlot(stackSeqCount);
                StackInfo cur;
                cur.kvOffset = kvOffset;
                cur.computeKvOffset = kvOffset;
                cur.computeVOffset = kvOffset;
                cur.segLen = seg.len;
                cur.computeSegLen = seg.len;
                cur.kvLoopNoMask = kvLoopNoMask;
                cur.computeKvLoopNoMask = kvLoopNoMask;
                cur.kvIdx = kvIdx;
                cur.computeKvIdx = kvIdx;
                cur.stackSeqTile = StackSeqTile(seg.len, kvLoopNoMask, kvIdx);
                cur.stackSeqCount = stackSeqCount;
                ring[slot] = cur;

                if (UsesGroupedD8QkPath()) {
                    StageKv(cur, GroupedChunkTaskIndex(groupedTaskIdx, 0), coreIdx, slot);
                }

                for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                    uint64_t workspaceOffset = WorkspaceOffset(coreIdx, slot, chunk);
                    if (stackSeqCount != 0) {
                        if (UsesGroupedD72PvPath()) {
                            RestoreGroupedSoftmaxStateUb(chunk, true);
                        } else {
                            RestoreSoftmaxState(coreIdx, chunk, FAI_QUERY_ROWS, true);
                        }
                    }
                    LayoutS layoutS(FAI_QUERY_ROWS, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                    LayoutP layoutP(FAI_QUERY_ROWS, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                    GemmCoord qkShape{FAI_QUERY_ROWS, cur.stackSeqTile, embed};
                    Arch::CrossCoreWaitFlag(qkReady);
                    RunCompactOnlineSoftmax(epilogueOnlineSoftmax, workspaceOffset, workspaceOffset, layoutP, layoutS,
                                            qkShape, stackSeqCount == 0, FAI_QUERY_ROWS, slot);
                    if (UsesGroupedD72PvPath()) {
                        SaveGroupedSoftmaxStateUb(chunk);
                        SaveGroupedSoftmaxDmUb(chunk, slot);
                    } else {
                        SaveSoftmaxState(coreIdx, chunk, FAI_QUERY_ROWS);
                        SaveSoftmaxDm(coreIdx, chunk, slot, FAI_QUERY_ROWS);
                    }
                    Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
                }

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    RunAivGroupedRescaleStack(groupedTaskIdx, delayed, coreIdx, outOffsets, embed, embedRound,
                                              totalStackSeqNum, epilogueRescaleO);
                }
                ++stackSeqCount;
            }
        }

        uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivGroupedRescaleStack(groupedTaskIdx, delayed, coreIdx, outOffsets, embed, embedRound, totalStackSeqNum,
                                      epilogueRescaleO);
        }
    }

    CATLASS_DEVICE void RunAivGroupedPairD16Rescale(uint32_t groupedTaskIdx, const TaskInfo (&infos)[6],
                                                    uint32_t queryCount, const StackInfo &stack, uint32_t coreIdx,
                                                    const uint64_t (&outOffsets)[6], uint32_t embed,
                                                    uint32_t embedRound, uint32_t totalStackSeqNum,
                                                    EpilogueRescaleO &epilogueRescaleO)
    {
        for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
            RestoreGroupedSoftmaxStateUb(chunk, false);
            RestoreGroupedSoftmaxDmUb(chunk, StackSlot(stack.stackSeqCount));
            if (stack.stackSeqCount != 0) {
                RestoreGroupedGoState(coreIdx, chunk, FAI_QUERY_ROWS, embedRound);
            }
            RunAivRescaleStack(GroupedChunkTaskIndex(groupedTaskIdx, chunk), stack, coreIdx, outOffsets[chunk],
                               infos[chunk].qRows, embed, embedRound, totalStackSeqNum, epilogueRescaleO, chunk);
            if (stack.stackSeqCount + 1 < totalStackSeqNum) {
                SaveGroupedGoState(coreIdx, chunk, FAI_QUERY_ROWS, embedRound);
            }
            // pairSlotDrained: the five-slot age prevents overwrite until recurrence state is saved.
        }
    }

    CATLASS_DEVICE void ProcessAivGroupedTextD16Task(uint32_t groupedTaskIdx,
                                                     EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                     EpilogueRescaleO &epilogueRescaleO)
    {
        uint32_t queryCount = TextChunksPerHead();
        TaskInfo infos[6];
        uint32_t taskIndices[6];
        uint64_t outOffsets[6];
        for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
            taskIndices[chunk] = GroupedChunkTaskIndex(groupedTaskIdx, chunk);
            infos[chunk] = DecodeTask(taskIndices[chunk]);
            if (infos[chunk].isText == 0 || infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            outOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * meta.dim;
        }
        // OPFORGE_DEVICE_INLINE_LAYOUT_FORMULA: normalize logical AIV lanes to the physical
        // workspace owner proven by ValidateAivPhysicalWorkspaceOwner in resource_ledger.hpp.
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t embedRound = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        WindowRange range = GetWindowRange(infos[0]);
        uint32_t totalStackSeqNum = PackedImageTextStackCount(range);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < totalStackSeqNum; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedImageOrTextStack(infos[0], range, stackSeqCount, headBase, embed);
            ring[slot] = cur;
            for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
                uint64_t workspaceOffset = WorkspaceOffset(coreIdx, slot, chunk);
                if (stackSeqCount != 0) {
                    RestoreGroupedSoftmaxStateUb(chunk, true);
                }
                LayoutS layoutS(infos[chunk].qRows, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                LayoutP layoutP(infos[chunk].qRows, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                GemmCoord qkShape{infos[chunk].qRows, cur.stackSeqTile, embed};
                Arch::CrossCoreWaitFlag(qkReady);
                RunCompactOnlineSoftmax(epilogueOnlineSoftmax, workspaceOffset, workspaceOffset, layoutP, layoutS,
                                        qkShape, stackSeqCount == 0, infos[chunk].qRows, slot);
                SaveGroupedSoftmaxStateUb(chunk);
                SaveGroupedSoftmaxDmUb(chunk, slot);
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
            }
            if (stackSeqCount >= D16_PAIR_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - D16_PAIR_PRELAUNCH_DEPTH)];
                RunAivGroupedPairD16Rescale(groupedTaskIdx, infos, queryCount, delayed, coreIdx, outOffsets, embed,
                                            embedRound, totalStackSeqNum, epilogueRescaleO);
            }
        }
        uint32_t drainStart =
            totalStackSeqNum > D16_PAIR_PRELAUNCH_DEPTH ? totalStackSeqNum - D16_PAIR_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < totalStackSeqNum; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivGroupedPairD16Rescale(groupedTaskIdx, infos, queryCount, delayed, coreIdx, outOffsets, embed,
                                        embedRound, totalStackSeqNum, epilogueRescaleO);
        }
    }

    CATLASS_DEVICE void ProcessAivGroupedPairD16Task(uint32_t groupedTaskIdx,
                                                     EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                     EpilogueRescaleO &epilogueRescaleO)
    {
        uint32_t queryCount = GroupedQueryCount(groupedTaskIdx);
        TaskInfo infos[6];
        uint64_t outOffsets[6];
        for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
            infos[chunk] = DecodeTask(GroupedChunkTaskIndex(groupedTaskIdx, chunk));
            if (infos[chunk].qRows != FAI_QUERY_ROWS) {
                return;
            }
            outOffsets[chunk] =
                HeadBase(infos[chunk].batch, infos[chunk].head) + static_cast<uint64_t>(infos[chunk].qStart) * meta.dim;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t embedRound = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
        uint64_t headBase = HeadBase(infos[0].batch, infos[0].head);
        WindowRange range = GetWindowRange(infos[0]);
        uint32_t totalStackSeqNum = PackedImageTextStackCount(range);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < totalStackSeqNum; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedImageOrTextStack(infos[0], range, stackSeqCount, headBase, embed);
            ring[slot] = cur;
            for (uint32_t chunk = 0; chunk < queryCount; ++chunk) {
                uint64_t workspaceOffset = WorkspaceOffset(coreIdx, slot, chunk);
                if (stackSeqCount != 0) {
                    RestoreGroupedSoftmaxStateUb(chunk, true);
                }
                LayoutS layoutS(FAI_QUERY_ROWS, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                LayoutP layoutP(FAI_QUERY_ROWS, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                GemmCoord qkShape{FAI_QUERY_ROWS, cur.stackSeqTile, embed};
                Arch::CrossCoreWaitFlag(qkReady);
                RunCompactOnlineSoftmax(epilogueOnlineSoftmax, workspaceOffset, workspaceOffset, layoutP, layoutS,
                                        qkShape, stackSeqCount == 0, FAI_QUERY_ROWS, slot);
                SaveGroupedSoftmaxStateUb(chunk);
                SaveGroupedSoftmaxDmUb(chunk, slot);
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
            }
            if (stackSeqCount >= D16_PAIR_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - D16_PAIR_PRELAUNCH_DEPTH)];
                RunAivGroupedPairD16Rescale(groupedTaskIdx, infos, queryCount, delayed, coreIdx, outOffsets, embed,
                                            embedRound, totalStackSeqNum, epilogueRescaleO);
            }
        }
        uint32_t drainStart =
            totalStackSeqNum > D16_PAIR_PRELAUNCH_DEPTH ? totalStackSeqNum - D16_PAIR_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < totalStackSeqNum; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivGroupedPairD16Rescale(groupedTaskIdx, infos, queryCount, delayed, coreIdx, outOffsets, embed,
                                        embedRound, totalStackSeqNum, epilogueRescaleO);
        }
    }

    CATLASS_DEVICE void ProcessAivPackedD72Stacks(uint32_t groupedTaskIdx,
                                                  const TaskInfo (&infos)[QUERY_CHUNKS_PER_TILE],
                                                  const uint64_t (&outOffsets)[QUERY_CHUNKS_PER_TILE],
                                                  const WindowRange &range, uint32_t coreIdx, uint32_t embed,
                                                  uint32_t embedRound, uint64_t headBase,
                                                  EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                                  EpilogueRescaleO &epilogueRescaleO)
    {
        uint32_t totalStackSeqNum = PackedImageTextStackCount(range);
        StackInfo ring[FAI_RING_STAGE_COUNT];
        for (uint32_t stackSeqCount = 0; stackSeqCount < totalStackSeqNum; ++stackSeqCount) {
            uint32_t slot = StackSlot(stackSeqCount);
            StackInfo cur = MakePackedImageOrTextStack(infos[0], range, stackSeqCount, headBase, embed);
            ring[slot] = cur;

            for (uint32_t chunk = 0; chunk < QUERY_CHUNKS_PER_TILE; ++chunk) {
                uint64_t workspaceOffset = WorkspaceOffset(coreIdx, slot, chunk);
                if (stackSeqCount != 0) {
                    if (UsesGroupedD72PvPath()) {
                        RestoreGroupedSoftmaxStateUb(chunk, true);
                    } else {
                        RestoreSoftmaxState(coreIdx, chunk, FAI_QUERY_ROWS, true);
                    }
                }
                LayoutS layoutS(FAI_QUERY_ROWS, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                LayoutP layoutP(FAI_QUERY_ROWS, cur.stackSeqTile, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
                GemmCoord qkShape{FAI_QUERY_ROWS, cur.stackSeqTile, embed};
                Arch::CrossCoreWaitFlag(qkReady);
                RunCompactOnlineSoftmax(epilogueOnlineSoftmax, workspaceOffset, workspaceOffset, layoutP, layoutS,
                                        qkShape, stackSeqCount == 0, FAI_QUERY_ROWS, slot);
                if (UsesGroupedD72PvPath()) {
                    SaveGroupedSoftmaxStateUb(chunk);
                    SaveGroupedSoftmaxDmUb(chunk, slot);
                } else {
                    SaveSoftmaxState(coreIdx, chunk, FAI_QUERY_ROWS);
                    SaveSoftmaxDm(coreIdx, chunk, slot, FAI_QUERY_ROWS);
                }
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
            }

            if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                RunAivGroupedRescaleStack(groupedTaskIdx, delayed, coreIdx, outOffsets, embed, embedRound,
                                          totalStackSeqNum, epilogueRescaleO);
            }
        }

        uint32_t drainStart = totalStackSeqNum > FAI_PRELAUNCH_DEPTH ? totalStackSeqNum - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < totalStackSeqNum; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivGroupedRescaleStack(groupedTaskIdx, delayed, coreIdx, outOffsets, embed, embedRound, totalStackSeqNum,
                                      epilogueRescaleO);
        }
    }

    template <typename QkBlock, typename ScoreElement>
    CATLASS_DEVICE void ProcessAicTask(uint32_t taskIdx, QkBlock &blockMmadQK, BlockMmadQKPaged &blockMmadQKPaged,
                                       GlobalTensor<ScoreElement> &scoreGm, bool compactScore,
                                       BlockMmadQKSlicedAccum &blockMmadQKSlicedAccum,
                                       BlockMmadPVNormal &blockMmadPVNormal, BlockMmadPVPaged &blockMmadPVPaged,
                                       BlockMmadPVDynamicSingleBlock &blockMmadPVDynamicSingleBlock,
                                       BlockMmadPVDynamicMultiBlock &blockMmadPVDynamicMultiBlock)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(info.qStart) * meta.dim;
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t qkEmbed = QkDim();
        uint32_t qkStride = QkComputeStride();
        uint32_t strideKV = embed;
        uint32_t qNBlockSize = 1;
        uint32_t qStrideHeads = 1;
        uint32_t stackSeqCount = 0;
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        const bool debugFirstD256Task =
            meta.dim > FAI_QK_SLICE_DIM && taskIdx < AscendC::GetBlockNum() * QUERY_CHUNKS_PER_TILE;
        if (debugFirstD256Task) {
            AscendC::printf("STA_AIC_STAGE phase=task_enter dim=%u task=%u block=%u\\n",
                            static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
        }
#endif
        StackInfo ring[FAI_RING_STAGE_COUNT];
        // CATLASS loadQGM derives the ND source row stride from shape(1), so staged Q must expose its padded width.
        uint32_t qLoadEmbed = NeedsQkStaging() && qkEmbed <= FAI_KV_BASE_ROWS ? qkStride : embed;
        LayoutQ layoutQ(info.qRows, qLoadEmbed, qkStride);
        LayoutK layoutK(embed, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, qkStride);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, strideKV);
        WindowRange range;
        if (info.isText == 0) {
            range = GetWindowRange(info);
        }

        if (NeedsQkStaging()) {
            AscendC::CrossCoreWaitFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(inputQReady.id);
        }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        if (debugFirstD256Task) {
            AscendC::printf("STA_AIC_STAGE phase=q_ready dim=%u task=%u block=%u\\n", static_cast<uint32_t>(meta.dim),
                            taskIdx, AscendC::GetBlockIdx());
        }
#endif
        uint64_t computeQOffset = NeedsQkStaging() ? QueryStageOffset(coreIdx) : qOffset;
        if (qkEmbed <= FAI_KV_BASE_ROWS) {
            blockMmadQK.loadQGM(qGm[computeQOffset], layoutQ, info.qRows, qNBlockSize, qStrideHeads);
        }
        if (UsesPagedD64ImageStacks() && info.isText == 0) {
            uint32_t totalStackSeqNum = TotalStackSeqNumFromRange(info, range);
            for (stackSeqCount = 0; stackSeqCount < totalStackSeqNum; ++stackSeqCount) {
                uint32_t curSlot = StackSlot(stackSeqCount);
                StackInfo cur = MakePagedD64TextStack(info, range, stackSeqCount, headBase, embed);
                SetComputeStackInfo(cur, coreIdx, curSlot);
                if (cur.usesPagedTable) {
                    StoreStackBlockTable(cur, coreIdx, curSlot);
                }
                ring[curSlot] = cur;

                uint64_t sOffset = WorkspaceOffset(coreIdx, curSlot);
                GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
                uint32_t blockSize = FAI_KV_BASE_ROWS;
                if (cur.usesPagedTable) {
                    uint32_t pagedKvIdx = 0;
                    uint32_t pagedKvLoop = cur.packedBlockCount;
                    blockMmadQKPaged(qGm[computeQOffset], kGm[cur.computeKvOffset], scoreGm[sOffset],
                                     StackBlockTable(coreIdx, curSlot), layoutQ, layoutK, qkShape, pagedKvIdx,
                                     pagedKvLoop, blockSize, cur.computeSegLen, qkStride);
                } else {
                    blockMmadQK(qGm[computeQOffset], kGm[cur.computeKvOffset], scoreGm[sOffset], dummyBlockTable,
                                layoutQ, layoutK, qkShape, cur.computeKvIdx, cur.computeKvLoopNoMask, blockSize,
                                cur.computeSegLen, qkStride);
                }
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qkReady);

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    if (delayed.usesPagedTable) {
                        RunAicPagedPvStack(info, delayed, coreIdx, layoutV, blockMmadPVPaged);
                    } else {
                        RunAicPvStack(taskIdx, info, delayed, coreIdx, embed, strideKV, layoutV, blockMmadPVNormal,
                                      blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                    }
                }
            }
            uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
            for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
                const StackInfo &delayed = ring[StackSlot(delayedIdx)];
                if (delayed.usesPagedTable) {
                    RunAicPagedPvStack(info, delayed, coreIdx, layoutV, blockMmadPVPaged);
                } else {
                    RunAicPvStack(taskIdx, info, delayed, coreIdx, embed, strideKV, layoutV, blockMmadPVNormal,
                                  blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                }
            }
            return;
        }
        uint32_t segCount = SegmentCountFromRange(info, range);
        for (uint32_t segIdx = 0; segIdx < segCount; ++segIdx) {
            SegmentInfo seg = GetSegmentFromRange(info, range, segIdx);
            if (seg.len == 0) {
                continue;
            }
            uint64_t kvOffset = headBase + static_cast<uint64_t>(seg.start) * meta.dim;
            uint32_t kvLoopNoMask = StaCeilDiv(seg.len, FAI_KV_BASE_ROWS);
            for (uint32_t kvIdx = 0; kvIdx < kvLoopNoMask; kvIdx += FAI_STACK_BLOCK_COUNT) {
                uint32_t curSlot = StackSlot(stackSeqCount);
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
                const bool debugRingReuseStack = debugFirstD256Task && stackSeqCount == FAI_RING_STAGE_COUNT;
                if (debugRingReuseStack) {
                    AscendC::printf("STA_AIC_STAGE phase=ring_reuse_enter dim=%u task=%u block=%u\\n",
                                    static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
                }
#endif
                StackInfo cur;
                cur.kvOffset = kvOffset;
                cur.segLen = seg.len;
                cur.kvLoopNoMask = kvLoopNoMask;
                cur.kvIdx = kvIdx;
                cur.stackSeqTile = StackSeqTile(seg.len, kvLoopNoMask, kvIdx);
                cur.stackSeqCount = stackSeqCount;
                SetComputeStackInfo(cur, coreIdx, curSlot);
                ring[curSlot] = cur;

                uint32_t sPingPong = curSlot;
                uint64_t sOffset = compactScore ?
                                       WorkspaceOffset(coreIdx, sPingPong) * (sizeof(ElementS) / sizeof(ScoreElement)) :
                                       WorkspaceOffset(coreIdx, sPingPong);
                GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
                if (NeedsKvQkStaging()) {
                    AscendC::CrossCoreWaitFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(inputKvReady.id);
                }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
                const bool debugFirstQkStack = debugFirstD256Task && segIdx == 0 && kvIdx == 0;
                if (debugFirstQkStack) {
                    AscendC::printf("STA_AIC_STAGE phase=kv_ready dim=%u task=%u block=%u\\n",
                                    static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
                }
#endif
                uint64_t qkKvOffset = NeedsKvQkStaging() ? KvStageOffset(coreIdx, curSlot) : cur.computeKvOffset;
                uint32_t qkKvIdx = NeedsKvQkStaging() ? 0 : cur.computeKvIdx;
                uint32_t qkKvLoopNoMask =
                    NeedsKvQkStaging() ? StaCeilDiv(cur.stackSeqTile, FAI_KV_BASE_ROWS) : cur.computeKvLoopNoMask;
                uint32_t qkSegLen = NeedsKvQkStaging() ? cur.stackSeqTile : cur.computeSegLen;
                if (qkEmbed <= FAI_KV_BASE_ROWS) {
                    blockMmadQK(qGm[computeQOffset], kGm[qkKvOffset], scoreGm[sOffset], dummyBlockTable, layoutQ,
                                layoutK, qkShape, qkKvIdx, qkKvLoopNoMask, ConstU32(FAI_KV_BASE_ROWS), qkSegLen,
                                qkStride);
                } else {
                    uint32_t sliceCount = QkSliceCount();
                    blockMmadQKSlicedAccum(
                        qGm[computeQOffset], kGm[qkKvOffset], sGm[sOffset], info.qRows, cur.stackSeqTile, sliceCount,
                        FAI_QUERY_ROWS * FAI_QK_SLICE_DIM, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS * FAI_QK_SLICE_DIM,
                        FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, embed, UsesOwnedStridedKLoader() ? embed : 0);
                    if (NeedsKvQkStaging()) {
                        Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(kvStageFree);
                    }
                }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
                if (debugFirstQkStack) {
                    AscendC::printf("STA_AIC_STAGE phase=qk_done dim=%u task=%u block=%u\\n",
                                    static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
                }
                if (debugRingReuseStack) {
                    AscendC::printf("STA_AIC_STAGE phase=ring_reuse_qk_done dim=%u task=%u block=%u\\n",
                                    static_cast<uint32_t>(meta.dim), taskIdx, AscendC::GetBlockIdx());
                }
#endif
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qkReady);

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    RunAicPvStack(taskIdx, info, delayed, coreIdx, embed, strideKV, layoutV, blockMmadPVNormal,
                                  blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
                }
                ++stackSeqCount;
            }
        }

        uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAicPvStack(taskIdx, info, delayed, coreIdx, embed, strideKV, layoutV, blockMmadPVNormal,
                          blockMmadPVDynamicSingleBlock, blockMmadPVDynamicMultiBlock);
        }
        if (NeedsQkStaging() && embed > FAI_QK_SLICE_DIM) {
            Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qStageFree);
        }
#if defined(OPFORGE_STA_LIFECYCLE_DEBUG)
        if (debugFirstD256Task) {
            AscendC::printf("STA_AIC_STAGE phase=task_done dim=%u task=%u block=%u\\n", static_cast<uint32_t>(meta.dim),
                            taskIdx, AscendC::GetBlockIdx());
        }
#endif
    }

    CATLASS_DEVICE void ProcessAivCompactTask(uint32_t taskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                              EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t outOffset = headBase + static_cast<uint64_t>(info.qStart) * meta.dim;
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t embedRound = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
        uint32_t stackSeqCount = 0;
        WindowRange range = GetWindowRange(info);
        uint32_t totalStackSeqNum = TotalStackSeqNumFromRange(info, range);
        uint32_t stackSeqTilePad = FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS;
        StackInfo ring[FAI_RING_STAGE_COUNT];

        uint32_t segCount = SegmentCountFromRange(info, range);
        for (uint32_t segIdx = 0; segIdx < segCount; ++segIdx) {
            SegmentInfo seg = GetSegmentFromRange(info, range, segIdx);
            if (seg.len == 0) {
                continue;
            }
            uint64_t kvOffset = headBase + static_cast<uint64_t>(seg.start) * meta.dim;
            uint32_t kvLoopNoMask = StaCeilDiv(seg.len, FAI_KV_BASE_ROWS);
            for (uint32_t kvIdx = 0; kvIdx < kvLoopNoMask; kvIdx += FAI_STACK_BLOCK_COUNT) {
                uint32_t slot = StackSlot(stackSeqCount);
                StackInfo cur;
                cur.kvOffset = kvOffset;
                cur.computeKvOffset = kvOffset;
                cur.computeVOffset = kvOffset;
                cur.segLen = seg.len;
                cur.computeSegLen = seg.len;
                cur.kvLoopNoMask = kvLoopNoMask;
                cur.computeKvLoopNoMask = kvLoopNoMask;
                cur.kvIdx = kvIdx;
                cur.computeKvIdx = kvIdx;
                cur.stackSeqTile = StackSeqTile(seg.len, kvLoopNoMask, kvIdx);
                cur.stackSeqCount = stackSeqCount;
                ring[slot] = cur;

                uint64_t pOffset = WorkspaceOffset(coreIdx, slot);
                uint64_t compactScoreOffset = CompactScoreOffset(coreIdx, slot);
                LayoutP layoutP(info.qRows, cur.stackSeqTile, stackSeqTilePad);
                LayoutS layoutS(info.qRows, cur.stackSeqTile, stackSeqTilePad);
                GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
                Arch::CrossCoreWaitFlag(qkReady);
                RunCompactOnlineSoftmax(epilogueOnlineSoftmax, compactScoreOffset, pOffset, layoutP, layoutS, qkShape,
                                        stackSeqCount == 0, info.qRows, slot);
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
                if (UsesResidualProbability()) {
                    Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxLowReady);
                }

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound,
                                       totalStackSeqNum, epilogueRescaleO);
                }
                ++stackSeqCount;
            }
        }

        uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound, totalStackSeqNum,
                               epilogueRescaleO);
        }
    }

    CATLASS_DEVICE void ProcessAivTask(uint32_t taskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                       EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo info = DecodeTask(taskIdx);
        if (info.qRows == 0) {
            return;
        }
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t outOffset = headBase + static_cast<uint64_t>(info.qStart) * meta.dim;
        uint64_t qOffset = headBase + static_cast<uint64_t>(info.qStart) * meta.dim;
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t embedRound = StaAlignUp(embed, VECTOR_BLOCK_ELEMENTS);
        uint32_t stackSeqCount = 0;
        WindowRange range;
        if (info.isText == 0) {
            range = GetWindowRange(info);
        }
        uint32_t totalStackSeqNum = TotalStackSeqNumFromRange(info, range);
        uint32_t stackSeqTilePad = FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS;
        StackInfo ring[FAI_RING_STAGE_COUNT];

        StageQuery(info, taskIdx, coreIdx, qOffset);

        if (UsesPagedD64ImageStacks() && info.isText == 0) {
            for (stackSeqCount = 0; stackSeqCount < totalStackSeqNum; ++stackSeqCount) {
                uint32_t slot = StackSlot(stackSeqCount);
                StackInfo cur = MakePagedD64TextStack(info, range, stackSeqCount, headBase, embed);
                SetComputeStackInfo(cur, coreIdx, slot);
                ring[slot] = cur;

                uint64_t sOffset = WorkspaceOffset(coreIdx, slot);
                LayoutS layoutS(info.qRows, cur.stackSeqTile, stackSeqTilePad);
                LayoutP layoutP(info.qRows, cur.stackSeqTile, stackSeqTilePad);
                GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
                Arch::CrossCoreWaitFlag(qkReady);
                RunOnlineSoftmax(epilogueOnlineSoftmax, sOffset, layoutP, layoutS, qkShape, stackSeqCount == 0,
                                 info.qRows, slot);
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound,
                                       totalStackSeqNum, epilogueRescaleO);
                }
            }
            uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
            for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
                const StackInfo &delayed = ring[StackSlot(delayedIdx)];
                RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound,
                                   totalStackSeqNum, epilogueRescaleO);
            }
            return;
        }

        uint32_t segCount = SegmentCountFromRange(info, range);
        for (uint32_t segIdx = 0; segIdx < segCount; ++segIdx) {
            SegmentInfo seg = GetSegmentFromRange(info, range, segIdx);
            if (seg.len == 0) {
                continue;
            }
            uint64_t kvOffset = headBase + static_cast<uint64_t>(seg.start) * meta.dim;
            uint32_t kvLoopNoMask = StaCeilDiv(seg.len, FAI_KV_BASE_ROWS);
            for (uint32_t kvIdx = 0; kvIdx < kvLoopNoMask; kvIdx += FAI_STACK_BLOCK_COUNT) {
                uint32_t curStackMod = StackSlot(stackSeqCount);
                StackInfo cur;
                cur.kvOffset = kvOffset;
                cur.segLen = seg.len;
                cur.kvLoopNoMask = kvLoopNoMask;
                cur.kvIdx = kvIdx;
                cur.stackSeqTile = StackSeqTile(seg.len, kvLoopNoMask, kvIdx);
                cur.stackSeqCount = stackSeqCount;
                SetComputeStackInfo(cur, coreIdx, curStackMod);
                ring[curStackMod] = cur;

                StageKv(cur, taskIdx, coreIdx, curStackMod);

                uint64_t sOffset = WorkspaceOffset(coreIdx, curStackMod);
                LayoutS layoutS(info.qRows, cur.stackSeqTile, stackSeqTilePad);
                LayoutP layoutP(info.qRows, cur.stackSeqTile, stackSeqTilePad);
                GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
                Arch::CrossCoreWaitFlag(qkReady);
                RunOnlineSoftmax(epilogueOnlineSoftmax, sOffset, layoutP, layoutS, qkShape, stackSeqCount == 0,
                                 info.qRows, curStackMod);
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxReady);
                if (UsesResidualProbability()) {
                    Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_MTE3>(softmaxLowReady);
                }

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound,
                                       totalStackSeqNum, epilogueRescaleO);
                }
                ++stackSeqCount;
            }
        }

        uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAivRescaleStack(taskIdx, delayed, coreIdx, outOffset, info.qRows, embed, embedRound, totalStackSeqNum,
                               epilogueRescaleO);
        }
    }

    CATLASS_DEVICE uint32_t StackSeqTile(uint32_t kvSeqlen, uint32_t kvLoopNoMask, uint32_t kvIdx) const
    {
        if (kvIdx + FAI_STACK_BLOCK_COUNT > kvLoopNoMask - 1) {
            return kvSeqlen - kvIdx * FAI_KV_BASE_ROWS;
        }
        return FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS;
    }

    CATLASS_DEVICE uint32_t &ConstU32(uint32_t value)
    {
        tmpU32 = value;
        return tmpU32;
    }

    CATLASS_DEVICE void InitAicEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    CATLASS_DEVICE void WaitAicEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    }

    CATLASS_DEVICE void InitAivEvents()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        if (NEEDS_INPUT_STAGING || NeedsQkStaging()) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
    }

    CATLASS_DEVICE void WaitAivEvents()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        if (NEEDS_INPUT_STAGING || NeedsQkStaging()) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
    }

    SlidingTileAttentionMeta meta;
    int64_t totalJobs = 0;
    float softmaxScale = 0.0f;
    StaFaiWorkspace workspace;
    uint32_t tmpU32 = 0;
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_FLAG_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_FLAG_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_FLAG_ID};
    Arch::CrossCoreFlag softmaxLowReady{SOFTMAX_LOW_READY_FLAG_ID};
    Arch::CrossCoreFlag pvLowReady{PV_LOW_READY_FLAG_ID};
    Arch::CrossCoreFlag inputQReady{INPUT_Q_READY_FLAG_ID};
    Arch::CrossCoreFlag inputKvReady{INPUT_KV_READY_FLAG_ID};
    Arch::CrossCoreFlag qStageFree{Q_STAGE_FREE_FLAG_ID};
    Arch::CrossCoreFlag kvStageFree{KV_STAGE_FREE_FLAG_ID};
    GlobalTensor<ElementInput> qInputGm;
    GlobalTensor<ElementInput> kInputGm;
    GlobalTensor<ElementInput> vInputGm;
    GlobalTensor<ElementQ> qGm;
    GlobalTensor<ElementK> kGm;
    GlobalTensor<ElementV> vGm;
    GlobalTensor<ElementO> outGm;
    GlobalTensor<int32_t> dummyBlockTable;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<ElementS> sGm;
    GlobalTensor<ElementCompactS> compactSGm;
    GlobalTensor<ElementP> pGm;
    GlobalTensor<ElementP> pLoGm;
    GlobalTensor<float> correctionGm;
    GlobalTensor<ElementOTmp> oTmpGm;
    GlobalTensor<ElementOTmp> oTmpLoGm;
    GlobalTensor<ElementUpdate> oUpdateGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<float> softmaxDmGm;
    uint32_t softmaxScaleBits = 0;
};

template <typename InputT, typename OutputT, bool PACKED_D128, bool PACKED_D256, bool PACKED_D72_FULL,
          bool PACKED_D3SLICE, bool PACKED_D256_TEXT>
__aicore__ inline void SlidingTileAttentionFaiBody(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                                   uint32_t softmaxScaleBits, SlidingTileAttentionMeta meta,
                                                   int64_t totalJobs, float softmaxScale, StaFaiWorkspace workspace)
{
    if (totalJobs <= 0) {
        return;
    }
    SlidingTileAttentionFaiTextOp<InputT, OutputT, PACKED_D128, PACKED_D256, PACKED_D72_FULL, PACKED_D3SLICE,
                                  PACKED_D256_TEXT>
        op;
    op.Init(q, k, v, out, softmaxScaleBits, meta, totalJobs, softmaxScale, workspace);
    op.Process();
}

template <bool PACKED_D128, bool PACKED_D256, bool PACKED_D72_FULL, bool PACKED_D3SLICE = false,
          bool PACKED_D256_TEXT = false>
__aicore__ inline void SlidingTileAttentionFaiEntry(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    AscendC::SetMaskNorm();
    SlidingTileAttentionTilingData tilingData;
    LoadSlidingTileAttentionTilingData(tiling, tilingData);
    float softmaxScale = tilingData.softmax_scale;
    uint32_t softmaxScaleBits = *reinterpret_cast<uint32_t *>(&tilingData.softmax_scale);
    GM_ADDR userWorkspace = workspace;
    bool withQkStaging = (tilingData.meta.dim % VECTOR_BLOCK_ELEMENTS) != 0 || tilingData.meta.dim > FAI_QK_SLICE_DIM;
    bool withCorrection = tilingData.meta.dim > FAI_QK_SLICE_DIM || tilingData.meta.dim == 72;
    if (tilingData.dtype == 1) {
        auto ws = MakeFaiWorkspace<bfloat16_t>(userWorkspace, tilingData.workspace_elems, tilingData.workspace_blocks,
                                               tilingData.workspace_block_elems, tilingData.meta.dim, withQkStaging,
                                               true, withCorrection, tilingData.meta.grouped_query_chunks);
        SlidingTileAttentionFaiBody<bfloat16_t, bfloat16_t, PACKED_D128, PACKED_D256, PACKED_D72_FULL, PACKED_D3SLICE,
                                    PACKED_D256_TEXT>(q, k, v, out, softmaxScaleBits, tilingData.meta,
                                                      tilingData.total_work, softmaxScale, ws);
    } else {
        auto ws = MakeFaiWorkspace<half>(userWorkspace, tilingData.workspace_elems, tilingData.workspace_blocks,
                                         tilingData.workspace_block_elems, tilingData.meta.dim, withQkStaging, false,
                                         withCorrection, tilingData.meta.grouped_query_chunks);
        SlidingTileAttentionFaiBody<half, half, PACKED_D128, PACKED_D256, PACKED_D72_FULL, PACKED_D3SLICE,
                                    PACKED_D256_TEXT>(q, k, v, out, softmaxScaleBits, tilingData.meta,
                                                      tilingData.total_work, softmaxScale, ws);
    }
}

} // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
extern "C" __global__ __aicore__ void sliding_tile_attention_fai(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR windows,
                                                                 GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)windows;
    if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_GENERAL)) {
        SlidingTileAttentionFaiEntry<false, false, false>(q, k, v, out, workspace, tiling);
    } else if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_FP16_D128_N1_PACKED)) {
        SlidingTileAttentionFaiEntry<true, false, false>(q, k, v, out, workspace, tiling);
    } else if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_FP16_D256_N1_PACKED)) {
        SlidingTileAttentionFaiEntry<false, true, false>(q, k, v, out, workspace, tiling);
    } else if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_FP16_D3SLICE_N1_PACKED)) {
        SlidingTileAttentionFaiEntry<false, false, false, true>(q, k, v, out, workspace, tiling);
    } else if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_FP16_D256_TEXT_PACKED)) {
        SlidingTileAttentionFaiEntry<false, false, false, false, true>(q, k, v, out, workspace, tiling);
    } else if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_FP16_D72_N1_NONFULL_GROUPED)) {
        SlidingTileAttentionFaiEntry<false, false, true>(q, k, v, out, workspace, tiling);
    } else if (TILING_KEY_IS(OPFORGE_STA_TILING_KEY_D16_TEXT_GROUPED)) {
        SlidingTileAttentionFaiEntry<false, false, false>(q, k, v, out, workspace, tiling);
    }
}
