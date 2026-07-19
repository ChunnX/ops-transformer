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


#include "sliding_tile_attention_tiling_data.h"

namespace {

using namespace AscendC;
using namespace Catlass;

constexpr uint32_t QK_READY_FLAG_ID = 1;
constexpr uint32_t SOFTMAX_READY_FLAG_ID = 2;
constexpr uint32_t PV_READY_FLAG_ID = 3;
constexpr uint32_t INPUT_Q_READY_FLAG_ID = 4;
constexpr uint32_t INPUT_KV_READY_FLAG_ID = 6;
constexpr uint32_t FFTS_AIV_PAIR_MODE = 2;
static_assert(FFTS_AIV_PAIR_MODE <= 0x3, "FFTS mode must fit the two-bit message field");
constexpr uint32_t HEAD_COUNT = 8;
constexpr uint32_t HEAD_INDEX_MASK = HEAD_COUNT - 1;
constexpr uint32_t HEAD_INDEX_SHIFT = 3;
constexpr uint32_t WINDOW_VALUES_PER_HEAD = 3;
constexpr uint32_t WINDOW_TIME_OFFSET = 0;
constexpr uint32_t WINDOW_HEIGHT_OFFSET = 1;
constexpr uint32_t WINDOW_WIDTH_OFFSET = 2;
constexpr uint32_t HEAD_DIM_ELEMENTS = 64;
constexpr uint32_t FAI_QUERY_ROWS = 128;
constexpr uint32_t FAI_KV_BASE_ROWS = 128;
constexpr uint32_t TILE_ELEMENTS = 384;
constexpr uint32_t QUERY_CHUNKS_PER_TILE = TILE_ELEMENTS / FAI_QUERY_ROWS;
constexpr uint32_t GRID_HEIGHT_TILES = 6;
constexpr uint32_t UB_REGION_BYTES = 16 * 1024;
constexpr uint32_t UB_SUBREGION_BYTES = 1024;
constexpr uint32_t VECTOR_BLOCK_ELEMENTS = 16;
constexpr uint32_t FLOAT_BLOCK_ELEMENTS = 8;
constexpr uint32_t FLOAT_VECTOR_ELEMENTS = 64;
constexpr uint32_t BF16_CAST_SOURCE_REPEAT_STRIDE = 4;
constexpr uint32_t FAI_STACK_BLOCK_COUNT = 4;
constexpr uint32_t FAI_PRELAUNCH_DEPTH = 4;
constexpr uint32_t FAI_RING_STAGE_COUNT = FAI_PRELAUNCH_DEPTH + 1;
constexpr uint32_t FAI_WORKSPACE_BLOCK_ELEMENTS = 131072;
constexpr uint32_t FAI_AIV_SUBBLOCK_COUNT = 2;
constexpr uint32_t FAI_QUERY_STAGE_ELEMENTS = FAI_QUERY_ROWS * HEAD_DIM_ELEMENTS;
constexpr uint32_t FAI_KV_STAGE_ELEMENTS = FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS * HEAD_DIM_ELEMENTS;
constexpr uint32_t BF16_CONVERT_CHUNK_ELEMENTS = 4096;
constexpr uint32_t BF16_CONVERT_VECTOR_ELEMENTS = 2048;
constexpr uint32_t BF16_CONVERT_SOURCE_UB_OFFSET = 11 * UB_REGION_BYTES;
constexpr uint32_t BF16_CONVERT_DESTINATION_UB_OFFSET =
    BF16_CONVERT_SOURCE_UB_OFFSET + BF16_CONVERT_CHUNK_ELEMENTS * sizeof(bfloat16_t);
// StageKv drains this bridge before online softmax reuses CATLASS TV scratch.
constexpr uint32_t BF16_CONVERT_FLOAT_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_CENTER_UB_OFFSET = 6 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_TV_UB_OFFSET = 10 * UB_REGION_BYTES;
constexpr uint32_t SOFTMAX_LL_UB_OFFSET = 10 * UB_REGION_BYTES + 11 * UB_SUBREGION_BYTES;
constexpr uint32_t SUM_V_TILE_ROWS = 32;
constexpr uint32_t SUM_V_ACCUMULATOR_COUNT = 4;
constexpr uint32_t SUM_V_SOURCE_UB_OFFSET = 0;
constexpr uint32_t SUM_V_FLOAT_UB_OFFSET = 2 * UB_REGION_BYTES;
constexpr uint32_t SUM_V_ACCUM_UB_OFFSET = 3 * UB_REGION_BYTES;
constexpr uint32_t SUM_V_VECTOR_ELEMENTS = HEAD_DIM_ELEMENTS;
constexpr uint32_t SUM_V_ACCUM_STRIDE_BYTES = SUM_V_VECTOR_ELEMENTS * sizeof(float);
constexpr uint32_t CORRECTION_GO_STATE_OFFSET_ELEMENTS =
    FAI_QUERY_ROWS + FAI_AIV_SUBBLOCK_COUNT * SUM_V_VECTOR_ELEMENTS;
constexpr uint32_t CORRECTION_GO_STATE_ELEMENTS = FAI_QUERY_ROWS * SUM_V_VECTOR_ELEMENTS;
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
static_assert(CORRECTION_GO_STATE_OFFSET_ELEMENTS + CORRECTION_GO_STATE_ELEMENTS <= FAI_WORKSPACE_BLOCK_ELEMENTS,
              "bf16 correction GO state exceeds one per-core correction slot");

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
    GM_ADDR correction = nullptr;
    GM_ADDR o_temp = nullptr;
    GM_ADDR o_update = nullptr;
    GM_ADDR q_stage = nullptr;
    GM_ADDR k_stage = nullptr;
    GM_ADDR v_stage = nullptr;
    int64_t blocks = 0;
    int64_t block_elems = 0;
    int64_t staged_elems = 0;
};

template <typename T>
__aicore__ inline StaFaiWorkspace MakeFaiWorkspace(GM_ADDR workspace, int64_t stagedElems, int64_t blocks,
                                                   int64_t blockElems, bool withInputStaging, bool withCorrection)
{
    StaFaiWorkspace out;
    __gm__ uint8_t *base = (__gm__ uint8_t *)workspace;
    int64_t offset = 0;
    out.s = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(float));
    out.p = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(T));
    if (withCorrection) {
        out.correction = base + offset;
        offset += stagedElems * static_cast<int64_t>(sizeof(float));
    }
    out.o_temp = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(float));
    out.o_update = base + offset;
    offset += stagedElems * static_cast<int64_t>(sizeof(float));
    if (withInputStaging) {
        out.q_stage = base + offset;
        offset += blocks * static_cast<int64_t>(FAI_QUERY_STAGE_ELEMENTS) * static_cast<int64_t>(sizeof(half));
        out.k_stage = base + offset;
        offset += blocks * static_cast<int64_t>(FAI_RING_STAGE_COUNT) * static_cast<int64_t>(FAI_KV_STAGE_ELEMENTS) *
                  static_cast<int64_t>(sizeof(half));
        out.v_stage = base + offset;
    }
    out.blocks = blocks;
    out.block_elems = blockElems;
    out.staged_elems = stagedElems;
    return out;
}

template <typename InputT, typename OutputT>
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
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<false, true>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using UpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2OnlineSoftmax, PType, SType, MaskType>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<Epilogue::EpilogueAtlasA2RescaleO, OType, OTmpType, UpdateType>;
    static constexpr bool NEEDS_INPUT_STAGING = !std::is_same_v<ElementInput, half>;
    static constexpr bool USE_DUAL_PROBABILITY = false;

    static_assert((std::is_same_v<ElementInput, half> && std::is_same_v<ElementO, half>) ||
                      (std::is_same_v<ElementInput, bfloat16_t> && std::is_same_v<ElementO, bfloat16_t>),
                  "STA FAI supports matched fp16 or bf16 public input/output types");

    CATLASS_DEVICE SlidingTileAttentionFaiTextOp()
    {
    }

    CATLASS_DEVICE void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, SlidingTileAttentionMeta metaIn,
                             int64_t totalJobsIn, StaFaiWorkspace workspaceIn)
    {
        meta = metaIn;
        totalJobs = totalJobsIn;
        workspace = workspaceIn;
        qInputGm.SetGlobalBuffer((__gm__ ElementInput *)q);
        kInputGm.SetGlobalBuffer((__gm__ ElementInput *)k);
        vInputGm.SetGlobalBuffer((__gm__ ElementInput *)v);
        qGm.SetGlobalBuffer((__gm__ ElementQ *)q);
        kGm.SetGlobalBuffer((__gm__ ElementK *)k);
        if constexpr (NEEDS_INPUT_STAGING) {
            vGm.SetGlobalBuffer((__gm__ ElementV *)workspace.v_stage);
        } else {
            vGm.SetGlobalBuffer((__gm__ ElementV *)v);
        }
        outGm.SetGlobalBuffer((__gm__ ElementO *)out);
        dummyBlockTable.SetGlobalBuffer((__gm__ int32_t *)workspace.o_update);
        sGm.SetGlobalBuffer((__gm__ ElementS *)workspace.s);
        pGm.SetGlobalBuffer((__gm__ ElementP *)workspace.p);
        oTmpGm.SetGlobalBuffer((__gm__ ElementOTmp *)workspace.o_temp);
        if constexpr (USE_DUAL_PROBABILITY) {
            correctionGm.SetGlobalBuffer((__gm__ float *)workspace.correction);
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void Process();

    template <>
    CATLASS_DEVICE void Process<AscendC::AIC>()
    {
        InitAicEvents();
        if constexpr (NEEDS_INPUT_STAGING) {
            AscendC::SyncAll<false>();
        }
        static constexpr uint32_t L1_QK_BYTES =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_BYTES);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
            ProcessAicTask(taskIdx, blockMmadQK, blockMmadPV);
        }
        WaitAicEvents();
    }

    template <>
    CATLASS_DEVICE void Process<AscendC::AIV>()
    {
        InitAivEvents();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, D64_SOFTMAX_SCALE);
        EpilogueRescaleO epilogueRescaleO(resource);

        uint32_t subBlockNum = AscendC::GetSubBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        uint32_t coreNum = AscendC::GetBlockNum();
        if constexpr (NEEDS_INPUT_STAGING) {
            StageAllV();
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
        }
        for (uint32_t taskIdx = coreIdx; taskIdx < static_cast<uint32_t>(totalJobs); taskIdx += coreNum) {
            ProcessAivTask(taskIdx, epilogueOnlineSoftmax, epilogueRescaleO);
        }
        WaitAivEvents();
    }

private:
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
    };

    CATLASS_DEVICE TaskInfo DecodeTask(uint32_t taskIdx) const
    {
        if (meta.tile_numel == TILE_ELEMENTS && meta.heads == HEAD_COUNT && meta.dim == HEAD_DIM_ELEMENTS &&
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
            meta.has_text != 0 ? StaCeilDiv(static_cast<uint32_t>(meta.text_len), FAI_QUERY_ROWS) : 0;
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
            info.qRows = StaMin(FAI_QUERY_ROWS, static_cast<uint32_t>(meta.text_len) - textOffset);
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
        info.head = tmp & HEAD_INDEX_MASK;
        info.batch = tmp >> HEAD_INDEX_SHIFT;
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
        uint32_t winT = static_cast<uint32_t>(meta.windows[windowBase + WINDOW_TIME_OFFSET]);
        uint32_t winH = static_cast<uint32_t>(meta.windows[windowBase + WINDOW_HEIGHT_OFFSET]);
        uint32_t winW = static_cast<uint32_t>(meta.windows[windowBase + WINDOW_WIDTH_OFFSET]);
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
            seg.len = static_cast<uint32_t>(meta.seq);
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
            return StaCeilDiv(static_cast<uint32_t>(meta.seq), stackTile);
        }
        uint32_t imageSegments = (range.tEnd - range.tStart) * (range.hEnd - range.hStart);
        uint32_t imageSegLen = (range.wEnd - range.wStart) * static_cast<uint32_t>(meta.tile_numel);
        uint32_t total = imageSegments * StackCountForSegment(imageSegLen, stackTile);
        if (meta.has_text != 0 && meta.text_len > 0) {
            total += StaCeilDiv(static_cast<uint32_t>(meta.text_len), stackTile);
        }
        return total;
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

    CATLASS_DEVICE uint64_t WorkspaceOffset(uint32_t coreIdx, uint32_t pingpong) const
    {
        return static_cast<uint64_t>(coreIdx) * FAI_WORKSPACE_BLOCK_ELEMENTS * FAI_RING_STAGE_COUNT +
               static_cast<uint64_t>(pingpong) * FAI_WORKSPACE_BLOCK_ELEMENTS;
    }

    CATLASS_DEVICE uint32_t StackSlot(uint32_t stackSeqCount) const
    {
        return stackSeqCount % FAI_RING_STAGE_COUNT;
    }

    CATLASS_DEVICE uint64_t QueryStageOffset(uint32_t coreIdx) const
    {
        return static_cast<uint64_t>(coreIdx) * FAI_QUERY_STAGE_ELEMENTS;
    }

    CATLASS_DEVICE uint64_t KvStageOffset(uint32_t coreIdx, uint32_t slot) const
    {
        return (static_cast<uint64_t>(coreIdx) * FAI_RING_STAGE_COUNT + slot) * FAI_KV_STAGE_ELEMENTS;
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
        if constexpr (NEEDS_INPUT_STAGING) {
            (void)info;
            (void)taskIdx;
            (void)coreIdx;
            (void)qOffset;
        }
    }

    CATLASS_DEVICE void StageKv(const StackInfo &stack, uint32_t taskIdx, uint32_t coreIdx, uint32_t slot)
    {
        (void)stack;
        (void)taskIdx;
        (void)coreIdx;
        (void)slot;
    }

    CATLASS_DEVICE void DrainInputStageWrite()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
    }


    CATLASS_DEVICE void RunAicPvStack(uint32_t taskIdx, const TaskInfo &info, const StackInfo &stack, uint32_t coreIdx,
                                      uint32_t embed, uint32_t strideKV, const LayoutV &layoutV,
                                      BlockMmadPV &blockMmadPV)
    {
        uint32_t pvPingPong = StackSlot(stack.stackSeqCount);
        uint64_t pOffset = WorkspaceOffset(coreIdx, pvPingPong);
        LayoutP layoutP(info.qRows, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
        GemmCoord pvShape{info.qRows, embed, stack.stackSeqTile};
        uint64_t computeVOffset = stack.computeVOffset;
        uint32_t computeKvIdx = stack.computeKvIdx;
        uint32_t computeKvLoopNoMask = stack.computeKvLoopNoMask;
        uint32_t computeSegLen = stack.computeSegLen;
        if constexpr (USE_DUAL_PROBABILITY && !NEEDS_INPUT_STAGING) {
            computeVOffset += static_cast<uint64_t>(stack.computeKvIdx) * FAI_KV_BASE_ROWS * strideKV;
        }
        if constexpr (USE_DUAL_PROBABILITY) {
            computeKvIdx = 0;
            computeKvLoopNoMask = StaCeilDiv(stack.stackSeqTile, FAI_KV_BASE_ROWS);
            computeSegLen = stack.stackSeqTile;
        }
        uint32_t blockSize = FAI_KV_BASE_ROWS;
        blockMmadPV(pGm[pOffset], vGm[computeVOffset], oTmpGm[pOffset], dummyBlockTable, layoutP, layoutV, pvShape,
                    computeKvIdx, computeKvLoopNoMask, blockSize, computeSegLen, strideKV, softmaxReady);
        Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(pvReady);
    }

    CATLASS_DEVICE void RunAivRescaleStack(uint32_t taskIdx, const StackInfo &stack, uint32_t coreIdx,
                                           uint64_t outOffset, uint32_t qRows, uint32_t embed, uint32_t embedRound,
                                           uint32_t totalStackSeqNum, EpilogueRescaleO &epilogueRescaleO)
    {
        uint32_t pvStackMod = StackSlot(stack.stackSeqCount);
        uint64_t oTmpOffset = WorkspaceOffset(coreIdx, pvStackMod);
        LayoutO layoutO(qRows, embed);
        LayoutOTmp layoutOTmp(qRows, embed, embedRound);
        GemmCoord pvShape{qRows, embed, stack.stackSeqTile};
        Arch::CrossCoreWaitFlag(pvReady);
        bool isFirstStack = stack.stackSeqCount == 0;
        bool isLastStack = stack.stackSeqCount == totalStackSeqNum - 1;
        if constexpr (USE_DUAL_PROBABILITY) {
            RunBf16RecurrenceStack(taskIdx, stack, oTmpOffset, outOffset, qRows, embed, embedRound, isFirstStack,
                                   isLastStack, pvStackMod);
        } else {
            epilogueRescaleO(outGm[outOffset], oTmpGm[oTmpOffset], layoutO, layoutOTmp, pvShape, qRows, 1, isFirstStack,
                             isLastStack, pvStackMod);
        }
        if constexpr (NEEDS_INPUT_STAGING) {
            if (stack.stackSeqCount == totalStackSeqNum - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            }
        }
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
        if constexpr (USE_DUAL_PROBABILITY) {
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
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::Duplicate(accum0, 0.0f, SUM_V_VECTOR_ELEMENTS);
            AscendC::Duplicate(accum1, 0.0f, SUM_V_VECTOR_ELEMENTS);
            AscendC::Duplicate(accum2, 0.0f, SUM_V_VECTOR_ELEMENTS);
            AscendC::Duplicate(accum3, 0.0f, SUM_V_VECTOR_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t rowBase = 0; rowBase < stack.stackSeqTile; rowBase += SUM_V_TILE_ROWS) {
                uint32_t currentRows = StaMin(SUM_V_TILE_ROWS, stack.stackSeqTile - rowBase);
                uint32_t currentElems = currentRows * strideKV;
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::DataCopy(srcUb, vGm[computeVOffset + static_cast<uint64_t>(rowBase) * strideKV], currentElems);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::Cast(floatUb, srcUb, AscendC::RoundMode::CAST_NONE, currentElems);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t row = 0; row < currentRows; row += SUM_V_ACCUMULATOR_COUNT) {
                    AscendC::Add(accum0, accum0, floatUb[row * strideKV], SUM_V_VECTOR_ELEMENTS);
                    if (row + 1 < currentRows) {
                        AscendC::Add(accum1, accum1, floatUb[(row + 1) * strideKV], SUM_V_VECTOR_ELEMENTS);
                    }
                    if (row + 2 < currentRows) {
                        AscendC::Add(accum2, accum2, floatUb[(row + 2) * strideKV], SUM_V_VECTOR_ELEMENTS);
                    }
                    if (row + 3 < currentRows) {
                        AscendC::Add(accum3, accum3, floatUb[(row + 3) * strideKV], SUM_V_VECTOR_ELEMENTS);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
            AscendC::Add(accum0, accum0, accum1, SUM_V_VECTOR_ELEMENTS);
            AscendC::Add(accum2, accum2, accum3, SUM_V_VECTOR_ELEMENTS);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accum0, accum0, accum2, SUM_V_VECTOR_ELEMENTS);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopy(correctionGm[correctionOffset + FAI_QUERY_ROWS + subBlockIdx * strideKV], accum0,
                              strideKV);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    CATLASS_DEVICE void RunOnlineSoftmax(EpilogueOnlineSoftmax &epilogueOnlineSoftmax, uint64_t pOffset,
                                         const LayoutP &layoutP, const LayoutS &layoutS, GemmCoord qkShape,
                                         bool isFirstStack, uint32_t qRows, uint32_t curStackMod)
    {
        if constexpr (!USE_DUAL_PROBABILITY) {
            epilogueOnlineSoftmax(pGm[pOffset], sGm[pOffset], layoutP, layoutS, qkShape, isFirstStack, qRows, 1,
                                  curStackMod);
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
        LocalTensor<float> centerUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_CENTER_UB_OFFSET);
        LocalTensor<float> centerBroadcastUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_TV_UB_OFFSET);
        LocalTensor<float> rowSumUb = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_LL_UB_OFFSET);

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
            uint32_t dmOffset = curStackMod * EpilogueOnlineSoftmax::MAX_ROW_NUM_SUB_CORE + rowInLoop;
            uint32_t elemCount = currentRows * columnNumRound;
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pingpong);
            epilogueOnlineSoftmax.ScaleS(sUbOffset, currentRows, columnNumRound);
            epilogueOnlineSoftmax.CalcLocalRowMax(sUbOffset, rowRound, columnNum, columnNumRound, rowInLoop);
            epilogueOnlineSoftmax.UpdateGlobalRowMax(currentRows, rowRound, columnNum, columnNumRound, dmOffset,
                                                     rowInLoop, isFirstStack);
            epilogueOnlineSoftmax.CalcExp(sUbOffset, currentRows, rowRound, columnNum, columnNumRound, rowInLoop);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pingpong);

            epilogueOnlineSoftmax.CalcLocalRowSum(sUbOffset, rowRound, columnNum, columnNumRound, rowInLoop);
            AscendC::Duplicate(centerUb.template ReinterpretCast<int32_t>(), static_cast<int32_t>(columnNum), rowRound);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast<float, int32_t>(centerBroadcastUb, centerUb.template ReinterpretCast<int32_t>(),
                                          AscendC::RoundMode::CAST_NONE, rowRound);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Div(centerUb, rowSumUb[rowInLoop], centerBroadcastUb, currentRows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Brcb(centerBroadcastUb.template ReinterpretCast<uint32_t>(),
                          centerUb.template ReinterpretCast<uint32_t>(), rowRound / FLOAT_BLOCK_ELEMENTS,
                          AscendC::BrcbRepeatParams(1, FLOAT_BLOCK_ELEMENTS));
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t subIdx = 0; subIdx < columnNum / FLOAT_VECTOR_ELEMENTS; ++subIdx) {
                AscendC::Sub<float, false>(pFloatUb[sUbOffset + subIdx * FLOAT_VECTOR_ELEMENTS],
                                           pFloatUb[sUbOffset + subIdx * FLOAT_VECTOR_ELEMENTS], centerBroadcastUb,
                                           static_cast<uint64_t>(0), currentRows,
                                           AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_ELEMENTS,
                                                                       columnNumRound / FLOAT_BLOCK_ELEMENTS, 1));
            }
            if (columnNum % FLOAT_VECTOR_ELEMENTS > 0) {
                epilogueOnlineSoftmax.SetVecMask(columnNum % FLOAT_VECTOR_ELEMENTS);
                AscendC::Sub<float, false>(
                    pFloatUb[sUbOffset + columnNum / FLOAT_VECTOR_ELEMENTS * FLOAT_VECTOR_ELEMENTS],
                    pFloatUb[sUbOffset + columnNum / FLOAT_VECTOR_ELEMENTS * FLOAT_VECTOR_ELEMENTS], centerBroadcastUb,
                    static_cast<uint64_t>(0), currentRows,
                    AscendC::BinaryRepeatParams(1, 1, 0, columnNumRound / FLOAT_BLOCK_ELEMENTS,
                                                columnNumRound / FLOAT_BLOCK_ELEMENTS, 1));
                AscendC::SetVectorMask<int8_t>(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            }
            AscendC::PipeBarrier<PIPE_V>();
            epilogueOnlineSoftmax.DownCastP(sUbOffset, currentRows, columnNumRound);
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
            AscendC::DataCopy(
                correctionGm[pOffset + subBlockIdx * (FAI_QUERY_ROWS / FAI_AIV_SUBBLOCK_COUNT) + rowInLoop], centerUb,
                rowRound);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pingpong);
            epilogueOnlineSoftmax.UpdateGlobalRowSum(sUbOffset, currentRows, rowRound, dmOffset, rowInLoop,
                                                     isFirstStack);
        }
    }

    CATLASS_DEVICE void ProcessAicTask(uint32_t taskIdx, BlockMmadQK &blockMmadQK, BlockMmadPV &blockMmadPV)
    {
        TaskInfo info = DecodeTask(taskIdx);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint64_t headBase = HeadBase(info.batch, info.head);
        uint64_t qOffset = headBase + static_cast<uint64_t>(info.qStart) * meta.dim;
        uint32_t embed = static_cast<uint32_t>(meta.dim);
        uint32_t strideKV = embed;
        uint32_t qNBlockSize = 1;
        uint32_t qStrideHeads = 1;
        uint32_t stackSeqCount = 0;
        StackInfo ring[FAI_RING_STAGE_COUNT];
        LayoutQ layoutQ(info.qRows, embed);
        LayoutK layoutK(strideKV, FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS);
        LayoutV layoutV(FAI_STACK_BLOCK_COUNT * FAI_KV_BASE_ROWS, strideKV);
        WindowRange range;
        if (info.isText == 0) {
            range = GetWindowRange(info);
        }

        uint64_t computeQOffset = qOffset;
        blockMmadQK.loadQGM(qGm[computeQOffset], layoutQ, info.qRows, qNBlockSize, qStrideHeads);
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
                uint64_t sOffset = WorkspaceOffset(coreIdx, sPingPong);
                GemmCoord qkShape{info.qRows, cur.stackSeqTile, embed};
                blockMmadQK(qGm[computeQOffset], kGm[cur.computeKvOffset], sGm[sOffset], dummyBlockTable, layoutQ,
                            layoutK, qkShape, cur.computeKvIdx, cur.computeKvLoopNoMask, ConstU32(FAI_KV_BASE_ROWS),
                            cur.computeSegLen, strideKV);
                Arch::CrossCoreSetFlag<FFTS_AIV_PAIR_MODE, PIPE_FIX>(qkReady);

                if (stackSeqCount >= FAI_PRELAUNCH_DEPTH) {
                    const StackInfo &delayed = ring[StackSlot(stackSeqCount - FAI_PRELAUNCH_DEPTH)];
                    RunAicPvStack(taskIdx, info, delayed, coreIdx, embed, strideKV, layoutV, blockMmadPV);
                }
                ++stackSeqCount;
            }
        }

        uint32_t drainStart = stackSeqCount > FAI_PRELAUNCH_DEPTH ? stackSeqCount - FAI_PRELAUNCH_DEPTH : 0;
        for (uint32_t delayedIdx = drainStart; delayedIdx < stackSeqCount; ++delayedIdx) {
            const StackInfo &delayed = ring[StackSlot(delayedIdx)];
            RunAicPvStack(taskIdx, info, delayed, coreIdx, embed, strideKV, layoutV, blockMmadPV);
        }
    }

    CATLASS_DEVICE void ProcessAivTask(uint32_t taskIdx, EpilogueOnlineSoftmax &epilogueOnlineSoftmax,
                                       EpilogueRescaleO &epilogueRescaleO)
    {
        TaskInfo info = DecodeTask(taskIdx);
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
                if constexpr (USE_DUAL_PROBABILITY) {
                    ComputeSumV(taskIdx, cur, sOffset, embed);
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
        if constexpr (USE_DUAL_PROBABILITY) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        if constexpr (NEEDS_INPUT_STAGING) {
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
        if constexpr (NEEDS_INPUT_STAGING) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        if constexpr (USE_DUAL_PROBABILITY) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);
        }
    }

    SlidingTileAttentionMeta meta;
    int64_t totalJobs = 0;
    StaFaiWorkspace workspace;
    uint32_t tmpU32 = 0;
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_FLAG_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_FLAG_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_FLAG_ID};
    Arch::CrossCoreFlag inputQReady{INPUT_Q_READY_FLAG_ID};
    Arch::CrossCoreFlag inputKvReady{INPUT_KV_READY_FLAG_ID};
    GlobalTensor<ElementInput> qInputGm;
    GlobalTensor<ElementInput> kInputGm;
    GlobalTensor<ElementInput> vInputGm;
    GlobalTensor<ElementQ> qGm;
    GlobalTensor<ElementK> kGm;
    GlobalTensor<ElementV> vGm;
    GlobalTensor<ElementO> outGm;
    GlobalTensor<int32_t> dummyBlockTable;
    GlobalTensor<ElementS> sGm;
    GlobalTensor<ElementP> pGm;
    GlobalTensor<float> correctionGm;
    GlobalTensor<ElementOTmp> oTmpGm;
};

template <typename InputT, typename OutputT>
__aicore__ inline void SlidingTileAttentionFaiBody(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                                   SlidingTileAttentionMeta meta, int64_t totalJobs,
                                                   StaFaiWorkspace workspace)
{
    if (totalJobs <= 0) {
        return;
    }
    SlidingTileAttentionFaiTextOp<InputT, OutputT> op;
    op.Init(q, k, v, out, meta, totalJobs, workspace);
    op.Process();
}

__aicore__ inline void SlidingTileAttentionFaiEntry(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    AscendC::SetMaskNorm();
    SlidingTileAttentionTilingData tilingData;
    LoadSlidingTileAttentionTilingData(tiling, tilingData);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    constexpr bool WITH_INPUT_STAGING = false;
    if (tilingData.dtype == 1) {
        auto ws = MakeFaiWorkspace<bfloat16_t>(userWorkspace, tilingData.workspace_elems, tilingData.workspace_blocks,
                                               tilingData.workspace_block_elems, true, false);
        SlidingTileAttentionFaiBody<bfloat16_t, bfloat16_t>(q, k, v, out, tilingData.meta, tilingData.total_work, ws);
    } else {
        auto ws = MakeFaiWorkspace<half>(userWorkspace, tilingData.workspace_elems, tilingData.workspace_blocks,
                                         tilingData.workspace_block_elems, WITH_INPUT_STAGING, false);
        SlidingTileAttentionFaiBody<half, half>(q, k, v, out, tilingData.meta, tilingData.total_work, ws);
    }
}

} // namespace

KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
extern "C" __global__ __aicore__ void sliding_tile_attention_fai(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                                                 GM_ADDR runtimeSignature, GM_ADDR out,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    (void)runtimeSignature;
    SlidingTileAttentionFaiEntry(q, k, v, out, workspace, tiling);
}
