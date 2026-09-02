/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"

#include "sliding_tile_attention_case_guard.h"
#define OPFORGE_RESOURCE_LEDGER_HOST_PROJECTION_ONLY
#include "../op_kernel/resource_ledger.hpp"
#undef OPFORGE_RESOURCE_LEDGER_HOST_PROJECTION_ONLY
#include "sliding_tile_attention_tiling.h"
#include "../op_kernel/sliding_tile_attention_tiling_key.h"

namespace optiling {

namespace {

constexpr int64_t FAI_QUERY_ROWS = 128;
constexpr int64_t FAI_STACK_BLOCK_COUNT = 4;
constexpr int64_t FAI_KV_ROWS = FAI_STACK_BLOCK_COUNT * FAI_QUERY_ROWS;
constexpr int64_t FAI_WORKSPACE_STAGES = 5;
constexpr int64_t FAI_BASE_WORKSPACE_BYTES_PER_ELEM = 14;
constexpr int64_t FAI_DUAL_PROBABILITY_BYTES_PER_ELEM = 6;
static_assert(OPFORGE_DUAL_PROBABILITY_WORKSPACE_VALID(FAI_DUAL_PROBABILITY_BYTES_PER_ELEM, sizeof(uint16_t),
                                                       sizeof(float)),
              "host residual workspace must match the device main/residual PV ownership");
constexpr int64_t FAI_QK_SLICE_DIM = 128;
constexpr int64_t FAI_QUERY_ROWS_PER_CORE = 128;
constexpr int64_t FAI_SCORE_WORKSPACE_BLOCK_ELEMS = FAI_QUERY_ROWS_PER_CORE * 512;
constexpr int64_t FAI_QUERY_CHUNKS_PER_TILE = 3;
constexpr int64_t FAI_QUERY_CHUNKS_PER_PAIR = 6;
constexpr int64_t FAI_SOFTMAX_DM_SLOTS = FAI_WORKSPACE_STAGES;
constexpr int64_t FAI_BLOCK_TABLE_BYTES_PER_CORE =
    FAI_WORKSPACE_STAGES * FAI_STACK_BLOCK_COUNT * static_cast<int64_t>(sizeof(int32_t));

constexpr size_t ATTR_BATCH = 0;
constexpr size_t ATTR_HEADS = 1;
constexpr size_t ATTR_SEQUENCE = 2;
constexpr size_t ATTR_HEAD_DIM = 3;
constexpr size_t ATTR_TEXT_LENGTH = 4;
constexpr size_t ATTR_HAS_TEXT = 5;
constexpr size_t ATTR_SEQ_SHAPE = 6;

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t WindowStart(int64_t coord, int64_t window, int64_t extent)
{
    if (window >= extent) {
        return 0;
    }
    const int64_t radius = window / 2;
    const int64_t center = std::max(radius, std::min(coord, extent - 1 - radius));
    return center - radius;
}

int64_t BuildAxisGroups(int64_t extent, int64_t window, int64_t *starts, int64_t *spans)
{
    int64_t count = 0;
    for (int64_t coord = 0; coord < extent;) {
        const bool pair =
            coord + 1 < extent && WindowStart(coord, window, extent) == WindowStart(coord + 1, window, extent);
        starts[count] = coord;
        spans[count] = pair ? 2 : 1;
        ++count;
        coord += pair ? 2 : 1;
    }
    return count;
}

void BuildD16PairPlan(SlidingTileAttentionMeta &meta, bool pairInputSupported)
{
    meta.pair_axis = -1;
    meta.pair_group_count = 0;
    meta.grouped_query_chunks = FAI_QUERY_CHUNKS_PER_TILE;
    if (!pairInputSupported || meta.dim != 16 || meta.has_text == 0 || meta.heads <= 0) {
        return;
    }
    for (int64_t head = 1; head < meta.heads; ++head) {
        for (int64_t axis = 0; axis < 3; ++axis) {
            if (meta.windows[head * 3 + axis] != meta.windows[axis]) {
                return;
            }
        }
    }
    const int64_t extents[3] = {meta.grid_t, meta.grid_h, meta.grid_w};
    const int64_t axisOrder[3] = {2, 1, 0};
    int64_t bestCount = std::numeric_limits<int64_t>::max();
    int64_t bestStarts[SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY] = {0};
    int64_t bestSpans[SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY] = {0};
    for (int64_t order = 0; order < 3; ++order) {
        const int64_t axis = axisOrder[order];
        int64_t starts[SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY] = {0};
        int64_t spans[SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY] = {0};
        const int64_t count = BuildAxisGroups(extents[axis], meta.windows[axis], starts, spans);
        if (count < bestCount) {
            bestCount = count;
            meta.pair_axis = axis;
            for (int64_t index = 0; index < count; ++index) {
                bestStarts[index] = starts[index];
                bestSpans[index] = spans[index];
            }
        }
    }
    if (meta.pair_axis < 0 || bestCount >= extents[meta.pair_axis]) {
        meta.pair_axis = -1;
        return;
    }
    meta.pair_group_count = bestCount;
    meta.grouped_query_chunks = FAI_QUERY_CHUNKS_PER_PAIR;
    for (int64_t index = 0; index < bestCount; ++index) {
        meta.pair_group_start[index] = bestStarts[index];
        meta.pair_group_span[index] = bestSpans[index];
    }
}

bool CheckedMultiply(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAdd(int64_t lhs, int64_t rhs, int64_t &result)
{
    if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool FillCanvasBySeqShape(int64_t seqShape, SlidingTileAttentionMeta &meta)
{
    meta.tile_t = sliding_tile_attention::TILE_TIME;
    meta.tile_h = sliding_tile_attention::TILE_HEIGHT;
    meta.tile_w = sliding_tile_attention::TILE_WIDTH;
    switch (seqShape) {
        case sliding_tile_attention::FASTVIDEO_SEQ_SHAPE_ID:
            meta.canvas_t = sliding_tile_attention::FASTVIDEO_CANVAS_TIME;
            meta.canvas_h = sliding_tile_attention::FASTVIDEO_CANVAS_HEIGHT;
            meta.canvas_w = sliding_tile_attention::FASTVIDEO_CANVAS_WIDTH;
            break;
        case sliding_tile_attention::STEPVIDEO_SEQ_SHAPE_ID:
            meta.canvas_t = sliding_tile_attention::STEPVIDEO_CANVAS_TIME;
            meta.canvas_h = sliding_tile_attention::STEPVIDEO_CANVAS_HEIGHT;
            meta.canvas_w = sliding_tile_attention::STEPVIDEO_CANVAS_WIDTH;
            break;
        case sliding_tile_attention::WAN_SEQ_SHAPE_ID:
            meta.canvas_t = sliding_tile_attention::WAN_CANVAS_TIME;
            meta.canvas_h = sliding_tile_attention::WAN_CANVAS_HEIGHT;
            meta.canvas_w = sliding_tile_attention::WAN_CANVAS_WIDTH;
            break;
        default:
            return false;
    }
    meta.grid_t = meta.canvas_t / meta.tile_t;
    meta.grid_h = meta.canvas_h / meta.tile_h;
    meta.grid_w = meta.canvas_w / meta.tile_w;
    meta.tile_numel = meta.tile_t * meta.tile_h * meta.tile_w;
    meta.image_seq = meta.canvas_t * meta.canvas_h * meta.canvas_w;
    return true;
}

bool TotalChunks(const SlidingTileAttentionMeta &meta, bool usesGroupedTileJobs, int64_t &totalChunks)
{
    const int64_t numTiles = meta.grid_t * meta.grid_h * meta.grid_w;
    if (usesGroupedTileJobs) {
        const int64_t textChunksPerHead = meta.has_text != 0 ? CeilDiv(meta.seq - meta.image_seq, FAI_QUERY_ROWS) : 0;
        // The grouped-text refinement is correctness-clean but rejected after an active-window regression.
        const bool groupedTextD16 = false;
        const int64_t textJobsPerHead = groupedTextD16 ? 1 : textChunksPerHead;
        int64_t jobsPerHead = 0;
        int64_t batchHeads = 0;
        int64_t imageJobs = numTiles;
        if (meta.pair_axis >= 0) {
            const int64_t axisExtent =
                meta.pair_axis == 0 ? meta.grid_t : (meta.pair_axis == 1 ? meta.grid_h : meta.grid_w);
            imageJobs = numTiles / axisExtent * meta.pair_group_count;
        }
        return CheckedAdd(imageJobs, textJobsPerHead, jobsPerHead) &&
               CheckedMultiply(meta.batch, meta.heads, batchHeads) &&
               CheckedMultiply(batchHeads, jobsPerHead, totalChunks);
    }
    const int64_t chunksPerTile = CeilDiv(meta.tile_numel, FAI_QUERY_ROWS);
    const int64_t imageChunksPerHead = numTiles * chunksPerTile;
    const int64_t textChunksPerHead = meta.has_text != 0 ? CeilDiv(meta.seq - meta.image_seq, FAI_QUERY_ROWS) : 0;
    int64_t batchHeads = 0;
    return CheckedMultiply(meta.batch, meta.heads, batchHeads) &&
           CheckedMultiply(batchHeads, imageChunksPerHead + textChunksPerHead, totalChunks);
}

int64_t AicCoreNum(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return 1;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    return std::max<int64_t>(platform.GetCoreNumAic(), 1);
}

int64_t SystemWorkspaceBytes(gert::TilingContext *context)
{
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return -1;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    return static_cast<int64_t>(platform.GetLibApiWorkSpaceSize());
}

ge::graphStatus FillMeta(gert::TilingContext *context, SlidingTileAttentionMeta &meta, ge::DataType &dtype)
{
    const auto *inputShape = context->GetInputShape(0);
    const auto *inputDesc = context->GetInputDesc(0);
    const auto *windowTensor = context->GetInputTensor(3);
    const auto *attrs = context->GetAttrs();
    if (inputShape == nullptr || inputDesc == nullptr || windowTensor == nullptr || attrs == nullptr) {
        OP_LOGE(context, "SlidingTileAttention tiling is missing q shape/desc, windows data, or attrs.");
        return ge::GRAPH_FAILED;
    }
    const int64_t *batch = attrs->GetAttrPointer<int64_t>(ATTR_BATCH);
    const int64_t *heads = attrs->GetAttrPointer<int64_t>(ATTR_HEADS);
    const int64_t *sequence = attrs->GetAttrPointer<int64_t>(ATTR_SEQUENCE);
    const int64_t *headDim = attrs->GetAttrPointer<int64_t>(ATTR_HEAD_DIM);
    const int64_t *textLength = attrs->GetAttrPointer<int64_t>(ATTR_TEXT_LENGTH);
    const bool *hasText = attrs->GetAttrPointer<bool>(ATTR_HAS_TEXT);
    const int64_t *seqShape = attrs->GetAttrPointer<int64_t>(ATTR_SEQ_SHAPE);
    if (batch == nullptr || heads == nullptr || sequence == nullptr || headDim == nullptr || textLength == nullptr ||
        hasText == nullptr || seqShape == nullptr) {
        OP_LOGE(context, "SlidingTileAttention runtime attrs are incomplete.");
        return ge::GRAPH_FAILED;
    }
    if (!FillCanvasBySeqShape(*seqShape, meta)) {
        OP_LOGE(context, "SlidingTileAttention seqShape is unsupported: %ld.", static_cast<long>(*seqShape));
        return ge::GRAPH_FAILED;
    }

    meta.batch = *batch;
    meta.heads = *heads;
    meta.seq = *sequence;
    meta.dim = *headDim;
    meta.text_len = *textLength;
    meta.has_text = *hasText ? 1 : 0;
    if (!sliding_tile_attention::IsValidSequenceMetadata(meta.seq, meta.text_len, *hasText, *seqShape) ||
        meta.batch <= 0 || !sliding_tile_attention::IsSupportedHeads(meta.heads) ||
        !sliding_tile_attention::IsSupportedHeadDim(meta.dim)) {
        OP_LOGE(context,
                "SlidingTileAttention requires B>0, 1<=N<=128, D>0 divisible by 8, and task-book sequence metadata; "
                "got B/N/S/D=[%ld,%ld,%ld,%ld], textLength=%ld, hasText=%d, seqShape=%ld.",
                static_cast<long>(meta.batch), static_cast<long>(meta.heads), static_cast<long>(meta.seq),
                static_cast<long>(meta.dim), static_cast<long>(meta.text_len), static_cast<int>(*hasText),
                static_cast<long>(*seqShape));
        return ge::GRAPH_FAILED;
    }

    const auto shape = inputShape->GetOriginShape();
    const int64_t dimNum = shape.GetDimNum();
    int64_t expectedElements = 0;
    int64_t batchHeads = 0;
    int64_t batchHeadSequence = 0;
    if (!CheckedMultiply(meta.batch, meta.heads, batchHeads) ||
        !CheckedMultiply(batchHeads, meta.seq, batchHeadSequence) ||
        !CheckedMultiply(batchHeadSequence, meta.dim, expectedElements)) {
        OP_LOGE(context, "SlidingTileAttention B*N*S*D overflows int64.");
        return ge::GRAPH_FAILED;
    }
    const bool exactBnsd = dimNum == sliding_tile_attention::SUPPORTED_RANK && shape.GetDim(0) == meta.batch &&
                           shape.GetDim(1) == meta.heads && shape.GetDim(2) == meta.seq && shape.GetDim(3) == meta.dim;
    const bool compatibleFlat = dimNum == 1 && shape.GetShapeSize() == expectedElements;
    if ((!exactBnsd && !compatibleFlat) || shape.GetShapeSize() != expectedElements) {
        OP_LOGE(context,
                "SlidingTileAttention q shape does not match runtime B/N/S/D; originRank=%ld, elements=%ld, "
                "attrs=[%ld,%ld,%ld,%ld], expectedElements=%ld.",
                static_cast<long>(dimNum), static_cast<long>(shape.GetShapeSize()), static_cast<long>(meta.batch),
                static_cast<long>(meta.heads), static_cast<long>(meta.seq), static_cast<long>(meta.dim),
                static_cast<long>(expectedElements));
        return ge::GRAPH_FAILED;
    }

    if (windowTensor->GetDataType() != ge::DT_INT64) {
        OP_LOGE(context, "SlidingTileAttention windows must be int64, got dtype=%d.",
                static_cast<int>(windowTensor->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    const int64_t rawWindowCount = windowTensor->GetShapeSize();
    const int64_t *rawWindows = windowTensor->GetData<int64_t>();
    sliding_tile_attention::WindowValues normalizedWindows;
    if (!sliding_tile_attention::NormalizeWindows(rawWindows, rawWindowCount, meta.heads, normalizedWindows)) {
        OP_LOGE(context, "SlidingTileAttention windows must contain three or 3*N positive values; values=%ld, N=%ld.",
                static_cast<long>(rawWindowCount), static_cast<long>(meta.heads));
        return ge::GRAPH_FAILED;
    }
    meta.window_count = static_cast<int64_t>(normalizedWindows.size());
    for (int64_t index = 0; index < meta.window_count; ++index) {
        meta.windows[index] = normalizedWindows[static_cast<size_t>(index)];
    }
    dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        OP_LOGE(context, "SlidingTileAttention dtype is unsupported: %d.", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    BuildD16PairPlan(meta, OPFORGE_D16_PAIR_INPUT_SUPPORTED(dtype, ge::DT_FLOAT16, ge::DT_BF16));
    if (!sliding_tile_attention::IsSupportedMetadata(meta.batch, meta.heads, meta.seq, meta.dim, meta.text_len,
                                                     *hasText, *seqShape, normalizedWindows)) {
        OP_LOGE(context, "SlidingTileAttention runtime metadata is unsupported.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetWorkspace(gert::TilingContext *context, int64_t bytes)
{
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = static_cast<size_t>(bytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SlidingTileAttentionFaiTilingImpl(gert::TilingContext *context, bool partialMode)
{
    SlidingTileAttentionMeta meta;
    ge::DataType dtype = ge::DT_FLOAT16;
    if (FillMeta(context, meta, dtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        OP_LOGE(context, "SlidingTileAttention raw tiling buffer is unavailable.");
        return ge::GRAPH_FAILED;
    }
    SlidingTileAttentionHostTilingData tiling;
    tiling.set_batch(meta.batch);
    tiling.set_heads(meta.heads);
    tiling.set_seq(meta.seq);
    tiling.set_dim(meta.dim);
    tiling.set_imageSeq(meta.image_seq);
    tiling.set_textLen(meta.text_len);
    tiling.set_hasText(meta.has_text);
    tiling.set_canvasT(meta.canvas_t);
    tiling.set_canvasH(meta.canvas_h);
    tiling.set_canvasW(meta.canvas_w);
    tiling.set_tileT(meta.tile_t);
    tiling.set_tileH(meta.tile_h);
    tiling.set_tileW(meta.tile_w);
    tiling.set_gridT(meta.grid_t);
    tiling.set_gridH(meta.grid_h);
    tiling.set_gridW(meta.grid_w);
    tiling.set_tileNumel(meta.tile_numel);
    tiling.set_windowCount(meta.window_count);
    tiling.set_pairAxis(meta.pair_axis);
    tiling.set_pairGroupCount(meta.pair_group_count);
    tiling.set_groupedQueryChunks(meta.grouped_query_chunks);
    tiling.set_pairGroupStart(meta.pair_group_start);
    tiling.set_pairGroupSpan(meta.pair_group_span);
    tiling.set_windows(meta.windows);
    tiling.set_dtype(dtype == ge::DT_BF16 ? 1 : 0);
    const float softmaxScale = 1.0f / std::sqrt(static_cast<float>(meta.dim));
    tiling.set_softmaxScale(softmaxScale);
    tiling.set_reserved(0);

    bool fullWindow = true;
    for (int64_t head = 0; head < meta.heads; ++head) {
        const int64_t base = head * 3;
        if (meta.windows[base] < meta.grid_t || meta.windows[base + 1] < meta.grid_h ||
            meta.windows[base + 2] < meta.grid_w) {
            fullWindow = false;
            break;
        }
    }
    const bool packedD72N1NonFull = dtype == ge::DT_FLOAT16 && meta.heads == 1 && meta.has_text == 0 &&
                                    meta.text_len == 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim == 72 &&
                                    !fullWindow;
    const bool packedD72MultiheadNonFull = dtype == ge::DT_FLOAT16 && meta.heads != 1 && meta.has_text == 0 &&
                                           meta.text_len == 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS &&
                                           meta.dim == 72 && !fullWindow;
    const bool groupedQkD72 = dtype == ge::DT_FLOAT16 && meta.has_text == 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS &&
                              meta.dim > 64 && meta.dim <= FAI_QK_SLICE_DIM && !fullWindow;
    const bool groupedQkD16 =
        (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) && meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim == 16;
    const bool groupedQkD8 =
        dtype == ge::DT_FLOAT16 && meta.has_text == 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim == 8;
    const bool packedD256Text = dtype == ge::DT_FLOAT16 && meta.heads == 1 && meta.has_text != 0 && meta.text_len > 0 &&
                                meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim == 256 && meta.window_count >= 3 &&
                                meta.windows[0] == 3 && meta.windows[1] == 3 && meta.windows[2] == 3;
    if (partialMode &&
        (!packedD256Text || meta.batch != 1 || meta.text_len > SLIDING_TILE_ATTENTION_PARTIAL_MAX_TEXT_ROWS)) {
        OP_LOGE(context, "D256 text partial mode received metadata outside its capacity guard.");
        return ge::GRAPH_FAILED;
    }
    const bool groupedQk = groupedQkD72 || groupedQkD16 || groupedQkD8;
    // D16 grouping is valid only when the semantic text tail is empty.
    const bool usesGroupedTileJobs =
        (dtype == ge::DT_FLOAT16 && meta.has_text == 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS) || groupedQkD16 ||
        packedD256Text;
    int64_t totalJobs = 0;
    if (!TotalChunks(meta, usesGroupedTileJobs, totalJobs)) {
        OP_LOGE(context, "SlidingTileAttention total work overflows int64.");
        return ge::GRAPH_FAILED;
    }
    const int64_t coreNum = AicCoreNum(context);
    const int64_t blockDim = totalJobs > 0 ? std::max<int64_t>(1, std::min(coreNum, totalJobs)) : 1;
    if (partialMode && (blockDim > SLIDING_TILE_ATTENTION_PARTIAL_MAX_PRODUCERS ||
                        CeilDiv(meta.seq, FAI_STACK_BLOCK_COUNT * FAI_QUERY_ROWS) < blockDim)) {
        OP_LOGE(context, "D256 text partial producer count exceeds capacity or would create an empty shard.");
        return ge::GRAPH_FAILED;
    }
    int64_t outputWorkspaceBlockElems = 0;
    int64_t workspaceBlockElems = 0;
    const int64_t outputDim = CeilDiv(meta.dim, static_cast<int64_t>(16)) * 16;
    if (!CheckedMultiply(FAI_QUERY_ROWS_PER_CORE, outputDim, outputWorkspaceBlockElems)) {
        OP_LOGE(context, "SlidingTileAttention per-core output workspace overflows int64.");
        return ge::GRAPH_FAILED;
    }
    workspaceBlockElems = std::max(FAI_SCORE_WORKSPACE_BLOCK_ELEMS, outputWorkspaceBlockElems);
    const int64_t workspaceChunkFactor = groupedQk ? meta.grouped_query_chunks : 1;
    int64_t workspaceElems = 0;
    int64_t blockWorkspaceElems = 0;
    if (!CheckedMultiply(blockDim, workspaceBlockElems, blockWorkspaceElems) ||
        !CheckedMultiply(blockWorkspaceElems, FAI_WORKSPACE_STAGES, workspaceElems) ||
        !CheckedMultiply(workspaceElems, workspaceChunkFactor, workspaceElems)) {
        OP_LOGE(context, "SlidingTileAttention workspace element count overflows int64.");
        return ge::GRAPH_FAILED;
    }
    tiling.set_totalWork(totalJobs);
    tiling.set_workPerBlock(totalJobs > 0 ? CeilDiv(totalJobs, blockDim) : 0);
    tiling.set_workspaceBlocks(blockDim);
    tiling.set_workspaceBlockElems(workspaceBlockElems);
    tiling.set_workspaceElems(workspaceElems);

    const int64_t systemWorkspaceBytes = SystemWorkspaceBytes(context);
    if (systemWorkspaceBytes < 0) {
        OP_LOGE(context, "SlidingTileAttention failed to query the system workspace size.");
        return ge::GRAPH_FAILED;
    }
    int64_t userWorkspaceBytes = 0;
    int64_t workspaceBytes = 0;
    const int64_t qkSliceCount = CeilDiv(outputDim, FAI_QK_SLICE_DIM);
    int64_t extraScoreBytesPerElem = 0;
    int64_t workspaceBytesPerElem = 0;
    const bool usesDualProbabilityWorkspace = meta.dim > FAI_QK_SLICE_DIM || meta.dim == 72;
    const int64_t dynamicProbabilityBytesPerElem =
        usesDualProbabilityWorkspace ? FAI_DUAL_PROBABILITY_BYTES_PER_ELEM : 0;
    if (!CheckedMultiply(qkSliceCount - 1, static_cast<int64_t>(sizeof(float)), extraScoreBytesPerElem) ||
        !CheckedAdd(FAI_BASE_WORKSPACE_BYTES_PER_ELEM + dynamicProbabilityBytesPerElem, extraScoreBytesPerElem,
                    workspaceBytesPerElem) ||
        !CheckedMultiply(workspaceElems, workspaceBytesPerElem, userWorkspaceBytes) ||
        !CheckedAdd(systemWorkspaceBytes, userWorkspaceBytes, workspaceBytes)) {
        OP_LOGE(context, "SlidingTileAttention workspace byte count overflows int64.");
        return ge::GRAPH_FAILED;
    }
    int64_t softmaxStateValues = 0;
    int64_t softmaxStateBytes = 0;
    const int64_t softmaxStateValuesPerCore =
        (2 * meta.grouped_query_chunks + meta.grouped_query_chunks * FAI_SOFTMAX_DM_SLOTS) * FAI_QUERY_ROWS_PER_CORE;
    if (!CheckedMultiply(blockDim, softmaxStateValuesPerCore, softmaxStateValues) ||
        !CheckedMultiply(softmaxStateValues, static_cast<int64_t>(sizeof(float)), softmaxStateBytes) ||
        !CheckedAdd(workspaceBytes, softmaxStateBytes, workspaceBytes)) {
        OP_LOGE(context, "SlidingTileAttention grouped softmax-state workspace overflows int64.");
        return ge::GRAPH_FAILED;
    }
    int64_t blockTableBytes = 0;
    if (!CheckedMultiply(blockDim, FAI_BLOCK_TABLE_BYTES_PER_CORE, blockTableBytes) ||
        !CheckedAdd(workspaceBytes, blockTableBytes, workspaceBytes)) {
        OP_LOGE(context, "SlidingTileAttention block-table workspace overflows int64.");
        return ge::GRAPH_FAILED;
    }
    const int64_t qkDim = ((meta.dim + 15) / 16) * 16;
    const int64_t qkStageDim = qkSliceCount > 1 ? qkSliceCount * FAI_QK_SLICE_DIM : qkDim;
    if (qkDim != meta.dim || qkSliceCount > 1) {
        int64_t queryStageElems = 0;
        int64_t kvStageElems = 0;
        int64_t stagedPrefixElems = 0;
        int64_t stagedPrefixBytes = 0;
        int64_t allBlockPrefixBytes = 0;
        const int64_t queryStageSlots = meta.dim == 8 ? FAI_QUERY_CHUNKS_PER_TILE : 1;
        if (!CheckedMultiply(FAI_QUERY_ROWS, qkStageDim, queryStageElems) ||
            !CheckedMultiply(queryStageElems, queryStageSlots, queryStageElems) ||
            !CheckedMultiply(FAI_KV_ROWS, qkStageDim, kvStageElems) ||
            !CheckedMultiply(FAI_WORKSPACE_STAGES, kvStageElems, stagedPrefixElems) ||
            !CheckedAdd(queryStageElems, stagedPrefixElems, stagedPrefixElems) ||
            !CheckedMultiply(stagedPrefixElems, static_cast<int64_t>(sizeof(uint16_t)), stagedPrefixBytes) ||
            !CheckedMultiply(blockDim, stagedPrefixBytes, allBlockPrefixBytes) ||
            !CheckedAdd(workspaceBytes, allBlockPrefixBytes, workspaceBytes)) {
            OP_LOGE(context, "SlidingTileAttention Q/K padding workspace overflows int64.");
            return ge::GRAPH_FAILED;
        }
    }
    if (dtype == ge::DT_BF16) {
        int64_t fullVElements = 0;
        int64_t fullVStageBytes = 0;
        if (!CheckedMultiply(meta.batch, meta.heads, fullVElements) ||
            !CheckedMultiply(fullVElements, meta.seq, fullVElements) ||
            !CheckedMultiply(fullVElements, meta.dim, fullVElements) ||
            !CheckedMultiply(fullVElements, static_cast<int64_t>(sizeof(uint16_t)), fullVStageBytes) ||
            !CheckedAdd(workspaceBytes, fullVStageBytes, workspaceBytes)) {
            OP_LOGE(context, "SlidingTileAttention BF16 staging workspace overflows int64.");
            return ge::GRAPH_FAILED;
        }
    }
    if (SetWorkspace(context, workspaceBytes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (tiling.GetDataSize() != sizeof(SlidingTileAttentionTilingData) ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        OP_LOGE(context, "SlidingTileAttention tiling ABI mismatch: host=%zu, device=%zu, capacity=%zu.",
                tiling.GetDataSize(), sizeof(SlidingTileAttentionTilingData), rawTiling->GetCapacity());
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    uint32_t expectedScaleBits = 0;
    uint32_t serializedScaleBits = 0;
    static_assert(offsetof(SlidingTileAttentionTilingData, reserved) ==
                      offsetof(SlidingTileAttentionTilingData, softmax_scale) + sizeof(float),
                  "SlidingTileAttention scale/reserved ABI must share one 64-bit word");
    std::memcpy(&expectedScaleBits, &softmaxScale, sizeof(expectedScaleBits));
    std::memcpy(
        &serializedScaleBits,
        static_cast<const uint8_t *>(rawTiling->GetData()) + offsetof(SlidingTileAttentionTilingData, softmax_scale),
        sizeof(serializedScaleBits));
    if (serializedScaleBits != expectedScaleBits) {
        OP_LOGE(context,
                "SlidingTileAttention softmax scale ABI mismatch: dim=%ld expected_bits=0x%08x serialized_bits=0x%08x "
                "offset=%zu.",
                meta.dim, expectedScaleBits, serializedScaleBits,
                offsetof(SlidingTileAttentionTilingData, softmax_scale));
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
    const bool packedD128N1 = dtype == ge::DT_FLOAT16 && meta.heads == 1 && meta.has_text == 0 && meta.text_len == 0 &&
                              meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim == 128;
    const bool packedD256N1 = dtype == ge::DT_FLOAT16 && meta.heads == 1 && meta.has_text == 0 && meta.text_len == 0 &&
                              meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim == 256 && meta.window_count >= 3 &&
                              meta.windows[0] == 3 && meta.windows[1] == 3 && meta.windows[2] == 3;
    const bool packedD3SliceN1 = dtype == ge::DT_FLOAT16 && meta.heads == 1 && meta.has_text == 0 &&
                                 meta.text_len == 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS && meta.dim > 256 &&
                                 meta.dim <= 384 && meta.window_count >= 3 && meta.windows[0] == 3 &&
                                 meta.windows[1] == 3 && meta.windows[2] == 3;
    const bool groupedD16Text =
        meta.dim == 16 && meta.has_text != 0 && meta.text_len > 0 && meta.tile_numel == 3 * FAI_QUERY_ROWS;
    context->SetTilingKey(partialMode               ? OPFORGE_STA_TILING_KEY_FP16_D256_TEXT_PARTIAL :
                          packedD128N1              ? OPFORGE_STA_TILING_KEY_FP16_D128_N1_PACKED :
                          packedD256N1              ? OPFORGE_STA_TILING_KEY_FP16_D256_N1_PACKED :
                          packedD256Text            ? OPFORGE_STA_TILING_KEY_FP16_D256_TEXT_PACKED :
                          packedD3SliceN1           ? OPFORGE_STA_TILING_KEY_FP16_D3SLICE_N1_PACKED :
                          packedD72N1NonFull        ? OPFORGE_STA_TILING_KEY_FP16_D72_N1_NONFULL_GROUPED :
                          packedD72MultiheadNonFull ? OPFORGE_STA_TILING_KEY_FP16_D72_N1_NONFULL_GROUPED :
                          groupedD16Text            ? OPFORGE_STA_TILING_KEY_D16_TEXT_GROUPED :
                                                      OPFORGE_STA_TILING_KEY_GENERAL);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SlidingTileAttentionFaiTiling(gert::TilingContext *context)
{
    return SlidingTileAttentionFaiTilingImpl(context, false);
}

ge::graphStatus SlidingTileAttentionFaiD256TextPartialTiling(gert::TilingContext *context)
{
    return SlidingTileAttentionFaiTilingImpl(context, true);
}

ge::graphStatus SlidingTileAttentionFaiD256TextMergeTiling(gert::TilingContext *context)
{
    SlidingTileAttentionMeta meta;
    ge::DataType dtype = ge::DT_FLOAT16;
    if (FillMeta(context, meta, dtype) != ge::GRAPH_SUCCESS || dtype != ge::DT_FLOAT16 || meta.batch != 1 ||
        meta.heads != 1 || meta.dim != 256 || meta.has_text == 0 || meta.text_len <= 0 ||
        meta.text_len > SLIDING_TILE_ATTENTION_PARTIAL_MAX_TEXT_ROWS) {
        OP_LOGE(context, "D256 text merge received metadata outside its capacity guard.");
        return ge::GRAPH_FAILED;
    }
    auto rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    SlidingTileAttentionHostTilingData tiling;
    tiling.set_batch(meta.batch);
    tiling.set_heads(meta.heads);
    tiling.set_seq(meta.seq);
    tiling.set_dim(meta.dim);
    tiling.set_imageSeq(meta.image_seq);
    tiling.set_textLen(meta.text_len);
    tiling.set_hasText(meta.has_text);
    tiling.set_canvasT(meta.canvas_t);
    tiling.set_canvasH(meta.canvas_h);
    tiling.set_canvasW(meta.canvas_w);
    tiling.set_tileT(meta.tile_t);
    tiling.set_tileH(meta.tile_h);
    tiling.set_tileW(meta.tile_w);
    tiling.set_gridT(meta.grid_t);
    tiling.set_gridH(meta.grid_h);
    tiling.set_gridW(meta.grid_w);
    tiling.set_tileNumel(meta.tile_numel);
    tiling.set_windowCount(meta.window_count);
    tiling.set_pairAxis(meta.pair_axis);
    tiling.set_pairGroupCount(meta.pair_group_count);
    tiling.set_groupedQueryChunks(meta.grouped_query_chunks);
    tiling.set_pairGroupStart(meta.pair_group_start);
    tiling.set_pairGroupSpan(meta.pair_group_span);
    tiling.set_windows(meta.windows);
    tiling.set_dtype(0);
    tiling.set_softmaxScale(1.0f / std::sqrt(static_cast<float>(meta.dim)));
    tiling.set_reserved(0);
    const int64_t producerCount = AicCoreNum(context);
    if (producerCount > SLIDING_TILE_ATTENTION_PARTIAL_MAX_PRODUCERS) {
        OP_LOGE(context, "D256 text merge producer count exceeds partial tensor capacity.");
        return ge::GRAPH_FAILED;
    }
    const int64_t mergeBlocks = std::max<int64_t>(1, std::min(producerCount, CeilDiv(meta.text_len, 2)));
    tiling.set_totalWork(meta.text_len);
    tiling.set_workPerBlock(CeilDiv(meta.text_len, mergeBlocks));
    tiling.set_workspaceBlocks(producerCount);
    tiling.set_workspaceBlockElems(0);
    tiling.set_workspaceElems(0);
    if (SetWorkspace(context, 0) != ge::GRAPH_SUCCESS ||
        tiling.GetDataSize() != sizeof(SlidingTileAttentionTilingData) ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(mergeBlocks);
    context->SetTilingKey(OPFORGE_STA_TILING_KEY_FP16_D256_TEXT_MERGE);
    return ge::GRAPH_SUCCESS;
}

struct SlidingTileAttentionCompileInfo {};

ge::graphStatus TilingParse([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(SlidingTileAttentionFai)
    .Tiling(SlidingTileAttentionFaiTiling)
    .TilingInputsDataDependency({3})
    .TilingParse<SlidingTileAttentionCompileInfo>(TilingParse);
IMPL_OP_OPTILING(SlidingTileAttentionFaiD256TextPartial)
    .Tiling(SlidingTileAttentionFaiD256TextPartialTiling)
    .TilingInputsDataDependency({3})
    .TilingParse<SlidingTileAttentionCompileInfo>(TilingParse);
IMPL_OP_OPTILING(SlidingTileAttentionFaiD256TextMerge)
    .Tiling(SlidingTileAttentionFaiD256TextMergeTiling)
    .TilingInputsDataDependency({3})
    .TilingParse<SlidingTileAttentionCompileInfo>(TilingParse);

} // namespace optiling
