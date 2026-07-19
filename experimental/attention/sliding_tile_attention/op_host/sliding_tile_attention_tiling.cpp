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
#include <cstdint>

#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "register/op_def_registry.h"

#include "sliding_tile_attention_case_guard.h"
#include "sliding_tile_attention_tiling.h"

namespace optiling {

namespace {

constexpr int64_t FAI_QUERY_ROWS = 128;
constexpr int64_t FAI_STACK_BLOCK_COUNT = 4;
constexpr int64_t FAI_KV_ROWS = FAI_STACK_BLOCK_COUNT * FAI_QUERY_ROWS;
constexpr int64_t FAI_WORKSPACE_BLOCK_ELEMS = 131072;
constexpr int64_t FAI_WORKSPACE_STAGES = 5;
constexpr int64_t FAI_WORKSPACE_BYTES_PER_ELEM = 14;
constexpr int64_t FAI_QUERY_STAGE_ELEMS = FAI_QUERY_ROWS * sliding_tile_attention::SUPPORTED_HEAD_DIM;
constexpr int64_t FAI_KV_STAGE_ELEMS = FAI_KV_ROWS * sliding_tile_attention::SUPPORTED_HEAD_DIM;
constexpr int64_t FAI_BF16_STAGE_PREFIX_BYTES_PER_BLOCK =
    (FAI_QUERY_STAGE_ELEMS + FAI_WORKSPACE_STAGES * FAI_KV_STAGE_ELEMS) * static_cast<int64_t>(sizeof(uint16_t));

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
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
            meta.text_len = sliding_tile_attention::FASTVIDEO_TEXT_LENGTH;
            meta.has_text = 1;
            break;
        case sliding_tile_attention::STEPVIDEO_SEQ_SHAPE_ID:
            meta.canvas_t = sliding_tile_attention::STEPVIDEO_CANVAS_TIME;
            meta.canvas_h = sliding_tile_attention::STEPVIDEO_CANVAS_HEIGHT;
            meta.canvas_w = sliding_tile_attention::STEPVIDEO_CANVAS_WIDTH;
            meta.text_len = 0;
            meta.has_text = 0;
            break;
        case sliding_tile_attention::WAN_SEQ_SHAPE_ID:
            meta.canvas_t = sliding_tile_attention::WAN_CANVAS_TIME;
            meta.canvas_h = sliding_tile_attention::WAN_CANVAS_HEIGHT;
            meta.canvas_w = sliding_tile_attention::WAN_CANVAS_WIDTH;
            meta.text_len = 0;
            meta.has_text = 0;
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

int64_t TotalChunks(const SlidingTileAttentionMeta &meta)
{
    const int64_t chunksPerTile = CeilDiv(meta.tile_numel, FAI_QUERY_ROWS);
    const int64_t numTiles = meta.grid_t * meta.grid_h * meta.grid_w;
    const int64_t imageChunksPerHead = numTiles * chunksPerTile;
    const int64_t textChunksPerHead = meta.has_text != 0 ? CeilDiv(meta.text_len, FAI_QUERY_ROWS) : 0;
    return meta.batch * meta.heads * (imageChunksPerHead + textChunksPerHead);
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
    auto inputShape = context->GetInputShape(0);
    auto signatureShape = context->GetInputShape(3);
    auto inputDesc = context->GetInputDesc(0);
    if (inputShape == nullptr || signatureShape == nullptr || inputDesc == nullptr) {
        OP_LOGE(context, "SlidingTileAttention tiling is missing input, signature, or input desc metadata.");
        return ge::GRAPH_FAILED;
    }

    const int64_t runtimeSignature = signatureShape->GetOriginShape().GetShapeSize();
    int64_t seqShape = 0;
    int64_t windowPattern = sliding_tile_attention::INVALID_WINDOW_PATTERN;
    if (!sliding_tile_attention::DecodeRuntimeSignature(runtimeSignature, seqShape, windowPattern)) {
        OP_LOGE(context, "SlidingTileAttention runtime signature is unsupported: %ld.",
                static_cast<long>(runtimeSignature));
        return ge::GRAPH_FAILED;
    }

    if (!FillCanvasBySeqShape(seqShape, meta)) {
        OP_LOGE(context, "SlidingTileAttention seqShape is unsupported: %ld.", static_cast<long>(seqShape));
        return ge::GRAPH_FAILED;
    }
    const int64_t expectedSeq = meta.has_text != 0 ? meta.image_seq + meta.text_len : meta.image_seq;
    if (expectedSeq <= 0) {
        OP_LOGE(context, "SlidingTileAttention text metadata is inconsistent.");
        return ge::GRAPH_FAILED;
    }

    const auto shape = inputShape->GetOriginShape();
    const int64_t dimNum = shape.GetDimNum();
    int64_t batch = -1;
    int64_t heads = -1;
    int64_t seq = -1;
    int64_t headDim = -1;
    if (dimNum == sliding_tile_attention::SUPPORTED_RANK) {
        batch = shape.GetDim(0);
        heads = shape.GetDim(1);
        seq = shape.GetDim(2);
        headDim = shape.GetDim(3);
    } else if (dimNum == 1) {
        // The ACLNN L0 view may reach tiling as one flat dimension. Recover BNSD only from the total element count
        // and the already validated closed-case N/S/D metadata; the normalized-signature guard below remains final.
        const int64_t totalElements = shape.GetShapeSize();
        const int64_t elementsPerBatch =
            sliding_tile_attention::SUPPORTED_HEADS * expectedSeq * sliding_tile_attention::SUPPORTED_HEAD_DIM;
        if (totalElements <= 0 || elementsPerBatch <= 0 || totalElements % elementsPerBatch != 0) {
            OP_LOGE(context, "SlidingTileAttention flat origin shape is incompatible: total=%ld, elementsPerBatch=%ld.",
                    static_cast<long>(totalElements), static_cast<long>(elementsPerBatch));
            return ge::GRAPH_FAILED;
        }
        batch = totalElements / elementsPerBatch;
        heads = sliding_tile_attention::SUPPORTED_HEADS;
        seq = expectedSeq;
        headDim = sliding_tile_attention::SUPPORTED_HEAD_DIM;
    }
    if ((batch != 1 && batch != 2) || heads != sliding_tile_attention::SUPPORTED_HEADS || seq != expectedSeq ||
        headDim != sliding_tile_attention::SUPPORTED_HEAD_DIM) {
        OP_LOGE(context,
                "SlidingTileAttention supports BNSD or its ACLNN flat origin view with B in {1,2}, N=8, D=64, "
                "and expected S; originRank=%ld, recovered=[%ld,%ld,%ld,%ld], expected S=%ld.",
                static_cast<long>(dimNum), static_cast<long>(batch), static_cast<long>(heads), static_cast<long>(seq),
                static_cast<long>(headDim), static_cast<long>(expectedSeq));
        return ge::GRAPH_FAILED;
    }
    meta.batch = batch;
    meta.heads = heads;
    meta.seq = seq;
    meta.dim = headDim;
    sliding_tile_attention::WindowArray windows{};
    if (!sliding_tile_attention::FillWindowPattern(windowPattern, windows)) {
        OP_LOGE(context, "SlidingTileAttention windowPattern is unsupported: %ld.", static_cast<long>(windowPattern));
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < sliding_tile_attention::WINDOW_VALUE_COUNT; ++i) {
        meta.windows[i] = windows[i];
    }

    dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        OP_LOGE(context, "SlidingTileAttention dtype is unsupported: %d.", static_cast<int>(dtype));
        return ge::GRAPH_FAILED;
    }
    if (!sliding_tile_attention::IsSupportedNormalizedSignature(batch, heads, seq, headDim, dtype == ge::DT_BF16,
                                                                meta.text_len, meta.has_text != 0, seqShape, windows)) {
        OP_LOGE(context, "SlidingTileAttention metadata is outside the fixed task acceptance case set.");
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

ge::graphStatus SlidingTileAttentionFaiTiling(gert::TilingContext *context)
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
    tiling.set_windows(meta.windows);
    tiling.set_dtype(dtype == ge::DT_BF16 ? 1 : 0);

    const int64_t totalJobs = TotalChunks(meta);
    const int64_t coreNum = AicCoreNum(context);
    const int64_t blockDim = totalJobs > 0 ? std::max<int64_t>(1, std::min(coreNum, totalJobs)) : 1;
    const int64_t workspaceElems = blockDim * FAI_WORKSPACE_BLOCK_ELEMS * FAI_WORKSPACE_STAGES;
    tiling.set_totalWork(totalJobs);
    tiling.set_workPerBlock(totalJobs > 0 ? CeilDiv(totalJobs, blockDim) : 0);
    tiling.set_workspaceBlocks(blockDim);
    tiling.set_workspaceBlockElems(FAI_WORKSPACE_BLOCK_ELEMS);
    tiling.set_workspaceElems(workspaceElems);

    const int64_t systemWorkspaceBytes = SystemWorkspaceBytes(context);
    if (systemWorkspaceBytes < 0) {
        OP_LOGE(context, "SlidingTileAttention failed to query the system workspace size.");
        return ge::GRAPH_FAILED;
    }
    int64_t workspaceBytes = systemWorkspaceBytes + workspaceElems * FAI_WORKSPACE_BYTES_PER_ELEM;
    if (dtype == ge::DT_BF16) {
        const int64_t fullVStageBytes =
            meta.batch * meta.heads * meta.seq * meta.dim * static_cast<int64_t>(sizeof(uint16_t));
        workspaceBytes += blockDim * FAI_BF16_STAGE_PREFIX_BYTES_PER_BLOCK + fullVStageBytes;
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
    rawTiling->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(blockDim);
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
    .TilingParse<SlidingTileAttentionCompileInfo>(TilingParse);

} // namespace optiling
