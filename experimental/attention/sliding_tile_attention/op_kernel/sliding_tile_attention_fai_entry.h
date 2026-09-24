/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_FAI_ENTRY_H
#define SLIDING_TILE_ATTENTION_FAI_ENTRY_H
#include "sliding_tile_attention_fai_op.h"

namespace {

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
    uint32_t softmaxScaleBits = *reinterpret_cast<uint32_t*>(&tilingData.softmax_scale);
    GM_ADDR userWorkspace = workspace;
    // Review 190252589: the workspace regions reserved here must be exactly the
    // regions the device consumes, so both sides use the shared layout rules.
    const bool dtypeIsFp16 = tilingData.dtype != 1;
    const bool withQkStaging = sliding_tile_attention::LayoutNeedsQkStaging(tilingData.meta.dim);
    const bool withCorrection = sliding_tile_attention::LayoutUsesResidualProbability(
        dtypeIsFp16, tilingData.meta.dim, tilingData.meta.has_text, tilingData.meta.tile_numel,
        sliding_tile_attention::LayoutWindowsCoverGrid(tilingData.meta), PACKED_D72_FULL);
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

#endif
