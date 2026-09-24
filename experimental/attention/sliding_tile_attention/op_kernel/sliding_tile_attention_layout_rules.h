/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_LAYOUT_RULES_H
#define SLIDING_TILE_ATTENTION_LAYOUT_RULES_H

#include <cstdint>

#include "sliding_tile_attention_launch.h"

// This header is shared by host tiling (gcc) and the device kernel (bisheng).
// Device code must see __aicore__ so the helpers are emitted for the AI Core
// instead of being referenced as unresolved host symbols; the host translation
// unit defines OPFORGE_STA_LAYOUT_RULES_HOST before including this header and
// therefore gets a plain inline definition.  Same convention as
// resource_ledger.hpp's OPFORGE_RESOURCE_LEDGER_HOST_PROJECTION_ONLY.
#ifdef OPFORGE_STA_LAYOUT_RULES_HOST
#define OPFORGE_STA_LAYOUT_RULES_INLINE inline
#else
#define OPFORGE_STA_LAYOUT_RULES_INLINE __aicore__ inline
#endif

// Single source of truth for the workspace decisions that host tiling (sizing)
// and the device kernel (region offsets plus compute paths) must agree on.
//
// Review 190252589: the host previously sized the Q/K padding stage with
// `dim % 16 != 0 || dim > 128` and the residual-probability region with
// `dim > 128 || dim == 72`, while the device only consumed those regions under
// `NeedsQkStaging()` / `UsesResidualProbability()`.  Those host tests are strict
// supersets of the device predicates, so unused workspace was reserved for
// head dims such as 72 and for every dim > 128.  Both sides now evaluate the
// same rule helpers instead of restating the conditions.
namespace sliding_tile_attention {

constexpr int64_t LAYOUT_VECTOR_ELEMENTS = 16; // must equal device VECTOR_BLOCK_ELEMENTS
constexpr int64_t LAYOUT_QK_SLICE_DIM = 128;   // must equal device FAI_QK_SLICE_DIM
constexpr int64_t LAYOUT_TILE_ELEMENTS = 384;  // must equal device TILE_ELEMENTS (6x8x8 tile)

OPFORGE_STA_LAYOUT_RULES_INLINE int64_t LayoutQkDim(int64_t dim)
{
    return ((dim + LAYOUT_VECTOR_ELEMENTS - 1) / LAYOUT_VECTOR_ELEMENTS) * LAYOUT_VECTOR_ELEMENTS;
}

OPFORGE_STA_LAYOUT_RULES_INLINE int64_t LayoutQkSliceCount(int64_t dim)
{
    return (LayoutQkDim(dim) + LAYOUT_QK_SLICE_DIM - 1) / LAYOUT_QK_SLICE_DIM;
}

OPFORGE_STA_LAYOUT_RULES_INLINE int64_t LayoutQkStageDim(int64_t dim)
{
    return LayoutQkSliceCount(dim) > 1 ? LayoutQkSliceCount(dim) * LAYOUT_QK_SLICE_DIM : LayoutQkDim(dim);
}

// The Q/K padding stage is consumed only when the aligned head dim spans more
// than one 128-element slice, or when a sub-64 head dim still needs padding up
// to one full vector block.
OPFORGE_STA_LAYOUT_RULES_INLINE bool LayoutNeedsQkStaging(int64_t dim)
{
    return LayoutQkSliceCount(dim) > 1 || (dim < 64 && LayoutQkDim(dim) != dim);
}

// True when every head window covers the whole canvas grid, i.e. the tile sees
// all image KV rows.  Mirrors the window test of the compact full-window path.
OPFORGE_STA_LAYOUT_RULES_INLINE bool LayoutWindowsCoverGrid(const SlidingTileAttentionMeta &meta)
{
    if (meta.heads <= 0 || meta.window_count < meta.heads * SLIDING_TILE_ATTENTION_WINDOW_VALUES_PER_HEAD) {
        return false;
    }
    for (int64_t head = 0; head < meta.heads; ++head) {
        const int64_t base = head * SLIDING_TILE_ATTENTION_WINDOW_VALUES_PER_HEAD;
        if (meta.windows[base] < meta.grid_t || meta.windows[base + 1] < meta.grid_h ||
            meta.windows[base + 2] < meta.grid_w) {
            return false;
        }
    }
    return true;
}

// The dedicated residual-probability region is written only by the FP16 D=72
// compact full-window path (device `UsesResidualProbability()`), never by the
// packed D=72 non-full-window specialisation or by any dim > 128 launch.
OPFORGE_STA_LAYOUT_RULES_INLINE bool LayoutUsesResidualProbability(bool isFp16, int64_t dim, int64_t hasText,
                                                                   int64_t tileNumel, bool fullWindow,
                                                                   bool packedD72FullWindow)
{
    return !packedD72FullWindow && isFp16 && dim == 72 && hasText == 0 && tileNumel == LAYOUT_TILE_ELEMENTS &&
           fullWindow;
}

} // namespace sliding_tile_attention

#endif
