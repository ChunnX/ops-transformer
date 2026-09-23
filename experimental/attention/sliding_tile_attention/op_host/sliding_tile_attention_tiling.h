/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_HOST_TILING_H
#define SLIDING_TILE_ATTENTION_HOST_TILING_H

#include "register/tilingdata_base.h"

#include "../op_kernel/sliding_tile_attention_tiling_data.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SlidingTileAttentionHostTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, heads);
TILING_DATA_FIELD_DEF(int64_t, seq);
TILING_DATA_FIELD_DEF(int64_t, dim);
TILING_DATA_FIELD_DEF(int64_t, imageSeq);
TILING_DATA_FIELD_DEF(int64_t, textLen);
TILING_DATA_FIELD_DEF(int64_t, hasText);
TILING_DATA_FIELD_DEF(int64_t, canvasT);
TILING_DATA_FIELD_DEF(int64_t, canvasH);
TILING_DATA_FIELD_DEF(int64_t, canvasW);
TILING_DATA_FIELD_DEF(int64_t, tileT);
TILING_DATA_FIELD_DEF(int64_t, tileH);
TILING_DATA_FIELD_DEF(int64_t, tileW);
TILING_DATA_FIELD_DEF(int64_t, gridT);
TILING_DATA_FIELD_DEF(int64_t, gridH);
TILING_DATA_FIELD_DEF(int64_t, gridW);
TILING_DATA_FIELD_DEF(int64_t, tileNumel);
TILING_DATA_FIELD_DEF(int64_t, windowCount);
TILING_DATA_FIELD_DEF(int64_t, pairAxis);
TILING_DATA_FIELD_DEF(int64_t, pairGroupCount);
TILING_DATA_FIELD_DEF(int64_t, groupedQueryChunks);
TILING_DATA_FIELD_DEF_ARR(int64_t, SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY, pairGroupStart);
TILING_DATA_FIELD_DEF_ARR(int64_t, SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY, pairGroupSpan);
TILING_DATA_FIELD_DEF_ARR(int64_t, SLIDING_TILE_ATTENTION_WINDOW_VALUE_COUNT, windows);
TILING_DATA_FIELD_DEF(int64_t, totalWork);
TILING_DATA_FIELD_DEF(int64_t, workPerBlock);
TILING_DATA_FIELD_DEF(int64_t, dtype);
TILING_DATA_FIELD_DEF(float, softmaxScale);
TILING_DATA_FIELD_DEF(uint32_t, reserved);
TILING_DATA_FIELD_DEF(int64_t, workspaceElems);
TILING_DATA_FIELD_DEF(int64_t, workspaceBlocks);
TILING_DATA_FIELD_DEF(int64_t, workspaceBlockElems);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SlidingTileAttentionFai, SlidingTileAttentionHostTilingData)
REGISTER_TILING_DATA_CLASS(SlidingTileAttentionFaiD256TextPartial, SlidingTileAttentionHostTilingData)
REGISTER_TILING_DATA_CLASS(SlidingTileAttentionFaiD256TextMerge, SlidingTileAttentionHostTilingData)

constexpr size_t SLIDING_TILE_ATTENTION_META_SCALAR_COUNT = 21;
constexpr size_t SLIDING_TILE_ATTENTION_PAIR_FIELD_COUNT = 2 * SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY;
constexpr size_t SLIDING_TILE_ATTENTION_RUNTIME_SCALAR_COUNT = 7;
constexpr size_t SLIDING_TILE_ATTENTION_TILING_FIELD_COUNT =
    SLIDING_TILE_ATTENTION_META_SCALAR_COUNT + SLIDING_TILE_ATTENTION_PAIR_FIELD_COUNT +
    SLIDING_TILE_ATTENTION_WINDOW_VALUE_COUNT + SLIDING_TILE_ATTENTION_RUNTIME_SCALAR_COUNT;
static_assert(sizeof(SlidingTileAttentionTilingData) == SLIDING_TILE_ATTENTION_TILING_FIELD_COUNT * sizeof(int64_t),
              "SlidingTileAttention host/device tiling ABI fields must stay aligned");

} // namespace optiling

#endif
