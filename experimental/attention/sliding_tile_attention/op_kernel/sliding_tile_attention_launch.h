/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_LAUNCH_H
#define SLIDING_TILE_ATTENTION_LAUNCH_H

#include <cstdint>

#ifndef GM_ADDR
#define GM_ADDR void *
#endif

constexpr int64_t SLIDING_TILE_ATTENTION_WINDOW_VALUES_PER_HEAD = 3;
constexpr int64_t SLIDING_TILE_ATTENTION_WINDOW_VALUE_COUNT = 128 * SLIDING_TILE_ATTENTION_WINDOW_VALUES_PER_HEAD;
constexpr int64_t SLIDING_TILE_ATTENTION_PARTIAL_MAX_TEXT_ROWS = 256;
constexpr int64_t SLIDING_TILE_ATTENTION_PARTIAL_MAX_PRODUCERS = 64;
constexpr int64_t SLIDING_TILE_ATTENTION_PARTIAL_HEADER_ELEMENTS = 16;
constexpr int64_t SLIDING_TILE_ATTENTION_PARTIAL_RECORD_ELEMENTS = SLIDING_TILE_ATTENTION_PARTIAL_HEADER_ELEMENTS + 256;
constexpr int64_t SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY = 10;

struct SlidingTileAttentionMeta {
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t seq = 0;
    int64_t dim = 0;
    int64_t image_seq = 0;
    int64_t text_len = 0;
    int64_t has_text = 0;
    int64_t canvas_t = 0;
    int64_t canvas_h = 0;
    int64_t canvas_w = 0;
    int64_t tile_t = 0;
    int64_t tile_h = 0;
    int64_t tile_w = 0;
    int64_t grid_t = 0;
    int64_t grid_h = 0;
    int64_t grid_w = 0;
    int64_t tile_numel = 0;
    int64_t window_count = 0;
    int64_t pair_axis = -1; // 0=T, 1=H, 2=W
    int64_t pair_group_count = 0;
    int64_t grouped_query_chunks = 3;
    int64_t pair_group_start[SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY] = {0};
    int64_t pair_group_span[SLIDING_TILE_ATTENTION_PAIR_GROUP_CAPACITY] = {0};
    int64_t windows[SLIDING_TILE_ATTENTION_WINDOW_VALUE_COUNT] = {0};
};

// Text queries cover the allocated storage tail; text_length only bounds valid text KV.
inline int64_t TextQueryRows(const SlidingTileAttentionMeta &meta)
{
    return meta.has_text != 0 ? (meta.seq - meta.image_seq) : 0;
}

#endif
