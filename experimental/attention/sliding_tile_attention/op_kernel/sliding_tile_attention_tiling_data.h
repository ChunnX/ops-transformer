/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SLIDING_TILE_ATTENTION_TILING_DATA_H
#define SLIDING_TILE_ATTENTION_TILING_DATA_H

#include <cstdint>

#include "sliding_tile_attention_launch.h"

struct SlidingTileAttentionTilingData {
    SlidingTileAttentionMeta meta;
    int64_t total_work = 0;
    int64_t work_per_block = 0;
    int64_t dtype = 0; // 0=fp16 direct, 1=bf16 input staging with half FAI compute
    float softmax_scale = 0.0f;
    uint32_t reserved = 0;
    int64_t workspace_elems = 0;
    int64_t workspace_blocks = 0;
    int64_t workspace_block_elems = 0;
};

#endif
