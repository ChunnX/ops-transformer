/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sliding_tile_attention_fai_entry.h"

// Review 187984911: this translation unit only instantiates the operator
// object and drives Init()/Process() through the entry helper; every
// implementation detail lives in the functional headers above.
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
