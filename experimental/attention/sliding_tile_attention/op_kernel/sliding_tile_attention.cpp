/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */

#include "sliding_tile_attention.h"
#include "sliding_tile_attention_legacy.h"
#include "sliding_tile_attention_tiling_key.h"

namespace {
template <bool IS_BF16>
__aicore__ inline void RunSlidingTileAttentionLegacy(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                                     const SlidingTileAttentionTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        SlidingTileAttentionLegacyKernel::SlidingTileAttentionCube<IS_BF16> op;
        op.Init(q, k, v, y, workspace, tilingData);
        op.Process();
    } else {
        SlidingTileAttentionLegacyKernel::SlidingTileAttention<IS_BF16> op;
        op.Init(q, k, v, y, workspace, tilingData);
        op.Process();
    }
}

template <bool IS_BF16>
__aicore__ inline void RunSlidingTileAttention2D(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                                 const SlidingTileAttentionTilingData *tilingData)
{
    if ASCEND_IS_AIC {
        SlidingTileAttentionKernel::SlidingTileAttentionCube<IS_BF16, true> op;
        op.Init(q, k, v, y, workspace, tilingData);
        op.Process();
    } else {
        SlidingTileAttentionKernel::SlidingTileAttention<IS_BF16, true> op;
        op.Init(q, k, v, y, workspace, tilingData);
        op.Process();
    }
}
} // namespace

extern "C" __global__ __aicore__ void sliding_tile_attention(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR y,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(SlidingTileAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(SlidingTileAttentionTilingData, tilingData, tiling);
    if (TILING_KEY_IS(SLIDING_TILE_ATTENTION_FP16_TILING_KEY)) {
        RunSlidingTileAttentionLegacy<false>(q, k, v, y, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(SLIDING_TILE_ATTENTION_BF16_TILING_KEY)) {
        RunSlidingTileAttentionLegacy<true>(q, k, v, y, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(SLIDING_TILE_ATTENTION_FP16_2D_SOFTMAX_TILING_KEY)) {
        RunSlidingTileAttention2D<false>(q, k, v, y, userWorkspace, &tilingData);
    } else if (TILING_KEY_IS(SLIDING_TILE_ATTENTION_BF16_2D_SOFTMAX_TILING_KEY)) {
        RunSlidingTileAttention2D<true>(q, k, v, y, userWorkspace, &tilingData);
    }
}
