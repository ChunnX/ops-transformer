/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "turbo_quant_attention_score_tiling_def.h"
#include "turbo_quant_attention_score_impl.hpp"

using namespace AscendC;

// asc_opc injects DTYPE_KEY_CACHE_NORM / DTYPE_KEY_CACHE_GAMMA per binary
// (bf16 → bfloat16_t, fp16 → half). Fallback keeps local/host builds working.
#ifndef DTYPE_KEY_CACHE_NORM
#define DTYPE_KEY_CACHE_NORM bfloat16_t
#endif

__aicore__ inline void CopyTilingData(optiling::TurboQuantAttentionScoreTilingData &dst, GM_ADDR tiling)
{
    uint32_t *local = reinterpret_cast<uint32_t *>(&dst);
    auto global = reinterpret_cast<__gm__ uint32_t *>(tiling);
    for (uint32_t i = 0; i < sizeof(optiling::TurboQuantAttentionScoreTilingData) / sizeof(uint32_t); ++i) {
        local[i] = global[i];
    }
}

extern "C" __global__ __aicore__ void turbo_quant_attention_score(GM_ADDR queryRotated, GM_ADDR queryQjl,
                                                                  GM_ADDR keyCacheIdx, GM_ADDR keyCacheQjl,
                                                                  GM_ADDR keyCacheNorm, GM_ADDR keyCacheGamma,
                                                                  GM_ADDR blockTable, GM_ADDR seqLens,
                                                                  GM_ADDR attnScores, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::TurboQuantAttentionScoreTilingData);
    optiling::TurboQuantAttentionScoreTilingData tilingDataIn;
    CopyTilingData(tilingDataIn, tiling);
    const auto *tilingData = &tilingDataIn;

    NsTurboQuantAttentionScore::TurboQuantAttentionScoreImpl<DTYPE_KEY_CACHE_NORM> op;
    if (TILING_KEY_IS(optiling::TQAS_TILING_KEY_MSE2)) {
        op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                attnScores, tilingData, 2U);
        op.Process();
    } else if (TILING_KEY_IS(optiling::TQAS_TILING_KEY_MSE3)) {
        op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                attnScores, tilingData, 3U);
        op.Process();
    } else if (TILING_KEY_IS(optiling::TQAS_TILING_KEY_MSE4)) {
        op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                attnScores, tilingData, 4U);
        op.Process();
    }
}
