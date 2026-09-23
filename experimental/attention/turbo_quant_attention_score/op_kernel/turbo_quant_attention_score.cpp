/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file turbo_quant_attention_score.cpp
 * \brief kernel 入口：tilingkey 102/103/104 分别对应 mse_bits=2/3/4 编译期特化
 */
#include "kernel_turbo_quant_attention_score.h"

using namespace AscendC;
using namespace TurboQuantAttentionScore;

extern "C" __global__ __aicore__ void turbo_quant_attention_score(GM_ADDR query_rotated, GM_ADDR query_qjl,
                                                                  GM_ADDR key_cache_idx, GM_ADDR key_cache_qjl,
                                                                  GM_ADDR key_cache_norm, GM_ADDR key_cache_gamma,
                                                                  GM_ADDR block_table, GM_ADDR seq_lens,
                                                                  GM_ADDR attn_scores, GM_ADDR workspace,
                                                                  GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TurboQuantAttentionScoreTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

    if (TILING_KEY_IS(102)) {
        KernelTurboQuantAttentionScore<2> op;
        op.Init(query_rotated, query_qjl, key_cache_idx, key_cache_qjl, key_cache_norm, key_cache_gamma,
                block_table, seq_lens, attn_scores, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(103)) {
        KernelTurboQuantAttentionScore<3> op;
        op.Init(query_rotated, query_qjl, key_cache_idx, key_cache_qjl, key_cache_norm, key_cache_gamma,
                block_table, seq_lens, attn_scores, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(104)) {
        KernelTurboQuantAttentionScore<4> op;
        op.Init(query_rotated, query_qjl, key_cache_idx, key_cache_qjl, key_cache_norm, key_cache_gamma,
                block_table, seq_lens, attn_scores, &tilingData, &pipe);
        op.Process();
    }
}
