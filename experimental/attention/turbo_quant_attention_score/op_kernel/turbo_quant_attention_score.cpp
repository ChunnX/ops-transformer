/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "turbo_quant_attention_score_prefill.h"

extern "C" __global__ __aicore__ void turbo_quant_attention_score(GM_ADDR queryRotated, GM_ADDR queryQjl,
                                                                  GM_ADDR keyCacheIdx, GM_ADDR keyCacheQjl,
                                                                  GM_ADDR keyCacheNorm, GM_ADDR keyCacheGamma,
                                                                  GM_ADDR blockTable, GM_ADDR seqLens,
                                                                  GM_ADDR attnScores, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(3, KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(2)) {
        TurboQuant::RunPrefill<bfloat16_t>(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm,
                                           keyCacheGamma, blockTable, seqLens, attnScores,
                                           AscendC::GetUserWorkspace(workspace), t);
        return;
    }
    if (TILING_KEY_IS(3)) {
        TurboQuant::RunPrefill<half>(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma,
                                     blockTable, seqLens, attnScores, AscendC::GetUserWorkspace(workspace), t);
        return;
    }

    if (TILING_KEY_IS(0)) {
        if (t.tokens == 1) {
            TurboQuant::Kernel<bfloat16_t, true> op;
            op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                    attnScores, t.tokens, t.qHeads, t.kvHeads, t.batch, t.blocks, t.tableWidth, t.maxLen, t.bits);
            op.Process();
        } else {
            TurboQuant::Kernel<bfloat16_t> op;
            op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                    attnScores, t.tokens, t.qHeads, t.kvHeads, t.batch, t.blocks, t.tableWidth, t.maxLen, t.bits);
            op.Process();
        }
    } else if (TILING_KEY_IS(1)) {
        if (t.tokens == 1) {
            TurboQuant::Kernel<half, true> op;
            op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                    attnScores, t.tokens, t.qHeads, t.kvHeads, t.batch, t.blocks, t.tableWidth, t.maxLen, t.bits);
            op.Process();
        } else {
            TurboQuant::Kernel<half> op;
            op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable, seqLens,
                    attnScores, t.tokens, t.qHeads, t.kvHeads, t.batch, t.blocks, t.tableWidth, t.maxLen, t.bits);
            op.Process();
        }
    }
}
