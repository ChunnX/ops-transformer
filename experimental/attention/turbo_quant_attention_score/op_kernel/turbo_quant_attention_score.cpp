/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"
#include "turbo_quant_attention_score_impl.h"

extern "C" __global__ __aicore__ void turbo_quant_attention_score(
    GM_ADDR queryRotated, GM_ADDR queryQjl, GM_ADDR keyCacheIdx, GM_ADDR keyCacheQjl,
    GM_ADDR keyCacheNorm, GM_ADDR keyCacheGamma, GM_ADDR blockTable, GM_ADDR seqLens,
    GM_ADDR attnScores, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t, tiling);
    if (TILING_KEY_IS(0)) {
        TurboQuant::Kernel<bfloat16_t> op;
        op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm,
                keyCacheGamma, blockTable, seqLens, attnScores, t.tokens, t.qHeads,
                t.kvHeads, t.batch, t.blocks, t.tableWidth, t.maxLen, t.bits);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        TurboQuant::Kernel<half> op;
        op.Init(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm,
                keyCacheGamma, blockTable, seqLens, attnScores, t.tokens, t.qHeads,
                t.kvHeads, t.batch, t.blocks, t.tableWidth, t.maxLen, t.bits);
        op.Process();
    }
}
