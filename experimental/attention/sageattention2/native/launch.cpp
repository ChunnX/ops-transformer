/*
 * Copyright (c) 2026 G_W_E. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "quant.h"
#include "attention.h"
extern "C" __global__ __aicore__ void sage_stats(GM_ADDR k, GM_ADDR v, GM_ADDR m, GM_ADDR vs, GM_ADDR partials,
                                                 SageParams p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (p.bf16)
        Sage::Stats<bfloat16_t>(k, v, m, vs, p, partials);
    else
        Sage::Stats<half>(k, v, m, vs, p, partials);
}
extern "C" __global__ __aicore__ void sage_quant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR m, GM_ADDR vs, GM_ADDR q8,
                                                 GM_ADDR k8, GM_ADDR v8, GM_ADDR rawQs, GM_ADDR rawKs, GM_ADDR qs,
                                                 GM_ADDR ks, SageParams p, uint32_t permuteV)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (permuteV) {
        if (p.bf16)
            Sage::Quant<bfloat16_t, true, true>(q, k, v, m, vs, q8, k8, v8, qs, ks, p, rawQs, rawKs);
        else
            Sage::Quant<half, true, true>(q, k, v, m, vs, q8, k8, v8, qs, ks, p, rawQs, rawKs);
    } else {
        if (p.bf16)
            Sage::Quant<bfloat16_t, false, true>(q, k, v, m, vs, q8, k8, v8, qs, ks, p, rawQs, rawKs);
        else
            Sage::Quant<half, false, true>(q, k, v, m, vs, q8, k8, v8, qs, ks, p, rawQs, rawKs);
    }
}
extern "C" __global__ __aicore__ void sage_attention(GM_ADDR q, GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs,
                                                     GM_ADDR ks, GM_ADDR m, GM_ADDR vs, GM_ADDR o, GM_ADDR lse,
                                                     SageParams p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (p.bf16)
        Sage::Attention<bfloat16_t>(q, q8, k8, v8, qs, ks, m, vs, o, lse, p);
    else
        Sage::Attention<half>(q, q8, k8, v8, qs, ks, m, vs, o, lse, p);
}
// Internal diagnostic ABI: 14 pointers; slots 12/13 are prepared FP16 Q/K scales.
// Stage bits: stats=1, quant (including FP16 scales)=2, attention=4; full pipeline=7.
extern "C" int sage_launch_stages_layout(void *stream, const SageParams *params, void **ptrs, int stages, int permuteV)
{
    if (!params || !ptrs || !params->cores || stages < 1 || stages > 7)
        return 1;
    SageParams p = *params;
    // Stats scratch fits in Q8 before quantization: SPLITS*4*sizeof(float) <= qp.
    static_assert(SAGE_STATS_SPLITS * 4 * sizeof(float) <= 128, "Stats scratch exceeds the minimum Q8 allocation");
    if (stages & 1)
        sage_stats<<<p.cores * 2, nullptr, stream>>>((uint8_t *)ptrs[1], (uint8_t *)ptrs[2], (uint8_t *)ptrs[3],
                                                     (uint8_t *)ptrs[4], (uint8_t *)ptrs[5], p);
    if (stages & 2)
        sage_quant<<<p.cores * 2, nullptr, stream>>>(
            (uint8_t *)ptrs[0], (uint8_t *)ptrs[1], (uint8_t *)ptrs[2], (uint8_t *)ptrs[3], (uint8_t *)ptrs[4],
            (uint8_t *)ptrs[5], (uint8_t *)ptrs[6], (uint8_t *)ptrs[7], (uint8_t *)ptrs[8], (uint8_t *)ptrs[9],
            (uint8_t *)ptrs[12], (uint8_t *)ptrs[13], p, permuteV);
    if (stages & 4)
        sage_attention<<<p.cores, nullptr, stream>>>(
            (uint8_t *)ptrs[0], (uint8_t *)ptrs[5], (uint8_t *)ptrs[6], (uint8_t *)ptrs[7], (uint8_t *)ptrs[12],
            (uint8_t *)ptrs[13], (uint8_t *)ptrs[3], (uint8_t *)ptrs[4], (uint8_t *)ptrs[10], (uint8_t *)ptrs[11], p);
    return 0;
}

extern "C" int sage_launch_stages(void *stream, const SageParams *params, void **ptrs, int stages)
{
    if (!params)
        return 1;
    const auto &p = *params;
    bool wide = p.dp == 128 && p.sq % 128 == 0 && p.sk % 512 == 0 && p.sk >= 4096;
    // Quant-only diagnostics retain their canonical V row order.
    return sage_launch_stages_layout(stream, params, ptrs, stages, wide && stages != 3);
}

extern "C" int sage_launch(void *stream, const SageParams *params, void **ptrs, int quantOnly)
{
    return sage_launch_stages(stream, params, ptrs, quantOnly ? 3 : 7);
}
