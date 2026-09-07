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
#include "attention_wide_stream.h"

extern "C" int sage_launch_stages_layout(void *stream, const SageParams *params, void **ptrs, int stages, int permuteV);

// Force the N512 path for isolated diagnostics, including shorter supported inputs.
extern "C" __global__ __aicore__ void sage_attention_wide(GM_ADDR q, GM_ADDR q8, GM_ADDR k8, GM_ADDR v8, GM_ADDR qs,
                                                          GM_ADDR ks, GM_ADDR mean, GM_ADDR vs, GM_ADDR out,
                                                          GM_ADDR lse, SageParams p)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (p.causal) {
        if (p.bf16)
            Sage::AttentionWideStream<bfloat16_t, true>(q, q8, k8, v8, qs, ks, mean, vs, out, lse, p);
        else
            Sage::AttentionWideStream<half, true>(q, q8, k8, v8, qs, ks, mean, vs, out, lse, p);
    } else if (p.bf16)
        Sage::AttentionWideStream<bfloat16_t>(q, q8, k8, v8, qs, ks, mean, vs, out, lse, p);
    else
        Sage::AttentionWideStream<half>(q, q8, k8, v8, qs, ks, mean, vs, out, lse, p);
}

extern "C" int sage_launch_wide(void *stream, const SageParams *params, void **ptrs, int quantOnly)
{
    if (!params || !ptrs || !params->cores || params->dp != 128 || params->sq % 128 || params->sk % 512 ||
        params->sk < 1024)
        return 1;
    int result = sage_launch_stages_layout(stream, params, ptrs, 3, !quantOnly);
    if (result || quantOnly)
        return result;
    SageParams p = *params;
    sage_attention_wide<<<p.cores, nullptr, stream>>>(
        (uint8_t *)ptrs[0], (uint8_t *)ptrs[5], (uint8_t *)ptrs[6], (uint8_t *)ptrs[7], (uint8_t *)ptrs[12],
        (uint8_t *)ptrs[13], (uint8_t *)ptrs[3], (uint8_t *)ptrs[4], (uint8_t *)ptrs[10], (uint8_t *)ptrs[11], p);
    return 0;
}
