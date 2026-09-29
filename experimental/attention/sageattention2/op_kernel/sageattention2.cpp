/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stats.h"
#include "quant.h"
#include "attention.h"
#include "sage_attention2_tiling_data.h"
#include "sage_attention2_template_tiling_key.h"

namespace Sage {
template <class T, bool WIDE>
__aicore__ inline __attribute__((always_inline)) void FusedAttention(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                                                     GM_ADDR lse, GM_ADDR work,
                                                                     const SageAttention2TilingData &t)
{
    const SageParams p = t.params;
    if ASCEND_IS_AIV {
        // GetBlockIdx() enumerates both AIV sub-blocks in MIX 1:2. Stats and
        // Quant stride by 2 * p.cores, including AIVs with no assigned work.
        Stats<T, false>(k, work + t.mean, p);
        // Each head has one writer in both passes, so they share the barrier
        // below. smooth_v is off for the two-level formats, where V is
        // quantised directly and the centring term is never read.
        if (p.smoothV)
            Stats<T, true>(v, work + t.vMean, p);
        SyncAll<true>();
        // The wide N512 PV stream consumes V as interleaved row pairs, so the
        // quantiser's store layout is selected by the same WIDE flag.
        Quant<T, WIDE>(q, k, v, work + t.mean, work + t.vMean, work + t.vMxScale, work + t.q8, work + t.k8, work + t.v8,
                       work + t.qScale, work + t.kScale, p);
    }
    // Drain preprocessing pipes and publish K/V GM writes to every AIC/AIV.
    // When Q is decoupled (WIDE path), Q preprocessing HBM roundtrips and global
    // sync stalls are eliminated; Q is quantized on-chip in Attention.
    // Idle cores also reach this barrier. Its flags (11..14) do not overlap
    // attention's MODE4 flags (0..4, where 4 is the streaming Q handoff).
    SyncAll<false>();
    if constexpr (WIDE) {
        Attention<T, WIDE>(q, work + t.k8, work + t.v8, work + t.qScale, work + t.kScale,
                           work + t.vMxScale, work + t.vMean, out, lse, p);
    } else {
        Attention<T, WIDE>(work + t.q8, work + t.k8, work + t.v8, work + t.qScale, work + t.kScale,
                           work + t.vMxScale, work + t.vMean, out, lse, p);
    }
}
} // namespace Sage

template <int INPUT_DTYPE, bool WIDE>
__global__ __aicore__ void sage_attention2(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SageAttention2TilingData, t, tiling);
    GM_ADDR work = GetUserWorkspace(workspace);
    static_assert(INPUT_DTYPE == SAGE_TPL_FP16 || INPUT_DTYPE == SAGE_TPL_BF16, "unsupported input dtype");
    if constexpr (INPUT_DTYPE == SAGE_TPL_BF16)
        Sage::FusedAttention<bfloat16_t, WIDE>(q, k, v, out, lse, work, t);
    else
        Sage::FusedAttention<half, WIDE>(q, k, v, out, lse, work, t);
}
