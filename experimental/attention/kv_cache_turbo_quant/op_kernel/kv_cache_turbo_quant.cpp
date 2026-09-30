/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file kv_cache_turbo_quant.cpp
 */
#include "kernel_kv_cache_turbo_quant.h"
#include "kernel_kv_cache_turbo_quant_mix.h"

// The v220 MIX wrapper auto-generates a matmul::clearWorkspace(workspace) call
// on every launch. On this stack BOTH active variants are broken:
//  - stock ClearWorkspaceImpl: its internal S_MTE3 EVENT_ID0 sync collides and
//    every 2nd launch dies 507057 (even for the pure-vector decode path);
//  - hand-rolled wipe: the unconditional SetAtomicNone/SetVectorMask resets
//    poison the pure-vector decode path (507015, bisected build242).
// A bare no-op keeps single launches of both paths alive. The MIX runtime's
// stale KFC-channel state after a cube launch is instead handled by the
// caller rotating to a fresh workspace per launch (the runtime re-inits the
// channel from scratch on a zeroed workspace - verified: warmup loops pass
// with per-launch workspaces, crash with a reused one).
namespace matmul {
__aicore__ inline void clearWorkspace(__gm__ uint8_t* workspace)
{
    (void)workspace;
}
} // namespace matmul

extern "C" __global__ __aicore__ void kv_cache_turbo_quant(GM_ADDR kv_vectors, GM_ADDR rotation_matrix,
                                                           GM_ADDR qjl_matrix, GM_ADDR quant_idx, GM_ADDR quant_qjl,
                                                           GM_ADDR quant_norm, GM_ADDR quant_gamma, GM_ADDR workspace,
                                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.totalVec >= 1024 && tilingData.qjlDim == 128) {
        // prefill path: AIC/AIV hybrid cube kernel. The MIX runtime owns the
        // leading libapi bytes of the workspace; user data starts after it.
        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        KernelKvCacheTurboQuantMix op;
        op.Init(kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl, quant_norm, quant_gamma, user,
                &tilingData);
        op.Process();
    } else {
        // decode / small-batch path: pure-vector kernel (AIC idles in mix mode)
        if ASCEND_IS_AIV {
            KernelKvCacheTurboQuant op;
            op.Init(kv_vectors, rotation_matrix, qjl_matrix, quant_idx, quant_qjl, quant_norm, quant_gamma,
                    &tilingData);
            op.Process();
        }
    }
}
