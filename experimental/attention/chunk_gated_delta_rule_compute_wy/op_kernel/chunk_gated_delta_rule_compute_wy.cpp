/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#include "arch20/compat_310p.h"
#endif
#include "arch20/compute_wy_kernel.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_compute_wy(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR q_kernel, GM_ADDR k_kernel, GM_ADDR w_kernel, GM_ADDR u_kernel, GM_ADDR g_kernel,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleComputeWyTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkGatedDeltaRuleComputeWyTilingData, tilingData, tiling);

    // MIX / default AiCore so Cube Matmul is available on Ascend 310P (not AIV_ONLY).
    TPipe pipe;
    ChunkGatedDeltaRuleComputeWy::KernelComputeWy op;
    op.Init(q, k, v, g, beta, q_kernel, k_kernel, w_kernel, u_kernel, g_kernel, workspace, &tilingData, &pipe);
    op.Process();
}
