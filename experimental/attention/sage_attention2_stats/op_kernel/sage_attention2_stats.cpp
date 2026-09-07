/*
 * Copyright (c) 2026 G_W_E. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../sageattention2/op_kernel/quant.h"
#include "../../sageattention2/op_kernel/attention.h"
extern "C" __global__ __aicore__ void sage_attention2_stats(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean,
                                                            GM_ADDR v_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SageParams);
    GET_TILING_DATA_WITH_STRUCT(SageParams, p, tiling);
    GM_ADDR partials = workspace ? GetUserWorkspace(workspace) : nullptr;
    if (p.bf16)
        Sage::Stats<bfloat16_t>(k, v, mean, v_scale, p, partials);
    else
        Sage::Stats<half>(k, v, mean, v_scale, p, partials);
}
