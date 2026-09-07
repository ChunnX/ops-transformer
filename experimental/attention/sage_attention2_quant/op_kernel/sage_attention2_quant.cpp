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
extern "C" __global__ __aicore__ void sage_attention2_quant(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mean,
                                                            GM_ADDR v_scale, GM_ADDR q8, GM_ADDR k8, GM_ADDR v8,
                                                            GM_ADDR q_scale, GM_ADDR k_scale, GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SageParams);
    GET_TILING_DATA_WITH_STRUCT(SageParams, p, tiling);
    bool wide = p.dp == 128 && p.sq % 128 == 0 && p.sk % 512 == 0 && p.sk >= 4096;
    if (wide) {
        if (p.bf16)
            Sage::Quant<bfloat16_t, true>(q, k, v, mean, v_scale, q8, k8, v8, q_scale, k_scale, p);
        else
            Sage::Quant<half, true>(q, k, v, mean, v_scale, q8, k8, v8, q_scale, k_scale, p);
    } else {
        if (p.bf16)
            Sage::Quant<bfloat16_t>(q, k, v, mean, v_scale, q8, k8, v8, q_scale, k_scale, p);
        else
            Sage::Quant<half>(q, k, v, mean, v_scale, q8, k8, v8, q_scale, k_scale, p);
    }
}
