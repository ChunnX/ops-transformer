/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../core/attention_engine.h"
#include "sage_attention2_tiling_data.h"
#include "sage_attention2_template_tiling_key.h"

template <int INPUT_DTYPE, bool WIDE>
__global__ __aicore__ void sage_attention2(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR lse, GM_ADDR workspace,
                                           GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(SageAttention2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SageAttention2TilingData, tilingData, tiling);
    GM_ADDR work = AscendC::GetUserWorkspace(workspace);

    static_assert(INPUT_DTYPE == SAGE_TPL_FP16 || INPUT_DTYPE == SAGE_TPL_BF16, "Unsupported input dtype for SageAttention2");

    if constexpr (INPUT_DTYPE == SAGE_TPL_BF16) {
        sage::core::AttentionEngine<bfloat16_t, WIDE>::Execute(q, k, v, out, lse, work, tilingData);
    } else {
        sage::core::AttentionEngine<half, WIDE>::Execute(q, k, v, out, lse, work, tilingData);
    }
}
