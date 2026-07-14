/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file apply_rotary_pos_emb_grad.cpp
 * \brief
 */

#include "apply_rotary_pos_emb_grad_half.h"

using namespace AscendC;
using namespace AiInfraApplyRotaryPosEmbGrad;

extern "C" __global__ __aicore__ void
apply_rotary_pos_emb_grad(GM_ADDR grad_query_embed, GM_ADDR grad_key_embed, GM_ADDR cos, GM_ADDR sin,
                                   GM_ADDR query, GM_ADDR key, GM_ADDR grad_query, GM_ADDR grad_key, GM_ADDR grad_cos,
                                   GM_ADDR grad_sin, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(AiInfraApplyRotaryPosEmbGradTilingData, tiling_data_in, tiling);
    TPipe pipe;
    const AiInfraApplyRotaryPosEmbGradTilingData *__restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(0)) {
        AiInfraApplyRotaryPosEmbGradHalf<DTYPE_GRAD_QUERY_EMBED, true, true> op;
        op.Init(grad_query_embed, grad_key_embed, cos, sin, query, key, grad_query, grad_key, grad_cos, grad_sin,
                workspace, *tilingData);
        op.Process();
    } else if (TILING_KEY_IS(100)) {
        AiInfraApplyRotaryPosEmbGradHalf<DTYPE_GRAD_QUERY_EMBED, true, false> op;
        op.Init(grad_query_embed, grad_key_embed, cos, sin, query, key, grad_query, grad_key, grad_cos, grad_sin,
                workspace, *tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000)) {
        AiInfraApplyRotaryPosEmbGradHalf<DTYPE_GRAD_QUERY_EMBED, false, true> op;
        op.Init(grad_query_embed, grad_key_embed, cos, sin, query, key, grad_query, grad_key, grad_cos, grad_sin,
                workspace, *tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1100)) {
        AiInfraApplyRotaryPosEmbGradHalf<DTYPE_GRAD_QUERY_EMBED, false, false> op;
        op.Init(grad_query_embed, grad_key_embed, cos, sin, query, key, grad_query, grad_key, grad_cos, grad_sin,
                workspace, *tilingData);
        op.Process();
    }
}
