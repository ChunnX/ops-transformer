/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_res_grad.cpp
 * \brief attention_res_grad kernel entry with template dispatch
 */

#include "attention_res_grad.h"

using namespace AscendC;

enum class AttentionResGradTilingKey : uint32_t {
    TILING_KEY_FP16 = static_cast<uint32_t>(TPL_SCH_MODE_FP16),
    TILING_KEY_BF16 = static_cast<uint32_t>(TPL_SCH_MODE_BF16),
};

template <uint32_t schMode>
__global__ __aicore__ void attention_res_grad(
    GM_ADDR prefix_sum, GM_ADDR block_residual, GM_ADDR proj_weight, GM_ADDR norm_weight,
    GM_ADDR grad_output, GM_ADDR inv_rms, GM_ADDR probs,
    GM_ADDR grad_prefix_sum, GM_ADDR grad_block_residual, GM_ADDR grad_proj_weight,
    GM_ADDR grad_norm_weight, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AttentionResGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(AttentionResGradTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(AttentionResGradTilingKey::TILING_KEY_FP16)) {
        NsAttentionResGrad::AttentionResGrad<half> op;
        op.Init(prefix_sum, block_residual, proj_weight, norm_weight,
                grad_output, inv_rms, probs,
                grad_prefix_sum, grad_block_residual,
                grad_proj_weight, grad_norm_weight, workspace,
                &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(AttentionResGradTilingKey::TILING_KEY_BF16)) {
        NsAttentionResGrad::AttentionResGrad<bfloat16_t> op;
        op.Init(prefix_sum, block_residual, proj_weight, norm_weight,
                grad_output, inv_rms, probs,
                grad_prefix_sum, grad_block_residual,
                grad_proj_weight, grad_norm_weight, workspace,
                &tilingData);
        op.Process();
    }
}
