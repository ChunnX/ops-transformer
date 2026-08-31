/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_attention_residuals_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockAttentionResidualsGrad);

const aclTensor *BlockAttentionResidualsGrad(
    const aclTensor *partial_block,
    const aclTensor *block_res,
    const aclTensor *proj_weight,
    const aclTensor *norm_weight,
    const aclTensor *grad_hidden_states,
    const aclTensor *inv_norm,
    const aclTensor *probs,
    int64_t valid_block_nums,
    aclTensor *grad_partial_block,
    aclTensor *grad_block_res,
    aclTensor *grad_proj_weight,
    aclTensor *grad_norm_weight,
    aclOpExecutor *executor)
{
    L0_DFX(BlockAttentionResidualsGrad,
        partial_block, block_res, proj_weight, norm_weight,
        grad_hidden_states, inv_norm, probs, valid_block_nums,
        grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight);

    auto ret = INFER_SHAPE(BlockAttentionResidualsGrad,
        OP_INPUT(partial_block, block_res, proj_weight, norm_weight, grad_hidden_states, inv_norm, probs),
        OP_OUTPUT(grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight),
        OP_ATTR(valid_block_nums));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
            "BlockAttentionResidualsGrad INFER_SHAPE failed.");
        return nullptr;
    }

    auto launcherRet = ADD_TO_LAUNCHER_LIST_AICORE(BlockAttentionResidualsGrad,
        OP_INPUT(partial_block, block_res, proj_weight, norm_weight,
                 grad_hidden_states, inv_norm, probs),
        OP_OUTPUT(grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight),
        OP_ATTR(valid_block_nums));
    if (launcherRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
            "BlockAttentionResidualsGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return grad_partial_block;
}

} // namespace l0op
