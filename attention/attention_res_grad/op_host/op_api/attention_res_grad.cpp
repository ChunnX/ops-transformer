/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attention_res_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AttentionResGrad);

const aclTensor *AttentionResGrad(
    const aclTensor *prefix_sum,
    const aclTensor *block_residual,
    const aclTensor *proj_weight,
    const aclTensor *norm_weight,
    const aclTensor *grad_output,
    const aclTensor *inv_rms,
    const aclTensor *probs,
    aclTensor *grad_prefix_sum,
    aclTensor *grad_block_residual,
    aclTensor *grad_proj_weight,
    aclTensor *grad_norm_weight,
    aclOpExecutor *executor)
{
    L0_DFX(AttentionResGrad,
        prefix_sum, block_residual, proj_weight, norm_weight,
        grad_output, inv_rms, probs,
        grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight);

    auto ret = INFER_SHAPE(AttentionResGrad,
        OP_INPUT(prefix_sum, block_residual, proj_weight, norm_weight, grad_output, inv_rms, probs),
        OP_OUTPUT(grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
            "AttentionResGrad INFER_SHAPE failed.");
        return nullptr;
    }

    auto launcherRet = ADD_TO_LAUNCHER_LIST_AICORE(AttentionResGrad,
        OP_INPUT(prefix_sum, block_residual, proj_weight, norm_weight,
                 grad_output, inv_rms, probs),
        OP_OUTPUT(grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight));
    if (launcherRet != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_STATIC_WORKSPACE_INVALID,
            "AttentionResGrad ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return nullptr;
    }

    return grad_prefix_sum;
}

} // namespace l0op
