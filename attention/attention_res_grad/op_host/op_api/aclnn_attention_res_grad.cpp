/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_attention_res_grad.h"
#include "attention_res_grad.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include "aclnn/aclnn_base.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAttentionResGradGetWorkspaceSize(
    const aclTensor *prefix_sum, const aclTensor *block_residual,
    const aclTensor *proj_weight, const aclTensor *norm_weight,
    const aclTensor *grad_output, const aclTensor *inv_rms,
    const aclTensor *probs,
    const aclTensor *grad_prefix_sum, const aclTensor *grad_block_residual,
    const aclTensor *grad_proj_weight, const aclTensor *grad_norm_weight,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{

    L2_DFX_PHASE_1(aclnnAttentionResGrad,
        DFX_IN(prefix_sum, block_residual, proj_weight, norm_weight, grad_output, inv_rms, probs),
        DFX_OUT(grad_prefix_sum, grad_block_residual, grad_proj_weight, grad_norm_weight));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto output = l0op::AttentionResGrad(
        prefix_sum, block_residual, proj_weight, norm_weight,
        grad_output, inv_rms, probs,
        const_cast<aclTensor*>(grad_prefix_sum),
        const_cast<aclTensor*>(grad_block_residual),
        const_cast<aclTensor*>(grad_proj_weight),
        const_cast<aclTensor*>(grad_norm_weight),
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAttentionResGrad(void *workspace, uint64_t workspaceSize,
                           aclOpExecutor *executor, aclrtStream stream)
{
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != 0) {
        OP_LOGE(ACLNN_ERR_INNER, "AttentionResGrad launch failed, ret = %d.", ret);
        return ret;
    }
    return ret;
}

#ifdef __cplusplus
}
#endif
