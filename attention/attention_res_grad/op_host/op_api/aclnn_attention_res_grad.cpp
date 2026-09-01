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

#include "aclnn_kernels/contiguous.h"

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

    // 与正向算子对齐：输入统一转 Contiguous 后再下发
    auto prefixSum_ = l0op::Contiguous(prefix_sum, uniqueExecutor.get());
    CHECK_RET(prefixSum_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto blockResidual_ = l0op::Contiguous(block_residual, uniqueExecutor.get());
    CHECK_RET(blockResidual_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto projWeight_ = l0op::Contiguous(proj_weight, uniqueExecutor.get());
    CHECK_RET(projWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto normWeight_ = l0op::Contiguous(norm_weight, uniqueExecutor.get());
    CHECK_RET(normWeight_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradOutput_ = l0op::Contiguous(grad_output, uniqueExecutor.get());
    CHECK_RET(gradOutput_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto invRms_ = l0op::Contiguous(inv_rms, uniqueExecutor.get());
    CHECK_RET(invRms_ != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto probs_ = l0op::Contiguous(probs, uniqueExecutor.get());
    CHECK_RET(probs_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::AttentionResGrad(
        prefixSum_, blockResidual_, projWeight_, normWeight_,
        gradOutput_, invRms_, probs_,
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
