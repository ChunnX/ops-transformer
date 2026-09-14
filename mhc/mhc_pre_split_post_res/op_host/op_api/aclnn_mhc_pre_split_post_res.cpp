/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "mhc_pre_split_post_res.h"
#include "aclnn_mhc_pre_split_post_res.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus MhcPreSplitPostResCommonProcess(const aclTensor *x, const aclTensor *phi,
    const aclTensor *alpha, const aclTensor *bias, float normEps,
    const aclTensor *hPostOut, const aclTensor *hResOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    // 输入非连续自动转连续；输出 h_post/h_res 要求连续（README 约束：非连续 ×）
    x = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(x != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous x failed.");

    phi = l0op::Contiguous(phi, uniqueExecutor.get());
    CHECK_COND(phi != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous phi failed.");

    alpha = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_COND(alpha != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous alpha failed.");

    bias = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_COND(bias != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous bias failed.");

    bool ok = l0op::MhcPreSplitPostRes(x, phi, alpha, bias, normEps, hPostOut, hResOut,
                                       uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnMhcPreSplitPostResGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    float normEps, const aclTensor *hPostOut, const aclTensor *hResOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreSplitPostRes,
                   DFX_IN(x, phi, alpha, bias),
                   DFX_OUT(hPostOut, hResOut));
    return MhcPreSplitPostResCommonProcess(x, phi, alpha, bias, normEps, hPostOut, hResOut,
                                           workspaceSize, executor);
}

aclnnStatus aclnnMhcPreSplitPostRes(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSplitPostRes);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
