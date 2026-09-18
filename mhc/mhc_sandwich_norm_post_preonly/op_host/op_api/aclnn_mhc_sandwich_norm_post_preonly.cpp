/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "mhc_sandwich_norm_post_preonly.h"
#include "aclnn_mhc_sandwich_norm_post_preonly.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static aclnnStatus MhcSandwichNormPostPreonlyCommonProcess(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost, const aclTensor *hResOptional,
    const aclTensor *phi, const aclTensor *alpha0, const aclTensor *bias, const aclTensor *gamma0Optional,
    const aclTensor *gamma1, const aclTensor *gamma2Optional, float normEps, float mhcPreNormEps, float hcEps,
    bool returnHInF32, const aclTensor *hInPrimeOut, const aclTensor *x2Out, const aclTensor *hInF32OutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    if (returnHInF32) {
        CHECK_COND(hInF32OutOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "hInF32OutOptional must not be nullptr when returnHInF32 is true.");
    } else {
        CHECK_COND(hInF32OutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "hInF32OutOptional must be nullptr when returnHInF32 is false.");
    }

    hOut = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_COND(hOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hOut failed.");

    residual = l0op::Contiguous(residual, uniqueExecutor.get());
    CHECK_COND(residual != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous residual failed.");
    CHECK_COND(hOut->GetDataType() == residual->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "hOut and residual must have the same dtype.");

    hPost = l0op::Contiguous(hPost, uniqueExecutor.get());
    CHECK_COND(hPost != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hPost failed.");

    CHECK_COND(hResOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "hResOptional is an interface-reserved optional input and must not be nullptr currently.");
    hResOptional = l0op::Contiguous(hResOptional, uniqueExecutor.get());
    CHECK_COND(hResOptional != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hResOptional failed.");

    phi = l0op::Contiguous(phi, uniqueExecutor.get());
    CHECK_COND(phi != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous phi failed.");

    alpha0 = l0op::Contiguous(alpha0, uniqueExecutor.get());
    CHECK_COND(alpha0 != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous alpha0 failed.");

    bias = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_COND(bias != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous bias failed.");

    gamma0Optional = l0op::Contiguous(gamma0Optional, uniqueExecutor.get());
    CHECK_COND(gamma0Optional != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma0Optional failed.");

    gamma1 = l0op::Contiguous(gamma1, uniqueExecutor.get());
    CHECK_COND(gamma1 != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma1 failed.");

    if (gamma2Optional != nullptr) {
        gamma2Optional = l0op::Contiguous(gamma2Optional, uniqueExecutor.get());
        CHECK_COND(gamma2Optional != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma2 failed.");
    }

    bool ok = MhcSandwichNormPostPreonlyl0op::MhcSandwichNormPostPreonly(
        hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma0Optional, gamma1, gamma2Optional, normEps,
        mhcPreNormEps, hcEps, returnHInF32, hInPrimeOut, x2Out, hInF32OutOptional, uniqueExecutor.get());

    CHECK_RET(ok, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnMhcSandwichNormPostPreonlyGetWorkspaceSize(
    const aclTensor *hOut, const aclTensor *residual, const aclTensor *hPost, const aclTensor *hResOptional,
    const aclTensor *phi, const aclTensor *alpha0, const aclTensor *bias, const aclTensor *gamma1,
    const aclTensor *gamma0Optional, const aclTensor *gamma2Optional, float normEps, float mhcPreNormEps, float hcEps,
    bool returnHInF32, const aclTensor *hInPrimeOut, const aclTensor *x2Out, const aclTensor *hInF32OutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnMhcSandwichNormPostPreonly,
        DFX_IN(hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma1, gamma0Optional, gamma2Optional),
        DFX_OUT(hInPrimeOut, x2Out, hInF32OutOptional));
    return MhcSandwichNormPostPreonlyCommonProcess(
        hOut, residual, hPost, hResOptional, phi, alpha0, bias, gamma0Optional, gamma1, gamma2Optional, normEps,
        mhcPreNormEps, hcEps, returnHInF32, hInPrimeOut, x2Out, hInF32OutOptional, workspaceSize, executor);
}

aclnnStatus aclnnMhcSandwichNormPostPreonly(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcSandwichNormPostPreonly);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
