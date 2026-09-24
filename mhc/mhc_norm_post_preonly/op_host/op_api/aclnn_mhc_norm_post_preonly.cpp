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
#include "mhc_norm_post_preonly.h"
#include "aclnn_mhc_norm_post_preonly.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr int64_t SUPPORTED_D_2560 = 2560;
constexpr int64_t SUPPORTED_D_5120 = 5120;
constexpr int64_t SUPPORTED_D_6144 = 6144;
constexpr int64_t SUPPORTED_D_7168 = 7168;

static bool IsSupportedD(int64_t d)
{
    return d == SUPPORTED_D_2560 || d == SUPPORTED_D_5120 || d == SUPPORTED_D_6144 || d == SUPPORTED_D_7168;
}

static aclnnStatus MhcNormPostPreonlyCommonProcess(const aclTensor *hOut, const aclTensor *residual,
                                                   const aclTensor *hPost, const aclTensor *hResOptional,
                                                   const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                                                   const aclTensor *preMixOptional, const aclTensor *gamma0Optional,
                                                   const aclTensor *gamma1, const aclTensor *gamma2Optional,
                                                   float normEps, float mhcPreNormEps, float hcEps, bool returnHInF32,
                                                   const aclTensor *hInPrimeOut, const aclTensor *x2Out,
                                                   const aclTensor *hInF32OutOptional, const aclTensor *hPreOptional,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    // pre_mix、h_pre 为接口预留的可选输入/输出，当前版本暂不支持，先在此拦截。
    CHECK_COND(preMixOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "preMixOptional is an interface-reserved optional input and must be nullptr currently.");
    CHECK_COND(hPreOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
               "hPreOptional is an interface-reserved optional output and must be nullptr currently.");

    if (returnHInF32) {
        CHECK_COND(hInF32OutOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "hInF32OutOptional must not be nullptr when returnHInF32 is true.");
    } else {
        CHECK_COND(hInF32OutOptional == nullptr, ACLNN_ERR_PARAM_INVALID,
                   "hInF32OutOptional must be nullptr when returnHInF32 is false.");
    }

    hOut = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_COND(hOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hOut failed.");
    const auto &hOutShape = hOut->GetViewShape();
    const auto hOutDimNum = hOutShape.GetDimNum();
    CHECK_COND(hOutDimNum == 2U || hOutDimNum == 3U, ACLNN_ERR_PARAM_INVALID,
               "hOut must be 2D [T, D] or 3D [B, S, D].");
    const int64_t d = hOutShape.GetDim(hOutDimNum - 1U);
    CHECK_COND(IsSupportedD(d), ACLNN_ERR_PARAM_INVALID,
               "The last dimension D of hOut must be one of 2560, 5120, 6144, or 7168, but got %ld.", d);
    CHECK_COND(hOut->GetDataType() == DataType::DT_FLOAT16 || hOut->GetDataType() == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "hOut dtype must be FLOAT16 or BFLOAT16.");

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

    alpha = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_COND(alpha != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous alpha failed.");

    bias = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_COND(bias != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous bias failed.");

    CHECK_COND(gamma0Optional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "gamma0Optional is an interface-reserved optional input and must not be nullptr currently.");
    gamma0Optional = l0op::Contiguous(gamma0Optional, uniqueExecutor.get());
    CHECK_COND(gamma0Optional != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma0Optional failed.");

    gamma1 = l0op::Contiguous(gamma1, uniqueExecutor.get());
    CHECK_COND(gamma1 != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma1 failed.");

    if (gamma2Optional != nullptr) {
        gamma2Optional = l0op::Contiguous(gamma2Optional, uniqueExecutor.get());
        CHECK_COND(gamma2Optional != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous gamma2 failed.");
    }

    bool isLaunchPrepared = MhcNormPostPreonlyl0op::MhcNormPostPreonly(
        hOut, residual, hPost, hResOptional, phi, alpha, bias, preMixOptional, gamma0Optional, gamma1, gamma2Optional,
        normEps, mhcPreNormEps, hcEps, returnHInF32, hInPrimeOut, x2Out, hInF32OutOptional, hPreOptional,
        uniqueExecutor.get());

    CHECK_RET(isLaunchPrepared, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnMhcNormPostPreonlyGetWorkspaceSize(const aclTensor *hOut, const aclTensor *residual,
                                                    const aclTensor *hPost, const aclTensor *hResOptional,
                                                    const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                                                    const aclTensor *gamma1, const aclTensor *preMixOptional,
                                                    const aclTensor *gamma0Optional, const aclTensor *gamma2Optional,
                                                    float normEps, float mhcPreNormEps, float hcEps, bool returnHInF32,
                                                    const aclTensor *hInPrimeOut, const aclTensor *x2Out,
                                                    const aclTensor *hInF32OutOptional, const aclTensor *hPreOptional,
                                                    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcNormPostPreonly,
                   DFX_IN(hOut, residual, hPost, hResOptional, phi, alpha, bias, gamma1, preMixOptional, gamma0Optional,
                          gamma2Optional),
                   DFX_OUT(hInPrimeOut, x2Out, hInF32OutOptional, hPreOptional));
    return MhcNormPostPreonlyCommonProcess(hOut, residual, hPost, hResOptional, phi, alpha, bias, preMixOptional,
                                           gamma0Optional, gamma1, gamma2Optional, normEps, mhcPreNormEps, hcEps,
                                           returnHInF32, hInPrimeOut, x2Out, hInF32OutOptional, hPreOptional,
                                           workspaceSize, executor);
}

aclnnStatus aclnnMhcNormPostPreonly(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcNormPostPreonly);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
