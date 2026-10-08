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

#include <cstddef>
#include <initializer_list>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

constexpr int64_t SUPPORTED_D_2560 = 2560;
constexpr int64_t SUPPORTED_D_5120 = 5120;
constexpr int64_t SUPPORTED_D_6144 = 6144;
constexpr int64_t SUPPORTED_D_7168 = 7168;
constexpr int64_t SUPPORTED_N = 4;
constexpr int64_t MAX_TOKENS = 1024 * 1024;

static bool CheckSupportedD(int64_t d)
{
    return d == SUPPORTED_D_2560 || d == SUPPORTED_D_5120 || d == SUPPORTED_D_6144 || d == SUPPORTED_D_7168;
}

static bool CheckShape(const aclTensor *tensor, std::initializer_list<int64_t> expected)
{
    if (tensor == nullptr || tensor->GetViewShape().GetDimNum() != expected.size()) {
        return false;
    }
    size_t i = 0;
    for (int64_t dim : expected) {
        if (tensor->GetViewShape().GetDim(i++) != dim) {
            return false;
        }
    }
    return true;
}

struct CheckContext {
    const aclTensor *hOut;
    const aclTensor *residual;
    const aclTensor *hPost;
    const aclTensor *hRes;
    const aclTensor *phi;
    const aclTensor *alpha;
    const aclTensor *bias;
    const aclTensor *preMix;
    const aclTensor *gamma0;
    const aclTensor *gamma1;
    const aclTensor *gamma2;
    const aclTensor *hInPrime;
    const aclTensor *x2;
    const aclTensor *hInF32;
    const aclTensor *hPre;
    float normEps;
    float mhcPreNormEps;
    float hcEps;
    bool returnHInF32;
};

static aclnnStatus CheckRequiredParams(const CheckContext &p, const uint64_t *workspaceSize,
                                       aclOpExecutor *const *executor)
{
    CHECK_COND(p.hOut != nullptr && p.residual != nullptr && p.hPost != nullptr && p.phi != nullptr &&
                   p.alpha != nullptr && p.bias != nullptr && p.gamma1 != nullptr && p.hInPrime != nullptr &&
                   p.x2 != nullptr && workspaceSize != nullptr && executor != nullptr,
               ACLNN_ERR_PARAM_NULLPTR, "Required tensor or output pointer must not be nullptr.");
    CHECK_COND(p.hRes != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "hResOptional is an interface-reserved optional input and must not be nullptr currently.");
    CHECK_COND(p.gamma0 != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "gamma0Optional is an interface-reserved optional input and must not be nullptr currently.");
    CHECK_COND(p.preMix == nullptr, ACLNN_ERR_PARAM_INVALID,
               "preMixOptional is an interface-reserved optional input and must be nullptr currently.");
    CHECK_COND(p.hPre == nullptr, ACLNN_ERR_PARAM_INVALID,
               "hPreOptional is an interface-reserved optional output and must be nullptr currently.");
    CHECK_COND(p.returnHInF32 ? p.hInF32 != nullptr : p.hInF32 == nullptr,
               p.returnHInF32 ? ACLNN_ERR_PARAM_NULLPTR : ACLNN_ERR_PARAM_INVALID,
               "hInF32OutOptional must be present exactly when returnHInF32 is true.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttrs(const CheckContext &p)
{
    CHECK_COND(p.normEps >= 0.0F && p.mhcPreNormEps >= 0.0F && p.hcEps >= 0.0F,
               ACLNN_ERR_PARAM_INVALID, "normEps, mhcPreNormEps and hcEps must be non-negative.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputDtype(const CheckContext &p)
{
    const auto inputDtype = p.hOut->GetDataType();
    CHECK_COND(inputDtype == DataType::DT_FLOAT16 || inputDtype == DataType::DT_BF16,
               ACLNN_ERR_PARAM_INVALID, "hOut dtype must be FLOAT16 or BFLOAT16.");
    CHECK_COND(p.residual->GetDataType() == inputDtype, ACLNN_ERR_PARAM_INVALID,
               "hOut and residual must have the same dtype.");
    const auto fp32 = DataType::DT_FLOAT;
    CHECK_COND(p.hPost->GetDataType() == fp32 && p.hRes->GetDataType() == fp32 &&
                   p.phi->GetDataType() == fp32 && p.alpha->GetDataType() == fp32 &&
                   p.bias->GetDataType() == fp32 && p.gamma0->GetDataType() == fp32 &&
                   p.gamma1->GetDataType() == fp32 &&
                   (p.gamma2 == nullptr || p.gamma2->GetDataType() == fp32),
               ACLNN_ERR_PARAM_INVALID, "hPost, hRes, phi, alpha, bias and gamma inputs must be FLOAT32.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutputDtype(const CheckContext &p)
{
    CHECK_COND(p.hInPrime->GetDataType() == p.hOut->GetDataType() &&
                   p.x2->GetDataType() == p.hOut->GetDataType() &&
                   (p.hInF32 == nullptr || p.hInF32->GetDataType() == DataType::DT_FLOAT),
               ACLNN_ERR_PARAM_INVALID, "Output dtypes must match hOut; hInF32OutOptional must be FLOAT32.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputShape(const CheckContext &p)
{
    const auto &hOutShape = p.hOut->GetViewShape();
    const auto rank = hOutShape.GetDimNum();
    CHECK_COND(rank == 2U || rank == 3U, ACLNN_ERR_PARAM_INVALID,
               "hOut must be 2D [T, D] or 3D [B, S, D].");
    const int64_t d = hOutShape.GetDim(rank - 1U);
    CHECK_COND(CheckSupportedD(d), ACLNN_ERR_PARAM_INVALID,
               "The last dimension D of hOut must be one of 2560, 5120, 6144, or 7168, but got %ld.", d);
    const int64_t tokenDim0 = hOutShape.GetDim(0);
    const int64_t tokenDim1 = rank == 3U ? hOutShape.GetDim(1) : 1;
    CHECK_COND(tokenDim0 > 0 && tokenDim1 > 0 && tokenDim0 <= MAX_TOKENS &&
                   tokenDim1 <= MAX_TOKENS / tokenDim0,
               ACLNN_ERR_PARAM_INVALID, "B*S must be in [1, %ld].", MAX_TOKENS);
    const bool is3D = rank == 3U;
    CHECK_COND(is3D ? CheckShape(p.residual, {tokenDim0, tokenDim1, SUPPORTED_N, d})
                    : CheckShape(p.residual, {tokenDim0, SUPPORTED_N, d}),
               ACLNN_ERR_PARAM_INVALID, "residual shape must match hOut with an additional N=4 dimension.");
    CHECK_COND(is3D ? CheckShape(p.hPost, {tokenDim0, tokenDim1, SUPPORTED_N})
                    : CheckShape(p.hPost, {tokenDim0, SUPPORTED_N}),
               ACLNN_ERR_PARAM_INVALID, "hPost shape must be [T,4] or [B,S,4].");
    CHECK_COND(is3D ? CheckShape(p.hRes, {tokenDim0, tokenDim1, SUPPORTED_N, SUPPORTED_N})
                    : CheckShape(p.hRes, {tokenDim0, SUPPORTED_N, SUPPORTED_N}),
               ACLNN_ERR_PARAM_INVALID, "hResOptional shape must be [T,4,4] or [B,S,4,4].");
    CHECK_COND(CheckShape(p.phi, {SUPPORTED_N, SUPPORTED_N * d}) && CheckShape(p.alpha, {1}) &&
                   CheckShape(p.bias, {SUPPORTED_N}) && CheckShape(p.gamma0, {d}) && CheckShape(p.gamma1, {d}),
               ACLNN_ERR_PARAM_INVALID, "phi, alpha, bias, gamma0 and gamma1 shapes must match N=4 and D.");
    CHECK_COND(p.gamma2 == nullptr || CheckShape(p.gamma2, {SUPPORTED_N * d}),
               ACLNN_ERR_PARAM_INVALID, "gamma2Optional shape must be [4*D] when present.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckOutputShape(const CheckContext &p)
{
    const auto &hOutShape = p.hOut->GetViewShape();
    const bool is3D = hOutShape.GetDimNum() == 3U;
    const int64_t tokenDim0 = hOutShape.GetDim(0);
    const int64_t tokenDim1 = is3D ? hOutShape.GetDim(1) : 1;
    const int64_t d = hOutShape.GetDim(hOutShape.GetDimNum() - 1U);
    CHECK_COND(is3D ? CheckShape(p.hInPrime, {tokenDim0, tokenDim1, d})
                    : CheckShape(p.hInPrime, {tokenDim0, d}),
               ACLNN_ERR_PARAM_INVALID, "hInPrimeOut shape must match hOut.");
    CHECK_COND(is3D ? CheckShape(p.x2, {tokenDim0, tokenDim1, SUPPORTED_N, d})
                    : CheckShape(p.x2, {tokenDim0, SUPPORTED_N, d}),
               ACLNN_ERR_PARAM_INVALID, "x2Out shape must match residual.");
    CHECK_COND(p.hInF32 == nullptr ||
                   (is3D ? CheckShape(p.hInF32, {tokenDim0, tokenDim1, d})
                         : CheckShape(p.hInF32, {tokenDim0, d})),
               ACLNN_ERR_PARAM_INVALID, "hInF32OutOptional shape must match hOut when present.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const CheckContext &p, const uint64_t *workspaceSize,
                               aclOpExecutor *const *executor)
{
    aclnnStatus status = CheckRequiredParams(p, workspaceSize, executor);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckAttrs(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckInputDtype(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckInputShape(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckOutputDtype(p);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckOutputShape(p);
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
    const CheckContext params = {hOut, residual, hPost, hResOptional, phi, alpha, bias, preMixOptional, gamma0Optional,
                                 gamma1, gamma2Optional, hInPrimeOut, x2Out, hInF32OutOptional, hPreOptional,
                                 normEps, mhcPreNormEps, hcEps, returnHInF32};
    const aclnnStatus checkStatus = CheckParams(params, workspaceSize, executor);
    if (checkStatus != ACLNN_SUCCESS) {
        return checkStatus;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();

    hOut = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_COND(hOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hOut failed.");
    residual = l0op::Contiguous(residual, uniqueExecutor.get());
    CHECK_COND(residual != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous residual failed.");

    hPost = l0op::Contiguous(hPost, uniqueExecutor.get());
    CHECK_COND(hPost != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hPost failed.");

    hResOptional = l0op::Contiguous(hResOptional, uniqueExecutor.get());
    CHECK_COND(hResOptional != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous hResOptional failed.");

    phi = l0op::Contiguous(phi, uniqueExecutor.get());
    CHECK_COND(phi != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous phi failed.");

    alpha = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_COND(alpha != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous alpha failed.");

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
