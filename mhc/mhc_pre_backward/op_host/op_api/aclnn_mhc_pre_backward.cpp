/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <new>
#include <memory>
#include <unordered_map>
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mhc_pre_backward.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct AclnnMhcPreBackwardParams {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *hInGrad = nullptr;
    const aclTensor *hPostGrad = nullptr;
    const aclTensor *hResGrad = nullptr;
    const aclTensor *invRms = nullptr;
    const aclTensor *mmRes = nullptr;
    const aclTensor *hPre = nullptr;
    const aclTensor *hPost = nullptr;
    const aclTensor *gamma = nullptr;
    const aclTensor *gradXPost = nullptr;
    double hcEps;

    const aclTensor *xGrad = nullptr;
    const aclTensor *hcWeightGrad = nullptr;
    const aclTensor *alphaGrad = nullptr;
    const aclTensor *biasPostGrad = nullptr;
    const aclTensor *gammaGrad = nullptr;

    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *hInGradContiguous = nullptr;
    const aclTensor *hPostGradContiguous = nullptr;
    const aclTensor *hResGradContiguous = nullptr;
    const aclTensor *invRmsContiguous = nullptr;
    const aclTensor *mmResContiguous = nullptr;
    const aclTensor *hPreContiguous = nullptr;
    const aclTensor *hPostContiguous = nullptr;
    const aclTensor *gammaContiguous = nullptr;
    const aclTensor *gradXPostContiguous = nullptr;
};

class AclnnMhcPreBackward {
public:
    static AclnnMhcPreBackward Create()
    {
        AclnnMhcPreBackward obj;

        return obj;
    }

    AclnnMhcPreBackward &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                    const aclTensor *gamma)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.gamma = gamma;

        return *this;
    }

    AclnnMhcPreBackward &SetForwardInput(const aclTensor *invRms, const aclTensor *mmRes, const aclTensor *hPre,
                                           const aclTensor *hPost)
    {
        obj_.invRms = invRms;
        obj_.mmRes = mmRes;
        obj_.hPre = hPre;
        obj_.hPost = hPost;

        return *this;
    }

    AclnnMhcPreBackward &SetGradInput(const aclTensor *hInGrad, const aclTensor *hPostGrad,
                                        const aclTensor *hResGrad)
    {
        obj_.hInGrad = hInGrad;
        obj_.hPostGrad = hPostGrad;
        obj_.hResGrad = hResGrad;
        return *this;
    }

    AclnnMhcPreBackward &SetXPostGradInput(const aclTensor *gradXPost)
    {
        obj_.gradXPost = gradXPost;
        return *this;
    }

    AclnnMhcPreBackward &SetAttr(double hcEps)
    {
        obj_.hcEps = hcEps;

        return *this;
    }

    AclnnMhcPreBackward &SetOutput(const aclTensor *xGrad, const aclTensor *hcWeightGrad,
                                     const aclTensor *alphaGrad, const aclTensor *biasPostGrad,
                                     const aclTensor *gammaGrad)
    {
        obj_.xGrad = xGrad;
        obj_.hcWeightGrad = hcWeightGrad;
        obj_.alphaGrad = alphaGrad;
        obj_.biasPostGrad = biasPostGrad;
        obj_.gammaGrad = gammaGrad;
        return *this;
    }

    AclnnMhcPreBackwardParams Build() const
    {
        return obj_;
    }

private:
    AclnnMhcPreBackwardParams obj_;
};

bool MhcGradCheckInputNotNull(const AclnnMhcPreBackwardParams &params)
{
    if (params.x == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x tensor is nullptr");
        return false;
    }
    if (params.phi == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "phi tensor is nullptr");
        return false;
    }
    if (params.alpha == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "alpha tensor is nullptr");
        return false;
    }
    if (params.hInGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hInGrad tensor is nullptr");
        return false;
    }
    if (params.hPostGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hPostGrad tensor is nullptr");
        return false;
    }
    if (params.hResGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hResGrad tensor is nullptr");
        return false;
    }
    if (params.invRms == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "invRms tensor is nullptr");
        return false;
    }
    if (params.mmRes == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mmRes tensor is nullptr");
        return false;
    }
    if (params.hPre == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hPre tensor is nullptr");
        return false;
    }
    if (params.hPost == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hPost tensor is nullptr");
        return false;
    }
    return true;
}

bool MhcGradCheckOutputNotNull(const AclnnMhcPreBackwardParams &params)
{
    if (params.xGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "xGrad tensor is nullptr");
        return false;
    }
    if (params.hcWeightGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hcWeightGrad tensor is nullptr");
        return false;
    }
    if (params.alphaGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "alphaGrad tensor is nullptr");
        return false;
    }
    if (params.biasPostGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "biasPostGrad tensor is nullptr");
        return false;
    }
    if (params.gammaGrad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gammaGrad tensor is nullptr");
        return false;
    }

    return true;
}

bool MhcGradCheckNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNull(params) && MhcGradCheckOutputNotNull(params);
}

bool MhcGradCheckEmptyTensor(const AclnnMhcPreBackwardParams &params)
{
    if (params.x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor is empty");
        return false;
    }
    if (params.phi->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor is empty");
        return false;
    }
    if (params.alpha->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor is empty");
        return false;
    }
    if (params.hInGrad->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hInGrad tensor is empty");
        return false;
    }
    if (params.hPostGrad->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPostGrad tensor is empty");
        return false;
    }
    if (params.hResGrad->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResGrad tensor is empty");
        return false;
    }
    if (params.invRms->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms tensor is empty");
        return false;
    }
    if (params.mmRes->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmRes tensor is empty");
        return false;
    }
    if (params.hPre->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre tensor is empty");
        return false;
    }
    if (params.hPost->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost tensor is empty");
        return false;
    }
    if (params.gradXPost != nullptr && params.gradXPost->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost tensor is empty");
        return false;
    }
    return true;
}

bool MhcGradCheckInputDims(const AclnnMhcPreBackwardParams &params)
{
    constexpr size_t DIM_NUM_1 = 1UL;
    constexpr size_t DIM_NUM_2 = 2UL;
    constexpr size_t DIM_NUM_3 = 3UL;
    constexpr size_t DIM_NUM_4 = 4UL;

    auto xDimNum = params.x->GetViewShape().GetDimNum();
    if (xDimNum != DIM_NUM_3 && xDimNum != DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dim num must be 3 or 4, but got %zu", xDimNum);
        return false;
    }

    auto phiDimNum = params.phi->GetViewShape().GetDimNum();
    if (phiDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor dim num must be 2, but got %zu", phiDimNum);
        return false;
    }

    auto alphaDimNum = params.alpha->GetViewShape().GetDimNum();
    if (alphaDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor dim num must be 1, but got %zu", alphaDimNum);
        return false;
    }

    if (params.gamma != nullptr) {
        auto gammaDimNum = params.gamma->GetViewShape().GetDimNum();
        if (gammaDimNum != DIM_NUM_2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor dim num must be 2, but got %zu", gammaDimNum);
            return false;
        }
    }

    auto hInGradDimNum = params.hInGrad->GetViewShape().GetDimNum();
    if (hInGradDimNum != DIM_NUM_2 && hInGradDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hInGrad tensor dim num must be 2 or 3, but got %zu", hInGradDimNum);
        return false;
    }

    auto hPostGradDimNum = params.hPostGrad->GetViewShape().GetDimNum();
    if (hPostGradDimNum != DIM_NUM_2 && hPostGradDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPostGrad tensor dim num must be 2 or 3, but got %zu", hPostGradDimNum);
        return false;
    }

    if (hInGradDimNum != hPostGradDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hInGrad and hPostGrad must have the same dim num");
        return false;
    }

    auto hResGradDimNum = params.hResGrad->GetViewShape().GetDimNum();
    if (hResGradDimNum != DIM_NUM_3 && hResGradDimNum != DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResGrad tensor dim num must be 3 or 4, but got %zu", hResGradDimNum);
        return false;
    }

    auto invRmsDimNum = params.invRms->GetViewShape().GetDimNum();
    if (invRmsDimNum != DIM_NUM_1 && invRmsDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms tensor dim num must be 1 or 2, but got %zu", invRmsDimNum);
        return false;
    }

    auto mmResDimNum = params.mmRes->GetViewShape().GetDimNum();
    if (mmResDimNum != DIM_NUM_2 && mmResDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmRes tensor dim num must be 2 or 3, but got %zu", mmResDimNum);
        return false;
    }

    auto hPreDimNum = params.hPre->GetViewShape().GetDimNum();
    if (hPreDimNum != DIM_NUM_2 && hPreDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre tensor dim num must be 2 or 3, but got %zu", hPreDimNum);
        return false;
    }

    auto hPostDimNum = params.hPost->GetViewShape().GetDimNum();
    if (hPostDimNum != DIM_NUM_2 && hPostDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost tensor dim num must be 2 or 3, but got %zu", hPostDimNum);
        return false;
    }

    if (params.gradXPost != nullptr) {
        auto xPostGradDimNum = params.gradXPost->GetViewShape().GetDimNum();
        if (xPostGradDimNum != DIM_NUM_3 && xPostGradDimNum != DIM_NUM_4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost tensor dim num must be 3 or 4, but got %zu", xPostGradDimNum);
            return false;
        }
    }

    return true;
}

bool MhcGradCheckOutputDims(const AclnnMhcPreBackwardParams &params)
{
    constexpr size_t DIM_NUM_1 = 1UL;
    constexpr size_t DIM_NUM_2 = 2UL;
    constexpr size_t DIM_NUM_3 = 3UL;
    constexpr size_t DIM_NUM_4 = 4UL;

    auto hInGradDimNum = params.hInGrad->GetViewShape().GetDimNum();
    auto xGradDimNum = params.xGrad->GetViewShape().GetDimNum();
    if (hInGradDimNum == DIM_NUM_3) {
        if (xGradDimNum != DIM_NUM_4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad tensor dim num must be 4 for BSND format, but got %zu",
                    xGradDimNum);
            return false;
        }
    } else {
        if (xGradDimNum != DIM_NUM_3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad tensor dim num must be 3 for TND format, but got %zu",
                    xGradDimNum);
            return false;
        }
    }

    auto hcWeightGradDimNum = params.hcWeightGrad->GetViewShape().GetDimNum();
    if (hcWeightGradDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcWeightGrad tensor dim num must be 2, but got %zu", hcWeightGradDimNum);
        return false;
    }

    auto alphaGradDimNum = params.alphaGrad->GetViewShape().GetDimNum();
    if (alphaGradDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alphaGrad tensor dim num must be 1, but got %zu", alphaGradDimNum);
        return false;
    }

    auto biasPostGradDimNum = params.biasPostGrad->GetViewShape().GetDimNum();
    if (biasPostGradDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "biasPostGrad tensor dim num must be 1, but got %zu", biasPostGradDimNum);
        return false;
    }

    auto gammaGradDimNum = params.gammaGrad->GetViewShape().GetDimNum();
    if (gammaGradDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gammaGrad tensor dim num must be 2, but got %zu", gammaGradDimNum);
        return false;
    }

    return true;
}

bool MhcGradCheckInputOutDims(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputDims(params) && MhcGradCheckOutputDims(params);
}

bool MhcGradCheckBSNDShape(const AclnnMhcPreBackwardParams &params, uint64_t batch, uint64_t sequence, uint64_t dimen,
                           uint64_t numsResidual)
{
    auto xShape = params.x->GetViewShape();
    auto hInGradShape = params.hInGrad->GetViewShape();
    auto hPostGradShape = params.hPostGrad->GetViewShape();
    auto hResGradShape = params.hResGrad->GetViewShape();
    auto invRmsShape = params.invRms->GetViewShape();
    auto mmResShape = params.mmRes->GetViewShape();
    auto hPreShape = params.hPre->GetViewShape();
    auto hPostShape = params.hPost->GetViewShape();
    auto xGradShape = params.xGrad->GetViewShape();

    if (xShape.GetDim(0) != batch || xShape.GetDim(1) != sequence || xShape.GetDim(2) != numsResidual ||
        xShape.GetDim(3) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor shape must be [B, S, N, D]");
        return false;
    }

    if (hPostGradShape.GetDim(0) != batch || hPostGradShape.GetDim(1) != sequence) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPostGrad shape must align with hInGrad on B and S dims");
        return false;
    }

    if (hResGradShape.GetDim(0) != batch || hResGradShape.GetDim(1) != sequence ||
        hResGradShape.GetDim(2) != numsResidual || hResGradShape.GetDim(3) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResGrad shape must be [B, S, N, N]");
        return false;
    }

    if (invRmsShape.GetDim(0) != batch || invRmsShape.GetDim(1) != sequence) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms shape must be [B, S]");
        return false;
    }

    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    if (mmResShape.GetDim(0) != batch || mmResShape.GetDim(1) != sequence || mmResShape.GetDim(2) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmRes shape must be [B, S, N^2+2N]");
        return false;
    }

    if (hPreShape.GetDim(0) != batch || hPreShape.GetDim(1) != sequence || hPreShape.GetDim(2) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre shape must be [B, S, N]");
        return false;
    }

    if (hPostShape.GetDim(0) != batch || hPostShape.GetDim(1) != sequence || hPostShape.GetDim(2) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost shape must be [B, S, N]");
        return false;
    }

    if (xGradShape.GetDim(0) != batch || xGradShape.GetDim(1) != sequence || xGradShape.GetDim(2) != numsResidual ||
        xGradShape.GetDim(3) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad shape must be [B, S, N, D]");
        return false;
    }

    if (params.gradXPost != nullptr) {
        auto xPostGradShape = params.gradXPost->GetViewShape();
        if (xPostGradShape.GetDim(0) != batch || xPostGradShape.GetDim(1) != sequence ||
            xPostGradShape.GetDim(2) != numsResidual || xPostGradShape.GetDim(3) != dimen) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost shape must be [B, S, N, D]");
            return false;
        }
    }

    return true;
}

bool MhcGradCheckTNDShape(const AclnnMhcPreBackwardParams &params, uint64_t t, uint64_t dimen, uint64_t numsResidual)
{
    auto xShape = params.x->GetViewShape();
    auto hPostGradShape = params.hPostGrad->GetViewShape();
    auto hResGradShape = params.hResGrad->GetViewShape();
    auto invRmsShape = params.invRms->GetViewShape();
    auto mmResShape = params.mmRes->GetViewShape();
    auto hPreShape = params.hPre->GetViewShape();
    auto hPostShape = params.hPost->GetViewShape();
    auto xGradShape = params.xGrad->GetViewShape();

    if (xShape.GetDim(0) != t || xShape.GetDim(1) != numsResidual || xShape.GetDim(2) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor shape must be [T, N, D]");
        return false;
    }

    if (hPostGradShape.GetDim(0) != t) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPostGrad shape must align with hInGrad on T dim");
        return false;
    }

    if (hResGradShape.GetDim(0) != t || hResGradShape.GetDim(1) != numsResidual ||
        hResGradShape.GetDim(2) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResGrad shape must be [T, N, N]");
        return false;
    }

    if (invRmsShape.GetDim(0) != t) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms shape must be [T]");
        return false;
    }

    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    if (mmResShape.GetDim(0) != t || mmResShape.GetDim(1) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmRes shape must be [T, N^2+2N]");
        return false;
    }

    if (hPreShape.GetDim(0) != t || hPreShape.GetDim(1) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre shape must be [T, N]");
        return false;
    }

    if (hPostShape.GetDim(0) != t || hPostShape.GetDim(1) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost shape must be [T, N]");
        return false;
    }

    if (xGradShape.GetDim(0) != t || xGradShape.GetDim(1) != numsResidual || xGradShape.GetDim(2) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad shape must be [T, N, D]");
        return false;
    }

    if (params.gradXPost != nullptr) {
        auto xPostGradShape = params.gradXPost->GetViewShape();
        if (xPostGradShape.GetDim(0) != t || xPostGradShape.GetDim(1) != numsResidual ||
            xPostGradShape.GetDim(2) != dimen) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost shape must be [T, N, D]");
            return false;
        }
    }

    return true;
}

bool MhcGradCheckCommonShape(const AclnnMhcPreBackwardParams &params, uint64_t numsResidual, uint64_t dimen)
{
    auto phiShape = params.phi->GetViewShape();
    auto alphaShape = params.alpha->GetViewShape();
    auto hcWeightGradShape = params.hcWeightGrad->GetViewShape();
    auto alphaGradShape = params.alphaGrad->GetViewShape();
    auto biasPostGradShape = params.biasPostGrad->GetViewShape();
    auto gammaGradShape = params.gammaGrad->GetViewShape();

    if (alphaShape.GetDim(0) != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor shape must be (3), but got (%ld)", alphaShape.GetDim(0));
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;

    if (phiShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor first dim must be n^2+2n=%ld, but got %ld", n2Plus2n,
                phiShape.GetDim(0));
        return false;
    }
    if (phiShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor second dim must be nD=%ld, but got %ld", nD, phiShape.GetDim(1));
        return false;
    }

    if (hcWeightGradShape.GetDim(0) != n2Plus2n || hcWeightGradShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcWeightGrad shape must be [2N+N^2, N*D]");
        return false;
    }

    if (alphaGradShape.GetDim(0) != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alphaGrad shape must be [3]");
        return false;
    }

    if (biasPostGradShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "biasPostGrad shape must be [2N+N^2]");
        return false;
    }

    if (gammaGradShape.GetDim(0) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gammaGrad shape must be [N]");
        return false;
    }

    if (gammaGradShape.GetDim(1) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gammaGrad shape must be [D]");
        return false;
    }

    return true;
}

bool MhcGradCheckInputOutShape(const AclnnMhcPreBackwardParams &params)
{
    auto hInGradShape = params.hInGrad->GetViewShape();
    auto hPostGradShape = params.hPostGrad->GetViewShape();
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    auto hInGradDimNum = hInGradShape.GetDimNum();

    uint64_t batch = 0;
    uint64_t sequence = 0;
    uint64_t t = 0;
    uint64_t dimen = 0;
    uint64_t numsResidual = 0;

    if (hInGradDimNum == 3) {
        batch = hInGradShape.GetDim(0);
        sequence = hInGradShape.GetDim(1);
        dimen = hInGradShape.GetDim(2);
        numsResidual = hPostGradShape.GetDim(2);

        if (xDimNum != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dim num must be 4 for BSND format, but got %zu", xDimNum);
            return false;
        }

        if (!MhcGradCheckBSNDShape(params, batch, sequence, dimen, numsResidual)) {
            return false;
        }
    } else if (hInGradDimNum == 2) {
        t = hInGradShape.GetDim(0);
        dimen = hInGradShape.GetDim(1);
        numsResidual = hPostGradShape.GetDim(1);

        if (xDimNum != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dim num must be 3 for TND format, but got %zu", xDimNum);
            return false;
        }

        if (!MhcGradCheckTNDShape(params, t, dimen, numsResidual)) {
            return false;
        }
    } else {
        return false;
    }

    return MhcGradCheckCommonShape(params, numsResidual, dimen);
}

bool MhcGradIsValidXType(DataType dtype)
{
    return dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16;
}

bool MhcGradCheckInputDtype(const AclnnMhcPreBackwardParams &params)
{
    if (!MhcGradIsValidXType(params.x->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor dtype must be BF16 or FP16");
        return false;
    }

    if (params.phi->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor dtype must be FP32");
        return false;
    }

    if (params.alpha->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor dtype must be FP32");
        return false;
    }

    if (params.gamma != nullptr) {
        if (params.gamma->GetDataType() != DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor dtype must be FP32");
            return false;
        }
    }

    if (!MhcGradIsValidXType(params.hInGrad->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hInGrad tensor dtype must be BF16 or FP16");
        return false;
    }

    if (params.hPostGrad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPostGrad tensor dtype must be FP32");
        return false;
    }
    if (params.hResGrad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResGrad tensor dtype must be FP32");
        return false;
    }
    if (params.invRms->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms tensor dtype must be FP32");
        return false;
    }
    if (params.mmRes->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmRes tensor dtype must be FP32");
        return false;
    }
    if (params.hPre->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre tensor dtype must be FP32");
        return false;
    }
    if (params.hPost->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost tensor dtype must be FP32");
        return false;
    }

    if (params.gradXPost != nullptr) {
        if (!MhcGradIsValidXType(params.gradXPost->GetDataType())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost tensor dtype must be BF16 or FP16");
            return false;
        }
        if (params.gradXPost->GetDataType() != params.x->GetDataType()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost tensor dtype must be the same as x");
            return false;
        }
    }

    return true;
}

bool MhcGradCheckOutputDtype(const AclnnMhcPreBackwardParams &params)
{
    auto hInGradDtype = params.hInGrad->GetDataType();
    if (params.xGrad->GetDataType() != hInGradDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad tensor dtype must match hInGrad");
        return false;
    }
    if (params.hcWeightGrad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hcWeightGrad tensor dtype must be FP32");
        return false;
    }
    if (params.alphaGrad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alphaGrad tensor dtype must be FP32");
        return false;
    }
    if (params.biasPostGrad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "biasPostGrad tensor dtype must be FP32");
        return false;
    }
    if (params.gammaGrad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gammaGrad tensor dtype must be FP32");
        return false;
    }

    return true;
}

bool MhcGradCheckDtypeValid(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputDtype(params) && MhcGradCheckOutputDtype(params);
}

static bool MhcGradIsPrivateFormat(ge::Format format)
{
    if (format == ge::FORMAT_NC1HWC0 || format == ge::FORMAT_FRACTAL_Z || format == ge::FORMAT_NDC1HWC0 ||
        format == ge::FORMAT_FRACTAL_Z_3D || format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_NC1HWC0_C04) {
        return true;
    }

    return false;
}

bool MhcGradCheckInputFormat(const AclnnMhcPreBackwardParams &params)
{
    if (IsPrivateFormat(params.x->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.phi->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.alpha->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor format must be ND");
        return false;
    }
    if (params.gamma != nullptr) {
        if (IsPrivateFormat(params.gamma->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma tensor format must be ND");
            return false;
        }
    }
    if (IsPrivateFormat(params.hInGrad->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hInGrad tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.hPostGrad->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPostGrad tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.hResGrad->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResGrad tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.invRms->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "invRms tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.mmRes->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mmRes tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.hPre->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPre tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.hPost->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hPost tensor format must be ND");
        return false;
    }
    if (params.gradXPost != nullptr) {
        if (IsPrivateFormat(params.gradXPost->GetViewFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXPost tensor format must be ND");
            return false;
        }
    }

    return true;
}

bool MhcGradCheckFormat(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputFormat(params);
}

aclnnStatus MhcGradCheckParams(const AclnnMhcPreBackwardParams &params)
{
    // 1. 检查参数是否为空指针、空tensor
    CHECK_RET(MhcGradCheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(MhcGradCheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);

    // 2. 校验输入、输出参数维度
    CHECK_RET(MhcGradCheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验输入、输出shape参数
    CHECK_RET(MhcGradCheckInputOutShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(MhcGradCheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查数据形状是否支持
    CHECK_RET(MhcGradCheckFormat(params), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus MhcGradCovertDataContiguous(AclnnMhcPreBackwardParams &params, aclOpExecutor *executor)
{
    // 将输入tensor转换为连续格式
    params.xContiguous = l0op::Contiguous(params.x, executor);
    CHECK_RET(params.xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.phiContiguous = l0op::Contiguous(params.phi, executor);
    CHECK_RET(params.phiContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.alphaContiguous = l0op::Contiguous(params.alpha, executor);
    CHECK_RET(params.alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gamma != nullptr) {
        params.gammaContiguous = l0op::Contiguous(params.gamma, executor);
        CHECK_RET(params.gammaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    params.hInGradContiguous = l0op::Contiguous(params.hInGrad, executor);
    CHECK_RET(params.hInGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.hPostGradContiguous = l0op::Contiguous(params.hPostGrad, executor);
    CHECK_RET(params.hPostGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.hResGradContiguous = l0op::Contiguous(params.hResGrad, executor);
    CHECK_RET(params.hResGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.invRmsContiguous = l0op::Contiguous(params.invRms, executor);
    CHECK_RET(params.invRmsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.mmResContiguous = l0op::Contiguous(params.mmRes, executor);
    CHECK_RET(params.mmResContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.hPreContiguous = l0op::Contiguous(params.hPre, executor);
    CHECK_RET(params.hPreContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.hPostContiguous = l0op::Contiguous(params.hPost, executor);
    CHECK_RET(params.hPostContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gradXPost != nullptr) {
        params.gradXPostContiguous = l0op::Contiguous(params.gradXPost, executor);
        CHECK_RET(params.gradXPostContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus mhcPreBackwardCommonProcess(AclnnMhcPreBackwardParams &params, aclOpExecutor *executor)
{
    auto ret = MhcGradCheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = MhcGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 使用转换后的连续tensor
    auto outParams = l0op::MhcPreBackward(
        params.xContiguous, params.phiContiguous, params.alphaContiguous, params.hInGradContiguous,
        params.hPostGradContiguous, params.hResGradContiguous, params.invRmsContiguous,
        params.mmResContiguous, params.hPreContiguous, params.hPostContiguous, params.gammaContiguous,
        params.gradXPostContiguous, params.hcEps, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    auto out0 = std::get<0>(outParams);
    auto ret0 = l0op::ViewCopy(out0, params.xGrad, executor);
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out1 = std::get<1>(outParams);
    auto ret1 = l0op::ViewCopy(out1, params.hcWeightGrad, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out2 = std::get<2>(outParams);
    auto ret2 = l0op::ViewCopy(out2, params.alphaGrad, executor);
    CHECK_RET(ret2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out3 = std::get<3>(outParams);
    auto ret3 = l0op::ViewCopy(out3, params.biasPostGrad, executor);
    CHECK_RET(ret3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out4 = std::get<4>(outParams);
    auto ret4 = l0op::ViewCopy(out4, params.gammaGrad, executor);
    CHECK_RET(ret4 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *hInGrad,
    const aclTensor *hPostGrad, const aclTensor *hResGrad, const aclTensor *invRms, const aclTensor *mmRes,
    const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma, const aclTensor *gradXPost,
    double hcEps, const aclTensor *xGrad, const aclTensor *hcWeightGrad, const aclTensor *alphaGrad,
    const aclTensor *biasPostGrad, const aclTensor *gammaGrad, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnMhcPreBackward,
        DFX_IN(x, phi, alpha, hInGrad, hPostGrad, hResGrad, invRms, mmRes, hPre, hPost, gamma, gradXPost),
        DFX_OUT(xGrad, hcWeightGrad, alphaGrad, biasPostGrad, gammaGrad));
    auto uniqueExecutor = CREATE_EXECUTOR();

    AclnnMhcPreBackwardParams params = AclnnMhcPreBackward::Create()
                                          .SetInput(x, phi, alpha, gamma)
                                          .SetGradInput(hInGrad, hPostGrad, hResGrad)
                                          .SetXPostGradInput(gradXPost)
                                          .SetForwardInput(invRms, mmRes, hPre, hPost)
                                          .SetAttr(hcEps)
                                          .SetOutput(xGrad, hcWeightGrad, alphaGrad, biasPostGrad, gammaGrad)
                                          .Build();

    auto ret = mhcPreBackwardCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
