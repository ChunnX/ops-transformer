/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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

namespace mhc_pre_grad {

struct MhcPreBackwardParamsBase {
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *h_in_grad = nullptr;
    const aclTensor *h_post_grad = nullptr;
    const aclTensor *h_res_grad = nullptr;
    const aclTensor *inv_rms = nullptr;
    const aclTensor *mm_res = nullptr;
    const aclTensor *h_pre = nullptr;
    const aclTensor *h_post = nullptr;
    const aclTensor *gamma = nullptr;
    double hc_eps;

    const aclTensor *x_grad = nullptr;
    const aclTensor *hc_weight_grad = nullptr;
    const aclTensor *alpha_grad = nullptr;
    const aclTensor *bias_post_grad = nullptr;
    const aclTensor *gamma_grad = nullptr;
    
    // 用于存储转换后的连续tensor（在CovertDataContiguous中使用）
    const aclTensor *x_contiguous = nullptr;
    const aclTensor *phi_contiguous = nullptr;
    const aclTensor *alpha_contiguous = nullptr;
    const aclTensor *h_in_grad_contiguous = nullptr;
    const aclTensor *h_post_grad_contiguous = nullptr;
    const aclTensor *h_res_grad_contiguous = nullptr;
    const aclTensor *inv_rms_contiguous = nullptr;
    const aclTensor *mm_res_contiguous = nullptr;
    const aclTensor *h_pre_contiguous = nullptr;
    const aclTensor *h_post_contiguous = nullptr;
    const aclTensor *gamma_contiguous = nullptr;
};

class MhcPreBackwardBuilder {
public:
    static MhcPreBackwardBuilder Create()
    {
        MhcPreBackwardBuilder obj;

        return obj;
    } 

    MhcPreBackwardBuilder &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *gamma)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.gamma = gamma;

        return *this;
    }

    MhcPreBackwardBuilder &SetForwardInput(const aclTensor *inv_rms, const aclTensor *mm_res, const aclTensor *h_pre, const aclTensor *h_post)
    {
        obj_.inv_rms = inv_rms;
        obj_.mm_res = mm_res;
        obj_.h_pre = h_pre;
        obj_.h_post = h_post;

        return *this;
    }

    MhcPreBackwardBuilder &SetGradInput(
        const aclTensor *h_in_grad, const aclTensor *h_post_grad, const aclTensor *h_res_grad)
    {
        obj_.h_in_grad = h_in_grad;
        obj_.h_post_grad = h_post_grad;
        obj_.h_res_grad = h_res_grad;
        return *this;
    }

    MhcPreBackwardBuilder &SetAttr(double hc_eps)
    {
        obj_.hc_eps = hc_eps;

        return *this;
    }

    MhcPreBackwardBuilder &SetOutput(
        const aclTensor *x_grad, const aclTensor *hc_weight_grad, const aclTensor *alpha_grad, const aclTensor *bias_post_grad, const aclTensor *gamma_grad)
    {
        obj_.x_grad = x_grad;
        obj_.hc_weight_grad = hc_weight_grad;
        obj_.alpha_grad = alpha_grad;
        obj_.bias_post_grad = bias_post_grad;
        obj_.gamma_grad = gamma_grad;
        return *this;
    }

    MhcPreBackwardParamsBase Build() const
    {
        return obj_;
    }
private:
    MhcPreBackwardParamsBase obj_;
};

bool MhcGradCheckInputNotNull(const MhcPreBackwardParamsBase &params)
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
    if (params.h_in_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "h_in_grad tensor is nullptr");
        return false;
    }
    if (params.h_post_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "h_post_grad tensor is nullptr");
        return false;
    }
    if (params.h_res_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "h_res_grad tensor is nullptr");
        return false;
    }
    if (params.inv_rms == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "inv_rms tensor is nullptr");
        return false;
    }
    if (params.mm_res == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "mm_res tensor is nullptr");
        return false;
    }
    if (params.h_pre == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "h_pre tensor is nullptr");
        return false;
    }
    if (params.h_post == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "h_post tensor is nullptr");
        return false;
    }
    return true;
}

bool MhcGradCheckOutputNotNull(const MhcPreBackwardParamsBase &params)
{
    if (params.x_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "x_grad tensor is nullptr");
        return false;
    }
    if (params.hc_weight_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "hc_weight_grad tensor is nullptr");
        return false;
    }
    if (params.alpha_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "alpha_grad tensor is nullptr");
        return false;
    }
    if (params.bias_post_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "bias_post_grad tensor is nullptr");
        return false;
    }
    if (params.gamma_grad == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "gamma_grad tensor is nullptr");
        return false;
    }

    return true;
}

bool MhcGradCheckNotNull(const MhcPreBackwardParamsBase &params)
{
    return MhcGradCheckInputNotNull(params) && MhcGradCheckOutputNotNull(params);
}

bool MhcGradCheckEmptyTensor(const MhcPreBackwardParamsBase &params)
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
    if (params.h_in_grad->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_in_grad tensor is empty");
        return false;
    }
    if (params.h_post_grad->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post_grad tensor is empty");
        return false;
    }
    if (params.h_res_grad->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_res_grad tensor is empty");
        return false;
    }
    if (params.inv_rms->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms tensor is empty");
        return false;
    }
    if (params.mm_res->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mm_res tensor is empty");
        return false;
    }
    if (params.h_pre->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre tensor is empty");
        return false;
    }
    if (params.h_post->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post tensor is empty");
        return false;
    }
    return true;
}

bool MhcGradCheckInputDims(const MhcPreBackwardParamsBase &params)
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
    
    auto hInGradDimNum = params.h_in_grad->GetViewShape().GetDimNum();
    if (hInGradDimNum != DIM_NUM_2 && hInGradDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_in_grad tensor dim num must be 2 or 3, but got %zu", hInGradDimNum);
        return false;
    }
    
    auto hPostGradDimNum = params.h_post_grad->GetViewShape().GetDimNum();
    if (hPostGradDimNum != DIM_NUM_2 && hPostGradDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post_grad tensor dim num must be 2 or 3, but got %zu", hPostGradDimNum);
        return false;
    }
    
    if (hInGradDimNum != hPostGradDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_in_grad and h_post_grad must have the same dim num");
        return false;
    }
    
    auto hResGradDimNum = params.h_res_grad->GetViewShape().GetDimNum();
    if (hResGradDimNum != DIM_NUM_3 && hResGradDimNum != DIM_NUM_4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_res_grad tensor dim num must be 3 or 4, but got %zu", hResGradDimNum);
        return false;
    }
    
    auto invRmsDimNum = params.inv_rms->GetViewShape().GetDimNum();
    if (invRmsDimNum != DIM_NUM_1 && invRmsDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms tensor dim num must be 1 or 2, but got %zu", invRmsDimNum);
        return false;
    }
    
    auto mmResDimNum = params.mm_res->GetViewShape().GetDimNum();
    if (mmResDimNum != DIM_NUM_2 && mmResDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mm_res tensor dim num must be 2 or 3, but got %zu", mmResDimNum);
        return false;
    }
    
    auto hPreDimNum = params.h_pre->GetViewShape().GetDimNum();
    if (hPreDimNum != DIM_NUM_2 && hPreDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre tensor dim num must be 2 or 3, but got %zu", hPreDimNum);
        return false;
    }
    
    auto hPostDimNum = params.h_post->GetViewShape().GetDimNum();
    if (hPostDimNum != DIM_NUM_2 && hPostDimNum != DIM_NUM_3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post tensor dim num must be 2 or 3, but got %zu", hPostDimNum);
        return false;
    }
    
    return true;
}

bool MhcGradCheckOutputDims(const MhcPreBackwardParamsBase &params)
{
    constexpr size_t DIM_NUM_1 = 1UL;
    constexpr size_t DIM_NUM_2 = 2UL;
    constexpr size_t DIM_NUM_3 = 3UL;
    constexpr size_t DIM_NUM_4 = 4UL;
    
    auto hInGradDimNum = params.h_in_grad->GetViewShape().GetDimNum();
    auto xGradDimNum = params.x_grad->GetViewShape().GetDimNum();
    if (hInGradDimNum == DIM_NUM_3) {
        if (xGradDimNum != DIM_NUM_4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x_grad tensor dim num must be 4 for BSND format, but got %zu", xGradDimNum);
            return false;
        }
    } else {
        if (xGradDimNum != DIM_NUM_3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x_grad tensor dim num must be 3 for TND format, but got %zu", xGradDimNum);
            return false;
        }
    }

    auto hcWeightGradDimNum = params.hc_weight_grad->GetViewShape().GetDimNum();
    if (hcWeightGradDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hc_weight_grad tensor dim num must be 2, but got %zu", hcWeightGradDimNum);
        return false;
    }

    auto alphaGradDimNum = params.alpha_grad->GetViewShape().GetDimNum();
    if (alphaGradDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha_grad tensor dim num must be 1, but got %zu", alphaGradDimNum);
        return false;
    }

    auto biasPostGradDimNum = params.bias_post_grad->GetViewShape().GetDimNum();
    if (biasPostGradDimNum != DIM_NUM_1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias_post_grad tensor dim num must be 1, but got %zu", biasPostGradDimNum);
        return false;
    }

    auto gammaGradDimNum = params.gamma_grad->GetViewShape().GetDimNum();
    if (gammaGradDimNum != DIM_NUM_2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma_grad tensor dim num must be 2, but got %zu", gammaGradDimNum);
        return false;
    }
    
    return true;
}

bool MhcGradCheckInputOutDims(const MhcPreBackwardParamsBase &params)
{
    return MhcGradCheckInputDims(params) && MhcGradCheckOutputDims(params);
}

bool MhcGradCheckBSNDShape(const MhcPreBackwardParamsBase &params,
                    uint64_t batch, uint64_t sequence, uint64_t dimen, uint64_t numsResidual)
{
    auto xShape = params.x->GetViewShape();
    auto hInGradShape = params.h_in_grad->GetViewShape();
    auto hPostGradShape = params.h_post_grad->GetViewShape();
    auto hResGradShape = params.h_res_grad->GetViewShape();
    auto invRmsShape = params.inv_rms->GetViewShape();
    auto mmResShape = params.mm_res->GetViewShape();
    auto hPreShape = params.h_pre->GetViewShape();
    auto hPostShape = params.h_post->GetViewShape();
    auto xGradShape = params.x_grad->GetViewShape();

    if (xShape.GetDim(0) != batch || xShape.GetDim(1) != sequence ||
        xShape.GetDim(2) != numsResidual || xShape.GetDim(3) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor shape must be [B, S, N, D]");
        return false;
    }

    if (hPostGradShape.GetDim(0) != batch || hPostGradShape.GetDim(1) != sequence) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post_grad shape must align with h_in_grad on B and S dims");
        return false;
    }

    if (hResGradShape.GetDim(0) != batch || hResGradShape.GetDim(1) != sequence ||
        hResGradShape.GetDim(2) != numsResidual || hResGradShape.GetDim(3) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_res_grad shape must be [B, S, N, N]");
        return false;
    }

    if (invRmsShape.GetDim(0) != batch || invRmsShape.GetDim(1) != sequence) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms shape must be [B, S]");
        return false;
    }

    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    if (mmResShape.GetDim(0) != batch || mmResShape.GetDim(1) != sequence ||
        mmResShape.GetDim(2) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mm_res shape must be [B, S, N^2+2N]");
        return false;
    }

    if (hPreShape.GetDim(0) != batch || hPreShape.GetDim(1) != sequence ||
        hPreShape.GetDim(2) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre shape must be [B, S, N]");
        return false;
    }

    if (hPostShape.GetDim(0) != batch || hPostShape.GetDim(1) != sequence ||
        hPostShape.GetDim(2) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post shape must be [B, S, N]");
        return false;
    }

    if (xGradShape.GetDim(0) != batch || xGradShape.GetDim(1) != sequence ||
        xGradShape.GetDim(2) != numsResidual || xGradShape.GetDim(3) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x_grad shape must be [B, S, N, D]");
        return false;
    }

    return true;
}

bool MhcGradCheckTNDShape(const MhcPreBackwardParamsBase &params,
                   uint64_t t, uint64_t dimen, uint64_t numsResidual)
{
    auto xShape = params.x->GetViewShape();
    auto hPostGradShape = params.h_post_grad->GetViewShape();
    auto hResGradShape = params.h_res_grad->GetViewShape();
    auto invRmsShape = params.inv_rms->GetViewShape();
    auto mmResShape = params.mm_res->GetViewShape();
    auto hPreShape = params.h_pre->GetViewShape();
    auto hPostShape = params.h_post->GetViewShape();
    auto xGradShape = params.x_grad->GetViewShape();

    if (xShape.GetDim(0) != t || xShape.GetDim(1) != numsResidual || xShape.GetDim(2) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x tensor shape must be [T, N, D]");
        return false;
    }

    if (hPostGradShape.GetDim(0) != t) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post_grad shape must align with h_in_grad on T dim");
        return false;
    }

    if (hResGradShape.GetDim(0) != t || hResGradShape.GetDim(1) != numsResidual ||
        hResGradShape.GetDim(2) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_res_grad shape must be [T, N, N]");
        return false;
    }

    if (invRmsShape.GetDim(0) != t) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms shape must be [T]");
        return false;
    }

    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    if (mmResShape.GetDim(0) != t || mmResShape.GetDim(1) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mm_res shape must be [T, N^2+2N]");
        return false;
    }

    if (hPreShape.GetDim(0) != t || hPreShape.GetDim(1) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre shape must be [T, N]");
        return false;
    }

    if (hPostShape.GetDim(0) != t || hPostShape.GetDim(1) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post shape must be [T, N]");
        return false;
    }

    if (xGradShape.GetDim(0) != t || xGradShape.GetDim(1) != numsResidual ||
        xGradShape.GetDim(2) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x_grad shape must be [T, N, D]");
        return false;
    }

    return true;
}

bool MhcGradCheckCommonShape(const MhcPreBackwardParamsBase &params,
                      uint64_t numsResidual, uint64_t dimen)
{
    auto phiShape = params.phi->GetViewShape();
    auto alphaShape = params.alpha->GetViewShape();
    auto hcWeightGradShape = params.hc_weight_grad->GetViewShape();
    auto alphaGradShape = params.alpha_grad->GetViewShape();
    auto biasPostGradShape = params.bias_post_grad->GetViewShape();
    auto gammaGradShape = params.gamma_grad->GetViewShape();

    if (alphaShape.GetDim(0) != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha tensor shape must be (3), but got (%ld)", alphaShape.GetDim(0));
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    
    if (phiShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor first dim must be n^2+2n=%ld, but got %ld", n2Plus2n, phiShape.GetDim(0));
        return false;
    }
    if (phiShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "phi tensor second dim must be nD=%ld, but got %ld", nD, phiShape.GetDim(1));
        return false;
    }
    
    if (hcWeightGradShape.GetDim(0) != n2Plus2n || hcWeightGradShape.GetDim(1) != nD) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hc_weight_grad shape must be [2N+N^2, N*D]");
        return false;
    }

    if (alphaGradShape.GetDim(0) != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha_grad shape must be [3]");
        return false;
    }

    if (biasPostGradShape.GetDim(0) != n2Plus2n) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias_post_grad shape must be [2N+N^2]");
        return false;
    }

    if (gammaGradShape.GetDim(0) != numsResidual) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma_grad shape must be [N]");
        return false;
    }

    if (gammaGradShape.GetDim(1) != dimen) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma_grad shape must be [D]");
        return false;
    }
    
    return true;
}

bool MhcGradCheckInputOutShape(const MhcPreBackwardParamsBase &params)
{
    auto hInGradShape = params.h_in_grad->GetViewShape();
    auto hPostGradShape = params.h_post_grad->GetViewShape();
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

bool MhcGradCheckInputDtype(const MhcPreBackwardParamsBase &params)
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
    
    if (!MhcGradIsValidXType(params.h_in_grad->GetDataType())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_in_grad tensor dtype must be BF16 or FP16");
        return false;
    }
    
    if (params.h_post_grad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post_grad tensor dtype must be FP32");
        return false;
    }
    if (params.h_res_grad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_res_grad tensor dtype must be FP32");
        return false;
    }
    if (params.inv_rms->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms tensor dtype must be FP32");
        return false;
    }
    if (params.mm_res->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mm_res tensor dtype must be FP32");
        return false;
    }
    if (params.h_pre->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre tensor dtype must be FP32");
        return false;
    }
    if (params.h_post->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post tensor dtype must be FP32");
        return false;
    }
    
    return true;
}

bool MhcGradCheckOutputDtype(const MhcPreBackwardParamsBase &params)
{
    auto hInGradDtype = params.h_in_grad->GetDataType();
    if (params.x_grad->GetDataType() != hInGradDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x_grad tensor dtype must match h_in_grad");
        return false;
    }
    if (params.hc_weight_grad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hc_weight_grad tensor dtype must be FP32");
        return false;
    }
    if (params.alpha_grad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "alpha_grad tensor dtype must be FP32");
        return false;
    }
    if (params.bias_post_grad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias_post_grad tensor dtype must be FP32");
        return false;
    }
    if (params.gamma_grad->GetDataType() != DataType::DT_FLOAT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gamma_grad tensor dtype must be FP32");
        return false;
    }
    
    return true;
}

bool MhcGradCheckDtypeValid(const MhcPreBackwardParamsBase &params)
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

bool MhcGradCheckInputFormat(const MhcPreBackwardParamsBase &params)
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
    if (IsPrivateFormat(params.h_in_grad->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_in_grad tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.h_post_grad->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post_grad tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.h_res_grad->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_res_grad tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.inv_rms->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inv_rms tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.mm_res->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mm_res tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.h_pre->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_pre tensor format must be ND");
        return false;
    }
    if (IsPrivateFormat(params.h_post->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "h_post tensor format must be ND");
        return false;
    }
    
    return true;
}

bool MhcGradCheckFormat(const MhcPreBackwardParamsBase &params)
{
    return MhcGradCheckInputFormat(params);
}

aclnnStatus MhcGradCheckParams(const MhcPreBackwardParamsBase &params)
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

aclnnStatus MhcGradCovertDataContiguous(MhcPreBackwardParamsBase &params, aclOpExecutor *executor)
{
    // 将输入tensor转换为连续格式
    params.x_contiguous = l0op::Contiguous(params.x, executor);
    CHECK_RET(params.x_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    params.phi_contiguous = l0op::Contiguous(params.phi, executor);
    CHECK_RET(params.phi_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    params.alpha_contiguous = l0op::Contiguous(params.alpha, executor);
    CHECK_RET(params.alpha_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (params.gamma != nullptr) {
        params.gamma_contiguous = l0op::Contiguous(params.gamma, executor);
        CHECK_RET(params.gamma_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    params.h_in_grad_contiguous = l0op::Contiguous(params.h_in_grad, executor);
    CHECK_RET(params.h_in_grad_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.h_post_grad_contiguous = l0op::Contiguous(params.h_post_grad, executor);
    CHECK_RET(params.h_post_grad_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.h_res_grad_contiguous = l0op::Contiguous(params.h_res_grad, executor);
    CHECK_RET(params.h_res_grad_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.inv_rms_contiguous = l0op::Contiguous(params.inv_rms, executor);
    CHECK_RET(params.inv_rms_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.mm_res_contiguous = l0op::Contiguous(params.mm_res, executor);
    CHECK_RET(params.mm_res_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.h_pre_contiguous = l0op::Contiguous(params.h_pre, executor);
    CHECK_RET(params.h_pre_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    params.h_post_contiguous = l0op::Contiguous(params.h_post, executor);
    CHECK_RET(params.h_post_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static aclnnStatus mhcPreBackwardCommonProcess(MhcPreBackwardParamsBase &params, aclOpExecutor *executor)
{
    auto ret = MhcGradCheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = MhcGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 使用转换后的连续tensor
    auto outParams = l0op::MhcPreBackward(
        params.x_contiguous, params.phi_contiguous, params.alpha_contiguous,
        params.h_in_grad_contiguous, params.h_post_grad_contiguous, params.h_res_grad_contiguous,
        params.inv_rms_contiguous, params.mm_res_contiguous, params.h_pre_contiguous, params.h_post_contiguous,
        params.gamma_contiguous, params.hc_eps,
        executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    auto out0 = std::get<0>(outParams);
    auto ret0 = l0op::ViewCopy(out0, params.x_grad, executor);
    CHECK_RET(ret0 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out1 = std::get<1>(outParams);
    auto ret1 = l0op::ViewCopy(out1, params.hc_weight_grad, executor);
    CHECK_RET(ret1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out2 = std::get<2>(outParams);
    auto ret2 = l0op::ViewCopy(out2, params.alpha_grad, executor);
    CHECK_RET(ret2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out3 = std::get<3>(outParams);
    auto ret3 = l0op::ViewCopy(out3, params.bias_post_grad, executor);
    CHECK_RET(ret3 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out4 = std::get<4>(outParams);
    auto ret4 = l0op::ViewCopy(out4, params.gamma_grad, executor);
    CHECK_RET(ret4 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
}

aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
    const aclTensor *h_in_grad, const aclTensor *h_post_grad, const aclTensor *h_res_grad,
    const aclTensor *inv_rms, const aclTensor *mm_res, const aclTensor *h_pre, const aclTensor *h_post, const aclTensor *gamma,
    double hc_eps,
    const aclTensor *x_grad, const aclTensor *hc_weight_grad, const aclTensor *alpha_grad, const aclTensor *bias_post_grad,
    const aclTensor *gamma_grad, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreBackward,
        DFX_IN(x, phi, alpha, h_in_grad, h_post_grad, h_res_grad, inv_rms, mm_res, h_pre, h_post, gamma),
        DFX_OUT(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad));
    auto uniqueExecutor = CREATE_EXECUTOR();

    mhc_pre_grad::MhcPreBackwardParamsBase params =
        mhc_pre_grad::MhcPreBackwardBuilder::Create()
        .SetInput(x, phi, alpha, gamma)
        .SetGradInput(h_in_grad, h_post_grad, h_res_grad)
        .SetForwardInput(inv_rms, mm_res, h_pre, h_post)
        .SetAttr(hc_eps)
        .SetOutput(x_grad, hc_weight_grad, alpha_grad, bias_post_grad, gamma_grad)
        .Build();

    auto ret = mhc_pre_grad::mhcPreBackwardCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
