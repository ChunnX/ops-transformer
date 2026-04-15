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

    AclnnMhcPreBackward &SetGradInput(const aclTensor *hInGrad, const aclTensor *hPostGrad, const aclTensor *hResGrad)
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

    AclnnMhcPreBackward &SetOutput(const aclTensor *xGrad, const aclTensor *hcWeightGrad, const aclTensor *alphaGrad,
                                   const aclTensor *biasPostGrad, const aclTensor *gammaGrad)
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

bool MhcGradCheckInputNotNullImpl(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensor is nullptr", name);
        return false;
    }
    return true;
}

bool MhcGradCheckInputNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNullImpl(params.x, "x") && MhcGradCheckInputNotNullImpl(params.phi, "phi") &&
           MhcGradCheckInputNotNullImpl(params.alpha, "alpha") &&
           MhcGradCheckInputNotNullImpl(params.hInGrad, "hInGrad") &&
           MhcGradCheckInputNotNullImpl(params.hPostGrad, "hPostGrad") &&
           MhcGradCheckInputNotNullImpl(params.hResGrad, "hResGrad") &&
           MhcGradCheckInputNotNullImpl(params.invRms, "invRms") &&
           MhcGradCheckInputNotNullImpl(params.mmRes, "mmRes") && MhcGradCheckInputNotNullImpl(params.hPre, "hPre") &&
           MhcGradCheckInputNotNullImpl(params.hPost, "hPost");
}

bool MhcGradCheckOutputNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNullImpl(params.xGrad, "xGrad") &&
           MhcGradCheckInputNotNullImpl(params.hcWeightGrad, "hcWeightGrad") &&
           MhcGradCheckInputNotNullImpl(params.alphaGrad, "alphaGrad") &&
           MhcGradCheckInputNotNullImpl(params.biasPostGrad, "biasPostGrad") &&
           MhcGradCheckInputNotNullImpl(params.gammaGrad, "gammaGrad");
}

bool MhcGradCheckNotNull(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputNotNull(params) && MhcGradCheckOutputNotNull(params);
}

bool MhcGradCheckEmptyTensorImpl(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr || tensor->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor is empty or nullptr", name);
        return false;
    }
    return true;
}

bool MhcGradCheckEmptyTensor(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckEmptyTensorImpl(params.x, "x") && MhcGradCheckEmptyTensorImpl(params.phi, "phi") &&
           MhcGradCheckEmptyTensorImpl(params.alpha, "alpha") &&
           MhcGradCheckEmptyTensorImpl(params.hInGrad, "hInGrad") &&
           MhcGradCheckEmptyTensorImpl(params.hPostGrad, "hPostGrad") &&
           MhcGradCheckEmptyTensorImpl(params.hResGrad, "hResGrad") &&
           MhcGradCheckEmptyTensorImpl(params.invRms, "invRms") && MhcGradCheckEmptyTensorImpl(params.mmRes, "mmRes") &&
           MhcGradCheckEmptyTensorImpl(params.hPre, "hPre") && MhcGradCheckEmptyTensorImpl(params.hPost, "hPost") &&
           (params.gradXPost == nullptr || MhcGradCheckEmptyTensorImpl(params.gradXPost, "gradXPost"));
}

bool CheckDimNum(const aclTensor *tensor, size_t expected, const char *name, bool isRange = false, size_t expected2 = 0)
{
    auto dimNum = tensor->GetViewShape().GetDimNum();
    bool valid = isRange ? (dimNum == expected || dimNum == expected2) : (dimNum == expected);
    if (!valid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dim num must be %s, but got %zu", name,
                isRange ? (std::to_string(expected) + " or " + std::to_string(expected2)).c_str() :
                          std::to_string(expected).c_str(),
                dimNum);
        return false;
    }
    return true;
}

bool MhcGradCheckInputDims(const AclnnMhcPreBackwardParams &params)
{
    return CheckDimNum(params.x, 3, "x", true, 4) && CheckDimNum(params.phi, 2, "phi") &&
           CheckDimNum(params.alpha, 1, "alpha") &&
           (params.gamma == nullptr || CheckDimNum(params.gamma, 2, "gamma")) &&
           CheckDimNum(params.hInGrad, (params.x)->GetViewShape().GetDimNum() - 1, "hInGrad") &&
           CheckDimNum(params.hPostGrad, (params.x)->GetViewShape().GetDimNum() - 1, "hPostGrad") &&
           CheckDimNum(params.hResGrad, (params.x)->GetViewShape().GetDimNum(), "hResGrad") &&
           CheckDimNum(params.invRms, (params.x)->GetViewShape().GetDimNum() - 2, "invRms") &&
           CheckDimNum(params.mmRes, (params.x)->GetViewShape().GetDimNum() - 1, "mmRes") &&
           CheckDimNum(params.hPre, (params.x)->GetViewShape().GetDimNum() - 1, "hPre") &&
           CheckDimNum(params.hPost, (params.x)->GetViewShape().GetDimNum() - 1, "hPost") &&
           (params.gradXPost == nullptr ||
            CheckDimNum(params.gradXPost, (params.x)->GetViewShape().GetDimNum(), "gradXPost"));
}

bool MhcGradCheckOutputDims(const AclnnMhcPreBackwardParams &params)
{
    auto hInGradDimNum = params.hInGrad->GetViewShape().GetDimNum();
    auto xGradDimNum = params.xGrad->GetViewShape().GetDimNum();
    if (hInGradDimNum == 3) {
        if (xGradDimNum != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad tensor dim num must be 4 for BSND format, but got %zu",
                    xGradDimNum);
            return false;
        }
    } else {
        if (xGradDimNum != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad tensor dim num must be 3 for TND format, but got %zu", xGradDimNum);
            return false;
        }
    }
    return CheckDimNum(params.hcWeightGrad, 2, "hcWeightGrad") && CheckDimNum(params.alphaGrad, 1, "alphaGrad") &&
           CheckDimNum(params.biasPostGrad, 1, "biasPostGrad") && CheckDimNum(params.gammaGrad, 2, "gammaGrad");
}

bool MhcGradCheckInputOutDims(const AclnnMhcPreBackwardParams &params)
{
    return MhcGradCheckInputDims(params) && MhcGradCheckOutputDims(params);
}

bool CheckShapeDim(const gert::Shape &shape, uint32_t index, uint64_t expected, const char *msg)
{
    if (shape.GetDim(index) != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s", msg);
        return false;
    }
    return true;
}

bool MhcGradCheckBSNDShape(const AclnnMhcPreBackwardParams &params, uint64_t batch, uint64_t sequence, uint64_t dimen,
                           uint64_t numsResidual)
{
    auto &xShape = params.x->GetViewShape();
    if (!CheckShapeDim(xShape, 0, batch, "x tensor shape must be [B, S, N, D]") ||
        !CheckShapeDim(xShape, 1, sequence, "x tensor shape must be [B, S, N, D]") ||
        !CheckShapeDim(xShape, 2, numsResidual, "x tensor shape must be [B, S, N, D]") ||
        !CheckShapeDim(xShape, 3, dimen, "x tensor shape must be [B, S, N, D]")) {
        return false;
    }

    auto &hPostGradShape = params.hPostGrad->GetViewShape();
    if (!CheckShapeDim(hPostGradShape, 0, batch, "hPostGrad shape must align with hInGrad on B and S dims") ||
        !CheckShapeDim(hPostGradShape, 1, sequence, "hPostGrad shape must align with hInGrad on B and S dims")) {
        return false;
    }

    auto &hResGradShape = params.hResGrad->GetViewShape();
    if (!CheckShapeDim(hResGradShape, 0, batch, "hResGrad shape must be [B, S, N, N]") ||
        !CheckShapeDim(hResGradShape, 1, sequence, "hResGrad shape must be [B, S, N, N]") ||
        !CheckShapeDim(hResGradShape, 2, numsResidual, "hResGrad shape must be [B, S, N, N]") ||
        !CheckShapeDim(hResGradShape, 3, numsResidual, "hResGrad shape must be [B, S, N, N]")) {
        return false;
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [B, S]") ||
        !CheckShapeDim(invRmsShape, 1, sequence, "invRms shape must be [B, S]")) {
        return false;
    }

    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    auto &mmResShape = params.mmRes->GetViewShape();
    if (!CheckShapeDim(mmResShape, 0, batch, "mmRes shape must be [B, S, N^2+2N]") ||
        !CheckShapeDim(mmResShape, 1, sequence, "mmRes shape must be [B, S, N^2+2N]") ||
        !CheckShapeDim(mmResShape, 2, n2Plus2n, "mmRes shape must be [B, S, N^2+2N]")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, batch, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 1, sequence, "hPre shape must be [B, S, N]") ||
        !CheckShapeDim(hPreShape, 2, numsResidual, "hPre shape must be [B, S, N]")) {
        return false;
    }

    auto &hPostShape = params.hPost->GetViewShape();
    if (!CheckShapeDim(hPostShape, 0, batch, "hPost shape must be [B, S, N]") ||
        !CheckShapeDim(hPostShape, 1, sequence, "hPost shape must be [B, S, N]") ||
        !CheckShapeDim(hPostShape, 2, numsResidual, "hPost shape must be [B, S, N]")) {
        return false;
    }

    auto &xGradShape = params.xGrad->GetViewShape();
    if (!CheckShapeDim(xGradShape, 0, batch, "xGrad shape must be [B, S, N, D]") ||
        !CheckShapeDim(xGradShape, 1, sequence, "xGrad shape must be [B, S, N, D]") ||
        !CheckShapeDim(xGradShape, 2, numsResidual, "xGrad shape must be [B, S, N, D]") ||
        !CheckShapeDim(xGradShape, 3, dimen, "xGrad shape must be [B, S, N, D]")) {
        return false;
    }

    if (params.gradXPost != nullptr) {
        auto &xPostGradShape = params.gradXPost->GetViewShape();
        if (!CheckShapeDim(xPostGradShape, 0, batch, "gradXPost shape must be [B, S, N, D]") ||
            !CheckShapeDim(xPostGradShape, 1, sequence, "gradXPost shape must be [B, S, N, D]") ||
            !CheckShapeDim(xPostGradShape, 2, numsResidual, "gradXPost shape must be [B, S, N, D]") ||
            !CheckShapeDim(xPostGradShape, 3, dimen, "gradXPost shape must be [B, S, N, D]")) {
            return false;
        }
    }
    return true;
}

bool MhcGradCheckTNDShape(const AclnnMhcPreBackwardParams &params, uint64_t t, uint64_t dimen, uint64_t numsResidual)
{
    auto &xShape = params.x->GetViewShape();
    if (!CheckShapeDim(xShape, 0, t, "x tensor shape must be [T, N, D]") ||
        !CheckShapeDim(xShape, 1, numsResidual, "x tensor shape must be [T, N, D]") ||
        !CheckShapeDim(xShape, 2, dimen, "x tensor shape must be [T, N, D]")) {
        return false;
    }

    auto &hPostGradShape = params.hPostGrad->GetViewShape();
    if (!CheckShapeDim(hPostGradShape, 0, t, "hPostGrad shape must align with hInGrad on T dim")) {
        return false;
    }

    auto &hResGradShape = params.hResGrad->GetViewShape();
    if (!CheckShapeDim(hResGradShape, 0, t, "hResGrad shape must be [T, N, N]") ||
        !CheckShapeDim(hResGradShape, 1, numsResidual, "hResGrad shape must be [T, N, N]") ||
        !CheckShapeDim(hResGradShape, 2, numsResidual, "hResGrad shape must be [T, N, N]")) {
        return false;
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!CheckShapeDim(invRmsShape, 0, t, "invRms shape must be [T]")) {
        return false;
    }

    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    auto &mmResShape = params.mmRes->GetViewShape();
    if (!CheckShapeDim(mmResShape, 0, t, "mmRes shape must be [T, N^2+2N]") ||
        !CheckShapeDim(mmResShape, 1, n2Plus2n, "mmRes shape must be [T, N^2+2N]")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!CheckShapeDim(hPreShape, 0, t, "hPre shape must be [T, N]") ||
        !CheckShapeDim(hPreShape, 1, numsResidual, "hPre shape must be [T, N]")) {
        return false;
    }

    auto &hPostShape = params.hPost->GetViewShape();
    if (!CheckShapeDim(hPostShape, 0, t, "hPost shape must be [T, N]") ||
        !CheckShapeDim(hPostShape, 1, numsResidual, "hPost shape must be [T, N]")) {
        return false;
    }

    auto &xGradShape = params.xGrad->GetViewShape();
    if (!CheckShapeDim(xGradShape, 0, t, "xGrad shape must be [T, N, D]") ||
        !CheckShapeDim(xGradShape, 1, numsResidual, "xGrad shape must be [T, N, D]") ||
        !CheckShapeDim(xGradShape, 2, dimen, "xGrad shape must be [T, N, D]")) {
        return false;
    }

    if (params.gradXPost != nullptr) {
        auto &xPostGradShape = params.gradXPost->GetViewShape();
        if (!CheckShapeDim(xPostGradShape, 0, t, "gradXPost shape must be [T, N, D]") ||
            !CheckShapeDim(xPostGradShape, 1, numsResidual, "gradXPost shape must be [T, N, D]") ||
            !CheckShapeDim(xPostGradShape, 2, dimen, "gradXPost shape must be [T, N, D]")) {
            return false;
        }
    }
    return true;
}

bool MhcGradCheckCommonShape(const AclnnMhcPreBackwardParams &params, uint64_t numsResidual, uint64_t dimen)
{
    auto &alphaShape = params.alpha->GetViewShape();
    if (!CheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    uint64_t n2Plus2n = numsResidual * numsResidual + 2 * numsResidual;
    auto &phiShape = params.phi->GetViewShape();
    if (!CheckShapeDim(phiShape, 0, n2Plus2n, "phi tensor first dim must be n^2+2n") ||
        !CheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be nD")) {
        return false;
    }

    auto &hcWeightGradShape = params.hcWeightGrad->GetViewShape();
    if (!CheckShapeDim(hcWeightGradShape, 0, n2Plus2n, "hcWeightGrad shape must be [2N+N^2, N*D]") ||
        !CheckShapeDim(hcWeightGradShape, 1, nD, "hcWeightGrad shape must be [2N+N^2, N*D]")) {
        return false;
    }

    auto &alphaGradShape = params.alphaGrad->GetViewShape();
    if (!CheckShapeDim(alphaGradShape, 0, 3, "alphaGrad shape must be [3]")) {
        return false;
    }

    auto &biasPostGradShape = params.biasPostGrad->GetViewShape();
    if (!CheckShapeDim(biasPostGradShape, 0, n2Plus2n, "biasPostGrad shape must be [2N+N^2]")) {
        return false;
    }

    auto &gammaGradShape = params.gammaGrad->GetViewShape();
    if (!CheckShapeDim(gammaGradShape, 0, numsResidual, "gammaGrad shape must be [N]") ||
        !CheckShapeDim(gammaGradShape, 1, dimen, "gammaGrad shape must be [D]")) {
        return false;
    }
    return true;
}

bool MhcGradCheckInputOutShape(const AclnnMhcPreBackwardParams &params)
{
    auto &hInGradShape = params.hInGrad->GetViewShape();
    auto &hPostGradShape = params.hPostGrad->GetViewShape();
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

bool CheckDtype(const aclTensor *tensor, DataType expected, const char *name, bool isValidX = false)
{
    auto dtype = tensor->GetDataType();
    bool valid = isValidX ? (dtype == DataType::DT_BF16 || dtype == DataType::DT_FLOAT16) : (dtype == expected);
    if (!valid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor dtype must be %s. actual is [%s].", name,
            isValidX ? "BF16 or FP16" : op::ToString(expected).GetString(), op::ToString(dtype).GetString());
        return false;
    }
    return true;
}

bool MhcGradCheckInputDtype(const AclnnMhcPreBackwardParams &params)
{
    return CheckDtype(params.x, DataType::DT_FLOAT, "x", true) && CheckDtype(params.phi, DataType::DT_FLOAT, "phi") &&
           CheckDtype(params.alpha, DataType::DT_FLOAT, "alpha") &&
           (params.gamma == nullptr || CheckDtype(params.gamma, DataType::DT_FLOAT, "gamma")) &&
           CheckDtype(params.hInGrad, DataType::DT_FLOAT, "hInGrad", true) &&
           CheckDtype(params.hInGrad, params.x->GetDataType(), "hInGrad") &&
           CheckDtype(params.hPostGrad, DataType::DT_FLOAT, "hPostGrad") &&
           CheckDtype(params.hResGrad, DataType::DT_FLOAT, "hResGrad") &&
           CheckDtype(params.invRms, DataType::DT_FLOAT, "invRms") &&
           CheckDtype(params.mmRes, DataType::DT_FLOAT, "mmRes") &&
           CheckDtype(params.hPre, DataType::DT_FLOAT, "hPre") &&
           CheckDtype(params.hPost, DataType::DT_FLOAT, "hPost") &&
           (params.gradXPost == nullptr || (CheckDtype(params.gradXPost, params.x->GetDataType(), "gradXPost", true) &&
                                            CheckDtype(params.gradXPost, params.x->GetDataType(), "gradXPost")));
}

bool MhcGradCheckOutputDtype(const AclnnMhcPreBackwardParams &params)
{
    auto hInGradDtype = params.hInGrad->GetDataType();
    if (params.xGrad->GetDataType() != hInGradDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xGrad tensor dtype must match hInGrad");
        return false;
    }
    return CheckDtype(params.hcWeightGrad, DataType::DT_FLOAT, "hcWeightGrad") &&
           CheckDtype(params.alphaGrad, DataType::DT_FLOAT, "alphaGrad") &&
           CheckDtype(params.biasPostGrad, DataType::DT_FLOAT, "biasPostGrad") &&
           CheckDtype(params.gammaGrad, DataType::DT_FLOAT, "gammaGrad");
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

static bool CheckFormat(const aclTensor *tensor, const char *name)
{
    if (MhcGradIsPrivateFormat(tensor->GetViewFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s tensor format must be ND", name);
        return false;
    }
    return true;
}

bool MhcGradCheckInputFormat(const AclnnMhcPreBackwardParams &params)
{
    return CheckFormat(params.x, "x") && CheckFormat(params.phi, "phi") && CheckFormat(params.alpha, "alpha") &&
           (params.gamma == nullptr || CheckFormat(params.gamma, "gamma")) && CheckFormat(params.hInGrad, "hInGrad") &&
           CheckFormat(params.hPostGrad, "hPostGrad") && CheckFormat(params.hResGrad, "hResGrad") &&
           CheckFormat(params.invRms, "invRms") && CheckFormat(params.mmRes, "mmRes") &&
           CheckFormat(params.hPre, "hPre") && CheckFormat(params.hPost, "hPost") &&
           (params.gradXPost == nullptr || CheckFormat(params.gradXPost, "gradXPost"));
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

const aclTensor *ConvertToContiguous(const aclTensor *tensor, aclOpExecutor *executor)
{
    auto result = l0op::Contiguous(tensor, executor);
    if (result == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Contiguous result is nullptr");
    }
    return result;
}

aclnnStatus MhcGradCovertDataContiguous(AclnnMhcPreBackwardParams &params, aclOpExecutor *executor)
{
    params.xContiguous = ConvertToContiguous(params.x, executor);
    params.phiContiguous = ConvertToContiguous(params.phi, executor);
    params.alphaContiguous = ConvertToContiguous(params.alpha, executor);
    params.hInGradContiguous = ConvertToContiguous(params.hInGrad, executor);
    params.hPostGradContiguous = ConvertToContiguous(params.hPostGrad, executor);
    params.hResGradContiguous = ConvertToContiguous(params.hResGrad, executor);
    params.invRmsContiguous = ConvertToContiguous(params.invRms, executor);
    params.mmResContiguous = ConvertToContiguous(params.mmRes, executor);
    params.hPreContiguous = ConvertToContiguous(params.hPre, executor);
    params.hPostContiguous = ConvertToContiguous(params.hPost, executor);

    if (params.xContiguous == nullptr || params.phiContiguous == nullptr || params.alphaContiguous == nullptr ||
        params.hInGradContiguous == nullptr || params.hPostGradContiguous == nullptr ||
        params.hResGradContiguous == nullptr || params.invRmsContiguous == nullptr ||
        params.mmResContiguous == nullptr || params.hPreContiguous == nullptr || params.hPostContiguous == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    if (params.gamma != nullptr) {
        params.gammaContiguous = ConvertToContiguous(params.gamma, executor);
        if (params.gammaContiguous == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    if (params.gradXPost != nullptr) {
        params.gradXPostContiguous = ConvertToContiguous(params.gradXPost, executor);
        if (params.gradXPostContiguous == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    return ACLNN_SUCCESS;
}

bool CopyOutput(const aclTensor *out, const aclTensor *dst, aclOpExecutor *executor)
{
    auto ret = l0op::ViewCopy(out, dst, executor);
    return ret != nullptr;
}

static aclnnStatus mhcPreBackwardCommonProcess(AclnnMhcPreBackwardParams &params, aclOpExecutor *executor)
{
    auto ret = MhcGradCheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = MhcGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto outParams =
        l0op::MhcPreBackward(params.xContiguous, params.phiContiguous, params.alphaContiguous, params.hInGradContiguous,
                             params.hPostGradContiguous, params.hResGradContiguous, params.invRmsContiguous,
                             params.mmResContiguous, params.hPreContiguous, params.hPostContiguous,
                             params.gammaContiguous, params.gradXPostContiguous, params.hcEps, executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(CopyOutput(std::get<0>(outParams), params.xGrad, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<1>(outParams), params.hcWeightGrad, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<2>(outParams), params.alphaGrad, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<3>(outParams), params.biasPostGrad, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CopyOutput(std::get<4>(outParams), params.gammaGrad, executor), ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnMhcPreBackwardGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *hInGrad,
    const aclTensor *hPostGrad, const aclTensor *hResGrad, const aclTensor *invRms, const aclTensor *mmRes,
    const aclTensor *hPre, const aclTensor *hPost, const aclTensor *gamma, const aclTensor *gradXPost, double hcEps,
    const aclTensor *xGrad, const aclTensor *hcWeightGrad, const aclTensor *alphaGrad, const aclTensor *biasPostGrad,
    const aclTensor *gammaGrad, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreBackward,
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
