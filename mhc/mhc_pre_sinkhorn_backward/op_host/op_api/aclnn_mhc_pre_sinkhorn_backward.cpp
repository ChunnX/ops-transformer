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
#include "mhc_pre_sinkhorn_backward.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct AclnnMhcPreSinkhornBackwardParams {
    const aclTensor *gradHin = nullptr;
    const aclTensor *gradHPost = nullptr;
    const aclTensor *gradHRes = nullptr;
    const aclTensor *x = nullptr;
    const aclTensor *phi = nullptr;
    const aclTensor *alpha = nullptr;
    const aclTensor *bias = nullptr;
    const aclTensor *hPre = nullptr;
    const aclTensor *hcBeforeNorm = nullptr;
    const aclTensor *invRms = nullptr;
    const aclTensor *sumOut = nullptr;
    const aclTensor *normOut = nullptr;
    double hcEps;

    const aclTensor *gradX = nullptr;
    const aclTensor *gradPhi = nullptr;
    const aclTensor *gradAlpha = nullptr;
    const aclTensor *gradBias = nullptr;

    const aclTensor *gradHinContiguous = nullptr;
    const aclTensor *gradHPostContiguous = nullptr;
    const aclTensor *gradHResContiguous = nullptr;
    const aclTensor *xContiguous = nullptr;
    const aclTensor *phiContiguous = nullptr;
    const aclTensor *alphaContiguous = nullptr;
    const aclTensor *biasContiguous = nullptr;
    const aclTensor *hPreContiguous = nullptr;
    const aclTensor *hcBeforeNormContiguous = nullptr;
    const aclTensor *invRmsContiguous = nullptr;
    const aclTensor *sumOutContiguous = nullptr;
    const aclTensor *normOutContiguous = nullptr;
};

class AclnnMhcPreSinkhornBackward {
public:
    static AclnnMhcPreSinkhornBackward Create()
    {
        AclnnMhcPreSinkhornBackward obj;
        return obj;
    }

    AclnnMhcPreSinkhornBackward &SetGradInput(const aclTensor *gradHin, const aclTensor *gradHPost,
                                              const aclTensor *gradHRes)
    {
        obj_.gradHin = gradHin;
        obj_.gradHPost = gradHPost;
        obj_.gradHRes = gradHRes;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetInput(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                          const aclTensor *bias)
    {
        obj_.x = x;
        obj_.phi = phi;
        obj_.alpha = alpha;
        obj_.bias = bias;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetForwardInput(const aclTensor *hPre, const aclTensor *hcBeforeNorm,
                                                 const aclTensor *invRms, const aclTensor *sumOut,
                                                 const aclTensor *normOut)
    {
        obj_.hPre = hPre;
        obj_.hcBeforeNorm = hcBeforeNorm;
        obj_.invRms = invRms;
        obj_.sumOut = sumOut;
        obj_.normOut = normOut;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetAttr(double hcEps)
    {
        obj_.hcEps = hcEps;
        return *this;
    }

    AclnnMhcPreSinkhornBackward &SetOutput(const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
                                           const aclTensor *gradBias)
    {
        obj_.gradX = gradX;
        obj_.gradPhi = gradPhi;
        obj_.gradAlpha = gradAlpha;
        obj_.gradBias = gradBias;
        return *this;
    }

    AclnnMhcPreSinkhornBackwardParams Build() const { return obj_; }

private:
    AclnnMhcPreSinkhornBackwardParams obj_;
};

bool MhcSinkhornGradCheckInputNotNullImpl(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "%s tensor is nullptr", name);
        return false;
    }
    return true;
}

bool MhcSinkhornGradCheckNotNull(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return MhcSinkhornGradCheckInputNotNullImpl(params.gradHin, "gradHin") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.gradHPost, "gradHPost") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.gradHRes, "gradHRes") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.x, "x") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.phi, "phi") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.alpha, "alpha") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.bias, "bias") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.hPre, "hPre") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.hcBeforeNorm, "hcBeforeNorm") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.invRms, "invRms") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.sumOut, "sumOut") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.normOut, "normOut") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.gradX, "gradX") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.gradPhi, "gradPhi") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.gradAlpha, "gradAlpha") &&
           MhcSinkhornGradCheckInputNotNullImpl(params.gradBias, "gradBias");
}

bool MhcSinkhornGradCheckDimNum(const aclTensor *tensor, size_t expected, const char *name,
                                bool isRange = false, size_t expected2 = 0)
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

bool MhcSinkhornGradCheckInputDims(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return MhcSinkhornGradCheckDimNum(params.gradHin, 3, "gradHin") &&
           MhcSinkhornGradCheckDimNum(params.gradHPost, 3, "gradHPost") &&
           MhcSinkhornGradCheckDimNum(params.gradHRes, 4, "gradHRes", true, 3) &&
           MhcSinkhornGradCheckDimNum(params.x, 4, "x") &&
           MhcSinkhornGradCheckDimNum(params.phi, 2, "phi") && MhcSinkhornGradCheckDimNum(params.alpha, 1, "alpha") &&
           MhcSinkhornGradCheckDimNum(params.bias, 1, "bias") && MhcSinkhornGradCheckDimNum(params.hPre, 3, "hPre") &&
           MhcSinkhornGradCheckDimNum(params.hcBeforeNorm, 3, "hcBeforeNorm") &&
           MhcSinkhornGradCheckDimNum(params.invRms, 3, "invRms") &&
           MhcSinkhornGradCheckDimNum(params.sumOut, 4, "sumOut") &&
           MhcSinkhornGradCheckDimNum(params.normOut, 5, "normOut");
}

bool MhcSinkhornGradCheckOutputDims(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return MhcSinkhornGradCheckDimNum(params.gradX, 4, "gradX") &&
           MhcSinkhornGradCheckDimNum(params.gradPhi, 2, "gradPhi") &&
           MhcSinkhornGradCheckDimNum(params.gradAlpha, 1, "gradAlpha") &&
           MhcSinkhornGradCheckDimNum(params.gradBias, 1, "gradBias");
}

bool MhcSinkhornGradCheckInputOutDims(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return MhcSinkhornGradCheckInputDims(params) && MhcSinkhornGradCheckOutputDims(params);
}

bool MhcSinkhornGradCheckShapeDim(const gert::Shape &shape, uint64_t index, uint64_t expected, const char *msg)
{
    if (shape.GetDim(index) != expected) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s, and dim %lu must be %lu, actual is %ld", msg, index, expected,
                shape.GetDim(index));
        return false;
    }
    return true;
}

bool MhcSinkhornGradCheckShape(const AclnnMhcPreSinkhornBackwardParams &params)
{
    auto &gradHInShape = params.gradHin->GetViewShape();
    uint64_t batch = gradHInShape.GetDim(0);
    uint64_t sequence = gradHInShape.GetDim(1);
    uint64_t dimen = gradHInShape.GetDim(2);

    auto &xShape = params.x->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(xShape, 0, batch, "x tensor shape must be [B, S, N, C]") ||
        !MhcSinkhornGradCheckShapeDim(xShape, 1, sequence, "x tensor shape must be [B, S, N, C]")) {
        return false;
    }
    uint64_t numsResidual = xShape.GetDim(2);
    if (!MhcSinkhornGradCheckShapeDim(xShape, 3, dimen, "x tensor shape must be [B, S, N, C]")) {
        return false;
    }

    uint64_t nD = numsResidual * dimen;
    auto &phiShape = params.phi->GetViewShape();
    uint64_t fusionSize = phiShape.GetDim(0);
    if (!MhcSinkhornGradCheckShapeDim(phiShape, 1, nD, "phi tensor second dim must be N*C")) {
        return false;
    }

    auto &gradHPostShape = params.gradHPost->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(gradHPostShape, 0, batch, "gradHPost shape must be [B, S, N]") ||
        !MhcSinkhornGradCheckShapeDim(gradHPostShape, 1, sequence, "gradHPost shape must be [B, S, N]") ||
        !MhcSinkhornGradCheckShapeDim(gradHPostShape, 2, numsResidual, "gradHPost shape must be [B, S, N]")) {
        return false;
    }

    auto &gradHResShape = params.gradHRes->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(gradHResShape, 0, batch,
                                       "gradHRes shape must be [B, S, N, N] or [B, S, N*N]") ||
        !MhcSinkhornGradCheckShapeDim(gradHResShape, 1, sequence,
                                       "gradHRes shape must be [B, S, N, N] or [B, S, N*N]")) {
        return false;
    }
    if (gradHResShape.GetDimNum() == 4) {
        if (!MhcSinkhornGradCheckShapeDim(gradHResShape, 2, numsResidual, "gradHRes shape must be [B, S, N, N]") ||
            !MhcSinkhornGradCheckShapeDim(gradHResShape, 3, numsResidual, "gradHRes shape must be [B, S, N, N]")) {
            return false;
        }
    } else {
        if (!MhcSinkhornGradCheckShapeDim(gradHResShape, 2, numsResidual * numsResidual,
                                           "gradHRes shape must be [B, S, N*N]")) {
            return false;
        }
    }

    auto &alphaShape = params.alpha->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(alphaShape, 0, 3, "alpha tensor shape must be (3)")) {
        return false;
    }

    auto &biasShape = params.bias->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(biasShape, 0, fusionSize, "bias tensor shape must be (2N+N^2)")) {
        return false;
    }

    auto &hPreShape = params.hPre->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(hPreShape, 0, batch, "hPre shape must be [B, S, N]") ||
        !MhcSinkhornGradCheckShapeDim(hPreShape, 1, sequence, "hPre shape must be [B, S, N]") ||
        !MhcSinkhornGradCheckShapeDim(hPreShape, 2, numsResidual, "hPre shape must be [B, S, N]")) {
        return false;
    }

    auto &hcBeforeNormShape = params.hcBeforeNorm->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(hcBeforeNormShape, 0, batch, "hcBeforeNorm shape must be [B, S, 2N+N^2]") ||
        !MhcSinkhornGradCheckShapeDim(hcBeforeNormShape, 1, sequence, "hcBeforeNorm shape must be [B, S, 2N+N^2]") ||
        !MhcSinkhornGradCheckShapeDim(hcBeforeNormShape, 2, fusionSize, "hcBeforeNorm shape must be [B, S, 2N+N^2]")) {
        return false;
    }

    auto &invRmsShape = params.invRms->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(invRmsShape, 0, batch, "invRms shape must be [B, S, 1]") ||
        !MhcSinkhornGradCheckShapeDim(invRmsShape, 1, sequence, "invRms shape must be [B, S, 1]") ||
        !MhcSinkhornGradCheckShapeDim(invRmsShape, 2, 1, "invRms shape must be [B, S, 1]")) {
        return false;
    }

    auto &sumOutShape = params.sumOut->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(sumOutShape, 1, batch,
                                       "sumOut shape must be [2*sk_iter_count, B, S, N]") ||
        !MhcSinkhornGradCheckShapeDim(sumOutShape, 2, sequence,
                                       "sumOut shape must be [2*sk_iter_count, B, S, N]") ||
        !MhcSinkhornGradCheckShapeDim(sumOutShape, 3, numsResidual,
                                       "sumOut shape must be [2*sk_iter_count, B, S, N]")) {
        return false;
    }

    auto &normOutShape = params.normOut->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(normOutShape, 1, batch,
                                       "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !MhcSinkhornGradCheckShapeDim(normOutShape, 2, sequence,
                                       "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !MhcSinkhornGradCheckShapeDim(normOutShape, 3, numsResidual,
                                       "normOut shape must be [2*sk_iter_count, B, S, N, N]") ||
        !MhcSinkhornGradCheckShapeDim(normOutShape, 4, numsResidual,
                                       "normOut shape must be [2*sk_iter_count, B, S, N, N]")) {
        return false;
    }

    auto &gradXShape = params.gradX->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(gradXShape, 0, batch, "gradX shape must be [B, S, N, C]") ||
        !MhcSinkhornGradCheckShapeDim(gradXShape, 1, sequence, "gradX shape must be [B, S, N, C]") ||
        !MhcSinkhornGradCheckShapeDim(gradXShape, 2, numsResidual, "gradX shape must be [B, S, N, C]") ||
        !MhcSinkhornGradCheckShapeDim(gradXShape, 3, dimen, "gradX shape must be [B, S, N, C]")) {
        return false;
    }

    auto &gradPhiShape = params.gradPhi->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(gradPhiShape, 0, fusionSize, "gradPhi shape must be [2N+N^2, N*C]") ||
        !MhcSinkhornGradCheckShapeDim(gradPhiShape, 1, nD, "gradPhi shape must be [2N+N^2, N*C]")) {
        return false;
    }

    auto &gradAlphaShape = params.gradAlpha->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(gradAlphaShape, 0, 3, "gradAlpha shape must be (3)")) {
        return false;
    }

    auto &gradBiasShape = params.gradBias->GetViewShape();
    if (!MhcSinkhornGradCheckShapeDim(gradBiasShape, 0, fusionSize, "gradBias shape must be (2N+N^2)")) {
        return false;
    }

    return true;
}

bool MhcSinkhornGradCheckDtype(const aclTensor *tensor, DataType expected, const char *name, bool isValidX = false)
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

bool MhcSinkhornGradCheckInputDtype(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return MhcSinkhornGradCheckDtype(params.gradHin, DataType::DT_FLOAT16, "gradHin", true) &&
           MhcSinkhornGradCheckDtype(params.gradHPost, DataType::DT_FLOAT, "gradHPost") &&
           MhcSinkhornGradCheckDtype(params.gradHRes, DataType::DT_FLOAT, "gradHRes") &&
           MhcSinkhornGradCheckDtype(params.x, DataType::DT_FLOAT16, "x", true) &&
           MhcSinkhornGradCheckDtype(params.phi, DataType::DT_FLOAT, "phi") &&
           MhcSinkhornGradCheckDtype(params.alpha, DataType::DT_FLOAT, "alpha") &&
           MhcSinkhornGradCheckDtype(params.bias, DataType::DT_FLOAT, "bias") &&
           MhcSinkhornGradCheckDtype(params.hPre, DataType::DT_FLOAT, "hPre") &&
           MhcSinkhornGradCheckDtype(params.hcBeforeNorm, DataType::DT_FLOAT, "hcBeforeNorm") &&
           MhcSinkhornGradCheckDtype(params.invRms, DataType::DT_FLOAT, "invRms") &&
           MhcSinkhornGradCheckDtype(params.sumOut, DataType::DT_FLOAT, "sumOut") &&
           MhcSinkhornGradCheckDtype(params.normOut, DataType::DT_FLOAT, "normOut");
}

bool MhcSinkhornGradCheckOutputDtype(const AclnnMhcPreSinkhornBackwardParams &params)
{
    auto gradHInDtype = params.gradHin->GetDataType();
    if (params.gradX->GetDataType() != gradHInDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradX tensor dtype must match gradHin");
        return false;
    }
    return MhcSinkhornGradCheckDtype(params.gradPhi, DataType::DT_FLOAT, "gradPhi") &&
           MhcSinkhornGradCheckDtype(params.gradAlpha, DataType::DT_FLOAT, "gradAlpha") &&
           MhcSinkhornGradCheckDtype(params.gradBias, DataType::DT_FLOAT, "gradBias");
}

bool MhcSinkhornGradCheckDtypeValid(const AclnnMhcPreSinkhornBackwardParams &params)
{
    return MhcSinkhornGradCheckInputDtype(params) && MhcSinkhornGradCheckOutputDtype(params);
}

aclnnStatus MhcSinkhornGradCheckParams(const AclnnMhcPreSinkhornBackwardParams &params)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(MhcSinkhornGradCheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 校验输入、输出参数维度
    CHECK_RET(MhcSinkhornGradCheckInputOutDims(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 校验输入、输出shape参数
    CHECK_RET(MhcSinkhornGradCheckShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入的数据类型是否在支持的数据类型范围之内
    CHECK_RET(MhcSinkhornGradCheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

const aclTensor *MhcSinkhornGradConvertToContiguous(const aclTensor *tensor, aclOpExecutor *executor)
{
    auto result = l0op::Contiguous(tensor, executor);
    if (result == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Contiguous result is nullptr");
    }
    return result;
}

aclnnStatus MhcSinkhornGradCovertDataContiguous(AclnnMhcPreSinkhornBackwardParams &params, aclOpExecutor *executor)
{
    params.gradHinContiguous = MhcSinkhornGradConvertToContiguous(params.gradHin, executor);
    params.gradHPostContiguous = MhcSinkhornGradConvertToContiguous(params.gradHPost, executor);
    params.gradHResContiguous = MhcSinkhornGradConvertToContiguous(params.gradHRes, executor);
    params.xContiguous = MhcSinkhornGradConvertToContiguous(params.x, executor);
    params.phiContiguous = MhcSinkhornGradConvertToContiguous(params.phi, executor);
    params.alphaContiguous = MhcSinkhornGradConvertToContiguous(params.alpha, executor);
    params.biasContiguous = MhcSinkhornGradConvertToContiguous(params.bias, executor);
    params.hPreContiguous = MhcSinkhornGradConvertToContiguous(params.hPre, executor);
    params.hcBeforeNormContiguous = MhcSinkhornGradConvertToContiguous(params.hcBeforeNorm, executor);
    params.invRmsContiguous = MhcSinkhornGradConvertToContiguous(params.invRms, executor);
    params.sumOutContiguous = MhcSinkhornGradConvertToContiguous(params.sumOut, executor);
    params.normOutContiguous = MhcSinkhornGradConvertToContiguous(params.normOut, executor);

    if (params.gradHinContiguous == nullptr || params.gradHPostContiguous == nullptr ||
        params.gradHResContiguous == nullptr || params.xContiguous == nullptr || params.phiContiguous == nullptr ||
        params.alphaContiguous == nullptr || params.biasContiguous == nullptr || params.hPreContiguous == nullptr ||
        params.hcBeforeNormContiguous == nullptr || params.invRmsContiguous == nullptr ||
        params.sumOutContiguous == nullptr || params.normOutContiguous == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

bool MhcSinkhornGradCopyOutput(const aclTensor *out, const aclTensor *dst, aclOpExecutor *executor)
{
    auto ret = l0op::ViewCopy(out, dst, executor);
    return ret != nullptr;
}

static aclnnStatus mhcPreSinkhornBackwardCommonProcess(AclnnMhcPreSinkhornBackwardParams &params,
                                                       aclOpExecutor *executor)
{
    auto ret = MhcSinkhornGradCheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = MhcSinkhornGradCovertDataContiguous(params, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto outParams = l0op::MhcPreSinkhornBackward(
        params.gradHinContiguous, params.gradHPostContiguous, params.gradHResContiguous, params.xContiguous,
        params.phiContiguous, params.alphaContiguous, params.biasContiguous, params.hPreContiguous,
        params.hcBeforeNormContiguous, params.invRmsContiguous, params.sumOutContiguous, params.normOutContiguous,
        static_cast<float>(params.hcEps), executor);
    CHECK_RET(outParams != std::tuple(nullptr, nullptr, nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(MhcSinkhornGradCopyOutput(std::get<0>(outParams), params.gradX, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MhcSinkhornGradCopyOutput(std::get<1>(outParams), params.gradPhi, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MhcSinkhornGradCopyOutput(std::get<2>(outParams), params.gradAlpha, executor), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MhcSinkhornGradCopyOutput(std::get<3>(outParams), params.gradBias, executor), ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
    const aclTensor *gradHin, const aclTensor *gradHPost, const aclTensor *gradHRes, const aclTensor *x,
    const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias, const aclTensor *hPre,
    const aclTensor *hcBeforeNorm, const aclTensor *invRms, const aclTensor *sumOut, const aclTensor *normOut,
    double hcEps, const aclTensor *gradX, const aclTensor *gradPhi, const aclTensor *gradAlpha,
    const aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnMhcPreSinkhornBackward,
        DFX_IN(gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm, invRms, sumOut, normOut),
        DFX_OUT(gradX, gradPhi, gradAlpha, gradBias));
    auto uniqueExecutor = CREATE_EXECUTOR();
    AclnnMhcPreSinkhornBackwardParams params = AclnnMhcPreSinkhornBackward::Create()
                                                   .SetGradInput(gradHin, gradHPost, gradHRes)
                                                   .SetInput(x, phi, alpha, bias)
                                                   .SetForwardInput(hPre, hcBeforeNorm, invRms, sumOut, normOut)
                                                   .SetAttr(hcEps)
                                                   .SetOutput(gradX, gradPhi, gradAlpha, gradBias)
                                                   .Build();
    auto ret = mhcPreSinkhornBackwardCommonProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhornBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
