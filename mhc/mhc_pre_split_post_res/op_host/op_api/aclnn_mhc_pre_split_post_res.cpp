/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cmath>
#include <vector>

#include "log/log.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "../mhc_pre_split_post_res_const.h"
#include "mhc_pre_split_post_res.h"
#include "aclnn_mhc_pre_split_post_res.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// n 固定值与 D 白名单统一来自 mhc_pre_split_post_res_const.h（与 tiling 单一事实来源）
using optiling::MHC_N_VALUE;
// 维度数（秩）与维度下标统一来自共享常量头，本文件不再自定义 DIM 常量
using optiling::MHC_DIM_NUM_1;
using optiling::MHC_DIM_NUM_2;
using optiling::MHC_DIM_NUM_3;
using optiling::MHC_DIM_NUM_4;
using optiling::MHC_DIM_ZERO;
using optiling::MHC_DIM_ONE;
using optiling::MHC_DIM_TWO;
using optiling::MHC_DIM_THREE;

static bool CheckRequiredTensor(const aclTensor *tensor, const char *name, const char *reason)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMhcPreSplitPostResGetWorkspaceSize", name, "nullptr", reason);
        return false;
    }
    return true;
}

// 强类型判空替代 void* 形参（extern "C" 块内不可用模板/重载，故拆为两个具名函数）
static bool CheckOutputSize(const uint64_t *value, const char *name)
{
    if (value == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMhcPreSplitPostResGetWorkspaceSize", name, "nullptr",
                                              "output parameter must not be nullptr");
        return false;
    }
    return true;
}

static bool CheckOutputExecutor(aclOpExecutor *const *value, const char *name)
{
    if (value == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnMhcPreSplitPostResGetWorkspaceSize", name, "nullptr",
                                              "output parameter must not be nullptr");
        return false;
    }
    return true;
}

static bool CheckTensorRank(const aclTensor *tensor, const char *name, size_t expectedRank)
{
    size_t actualRank = tensor->GetViewShape().GetDimNum();
    if (actualRank != expectedRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnMhcPreSplitPostResGetWorkspaceSize", name,
                                     std::to_string(actualRank).c_str(), std::to_string(expectedRank).c_str());
        return false;
    }
    return true;
}

static bool CheckTensorDtype(const aclTensor *tensor, const char *name, DataType expectedDtype)
{
    auto actualDtype = tensor->GetDataType();
    if (actualDtype != expectedDtype) {
        OP_LOGE_FOR_INVALID_DTYPE("aclnnMhcPreSplitPostResGetWorkspaceSize", name,
                                  op::ToString(actualDtype).GetString(), op::ToString(expectedDtype).GetString());
        return false;
    }
    return true;
}

static bool CheckTensorShape(const aclTensor *tensor, const char *name, const std::vector<int64_t> &expected)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() != expected.size()) {
        return false;
    }
    size_t index = 0;
    for (int64_t dim : expected) {
        if (shape.GetDim(index++) != dim) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape does not match the expected shape.", name);
            return false;
        }
    }
    return true;
}

static bool IsSupportedD(int64_t d)
{
    return d == optiling::MHC_D_2560 || d == optiling::MHC_D_5120 || d == optiling::MHC_D_6144 ||
           d == optiling::MHC_D_7168;
}

static bool CheckNonEmptyTensor(const aclTensor *tensor, const char *name)
{
    if (tensor->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must not be empty.", name);
        return false;
    }
    return true;
}

static bool CheckTensorFormat(const aclTensor *tensor, const char *name)
{
    switch (tensor->GetViewFormat()) {
        case ge::FORMAT_NC1HWC0:
        case ge::FORMAT_FRACTAL_Z:
        case ge::FORMAT_NDC1HWC0:
        case ge::FORMAT_FRACTAL_Z_3D:
        case ge::FORMAT_FRACTAL_NZ:
        case ge::FORMAT_NC1HWC0_C04:
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s private format is not supported.", name);
            return false;
        default:
            return true;
    }
}

static bool CheckOutputContiguous(const aclTensor *tensor, const char *name)
{
    if (!IsContiguous(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s output must be contiguous.", name);
        return false;
    }
    return true;
}

static bool CheckOutputShape(const aclTensor *x, const aclTensor *phi, const aclTensor *hPostOut,
                             const aclTensor *hResOut)
{
    auto xShape = x->GetViewShape();
    size_t xRank = xShape.GetDimNum();
    int64_t n = xRank == MHC_DIM_NUM_4 ? xShape.GetDim(MHC_DIM_TWO) : xShape.GetDim(MHC_DIM_ONE);
    int64_t t = xRank == MHC_DIM_NUM_4 ? xShape.GetDim(MHC_DIM_ZERO) * xShape.GetDim(MHC_DIM_ONE)
                                  : xShape.GetDim(MHC_DIM_ZERO);
    int64_t phiRows = phi->GetViewShape().GetDim(MHC_DIM_ZERO);
    // 输出前导维：4 维为 [B,S]，3 维为 [T]；h_post 期望 shape = 前导 + [n]
    std::vector<int64_t> expected = (xRank == MHC_DIM_NUM_4)
        ? std::vector<int64_t>{xShape.GetDim(MHC_DIM_ZERO), xShape.GetDim(MHC_DIM_ONE)}
        : std::vector<int64_t>{t};
    expected.push_back(n);
    if (!CheckTensorShape(hPostOut, "hPostOut", expected)) {
        return false;
    }
    // 完整形态（phiRows == n*n+n）：h_res 期望 shape = 前导 + [n, n]
    if (phiRows == n * n + n) {
        if (hResOut == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResOut must not be nullptr when phi has n*n+n rows.");
            return false;
        }
        expected.push_back(n);
        return CheckTensorShape(hResOut, "hResOut", expected);
    }
    // 仅 post 形态（phiRows == n）：hResOut 允许 nullptr（L0 层会分配 dummy 占位），
    // 传入时必须为 0 元素 Tensor（与 InferShape 该形态输出 0 元素的契约一致）
    if (hResOut != nullptr && !hResOut->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "hResOut must be nullptr or an empty tensor when phi has only n rows.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                               const aclTensor *bias, float normEps, const aclTensor *hPostOut,
                               const aclTensor *hResOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckRequiredTensor(x, "x", "required input tensor must not be nullptr"), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckRequiredTensor(phi, "phi", "required input tensor must not be nullptr"), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckRequiredTensor(alpha, "alpha", "required input tensor must not be nullptr"),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckRequiredTensor(bias, "bias", "required input tensor must not be nullptr"), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckRequiredTensor(hPostOut, "hPostOut", "required output tensor must not be nullptr"),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOutputSize(workspaceSize, "workspaceSize"), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckOutputExecutor(executor, "executor"), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(std::isfinite(normEps) && normEps > 0.0f, ACLNN_ERR_PARAM_INVALID);
    const size_t xRank = x->GetViewShape().GetDimNum();
    CHECK_RET(xRank == MHC_DIM_NUM_3 || xRank == MHC_DIM_NUM_4, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorRank(phi, "phi", MHC_DIM_NUM_2) && CheckTensorRank(alpha, "alpha", MHC_DIM_NUM_1) &&
                  CheckTensorRank(bias, "bias", MHC_DIM_NUM_1), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNonEmptyTensor(x, "x") && CheckNonEmptyTensor(phi, "phi") &&
                  CheckNonEmptyTensor(alpha, "alpha") && CheckNonEmptyTensor(bias, "bias"),
              ACLNN_ERR_PARAM_INVALID);
    auto xShape = x->GetViewShape();
    int64_t n = xRank == MHC_DIM_NUM_4 ? xShape.GetDim(MHC_DIM_TWO) : xShape.GetDim(MHC_DIM_ONE);
    int64_t d = xRank == MHC_DIM_NUM_4 ? xShape.GetDim(MHC_DIM_THREE) : xShape.GetDim(MHC_DIM_TWO);
    CHECK_RET(n == MHC_N_VALUE && IsSupportedD(d), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDtype(x, "x", DataType::DT_BF16) || CheckTensorDtype(x, "x", DataType::DT_FLOAT16),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDtype(phi, "phi", DataType::DT_FLOAT) &&
                  CheckTensorDtype(alpha, "alpha", DataType::DT_FLOAT) &&
                  CheckTensorDtype(bias, "bias", DataType::DT_FLOAT) &&
                  CheckTensorDtype(hPostOut, "hPostOut", DataType::DT_FLOAT),
              ACLNN_ERR_PARAM_INVALID);
    if (hResOut != nullptr) {
        CHECK_RET(CheckTensorDtype(hResOut, "hResOut", DataType::DT_FLOAT), ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckTensorFormat(x, "x") && CheckTensorFormat(phi, "phi") && CheckTensorFormat(alpha, "alpha") &&
                  CheckTensorFormat(bias, "bias") && CheckTensorFormat(hPostOut, "hPostOut"), ACLNN_ERR_PARAM_INVALID);
    if (hResOut != nullptr) {
        CHECK_RET(CheckTensorFormat(hResOut, "hResOut"), ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckOutputContiguous(hPostOut, "hPostOut"), ACLNN_ERR_PARAM_INVALID);
    if (hResOut != nullptr && !hResOut->IsEmpty()) {
        CHECK_RET(CheckOutputContiguous(hResOut, "hResOut"), ACLNN_ERR_PARAM_INVALID);
    }
    auto phiShape = phi->GetViewShape();
    int64_t phiRows = phiShape.GetDim(MHC_DIM_ZERO);
    CHECK_RET((phiRows == n || phiRows == n * n + n) && phiShape.GetDim(MHC_DIM_ONE) == n * d,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(bias->GetViewShape().GetDim(MHC_DIM_ZERO) == phiRows && CheckOutputShape(x, phi, hPostOut, hResOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus MhcPreSplitPostResCommonProcess(const aclTensor *x, const aclTensor *phi,
    const aclTensor *alpha, const aclTensor *bias, float normEps,
    const aclTensor *hPostOut, const aclTensor *hResOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckParams(x, phi, alpha, bias, normEps, hPostOut, hResOut, workspaceSize, executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 输入非连续自动转连续；输出 h_post/h_res 要求连续（README 约束：非连续 ×）
    x = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(x != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous x failed.");

    phi = l0op::Contiguous(phi, uniqueExecutor.get());
    CHECK_COND(phi != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous phi failed.");

    alpha = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_COND(alpha != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous alpha failed.");

    bias = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_COND(bias != nullptr, ACLNN_ERR_INNER_NULLPTR, "Contiguous bias failed.");

    // h_res 可选输出由 phi 行数决定：N^2+N（完整形态）时必选，N（仅 post 形态）时忽略
    // （形态 B 传 nullptr 或空 tensor 均可，kernel 按 phiRows 跳过 res 段）
    const size_t xRank = x->GetViewShape().GetDimNum();
    const int64_t nValue = (xRank == MHC_DIM_NUM_4) ? x->GetViewShape().GetDim(MHC_DIM_TWO)
                                                     : x->GetViewShape().GetDim(MHC_DIM_ONE);
    if (phi->GetViewShape().GetDim(MHC_DIM_ZERO) == nValue * nValue + nValue) {
        CHECK_COND(hResOut != nullptr, ACLNN_ERR_PARAM_INVALID,
                   "h_res output is required when the number of phi rows is N^2+N (%ld).", nValue * nValue + nValue);
    } else if (hResOut != nullptr) {
        CHECK_COND(hResOut->IsEmpty(), ACLNN_ERR_PARAM_INVALID,
                   "h_res output must be empty for post-only phi shape.");
    }

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
