/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_causal_conv1d_backward.cpp
 * \brief ACLNN L2 API 实现 - CausalConv1dBackward
 */

#include <cstring>
#include "aclnn_causal_conv1d_backward.h"
#include "causal_conv1d_backward.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

static const std::initializer_list<op::DataType> SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};

static int64_t ParseActivationStr(const char *str)
{
    if (str == nullptr || std::strcmp(str, "") == 0 || std::strcmp(str, "silu") == 0) {
        return 1;
    }
    if (std::strcmp(str, "none") == 0) {
        return 0;
    }
    return -1;
}

static bool CheckNotNull(const aclTensor *gradY, const aclTensor *x, const aclTensor *weight,
                         const aclTensor *gradXOut, const aclTensor *gradWeightOut,
                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(gradXOut, return false);
    OP_CHECK_NULL(gradWeightOut, return false);
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "expected a value of type uint64_t* but got null for argument workspaceSize.");
        return false;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "expected a value of type aclOpExecutor** but got null for argument executor.");
        return false;
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradY, const aclTensor *x, const aclTensor *weight,
                            const aclTensor *queryStartLocOptional, const aclTensor *gradXOut,
                            const aclTensor *gradWeightOut)
{
    OP_CHECK(CheckType(gradY->GetDataType(), SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY dtype %s not supported (float16/bfloat16).",
                     op::ToString(gradY->GetDataType()).GetString()),
             return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, gradY->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(weight, gradY->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradXOut, gradY->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradWeightOut, gradY->GetDataType(), return false);
    if (queryStartLocOptional != nullptr &&
        queryStartLocOptional->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "queryStartLocOptional dtype must be INT32, but got %s.",
                op::ToString(queryStartLocOptional->GetDataType()).GetString());
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *gradY, const aclTensor *x, const aclTensor *weight,
                       const aclTensor *queryStartLocOptional, const aclTensor *gradXOut,
                       const aclTensor *gradWeightOut)
{
    const auto& gyShape = gradY->GetViewShape();
    const auto& xShape = x->GetViewShape();
    const auto& wShape = weight->GetViewShape();

    OP_CHECK(gyShape.GetDimNum() == 2 || gyShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY must be 2D [cu_seq_len, D] or 3D [B, L, D], got %zuD.",
                     gyShape.GetDimNum()),
             return false);
    OP_CHECK(gyShape == xShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY.shape must equal x.shape."),
             return false);
    OP_CHECK(wShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight must be 2D [W, D], got %zuD.", wShape.GetDimNum()),
             return false);
    int64_t D = gyShape.GetDimNum() == 3 ? gyShape.GetDim(2) : gyShape.GetDim(1);
    int64_t W = wShape.GetDim(0);
    OP_CHECK(wShape.GetDim(1) == D,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight dim(%ld) must equal D(%ld).", wShape.GetDim(1), D),
             return false);
    OP_CHECK(W >= 2 && W <= 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernel width W must be in [2, 4], but got %ld.", W),
             return false);
    OP_CHECK(gradXOut->GetViewShape() == xShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradXOut shape must equal x.shape."),
             return false);
    OP_CHECK(gradWeightOut->GetViewShape() == wShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradWeightOut shape must equal weight.shape."),
             return false);
    if (gyShape.GetDimNum() == 2) {
        OP_CHECK(queryStartLocOptional != nullptr,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "2D grad_y (varlen) requires queryStartLocOptional."),
                 return false);
    }
    if (queryStartLocOptional != nullptr) {
        const auto& qslShape = queryStartLocOptional->GetViewShape();
        OP_CHECK(qslShape.GetDimNum() == 1 && qslShape.GetDim(0) >= 2,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                         "queryStartLocOptional must be 1D with size >= 2 (num_seq+1)."),
                 return false);
    }
    return true;
}

aclnnStatus CausalConv1dBackwardCommonProcess(
    const aclTensor *gradY, const aclTensor *x, const aclTensor *weight,
    const aclTensor *queryStartLocOptional, const char *activation,
    aclTensor *gradXOut, aclTensor *gradWeightOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();

    CHECK_RET(CheckNotNull(gradY, x, weight, gradXOut, gradWeightOut, workspaceSize, executor),
              ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(CheckDtypeValid(gradY, x, weight, queryStartLocOptional, gradXOut, gradWeightOut),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShape(gradY, x, weight, queryStartLocOptional, gradXOut, gradWeightOut),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_COND(ParseActivationStr(activation) >= 0, ACLNN_ERR_PARAM_INVALID, "Invalid activation: %s",
               activation ? activation : "null");

    const aclTensor *gradYFinal = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_COND(gradYFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous gradY failed.");

    const aclTensor *xFinal = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(xFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous x failed.");

    weight = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_COND(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous weight failed.");

    if (queryStartLocOptional != nullptr) {
        queryStartLocOptional = l0op::Contiguous(queryStartLocOptional, uniqueExecutor.get());
        CHECK_COND(queryStartLocOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "Contiguous queryStartLocOptional failed.");
    }

    // 输出通常为连续（torch 桥用 at::empty 分配）：Contiguous 直通，无拷贝；
    // 非连续时 kernel 写入临时副本，再 ViewCopy 拷回调用方张量。
    aclTensor *gradXFinal = const_cast<aclTensor *>(l0op::Contiguous(gradXOut, uniqueExecutor.get()));
    CHECK_COND(gradXFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous gradXOut failed.");

    aclTensor *gradWeightFinal = const_cast<aclTensor *>(l0op::Contiguous(gradWeightOut, uniqueExecutor.get()));
    CHECK_COND(gradWeightFinal != nullptr, ACLNN_ERR_PARAM_NULLPTR, "Contiguous gradWeightOut failed.");

    bool ok = l0op::CausalConv1dBackward(gradYFinal, xFinal, weight, queryStartLocOptional, activation,
                                         gradXFinal, gradWeightFinal, uniqueExecutor.get());
    CHECK_RET(ok, ACLNN_ERR_INNER_TILING_ERROR);

    if (gradXFinal != gradXOut) {
        auto gradXCopyResult = l0op::ViewCopy(gradXFinal, gradXOut, uniqueExecutor.get());
        CHECK_COND(gradXCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "ViewCopy gradXOut back failed.");
    }
    if (gradWeightFinal != gradWeightOut) {
        auto gradWeightCopyResult = l0op::ViewCopy(gradWeightFinal, gradWeightOut, uniqueExecutor.get());
        CHECK_COND(gradWeightCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR, "ViewCopy gradWeightOut back failed.");
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnCausalConv1dBackwardGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *queryStartLocOptional,
    const char *activation,
    aclTensor *gradXOut,
    aclTensor *gradWeightOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dBackward,
                   DFX_IN(gradY, x, weight, queryStartLocOptional, activation),
                   DFX_OUT(gradXOut, gradWeightOut));
    return CausalConv1dBackwardCommonProcess(gradY, x, weight, queryStartLocOptional, activation,
                                             gradXOut, gradWeightOut, workspaceSize, executor);
}

extern "C" aclnnStatus aclnnCausalConv1dBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
