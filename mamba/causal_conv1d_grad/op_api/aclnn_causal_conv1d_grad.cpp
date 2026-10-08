/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file aclnn_causal_conv1d_grad.cpp
 * @brief ACLNN L2 API 实现 - CausalConv1dGrad
 */

#include "aclnn_causal_conv1d_grad.h"
#include "causal_conv1d_grad.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

static const std::initializer_list<op::DataType> SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16
};

static bool CheckNotNull(const aclTensor* gradY, const aclTensor* x, const aclTensor* weight,
                         const aclTensor* bias, const aclTensor* dx, const aclTensor* dweight,
                         const aclTensor* dbias)
{
    OP_CHECK_NULL(gradY, return false);
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(dx, return false);
    OP_CHECK_NULL(dweight, return false);
    OP_CHECK_NULL(dbias, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* gradY, const aclTensor* x, const aclTensor* weight,
                            const aclTensor* bias)
{
    OP_CHECK(CheckType(gradY->GetDataType(), SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY dtype %d not supported (float16/bfloat16).",
                     static_cast<int>(gradY->GetDataType())),
             return false);
    OP_CHECK_DTYPE_NOT_MATCH(x, gradY->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(weight, gradY->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(bias, gradY->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* gradY, const aclTensor* x, const aclTensor* weight,
                       const aclTensor* bias)
{
    const auto& gyShape = gradY->GetViewShape();
    const auto& xShape = x->GetViewShape();
    const auto& wShape = weight->GetViewShape();
    const auto& bShape = bias->GetViewShape();

    OP_CHECK(gyShape.GetDimNum() == 3 && xShape.GetDimNum() == 3,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY/x must be 3D [B,T,D]."), return false);
    OP_CHECK(gyShape == xShape,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradY.shape must equal x.shape."), return false);
    OP_CHECK(wShape.GetDimNum() == 2,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight must be 2D [kW,D]."), return false);
    OP_CHECK(bShape.GetDimNum() == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "bias must be 1D [D]."), return false);

    int64_t D = gyShape.GetDim(2);
    OP_CHECK(wShape.GetDim(1) == D && bShape.GetDim(0) == D,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "weight[1]/bias[0] must equal D=%ld.", D), return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* gradY, const aclTensor* x, const aclTensor* weight,
                               const aclTensor* bias, const aclTensor* dx, const aclTensor* dweight,
                               const aclTensor* dbias)
{
    CHECK_COND(CheckNotNull(gradY, x, weight, bias, dx, dweight, dbias),
               ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(gradY, x, weight, bias),
               ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed");
    CHECK_COND(CheckShape(gradY, x, weight, bias),
               ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnCausalConv1dGradGetWorkspaceSize(
    const aclTensor* gradY,
    const aclTensor* x,
    const aclTensor* weight,
    const aclTensor* bias,
    const aclTensor* dx,
    const aclTensor* dweight,
    const aclTensor* dbias,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnCausalConv1dGrad,
                   DFX_IN(gradY, x, weight, bias), DFX_OUT(dx, dweight, dbias));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(gradY, x, weight, bias, dx, dweight, dbias);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto gradYC = l0op::Contiguous(gradY, uniqueExecutor.get());
    CHECK_RET(gradYC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xC = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto weightC = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto biasC = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasC != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto results = l0op::CausalConv1dGrad(gradYC, xC, weightC, biasC, uniqueExecutor.get());
    CHECK_RET(results[0] != nullptr && results[1] != nullptr && results[2] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    auto viewDx = l0op::ViewCopy(results[0], dx, uniqueExecutor.get());
    CHECK_RET(viewDx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewDw = l0op::ViewCopy(results[1], dweight, uniqueExecutor.get());
    CHECK_RET(viewDw != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewDb = l0op::ViewCopy(results[2], dbias, uniqueExecutor.get());
    CHECK_RET(viewDb != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnCausalConv1dGrad(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnCausalConv1dGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
