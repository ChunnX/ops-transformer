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
 * \file aclnn_mhc_pre_sinkhorn.cpp
 * \brief aclnn_mhc_pre_sinkhorn
 */

#include "aclnn_mhc_pre_sinkhorn.h"
#include "mhc_pre_sinkhorn.h"
#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> DTYPE_X_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_OTHER_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static bool CheckNotNull(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                         const aclTensor *y, const aclTensor *post, const aclTensor *combFrag)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(phi, return false);
    OP_CHECK_NULL(alpha, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(y, return false);
    OP_CHECK_NULL(post, return false);
    OP_CHECK_NULL(combFrag, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                            const aclTensor *y, const aclTensor *post, const aclTensor *combFrag)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_X_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(phi, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(alpha, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, DTYPE_X_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(post, DTYPE_OTHER_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(combFrag, DTYPE_OTHER_SUPPORT_LIST, return false);
    return true;
}

static bool CheckShape(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
                       const aclTensor *y, const aclTensor *post, const aclTensor *combFrag, int64_t numIters)
{
    OP_CHECK_SHAPE_NOT_EQUAL(x, y, return false);

    if (numIters <= 0 || numIters > 100) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "numIters value error, numIters must in 1 to 100, but got numIters = %ld .",
                numIters);
        return false;
    }

    auto xShape = x->GetViewShape();
    auto xDim = xShape.GetDimNum();
    if (xDim != 3 && xDim != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim value error, input dim must be 3 or 4, but got Dim = %ld .", xDim);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                      const aclTensor *bias, const aclTensor *y, const aclTensor *post,
                                      const aclTensor *combFrag, int64_t numIters)
{
    CHECK_RET(CheckNotNull(x, phi, alpha, bias, y, post, combFrag), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(x, phi, alpha, bias, y, post, combFrag), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(x, phi, alpha, bias, y, post, combFrag, numIters), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                                const aclTensor *bias, int64_t hcMult, int64_t numIters, float hcEps,
                                                float normEps, bool needBackward, aclTensor *y, aclTensor *post,
                                                aclTensor *combFrag, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMhcPreSinkhorn, DFX_IN(x, phi, alpha, bias, hcMult, numIters, hcEps, normEps, needBackward),
                   DFX_OUT(y, post, combFrag));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, phi, alpha, bias, y, post, combFrag, numIters);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support empty input tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const aclTensor *xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *phiContiguous = l0op::Contiguous(phi, uniqueExecutor.get());
    CHECK_RET(phiContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *alphaContiguous = l0op::Contiguous(alpha, uniqueExecutor.get());
    CHECK_RET(alphaContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *biasContiguous = l0op::Contiguous(bias, uniqueExecutor.get());
    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor *yContiguous = const_cast<aclTensor*>(l0op::Contiguous(y, uniqueExecutor.get()));
    CHECK_RET(yContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor *postContiguous = const_cast<aclTensor*>(l0op::Contiguous(post, uniqueExecutor.get()));
    CHECK_RET(postContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor *combFragContiguous = const_cast<aclTensor*>(l0op::Contiguous(combFrag, uniqueExecutor.get()));
    CHECK_RET(combFragContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *kernelOut =
        l0op::MhcPreSinkhorn(xContiguous, phiContiguous, alphaContiguous, biasContiguous, hcMult, numIters, hcEps,
                             normEps, needBackward, yContiguous, postContiguous, combFragContiguous,
                             uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(kernelOut, y, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPreSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPreSinkhorn);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
