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
 * \file aclnn_tri_inverse.cpp
 * \brief
 */

#include "aclnn_tri_inverse.h"
#include "tri_inverse.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct TriInverseParams {
    const aclTensor *ccIn{nullptr};
    const aclTensor *ccOut{nullptr};
};

static const std::initializer_list<op::DataType> BF16_SUPPORT_LIST = {op::DataType::DT_BF16};

static inline bool CheckNotNull(const TriInverseParams &params)
{
    OP_CHECK_NULL(params.ccIn, return false);
    OP_CHECK_NULL(params.ccOut, return false);
    return true;
}

static inline bool CheckDtypeValid(const TriInverseParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ccIn, BF16_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.ccOut, BF16_SUPPORT_LIST, return false);
    return true;
}

static aclnnStatus CheckParams(const TriInverseParams &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("TriInverse check params success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(const TriInverseParams &params)
{
    params.ccIn->SetOriginalShape(params.ccIn->GetViewShape());
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnTriInverseGetWorkspaceSize(const aclTensor *ccIn, const aclTensor *ccOut,
                                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnTriInverse, DFX_IN(ccIn), DFX_OUT(ccOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    TriInverseParams params{ccIn, ccOut};
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (ccOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ccInContiguous = l0op::Contiguous(ccIn, uniqueExecutor.get());
    CHECK_RET(ccInContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto ccOutContiguous = l0op::Contiguous(ccOut, uniqueExecutor.get());
    CHECK_RET(ccOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::TriInverse(ccInContiguous, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, ccOutContiguous, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTriInverse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTriInverse);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
