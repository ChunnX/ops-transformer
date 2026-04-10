/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file aclnn_mhc_post_backward.cpp
 * \brief MhcPostBackward ACLNN API implementation
 */

#include "aclnn_mhc_post_backward.h"
#include "mhc_post_backward.h"

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

static const int64_t DIM_NUM_2 = 2;
static const int64_t DIM_NUM_3 = 3;
static const int64_t DIM_NUM_4 = 4;
static const int64_t DIM_IDX_0 = 0;
static const int64_t DIM_IDX_1 = 1;
static const int64_t DIM_IDX_2 = 2;
static const int64_t DIM_IDX_3 = 3;

struct MhcPostBackwardParams {
    const aclTensor *gradOutput = nullptr;
    const aclTensor *x = nullptr;
    const aclTensor *hRes = nullptr;
    const aclTensor *hOut = nullptr;
    const aclTensor *hPost = nullptr;
    aclTensor *gradX = nullptr;
    aclTensor *gradHres = nullptr;
    aclTensor *gradHout = nullptr;
    aclTensor *gradHpost = nullptr;
};

// gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost
aclnnStatus aclnnMhcPostBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                                        const aclTensor *hPost, aclTensor *gradX, aclTensor *gradHres, aclTensor *gradHout, aclTensor *gradHpost, 
                                        uint64_t *workspaceSize,
                                        aclOpExecutor **executor)
{
    // CHECK_COND(CheckNotNull(gradOutput, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
    //            "one of required inputs for aclnnMhcPostBackwardGetWorkspaceSize is nullptr.");
    MhcPostBackwardParams params{gradOutput, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost};

    // aclnnStatus ret = CheckParam(params);
    // CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Check if input tensors are empty
    if (gradOutput->IsEmpty() || x->IsEmpty() || hRes->IsEmpty() || hOut->IsEmpty() || hPost->IsEmpty()) {

        OP_LOGE(ACLNN_ERR_INNER, "aclnn_mhc_post_backward do not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;



    }

    L2_DFX_PHASE_1(aclnnMhcPostBackward, DFX_IN(params.gradOutput, params.x, params.hRes, params.hOut, params.hPost), DFX_OUT(params.gradX, params.gradHres, params.gradHout, params.gradHpost));

    // Convert input tensors to contiguous
    auto reformatedGradOutput = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    CHECK_COND(reformatedGradOutput != nullptr, ACLNN_ERR_INNER_NULLPTR, "reformatedGradOutput Contiguous failed.");
    auto reformatedX = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(reformatedX != nullptr, ACLNN_ERR_INNER_NULLPTR, "x Contiguous failed.");
    auto reformatedHRes = l0op::Contiguous(hRes, uniqueExecutor.get());
    CHECK_COND(reformatedHRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "hRes Contiguous failed.");
    auto reformatedHOut = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_COND(reformatedHOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "hOut Contiguous failed.");
    auto reformatedHPost = l0op::Contiguous(hPost, uniqueExecutor.get());
    CHECK_COND(reformatedHPost != nullptr, ACLNN_ERR_INNER_NULLPTR, "hPost Contiguous failed.");

    // Call l0 interface: MhcPostBackward kernel
    const aclTensor *Result =
        l0op::MhcPostBackward(reformatedGradOutput, reformatedX, reformatedHRes, reformatedHOut, reformatedHPost, uniqueExecutor.get());
    
    CHECK_RET(std::get<0>(Result) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    outGradX = std::get<0>(Result);
    outGradHres = std::get<1>(Result);
    outGradHout = std::get<2>(Result);
    outGradHpost = std::get<3>(Result);
    CHECK_RET(outGradHres != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outGradHout != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(outGradHpost != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // gradOutput, x, hRes, hOut, hPost, gradX, gradHres, gradHout, gradHpost
    // Convert output tensor to contiguous tensor and copy to output
    auto viewCopygradXResult = l0op::ViewCopy(outGradX, gradX, uniqueExecutor.get());
    CHECK_RET(viewCopygradXResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto viewCopygradHresResult = l0op::ViewCopy(outGradHres, gradHres, uniqueExecutor.get());
    CHECK_RET(viewCopygradHresResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto viewCopygradHoutResult = l0op::ViewCopy(outGradHout, gradHout, uniqueExecutor.get());
    CHECK_RET(viewCopygradHoutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    
    auto viewCopygradHpostResult = l0op::ViewCopy(outGradHpost, gradHpost, uniqueExecutor.get());
    CHECK_RET(viewCopygradHpostResult != nullptr, ACLNN_ERR_INNER_NULLPTR);


    // Get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPostBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPostBackward);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in MhcPostBackward launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif