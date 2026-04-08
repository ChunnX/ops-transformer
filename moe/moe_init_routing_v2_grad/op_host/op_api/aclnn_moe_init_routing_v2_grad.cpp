/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_init_routing_v2_grad.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe_init_routing_v2_grad.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx, int64_t topK, int64_t dropPadMode,
    int64_t activeNum, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeInitRoutingV2Grad,
        DFX_IN(gradExpandedX, expandedRowIdx, topK, dropPadMode, activeNum),
        DFX_OUT(out));

    // 参数检查
    OP_CHECK_NULL(gradExpandedX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(expandedRowIdx, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入转换成连续的tensor
    auto gradExpandedXContiguous = l0op::Contiguous(gradExpandedX, uniqueExecutor.get());
    CHECK_RET(gradExpandedXContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto expandedRowIdxContiguous = l0op::Contiguous(expandedRowIdx, uniqueExecutor.get());
    CHECK_RET(expandedRowIdxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0接口进行计算，传入out参数
    auto out_ = l0op::MoeInitRoutingV2Grad(gradExpandedXContiguous, expandedRowIdxContiguous,
        topK, dropPadMode, activeNum,
        out, uniqueExecutor.get());
    CHECK_RET(out_ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyOutResult = l0op::ViewCopy(out_, out, uniqueExecutor.get());
    CHECK_RET(viewCopyOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeInitRoutingV2Grad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
