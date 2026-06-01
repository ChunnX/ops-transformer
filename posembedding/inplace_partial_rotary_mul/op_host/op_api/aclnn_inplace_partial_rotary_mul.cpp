/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_inplace_partial_rotary_mul.h"
#include "inplace_partial_rotary_mul.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus InplacePartialRotaryMulCommonProcess(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
                                                         int64_t rotary_mode, const aclIntArray *partialSlice,
                                                         aclTensor *out, aclOpExecutor *executor)
{
    auto result = l0op::InplacePartialRotaryMul(x, cos, sin, rotary_mode, partialSlice, executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplacePartialRotaryMulGetWorkspaceSize(const aclTensor* x, const aclTensor* cos, const aclTensor* sin,
                                                          int64_t rotary_mode, const aclIntArray *partialSlice,
                                                          aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplacePartialRotaryMul,
        DFX_IN(x, cos, sin, rotary_mode, partialSlice),
        DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto ret = InplacePartialRotaryMulCommonProcess(x, cos, sin, rotary_mode, partialSlice, out, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplacePartialRotaryMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplacePartialRotaryMul);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
