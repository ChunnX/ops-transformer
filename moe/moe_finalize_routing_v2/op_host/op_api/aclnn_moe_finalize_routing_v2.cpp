/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_moe_finalize_routing_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "common/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"

#ifdef __cplusplus
extern "C" {
#endif
 extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2GetWorkspaceSize(
 	     const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
 	     const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
 	     const aclTensor* expertIdxOptional, const aclTensor* xOptional, const aclTensor* a1Optional,
 	     const aclTensor* a2Optional, const aclTensor* vOptional, int64_t dropPadMode, const aclIntArray* zeroExpertRange,
 	     const aclIntArray* copyExpertRange, const aclIntArray* constantExpertRange, const aclTensor* out,
 	     uint64_t* workspaceSize, aclOpExecutor** executor);
extern aclnnStatus aclnnInnerMoeFinalizeRoutingV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(
        const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* x1Optional,
        const aclTensor* x2Optional, const aclTensor* biasOptional, const aclTensor* scalesOptional,
        const aclTensor* expertIdxOptional, const aclTensor* xOptional, const aclTensor* a1Optional,
        const aclTensor* a2Optional, const aclTensor* vOptional, int64_t dropPadMode, const aclIntArray* zeroExpertRange,
        const aclIntArray* copyExpertRange, const aclIntArray* constantExpertRange, const aclTensor* out,
        uint64_t* workspaceSize, aclOpExecutor** executor)
 {
    return aclnnInnerMoeFinalizeRoutingV2GetWorkspaceSize(
        expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional, expertIdxOptional, xOptional,
        a1Optional, a2Optional, vOptional, dropPadMode, zeroExpertRange, copyExpertRange, constantExpertRange, out,
        workspaceSize, executor);
}

aclnnStatus aclnnMoeFinalizeRoutingV2(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerMoeFinalizeRoutingV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif