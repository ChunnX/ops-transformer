/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_broadcast_nddma_bench.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnnInner_broadcast_nddma_bench.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnBroadcastNddmaBenchGetWorkspaceSize(
    const aclTensor* x, int64_t isMte3, int64_t scene, int64_t dim2, int64_t dim3, int64_t dim4,
    aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    return aclnnInnerBroadcastNddmaBenchGetWorkspaceSize(
        x, isMte3, scene, dim2, dim3, dim4, y, workspaceSize, executor);
}

aclnnStatus aclnnBroadcastNddmaBench(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerBroadcastNddmaBench(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
