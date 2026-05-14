/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

extern "C" {
aclnnStatus aclnnInnerDistributeBarrierExtend(void * /* workspace */, uint64_t /* workspaceSize */,
    aclOpExecutor * /* executor */, aclrtStream /* stream */)
{
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInnerDistributeBarrierExtendGetWorkspaceSize(const aclTensor * /* context */,
    const aclTensor * /* xRef */, const aclTensor * /* timeOut */, const aclTensor * /* elasticInfo */,
    const char * /* group */, int64_t /* worldSize */, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (workspaceSize != nullptr) {
        *workspaceSize = 0;
    }
    if (executor != nullptr) {
        *executor = nullptr;
    }
    return ACLNN_SUCCESS;
}
}
