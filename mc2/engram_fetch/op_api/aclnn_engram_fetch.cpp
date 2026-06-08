/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_engram_fetch.h"
#include <algorithm>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "aclnnInner_engram_fetch.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static bool CheckNotNull(const aclTensor *bufferHandle, const aclTensor *indices, aclTensor *fetched)
{
    OP_CHECK_NULL(bufferHandle, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(fetched, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *bufferHandle, const aclTensor *indices, aclTensor *fetched)
{
    CHECK_RET(CheckNotNull(bufferHandle, indices, fetched), ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnEngramFetchGetWorkspaceSize(const aclTensor *bufferHandle, const aclTensor *indices,
                                               aclTensor *fetched, int32_t hiddenSize,
                                               int64_t numEntriesPerRank, const char *group,
                                               int64_t outputDtype,
                                               uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto retParam = CheckParams(bufferHandle, indices, fetched);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret = aclnnInnerEngramFetchGetWorkspaceSize(
        bufferHandle, indices, hiddenSize, numEntriesPerRank,
        const_cast<char *>(group), outputDtype, fetched, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnEngramFetch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerEngramFetch(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif