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
 * \file aclnn_notify_dispatch.cpp
 * \brief ACLNN API for notify_dispatch operator
 */

#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

// HCCL server type definition
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, int64_t sendCount, int64_t numTokens,
    const char *commGroup, int64_t rankSize, int64_t rankId, int64_t localRankSize, int64_t localRankId,
    aclTensor *sendDataOffset, aclTensor *recvData, aclTensor *totalRecvTokens, aclTensor *recvCount,
    aclTensor *recvOffset, aclTensor *maxBs, aclTensor *recvTokensPerExpert, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNotifyDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                            aclrtStream stream);

aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(const aclTensor *sendData, const aclTensor *tokenPerExpertData,
                                                int64_t sendCount, int64_t numTokens, const char *commGroup,
                                                int64_t rankSize, int64_t rankId, int64_t localRankSize,
                                                int64_t localRankId, aclTensor *sendDataOffset, aclTensor *recvData,
                                                aclTensor *totalRecvTokens, aclTensor *recvCount, aclTensor *recvOffset,
                                                aclTensor *maxBs, aclTensor *recvTokensPerExpert,
                                                uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_notify_dispatch WorkspaceSize start");

    OP_CHECK_NULL(sendData, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(tokenPerExpertData, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sendDataOffset, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvData, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(totalRecvTokens, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvCount, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvOffset, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(maxBs, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvTokensPerExpert, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerNotifyDispatchGetWorkspaceSize(
        sendData, tokenPerExpertData, sendCount, numTokens, commGroup, rankSize, rankId, localRankSize, localRankId,
        sendDataOffset, recvData, totalRecvTokens, recvCount, recvOffset, maxBs, recvTokensPerExpert, workspaceSize,
        executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnNotifyDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_notify_dispatch start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerNotifyDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif