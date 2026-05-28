/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <string.h>
#include "graph/types.h"
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "aclnn_notify_dispatch.h"
#include "aclnnInner_notify_dispatch.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNullStatus(const aclTensor *sendData, const aclTensor *tokenPerExpertData, const aclTensor *sendDataOffset,
                            const aclTensor *recvData, const aclTensor *recvCount, const aclTensor *recvOffset,
                            const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset,
                            const aclTensor *rInSrcrankOffset, const aclTensor *totalRecvTokens, const aclTensor *maxBs,
                            const aclTensor *recvTokensPerExpert)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(sendData, return false);
    OP_CHECK_NULL(tokenPerExpertData, return false);
    OP_CHECK_NULL(sendDataOffset, return false);
    OP_CHECK_NULL(recvData, return false);
    OP_CHECK_NULL(recvCount, return false);
    OP_CHECK_NULL(recvOffset, return false);

    OP_CHECK_NULL(expertGlobalOffset, return false);
    OP_CHECK_NULL(srcrankInExpertOffset, return false);
    OP_CHECK_NULL(rInSrcrankOffset, return false);
    OP_CHECK_NULL(totalRecvTokens, return false);
    OP_CHECK_NULL(maxBs, return false);
    OP_CHECK_NULL(recvTokensPerExpert, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount,
                              int32_t numTokens, char *commGroup, int32_t rankSize, int32_t rankId,
                              int32_t localRankSize, int32_t localRankId, int32_t round, int32_t perRoundTokens,
                              const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensor *recvCount,
                              const aclTensor *recvOffset, const aclTensor *expertGlobalOffset,
                              const aclTensor *srcrankInExpertOffset, const aclTensor *rInSrcrankOffset,
                              const aclTensor *totalRecvTokens, const aclTensor *maxBs, const aclTensor *recvTokensPerExpert)
{
    // 判空
    CHECK_RET(CheckNullStatus(sendData, tokenPerExpertData, sendDataOffset, recvData, recvCount, recvOffset,
                              expertGlobalOffset, srcrankInExpertOffset, rInSrcrankOffset, totalRecvTokens, maxBs,
                              recvTokensPerExpert),
              ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount, int32_t numTokens,
    char *commGroup, int32_t rankSize, int32_t rankId, int32_t localRankSize, int32_t localRankId, int32_t round,
    int32_t perRoundTokens, const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensor *recvCount,
    const aclTensor *recvOffset, const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset,
    const aclTensor *rInSrcrankOffset, const aclTensor *totalRecvTokens, const aclTensor *maxBs,
    const aclTensor *recvTokensPerExpert, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_notify_dispatch WorkspaceSize start");
    auto retParam = CheckParams(sendData, tokenPerExpertData, sendCount, numTokens, commGroup, rankSize,
                                rankId, localRankSize, localRankId, round, perRoundTokens, sendDataOffset,
                                recvData, recvCount, recvOffset, expertGlobalOffset, srcrankInExpertOffset,
                                rInSrcrankOffset, totalRecvTokens, maxBs, recvTokensPerExpert);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerNotifyDispatchGetWorkspaceSize(
        sendData, tokenPerExpertData, sendCount, numTokens, commGroup, rankSize, rankId, localRankSize, localRankId,
        round, perRoundTokens, sendDataOffset, recvData, recvCount, recvOffset, expertGlobalOffset,
        srcrankInExpertOffset, rInSrcrankOffset, totalRecvTokens, maxBs, recvTokensPerExpert, workspaceSize, executor);
    
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
