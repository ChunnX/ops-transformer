/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
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

static bool CheckNullStatus(const aclTensor *context, const aclTensor *sendData, const aclTensor *tokenPerExpertData,
                            const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensor *recvCount,
                            const aclTensor *recvOffset, const aclTensor *expertGlobalOffset,
                            const aclTensor *srcrankInExpertOffset, const aclTensor *rInSrcrankOffset,
                            const aclTensor *totalRecvTokens, const aclTensor *maxBs,
                            const aclTensor *recvTokensPerExpert)
{
    OP_CHECK_NULL(context, return false);
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

static aclnnStatus CheckParams(const aclTensor *context, const aclTensor *sendData, const aclTensor *tokenPerExpertData,
                               int32_t sendCount, int32_t numTokens, int32_t epWorldSize, int32_t epRankId,
                               int32_t localRankSize, int32_t localRankId, int64_t cclBufferSize, int32_t round,
                               int32_t perRoundTokens, const aclTensor *sendDataOffset, const aclTensor *recvData,
                               const aclTensor *recvCount, const aclTensor *recvOffset,
                               const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset,
                               const aclTensor *rInSrcrankOffset, const aclTensor *totalRecvTokens,
                               const aclTensor *maxBs, const aclTensor *recvTokensPerExpert)
{
    CHECK_RET(CheckNullStatus(context, sendData, tokenPerExpertData, sendDataOffset, recvData, recvCount, recvOffset,
                              expertGlobalOffset, srcrankInExpertOffset, rInSrcrankOffset, totalRecvTokens, maxBs,
                              recvTokensPerExpert),
              ACLNN_ERR_PARAM_NULLPTR);
    (void)sendCount;
    (void)numTokens;
    (void)epWorldSize;
    (void)epRankId;
    (void)localRankSize;
    (void)localRankId;
    (void)cclBufferSize;
    (void)round;
    (void)perRoundTokens;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *context, const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount,
    int32_t numTokens, int32_t epWorldSize, int32_t epRankId, int32_t localRankSize, int32_t localRankId,
    int64_t cclBufferSize, int32_t round, int32_t perRoundTokens, const aclTensor *sendDataOffset,
    const aclTensor *recvData, const aclTensor *recvCount, const aclTensor *recvOffset,
    const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset, const aclTensor *rInSrcrankOffset,
    const aclTensor *totalRecvTokens, const aclTensor *maxBs, const aclTensor *recvTokensPerExpert,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_notify_dispatch WorkspaceSize start");
    auto retParam = CheckParams(context, sendData, tokenPerExpertData, sendCount, numTokens, epWorldSize, epRankId,
                                localRankSize, localRankId, cclBufferSize, round, perRoundTokens, sendDataOffset,
                                recvData, recvCount, recvOffset, expertGlobalOffset, srcrankInExpertOffset,
                                rInSrcrankOffset, totalRecvTokens, maxBs, recvTokensPerExpert);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    return aclnnInnerNotifyDispatchGetWorkspaceSize(
        context, sendData, tokenPerExpertData, sendCount, numTokens, epWorldSize, epRankId, localRankSize,
        localRankId, cclBufferSize, round, perRoundTokens, sendDataOffset, recvData, recvCount, recvOffset,
        expertGlobalOffset, srcrankInExpertOffset, rInSrcrankOffset, totalRecvTokens, maxBs, recvTokensPerExpert,
        workspaceSize, executor);
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
