/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_NOTIFY_DISPATCH_H_
#define ACLNN_NOTIFY_DISPATCH_H_

// #include "aclnn/acl_meta.h"
#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

/* function: aclnnNotifyDispatchGetWorkspaceSize
 * parameters :
 * sendData : required
 * tokenPerExpertData : required
 * sendCount : required
 * numTokens : required
 * commGroup : required
 * rankSize : required
 * rankId : required
 * localRankSize : required
 * localRankId : required
 * round : required
 * perRoundTokens : required
 * sendDataOffset : required
 * recvData : required
 * recvCount : required
 * recvOffset : required
 * expertGlobalOffset : required
 * srcrankInExpertOffset : required
 * rInSrcrankOffset : required
 * totalRecvTokens : required
 * maxBs : required
 * recvTokensPerExpert : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
ACLNN_API aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount, int32_t numTokens,
    char *commGroup, int32_t rankSize, int32_t rankId, int32_t localRankSize, int32_t localRankId, int32_t round,
    int32_t perRoundTokens, const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensor *recvCount,
    const aclTensor *recvOffset, const aclTensor *expertGlobalOffset, const aclTensor *srcrankInExpertOffset,
    const aclTensor *rInSrcrankOffset, const aclTensor *totalRecvTokens, const aclTensor *maxBs,
    const aclTensor *recvTokensPerExpert, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnNotifyDispatch的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnNotifyDispatchGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnNotifyDispatch(void *workspace, uint64_t workspaceSize,
                                          aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_NOTIFY_DISPATCH_
