/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_NOTIFY_DISPATCH_H_
#define ACLNN_NOTIFY_DISPATCH_H_

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

/**
 * @brief Get NotifyDispatch workspace size.
 *
 * recv_offset: 2 tensors [recvOffset, recvCount].
 * expert_output_offset: 3 tensors [expertGlobalOffset, srcrankInExpertOffset, rInSrcrankOffset].
 * ext_handle_data: optional placeholder, 1D int32 with numel 0 when provided.
 */
ACLNN_API aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *context, const aclTensor *sendData, const aclTensor *tokenPerExpertData, int32_t sendCount,
    int32_t numTokens, int32_t epWorldSize, int32_t epRankId, int32_t localRankSize, int32_t localRankId,
    int64_t cclBufferSize, int32_t round, int32_t perRoundTokens, const aclTensor *sendDataOffset,
    const aclTensor *recvData, const aclTensorList *recvOffset, const aclTensorList *expertOutputOffset,
    const aclTensor *totalRecvTokens, const aclTensor *maxBs, const aclTensor *recvTokensPerExpert,
    const aclTensor *extHandleData, uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnNotifyDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_NOTIFY_DISPATCH_H_
