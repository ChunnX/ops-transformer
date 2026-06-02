/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

namespace {
constexpr int64_t RECV_OFFSET_SIZE = 2;
constexpr int64_t EXPERT_OUTPUT_OFFSET_SIZE = 3;
}  // namespace

static bool CheckNullStatus(const aclTensor *context, const aclTensor *sendData, const aclTensor *tokenPerExpertData,
                            const aclTensor *sendDataOffset, const aclTensor *recvData, const aclTensorList *recvOffset,
                            const aclTensorList *expertOutputOffset, const aclTensor *totalRecvTokens,
                            const aclTensor *maxBs, const aclTensor *recvTokensPerExpert)
{
    OP_CHECK_NULL(context, return false);
    OP_CHECK_NULL(sendData, return false);
    OP_CHECK_NULL(tokenPerExpertData, return false);
    OP_CHECK_NULL(sendDataOffset, return false);
    OP_CHECK_NULL(recvData, return false);
    OP_CHECK_NULL(recvOffset, return false);
    OP_CHECK_NULL(expertOutputOffset, return false);
    OP_CHECK_NULL(totalRecvTokens, return false);
    OP_CHECK_NULL(maxBs, return false);
    OP_CHECK_NULL(recvTokensPerExpert, return false);
    return true;
}

static aclnnStatus CheckExtHandleData(const aclTensor *extHandleData)
{
    if (extHandleData == nullptr) {
        return ACLNN_SUCCESS;
    }
    CHECK_COND(extHandleData->GetDataType() == op::DataType::DT_INT32, ACLNN_ERR_PARAM_INVALID,
               "ext_handle_data dtype must be int32.");
    CHECK_COND(extHandleData->GetViewShape().GetDimNum() == 1, ACLNN_ERR_PARAM_INVALID,
               "ext_handle_data must be 1D tensor.");
    CHECK_COND(extHandleData->GetViewShape().GetDim(0) == 0, ACLNN_ERR_PARAM_INVALID,
               "ext_handle_data shape[0] must be 0.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor *context, const aclTensor *sendData, const aclTensor *tokenPerExpertData,
                               int32_t sendCount, int32_t numTokens, int32_t epWorldSize, int32_t epRankId,
                               int32_t localRankSize, int32_t localRankId, int64_t cclBufferSize, int32_t round,
                               int32_t perRoundTokens, const aclTensor *sendDataOffset, const aclTensor *recvData,
                               const aclTensorList *recvOffset, const aclTensorList *expertOutputOffset,
                               const aclTensor *totalRecvTokens, const aclTensor *maxBs,
                               const aclTensor *recvTokensPerExpert, const aclTensor *extHandleData)
{
    CHECK_RET(CheckNullStatus(context, sendData, tokenPerExpertData, sendDataOffset, recvData, recvOffset,
                              expertOutputOffset, totalRecvTokens, maxBs, recvTokensPerExpert),
              ACLNN_ERR_PARAM_NULLPTR);

    CHECK_COND(recvOffset->Size() == RECV_OFFSET_SIZE, ACLNN_ERR_PARAM_INVALID,
               "recv_offset must contain exactly 2 tensors, but got %lu.", recvOffset->Size());
    CHECK_COND((*recvOffset)[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR, "recv_offset[0] must not be nullptr.");
    CHECK_COND((*recvOffset)[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR, "recv_offset[1] must not be nullptr.");

    CHECK_COND(expertOutputOffset->Size() == EXPERT_OUTPUT_OFFSET_SIZE, ACLNN_ERR_PARAM_INVALID,
               "expert_output_offset must contain exactly 3 tensors, but got %lu.", expertOutputOffset->Size());
    CHECK_COND((*expertOutputOffset)[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "expert_output_offset[0] must not be nullptr.");
    CHECK_COND((*expertOutputOffset)[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "expert_output_offset[1] must not be nullptr.");
    CHECK_COND((*expertOutputOffset)[2] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "expert_output_offset[2] must not be nullptr.");

    CHECK_RET(CheckExtHandleData(extHandleData) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

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
    const aclTensor *recvData, const aclTensorList *recvOffset, const aclTensorList *expertOutputOffset,
    const aclTensor *totalRecvTokens, const aclTensor *maxBs, const aclTensor *recvTokensPerExpert,
    const aclTensor *extHandleData, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_notify_dispatch WorkspaceSize start");
    auto retParam = CheckParams(context, sendData, tokenPerExpertData, sendCount, numTokens, epWorldSize, epRankId,
                                localRankSize, localRankId, cclBufferSize, round, perRoundTokens, sendDataOffset,
                                recvData, recvOffset, expertOutputOffset, totalRecvTokens, maxBs, recvTokensPerExpert,
                                extHandleData);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    return aclnnInnerNotifyDispatchGetWorkspaceSize(
        context, sendData, tokenPerExpertData, sendCount, numTokens, epWorldSize, epRankId, localRankSize,
        localRankId, cclBufferSize, round, perRoundTokens, sendDataOffset, recvData, recvOffset, expertOutputOffset,
        totalRecvTokens, maxBs, recvTokensPerExpert, extHandleData, workspaceSize, executor);
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
