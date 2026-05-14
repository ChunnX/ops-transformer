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
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "common/op_host/op_api/matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace Ops::Transformer;
using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, char *groupEp, int64_t epWorldSize, int64_t epRankId,
    char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode,
    int64_t globalBs, const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats, uint64_t* workspaceSize, aclOpExecutor** executor);


extern aclnnStatus aclnnInnerMoeDispatchNormal(void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);

aclnnStatus aclnnMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, char *groupEp, int64_t epWorldSize, int64_t epRankId,
    char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode,
    int64_t globalBs, const aclTensor *recvX, const aclTensor *recvXScales, const aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStats, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_moe_dispatch_normal WorkspaceSize start");

    // OP_CHECK_NULL(context, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkIdx, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sendOffset, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sendTokenIdx, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvOffset, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvCount, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvXScales, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(assistInfoForCombine, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(waitRecvCostStats, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMoeDispatchNormalGetWorkspaceSize(
        x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount, groupEp, epWorldSize, epRankId,
        groupTpOptional, tpWorldSize, tpRankId, moeExpertNum, quantMode, globalBs, recvX, recvXScales,
        assistInfoForCombine, waitRecvCostStats, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDispatchNormal(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_moe_dispatch_normal start");
    return aclnnInnerMoeDispatchNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif