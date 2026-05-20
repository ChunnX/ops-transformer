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
 * \file aclnn_moe_dispatch_normal.cpp
 * \brief ACLNN API for moe_dispatch_normal operator
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

extern aclnnStatus aclnnInnerMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    const char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode,
    int64_t globalBs, aclTensor *recvX, aclTensor *recvXScales, aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStatsOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                               aclrtStream stream);

aclnnStatus aclnnMoeDispatchNormalGetWorkspaceSize(
    const aclTensor *x, const aclTensor *topkIdx, const aclTensor *sendOffset, const aclTensor *sendTokenIdx,
    const aclTensor *recvOffset, const aclTensor *recvCount, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    const char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t quantMode,
    int64_t globalBs, aclTensor *recvX, aclTensor *recvXScales, aclTensor *assistInfoForCombine,
    const aclTensor *waitRecvCostStatsOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_moe_dispatch_normal WorkspaceSize start");

    OP_CHECK_NULL(x, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkIdx, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sendOffset, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(sendTokenIdx, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvOffset, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvCount, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(recvXScales, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(assistInfoForCombine, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMoeDispatchNormalGetWorkspaceSize(
        x, topkIdx, sendOffset, sendTokenIdx, recvOffset, recvCount, groupEp, epWorldSize, epRankId, groupTpOptional,
        tpWorldSize, tpRankId, moeExpertNum, quantMode, globalBs, recvX, recvXScales, assistInfoForCombine,
        waitRecvCostStatsOutOptional, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDispatchNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_moe_dispatch_normal start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeDispatchNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif