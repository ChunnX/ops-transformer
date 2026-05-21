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
 * \file aclnn_moe_combine_normal.cpp
 * \brief ACLNN API for moe_combine_normal operator
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

extern aclnnStatus aclnnInnerMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCount, const aclTensor *topkWeights,
    const aclTensor *tpRecvCountOptional, const char *epGroupName, int64_t epWorldSize, int64_t epRankId,
    const char *tpGroupNameOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t globalBs,
    aclTensor *xOut, aclTensor *sendCostStatsOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeCombineNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream);

aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCount, const aclTensor *topkWeights,
    const aclTensor *tpRecvCountOptional, const char *epGroupName, int64_t epWorldSize, int64_t epRankId,
    const char *tpGroupNameOptional, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t globalBs,
    aclTensor *xOut, aclTensor *sendCostStatsOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_moe_combine_normal WorkspaceSize start");

    OP_CHECK_NULL(recvX, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(tokenSrcInfo, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(epRecvCount, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(topkWeights, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(xOut, return ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerMoeCombineNormalGetWorkspaceSize(
        recvX, tokenSrcInfo, epRecvCount, topkWeights, tpRecvCountOptional, epGroupName, epWorldSize, epRankId,
        tpGroupNameOptional, tpWorldSize, tpRankId, moeExpertNum, globalBs, xOut, sendCostStatsOutOptional,
        workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeCombineNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_moe_combine_normal start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeCombineNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif