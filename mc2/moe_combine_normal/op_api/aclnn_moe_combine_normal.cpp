/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_combine_normal.h"
#include "aclnnInner_moe_combine_normal.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

static aclnnStatus CheckNotNull(const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCounts,
                                const aclTensor *recvTopkWeights, const aclTensor *xOut, const char *epGroupName)
{
    OP_LOGD("aclnn_moe_combine_normal CheckNotNull start");
    if (recvX == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "recvX is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (tokenSrcInfo == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "tokenSrcInfo is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (epRecvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "epRecvCounts is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (recvTopkWeights == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "recvTopkWeights is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (xOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "xOut is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (epGroupName == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "epGroupName is nullptr.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    OP_LOGD("aclnn_moe_combine_normal CheckNotNull success");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCounts,
    const aclTensor *recvTopkWeights, const aclTensor *tpRecvCountsOptional, const char *epGroupName,
    int64_t epWorldSize, int64_t epRankId, const char *tpGroupNameOptional, int64_t tpWorldSize, int64_t tpRankId,
    int64_t moeExpertNum, int64_t globalBs, const aclTensor *xOut, const aclTensor *combineSendCostStatsOutOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret_param = CheckNotNull(recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, xOut, epGroupName);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerMoeCombineNormalGetWorkspaceSize(
        recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, tpRecvCountsOptional, const_cast<char *>(epGroupName),
        epWorldSize, epRankId, const_cast<char *>(tpGroupNameOptional), tpWorldSize, tpRankId, moeExpertNum, globalBs,
        xOut, combineSendCostStatsOutOptional, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMoeCombineNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    aclnnStatus ret = aclnnInnerMoeCombineNormal(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif