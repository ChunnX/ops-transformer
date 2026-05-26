/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string.h>
#include "graph/types.h"
#include "aclnn_moe_combine_normal.h"
#include "aclnnInner_moe_combine_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCounts,
    const aclTensor *recvTopkWeights, const aclTensor *tokenIdx, const aclTensor *tpRecvCountsOptional,
    char *epGroupName, int64_t epWorldSize, int64_t epRankId, char *tpGroupNameOptional, int64_t tpWorldSize,
    int64_t tpRankId, int64_t moeExpertNum, int64_t realMaxBs, int32_t round, int32_t per_round_tokens,
    const aclTensor *out, const aclTensor *sendCostStats, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerMoeCombineNormalGetWorkspaceSize(
        recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, tokenIdx, tpRecvCountsOptional, epGroupName, epWorldSize,
        epRankId, tpGroupNameOptional, tpWorldSize, tpRankId, moeExpertNum, realMaxBs, round, per_round_tokens, out,
        sendCostStats, workspaceSize, executor);
}

aclnnStatus aclnnMoeCombineNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerMoeCombineNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif