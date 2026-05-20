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
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "aclnn_combine_normal.h"
#include "aclnnInner_combine_normal.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNullStatus(const aclTensor *context, const aclTensor *recvX, const aclTensor *tokenSrcInfo,
                            const aclTensor *epRecvCounts, const aclTensor *recvTopkWeights, const aclTensor *tokenIdx,
                            const aclTensor *xOut)
{
    OP_CHECK_NULL(context, return false);
    OP_CHECK_NULL(recvX, return false);
    OP_CHECK_NULL(tokenSrcInfo, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    OP_CHECK_NULL(recvTopkWeights, return false);
    OP_CHECK_NULL(tokenIdx, return false);
    OP_CHECK_NULL(xOut, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *context, const aclTensor *recvX, const aclTensor *tokenSrcInfo,
                               const aclTensor *epRecvCounts, const aclTensor *recvTopkWeights, const aclTensor *tokenIdx,
                               const aclTensor *tpRecvCountsOptional, int64_t epWorldSize, int64_t epRankId,
                               int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum, int64_t cclBufferSize,
                               int64_t realMaxBs, int32_t round, int32_t perRoundTokens, const aclTensor *xOut,
                               const aclTensor *combineSendCostStatsOutOptional)
{
    CHECK_RET(CheckNullStatus(context, recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, tokenIdx, xOut),
              ACLNN_ERR_PARAM_NULLPTR);
    (void)tpRecvCountsOptional;
    (void)epWorldSize;
    (void)epRankId;
    (void)tpWorldSize;
    (void)tpRankId;
    (void)moeExpertNum;
    (void)cclBufferSize;
    (void)realMaxBs;
    (void)round;
    (void)perRoundTokens;
    (void)combineSendCostStatsOutOptional;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCombineNormalGetWorkspaceSize(
    const aclTensor *context, const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCounts,
    const aclTensor *recvTopkWeights, const aclTensor *tokenIdx, const aclTensor *tpRecvCountsOptional,
    int64_t epWorldSize, int64_t epRankId, int64_t tpWorldSize, int64_t tpRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t realMaxBs, int32_t round, int32_t perRoundTokens, const aclTensor *xOut,
    const aclTensor *combineSendCostStatsOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_combine_normal WorkspaceSize start");
    auto retParam = CheckParams(context, recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, tokenIdx,
                                tpRecvCountsOptional, epWorldSize, epRankId, tpWorldSize, tpRankId, moeExpertNum,
                                cclBufferSize, realMaxBs, round, perRoundTokens, xOut, combineSendCostStatsOutOptional);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    return aclnnInnerCombineNormalGetWorkspaceSize(
        context, recvX, tokenSrcInfo, epRecvCounts, recvTopkWeights, tokenIdx, tpRecvCountsOptional, epWorldSize,
        epRankId, tpWorldSize, tpRankId, moeExpertNum, cclBufferSize, realMaxBs, round, perRoundTokens, xOut,
        combineSendCostStatsOutOptional, workspaceSize, executor);
}

aclnnStatus aclnnCombineNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_combine_normal start");
    return aclnnInnerCombineNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
