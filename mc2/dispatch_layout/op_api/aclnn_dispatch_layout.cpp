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
#include "aclnn_dispatch_layout.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "opdev/data_type.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerDispatchLayoutGetWorkspaceSize(
    const aclTensor* topkIdx,
    int64_t num_tokens,
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_topk,
    int64_t local_ranksize,
    aclTensor* numTokensPerRank,
    aclTensor* numTokensPerExpert,
    aclTensor* isTokenInRank,
    aclTensor* notifySendData,
    aclTensor* sendTokenIdxSmall,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerDispatchLayout(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

static bool CheckNullStatus(const aclTensor* topkIdx,
                            aclTensor* numTokensPerRank,
                            aclTensor* numTokensPerExpert,
                            aclTensor* isTokenInRank,
                            aclTensor* notifySendData,
                            aclTensor* sendTokenIdxSmall)
{
    OP_CHECK_NULL(topkIdx, return false);
    OP_CHECK_NULL(numTokensPerRank, return false);
    OP_CHECK_NULL(numTokensPerExpert, return false);
    OP_CHECK_NULL(isTokenInRank, return false);
    OP_CHECK_NULL(notifySendData, return false);
    OP_CHECK_NULL(sendTokenIdxSmall, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* topkIdx,
                               int64_t num_tokens,
                               int64_t num_ranks,
                               int64_t num_experts,
                               int64_t num_topk)
{
    CHECK_RET(CheckNullStatus(topkIdx, nullptr, nullptr, nullptr, nullptr, nullptr),
              ACLNN_ERR_PARAM_NULLPTR);

    auto topkIdxDataType = topkIdx->GetDataType();
    OP_CHECK(topkIdxDataType == op::DataType::DT_INT64,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "topkIdx dtype must be INT64, but got %s.",
                     op::ToString(topkIdxDataType).c_str()),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(num_tokens > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "num_tokens must be greater than 0, but got %ld.",
                     num_tokens),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(num_ranks > 0 && num_ranks <= 384,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "num_ranks must be in [1, 384], but got %ld.",
                     num_ranks),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(num_experts > 0 && num_experts <= 512,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "num_experts must be in [1, 512], but got %ld.",
                     num_experts),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(num_experts % num_ranks == 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "num_experts (%ld) must be divisible by num_ranks (%ld).",
                     num_experts, num_ranks),
             return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK(num_topk > 0 && num_topk <= 16,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "num_topk must be in [1, 16], but got %ld.",
                     num_topk),
             return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDispatchLayoutGetWorkspaceSize(
    const aclTensor* topkIdx,
    int64_t num_tokens,
    int64_t num_ranks,
    int64_t num_experts,
    int64_t num_topk,
    int64_t local_ranksize,
    aclTensor* numTokensPerRank,
    aclTensor* numTokensPerExpert,
    aclTensor* isTokenInRank,
    aclTensor* notifySendData,
    aclTensor* sendTokenIdxSmall,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_LOGD("aclnn_dispatch_layout GetWorkspaceSize start");

    CHECK_RET(CheckNullStatus(topkIdx, numTokensPerRank, numTokensPerExpert,
                              isTokenInRank, notifySendData, sendTokenIdxSmall),
              ACLNN_ERR_PARAM_NULLPTR);

    aclnnStatus ret = CheckParams(topkIdx, num_tokens, num_ranks, num_experts, num_topk);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerDispatchLayoutGetWorkspaceSize(
        topkIdx, num_tokens, num_ranks, num_experts, num_topk, local_ranksize,
        numTokensPerRank, numTokensPerExpert, isTokenInRank, notifySendData,
        sendTokenIdxSmall, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnDispatchLayout(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch_layout start");
    return aclnnInnerDispatchLayout(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif