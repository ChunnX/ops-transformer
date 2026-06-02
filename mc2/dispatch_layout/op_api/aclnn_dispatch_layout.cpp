/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "aclnn_dispatch_layout.h"
#include "aclnnInner_dispatch_layout.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

namespace {
constexpr int64_t EXT_HANDLE_DATA_SIZE = 2;
}

static bool CheckNullStatus(const aclTensor *topkIdx, const aclTensor *numTokensPerRank,
                            const aclTensor *numTokensPerExpert, const aclTensor *isTokenInRank,
                            const aclTensorList *extHandleData)
{
    OP_CHECK_NULL(topkIdx, return false);
    OP_CHECK_NULL(numTokensPerRank, return false);
    OP_CHECK_NULL(numTokensPerExpert, return false);
    OP_CHECK_NULL(isTokenInRank, return false);
    OP_CHECK_NULL(extHandleData, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *topkIdx, int64_t numTokens, int64_t numRanks, int64_t numExperts,
                               int64_t numTopk, int64_t localRankSize, int32_t perRoundTokens, int32_t rankId,
                               const aclTensor *numTokensPerRank, const aclTensor *numTokensPerExpert,
                               const aclTensor *isTokenInRank, const aclTensorList *extHandleData)
{
    CHECK_RET(CheckNullStatus(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, extHandleData),
              ACLNN_ERR_PARAM_NULLPTR);

    if (numTopk > numExperts) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "num of Topk should be less than num of moe experts.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    CHECK_COND(extHandleData->Size() == EXT_HANDLE_DATA_SIZE, ACLNN_ERR_PARAM_INVALID,
               "ext_handle_data must contain exactly 2 tensors, but got %lu.", extHandleData->Size());
    CHECK_COND((*extHandleData)[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR, "ext_handle_data[0] must not be nullptr.");
    CHECK_COND((*extHandleData)[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR, "ext_handle_data[1] must not be nullptr.");

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDispatchLayoutGetWorkspaceSize(const aclTensor *topkIdx, int64_t numTokens, int64_t numRanks,
                                                int64_t numExperts, int64_t numTopk, int64_t localRankSize,
                                                int32_t perRoundTokens, int32_t rankId,
                                                const aclTensor *numTokensPerRank, const aclTensor *numTokensPerExpert,
                                                const aclTensor *isTokenInRank, const aclTensorList *extHandleData,
                                                uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("aclnn_dispatch_layout WorkspaceSize start");
    auto retParam = CheckParams(topkIdx, numTokens, numRanks, numExperts, numTopk, localRankSize, perRoundTokens, rankId,
                                numTokensPerRank, numTokensPerExpert, isTokenInRank, extHandleData);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclnnStatus getWorkspaceSizesRes = aclnnInnerDispatchLayoutGetWorkspaceSize(
        topkIdx, numTokens, numRanks, numExperts, numTopk, localRankSize, perRoundTokens, rankId, numTokensPerRank,
        numTokensPerExpert, isTokenInRank, extHandleData, workspaceSize, executor);

    return getWorkspaceSizesRes;
}

aclnnStatus aclnnDispatchLayout(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch_layout start");
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDispatchLayout(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
