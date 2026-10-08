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
 * \file aclnn_engram_fetch.cpp
 * \brief engram_fetch 算子 aclnn 接口实现
 */

#include <algorithm>
#include "aclnn/aclnn_base.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "aclnnInner_engram_fetch.h"
#include "engram_fetch_mode_def.h"

using namespace op;

static bool CheckNotNull(const aclTensor* commContext, const aclTensor* indices, const aclTensor* fetched)
{
    OP_CHECK_NULL(commContext, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(fetched, return false);
    return true;
}

// fetch_mode: 0 = sliced cross-NPU (default, train/infer), 1 = local full table (infer only)
static bool CheckFetchMode(int64_t withGrad, int64_t fetchMode)
{
    if (fetchMode != ENGRAM_FETCH_MODE_SLICED && fetchMode != ENGRAM_FETCH_MODE_LOCAL) {
        OP_LOGE_WITH_INVALID_ATTR("aclnnEngramFetch", "fetch_mode", std::to_string(fetchMode).c_str(), "[0, 1]");
        return false;
    }
    if (withGrad != 0 && fetchMode == ENGRAM_FETCH_MODE_LOCAL) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "fetch_mode=1 (local, full replication) only supports inference, with_grad=1 is rejected, "
                "got with_grad=%ld, fetch_mode=%ld.",
                withGrad, fetchMode);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* commContext, const aclTensor* indices, const aclTensor* fetched,
                               int64_t withGrad, int64_t fetchMode)
{
    CHECK_RET(CheckNotNull(commContext, indices, fetched), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckFetchMode(withGrad, fetchMode), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnEngramFetchGetWorkspaceSize(const aclTensor* commContext, const aclTensor* indices,
                                             const aclTensor* localStorageAddr, const aclTensor* sfTable,
                                             aclTensor* fetched, aclTensor* permOut, aclTensor* sendCountsOut,
                                             aclTensor* recvCountsOut, aclTensor* recvLocalEntryOut,
                                             aclTensor* numRecvOut, aclTensor* fetchedSf, int32_t hiddenSize,
                                             int64_t numEntriesPerRank, int64_t numMaxTokensPerRank,
                                             int64_t commBufferSize, int64_t withGrad, int64_t fetchMode,
                                             uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto retParam = CheckParams(commContext, indices, fetched, withGrad, fetchMode);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret = aclnnInnerEngramFetchGetWorkspaceSize(
        commContext, indices, localStorageAddr, sfTable, hiddenSize, numEntriesPerRank, numMaxTokensPerRank,
        commBufferSize, withGrad, fetchMode, fetched, permOut, sendCountsOut, recvCountsOut, recvLocalEntryOut,
        numRecvOut, fetchedSf, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnEngramFetch(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerEngramFetch(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
