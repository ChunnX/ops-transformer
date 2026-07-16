/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root directory of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_engram_fetch_train.cpp
 * \brief EngramFetchTrain 算子 aclnn 接口实现（训练态前向）
 */

#include <algorithm>
#include "aclnn/aclnn_base.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "aclnnInner_engram_fetch_train.h"

using namespace op;

static bool CheckNotNull(const aclTensor *commContext, const aclTensor *indices,
                         aclTensor *fetched, aclTensor *permOut, aclTensor *sendCountsOut,
                         aclTensor *recvCountsOut, aclTensor *recvLocalEntryOut, aclTensor *numRecvOut)
{
    OP_CHECK_NULL(commContext, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(fetched, return false);
    OP_CHECK_NULL(permOut, return false);
    OP_CHECK_NULL(sendCountsOut, return false);
    OP_CHECK_NULL(recvCountsOut, return false);
    OP_CHECK_NULL(recvLocalEntryOut, return false);
    OP_CHECK_NULL(numRecvOut, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor *commContext, const aclTensor *indices,
                               aclTensor *fetched, aclTensor *permOut, aclTensor *sendCountsOut,
                               aclTensor *recvCountsOut, aclTensor *recvLocalEntryOut, aclTensor *numRecvOut)
{
    CHECK_RET(CheckNotNull(commContext, indices, fetched, permOut, sendCountsOut,
                           recvCountsOut, recvLocalEntryOut, numRecvOut),
              ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnEngramFetchTrainGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *indices,
    aclTensor *fetched, aclTensor *permOut, aclTensor *sendCountsOut,
    aclTensor *recvCountsOut, aclTensor *recvLocalEntryOut, aclTensor *numRecvOut,
    int32_t hiddenSize, int64_t numEntriesPerRank, int64_t localStorageAddr,
    int64_t numMaxTokensPerRank, int64_t commBufferSize,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto retParam = CheckParams(commContext, indices, fetched, permOut, sendCountsOut,
                                recvCountsOut, recvLocalEntryOut, numRecvOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);
    aclnnStatus ret = aclnnInnerEngramFetchTrainGetWorkspaceSize(
        commContext, indices,
        hiddenSize, numEntriesPerRank, localStorageAddr, numMaxTokensPerRank, commBufferSize,
        fetched, permOut, sendCountsOut, recvCountsOut, recvLocalEntryOut, numRecvOut,
        workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnEngramFetchTrain(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerEngramFetchTrain(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
