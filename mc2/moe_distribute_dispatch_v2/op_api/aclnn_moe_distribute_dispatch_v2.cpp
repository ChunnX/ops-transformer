/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_moe_distribute_dispatch_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"

#include "aclnnInner_moe_distribute_dispatch_v2.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* expertIds, const aclTensor* scalesOptional,
    const aclTensor* xActiveMaskOptional, const aclTensor* expertScalesOptional,
    const aclTensor* elasticInfoOptional, const aclTensor* performanceInfoOptional,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t cclBufferSize,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType,
    const char* commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    int64_t yDtype, aclTensor* expandXOut, aclTensor* dynamicScalesOut, aclTensor* assistInfoForCombineOut,
    aclTensor* expertTokenNumsOut, aclTensor* epRecvCountsOut, aclTensor* tpRecvCountsOut,
    aclTensor* expandScalesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_LOGD("aclnn_dispatch v2 WorkspaceSize start");
    return aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
        context, x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional,
        elasticInfoOptional, performanceInfoOptional,
        epWorldSize, epRankId, moeExpertNum, cclBufferSize,
        tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType,
        const_cast<char*>(commAlg), zeroExpertNum, copyExpertNum, constExpertNum, yDtype,
        expandXOut, dynamicScalesOut, assistInfoForCombineOut,
        expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
        expandScalesOut, workspaceSize, executor);
}

aclnnStatus aclnnMoeDistributeDispatchV2(void* workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    OP_LOGD("aclnn_dispatch v2 start");
    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
