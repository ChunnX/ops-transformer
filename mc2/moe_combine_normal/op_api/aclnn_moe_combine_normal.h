/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef ACLNN_CAM_MOE_COMBINE_NORMAL_H_
#define ACLNN_CAM_MOE_COMBINE_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif


__attribute__((visibility("default"))) aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX, const aclTensor *tokenSrcInfo, const aclTensor *epRecvCounts,
    const aclTensor *recvTopkWeights, const aclTensor *tokenIdx, const aclTensor *tpRecvCountsOptional,
    char *epGroupName, int64_t epWorldSize, int64_t epRankId, char *tpGroupNameOptional, int64_t tpWorldSize,
    int64_t tpRankId, int64_t moeExpertNum, int64_t realMaxBs, int32_t round, int32_t per_round_tokens,
    const aclTensor *out, const aclTensor *sendCostStats, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnMoeCombineNormal(void *workspace, uint64_t workspaceSize,
                                                                    aclOpExecutor *executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif