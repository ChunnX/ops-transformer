/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include "op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_grouped_mat_mul_allto_allv.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, aclTensor* y, aclTensor* mmYOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerGroupedMatMulAlltoAllv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                    aclrtStream stream);

// check nullptr
static bool CheckNullStatus(const aclTensor* gmmX, const aclTensor* gmmWeight,
                            const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional,
                            const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, const char* group,
                            bool transGmmWeight, bool transMmWeight, aclTensor* y, const aclTensor* mmYOptional)
{
    (void)transGmmWeight;
    (void)transMmWeight;
    // 检查必选入参出参为非空
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(y, return false);
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, "
                "left: %u, right: %u, mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is "
                "nullptr: %u",
                (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))),
                (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))),
                mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* gmmX, const aclTensor* gmmWeight,
                               const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional,
                               const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, const char* group,
                               int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
                               bool transGmmWeight, bool transMmWeight, aclTensor* y, aclTensor* mmYOptional)
{
    (void)epWorldSize;
    (void)sendCounts;
    (void)recvCounts;
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                              mmWeightOptional, group, transGmmWeight, transMmWeight, y, mmYOptional),
              ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, aclTensor* y, aclTensor* mmYOptional, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto ret_param =
        CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                    group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group,
        epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnGroupedMatMulAlltoAllv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif