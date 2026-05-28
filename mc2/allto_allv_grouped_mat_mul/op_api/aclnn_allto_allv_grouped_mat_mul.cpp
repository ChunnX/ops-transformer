/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_allto_allv_grouped_mat_mul.h"
#include <algorithm>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnnInner_allto_allv_grouped_mat_mul.h"
#include "mc2_comm_utils.h"
#include "aclnn_allto_allv_grouped_mat_mul_v2.h"  // V2 aclnn 头文件

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

// check nullptr
static bool CheckNullStatus(const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
                            const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                            const char* group, bool permuteOutFlag, aclTensor* gmmY, const aclTensor* mmYOptional, const aclTensor* permuteOutOptional)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(gmmY, return false);
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sendCountsTensorOptional and recvCountsTensorOptional should be empty.");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) && (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))){
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "mmXOptional, mmWeightOptional and mmYOptional should all be null or all not be null, left: %u, right: %u, mmXOptional is nullptr: %u, mmWeightOptional is nullptr: %u, mmYOptional is nullptr: %u",
        (!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))), (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr))), mmXOptional == nullptr, mmWeightOptional == nullptr, mmYOptional == nullptr);
        return false;
    }
    if (permuteOutFlag == (permuteOutOptional == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
        "Optional output flag does not match optional output ptr!");
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
                               const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                               const char* group, int64_t epWorldSize, bool permuteOutFlag, aclTensor* gmmY, aclTensor* mmYOptional, 
                               aclTensor* permuteOutOptional)
{
    (void)epWorldSize;      // Unused
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional, group,
              permuteOutFlag, gmmY, mmYOptional, permuteOutOptional), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Required group name exceeds %zu.", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSendAndRecv(const aclIntArray* sendCounts, const aclIntArray* recvCounts)
{
    if (sendCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U;  // recvCounts的大小
    uint64_t sendSize = 0U;  // recvCounts的大小
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor* gmmX, const aclTensor* gmmWeight,
                                                        const aclTensor* sendCountsTensorOptional, 
                                                        const aclTensor* recvCountsTensorOptional, 
                                                        const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, 
                                                        const char* group, int64_t epWorldSize, 
                                                        const aclIntArray* sendCounts, const aclIntArray* recvCounts, 
                                                        bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, 
                                                        aclTensor* gmmY, aclTensor* mmYOptional, aclTensor* permuteOutOptional,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    auto ret_param = CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, 
        mmXOptional, mmWeightOptional, group, epWorldSize, permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    // 根据架构判断通信方式
    const char* commMode = "";
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        commMode = "ai_cpu";  // A3/arch22 使用 AICPU
    }
    // A5/arch35 使用默认 CCU，无需特殊处理

    // 调用 V2 aclnn 接口
    aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
        gmmX, gmmWeight,
        sendCountsTensorOptional, recvCountsTensorOptional,
        mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize,
        sendCounts, recvCounts,
        transGmmWeight, transMmWeight, permuteOutFlag,
        commMode,  // 新增参数
        gmmY, mmYOptional, permuteOutOptional,
        workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnAlltoAllvGroupedMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    // 直接调用 V2 Execute
    aclnnStatus ret = aclnnAlltoAllvGroupedMatMulV2(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif