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
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnn_grouped_mat_mul_allto_allv_v2.h"
#include "aclnnInner_grouped_mat_mul_allto_allv.h"
#include "mc2_comm_utils.h"

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
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

static void SetNnopbaseHcclServerTypeByArch(aclOpExecutor *executor, const char* commMode)
{
    if ((executor == nullptr) || (NnopbaseSetHcclServerType == nullptr)) {
        return;
    }
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
        NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        const bool isAicpu = (strncmp(commMode, "ai_cpu", 6) == 0);
        NnopbaseHcclServerType serverType = isAicpu
            ? NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU
            : NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU;
        OP_LOGD("Execute commMode is %s for DAV_3510", isAicpu ? "AICPU" : "CCU");
        NnopbaseSetHcclServerType(executor, serverType);
    }
}

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

aclnnStatus aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(
    const aclTensor* gmmX, const aclTensor* gmmWeight, const aclTensor* sendCountsTensorOptional,
    const aclTensor* recvCountsTensorOptional, const aclTensor* mmXOptional, const aclTensor* mmWeightOptional,
    const char* group, int64_t epWorldSize, const aclIntArray* sendCounts, const aclIntArray* recvCounts,
    bool transGmmWeight, bool transMmWeight, const char* commMode, aclTensor* y, aclTensor* mmYOptional,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 添加 commMode 参数校验
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "commMode parameter is required and cannot be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    auto ret_param =
        CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                    group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional,
        workspaceSize, executor);

    // 在 executor 中存储 commMode，供第二段接口使用
    if (ret == ACLNN_SUCCESS && *executor != nullptr) {
        char* commModePtr = new char[strlen(commMode) + 1];
        if (commModePtr == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[GroupedMatMulAlltoAllvV2] Failed to allocate memory for commMode");
            return ACLNN_ERR_INNER;
        } else {
            strcpy(commModePtr, commMode);
            NnopbaseSetUserHandle(*executor, commModePtr);
            OP_LOGD("[GroupedMatMulAlltoAllvV2] GetWorkspaceSize set commMode=%s", commMode);
        }
    }

    return ret;
}

aclnnStatus aclnnGroupedMatMulAlltoAllvV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    const char* commMode = reinterpret_cast<char*>(NnopbaseGetUserHandle(executor));
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "[GroupedMatMulAlltoAllvV2] commMode is nullptr from NnopbaseGetUserHandle");
        return ACLNN_ERR_INNER;
    }
    SetNnopbaseHcclServerTypeByArch(executor, commMode);
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif