/* *
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */
#include "aclnn_allto_allv_grouped_mat_mul_v2.h"
#include "allto_allv_grouped_mat_mul_check.h"
#include <algorithm>
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/common_types.h"
#include "aclnnInner_allto_allv_grouped_mat_mul.h"
#include "mc2_comm_utils.h"

using namespace op;

static const char *A2AV_GMM_V2_API = "aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize";

#ifdef __cplusplus
extern "C" {
#endif

enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_CCU,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

static constexpr uint32_t RANK_DIM_BOUNDARY = 8;

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

static aclnnStatus CheckAndHandleCommMode(const char *group, const char *commModeStr, uint8_t &commModeEnum)
{
    const size_t maxLength = 7UL;
    // 获取通信引擎参数
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_CCU;
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Currently, commMode only support 'ccu', 'ai_cpu', but got %s.",
                    commModeStr);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Currently, commMode only support 'ai_cpu', but got %s.", commModeStr);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const char *group, const char *commMode, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, bool permuteOutFlag, aclTensor *gmmY,
    aclTensor *mmYOptional, aclTensor *permuteOutOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret_param =
        checkParams(A2AV_GMM_V2_API, gmmX, gmmWeight, sendCountsTensorOptional,
                                          recvCountsTensorOptional, mmXOptional, mmWeightOptional, group, epWorldSize,
                                          permuteOutFlag, gmmY, mmYOptional, permuteOutOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_counts = checkCounts(A2AV_GMM_V2_API, gmmWeight, epWorldSize,
                                                        sendCountsTensorOptional, recvCountsTensorOptional, sendCounts,
                                                        recvCounts);
    CHECK_RET(ret_counts == ACLNN_SUCCESS, ret_counts);
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Optional commMode name is Empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    char *str_commMode = const_cast<char *>(commMode);
    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode(group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, permuteOutFlag,
        str_commMode, gmmY, mmYOptional, permuteOutOptional, workspaceSize, executor);
    OP_LOGD("AlltoAllvGroupedMatmul, aclnnInnerAlltoAllvGroupedMatMulGetWorkspaceSize ret %d.", ret);
    if (*executor != nullptr) {
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, args);
    }
    return ret;
}

aclnnStatus aclnnAlltoAllvGroupedMatMulV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                OP_LOGD("arch35, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                OP_LOGD("arch35, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        } else {
            OP_LOGD("Arch22 platform, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    aclnnStatus ret = aclnnInnerAlltoAllvGroupedMatMul(workspace, workspaceSize, executor, stream);
    OP_LOGD("AlltoAllvGroupedMatmul, aclnnInnerAlltoAllvGroupedMatMul ret %d.", ret);
    return ret;
}

#ifdef __cplusplus
}
#endif
