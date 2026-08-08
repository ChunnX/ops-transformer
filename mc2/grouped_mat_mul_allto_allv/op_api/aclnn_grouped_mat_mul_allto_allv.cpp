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
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "mc2_log_compat.h"
#include "opdev/common_types.h"
#include "aclnn_grouped_mat_mul_allto_allv.h"
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

// check nullptr
static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight,
                            const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                            const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const char *group,
                            bool transGmmWeight, bool transMmWeight, aclTensor *y, const aclTensor *mmYOptional)
{
    (void)transGmmWeight;
    (void)transMmWeight;
    (void)sendCountsTensorOptional;
    (void)recvCountsTensorOptional;
    // 检查必选入参出参为非空
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(y, return false);
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "group");
        return false;
    }
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize",
                                              "mmXOptional/mmWeightOptional/mmYOptional", "mixed null/non-null",
                                              "should all be null or all not be null");
        return false;
    }
    return true;
}

static aclnnStatus checkCounts(const char *apiName, const aclTensor *gmmWeight, int64_t epWorldSize,
                               const aclTensor *sendCountsTensorOptional,
                               const aclTensor *recvCountsTensorOptional,
                               const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    bool hasSendTensor = (sendCountsTensorOptional != nullptr);
    bool hasRecvTensor = (recvCountsTensorOptional != nullptr);
    bool hasSendAttr = false;
    bool hasRecvAttr = false;
    uint64_t sendSize = 0U;
    uint64_t recvSize = 0U;
    if (sendCounts != nullptr) {
        aclGetIntArraySize(sendCounts, &sendSize);
        hasSendAttr = (sendSize > 0U);
    }
    if (recvCounts != nullptr) {
        aclGetIntArraySize(recvCounts, &recvSize);
        hasRecvAttr = (recvSize > 0U);
    }
    // attr 与 tensor 互斥
    if (hasSendTensor && hasSendAttr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "sendCounts/sendCountsTensor", "both set",
                                              "sendCounts attr and sendCountsTensor are mutually exclusive.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // send/recv 必须同源
    if (hasSendTensor != hasRecvTensor) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "sendCountsTensor/recvCountsTensor",
            (std::to_string(hasSendTensor) + "/" + std::to_string(hasRecvTensor)).c_str(),
            "sendCountsTensor and recvCountsTensor must use the same source.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (hasSendAttr != hasRecvAttr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "sendCounts/recvCounts", (std::to_string(hasSendAttr) + "/" + std::to_string(hasRecvAttr)).c_str(),
            "sendCounts and recvCounts attrs must use the same source.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 至少提供一种
    if (!hasSendTensor && !hasSendAttr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "sendCounts/sendCountsTensor", "both null",
                                              "at least one of sendCounts attr or sendCountsTensor must be provided.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 counts tensor 元素个数 == ep * epWorldSize
    if (hasSendTensor) {
        int64_t ep = 0;
        if (gmmWeight != nullptr) {
            auto gmmWeightShape = gmmWeight->GetViewShape();
            ep = gmmWeightShape.GetDim(0);
        }
        int64_t expectedCount = ep * epWorldSize;
        if (expectedCount <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "ep*epWorldSize", std::to_string(expectedCount).c_str(),
                                                  "ep*epWorldSize must be positive when using counts tensor.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t sendTensorElems = 1;
        auto sendShape = sendCountsTensorOptional->GetViewShape();
        for (size_t i = 0; i < sendShape.GetDimNum(); i++) {
            sendTensorElems *= sendShape.GetDim(i);
        }
        int64_t recvTensorElems = 1;
        auto recvShape = recvCountsTensorOptional->GetViewShape();
        for (size_t i = 0; i < recvShape.GetDimNum(); i++) {
            recvTensorElems *= recvShape.GetDim(i);
        }
        if (sendTensorElems != expectedCount) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "sendCountsTensor", std::to_string(sendTensorElems).c_str(),
                (std::to_string(expectedCount) + " (ep*epWorldSize)").c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (recvTensorElems != expectedCount) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName, "recvCountsTensor", std::to_string(recvTensorElems).c_str(),
                (std::to_string(expectedCount) + " (ep*epWorldSize)").c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight,
                               const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional,
                               const aclTensor *mmXOptional, const aclTensor *mmWeightOptional, const char *group,
                               int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
                               bool transGmmWeight, bool transMmWeight, aclTensor *y, aclTensor *mmYOptional)
{
    (void)epWorldSize;
    (void)sendCounts;
    (void)recvCounts;
    CHECK_RET(CheckNullStatus(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
                              mmWeightOptional, group, transGmmWeight, transMmWeight, y, mmYOptional),
              ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(group, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "group", "too long",
            ("group name length must be less than " + std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, const aclTensor *sendCountsTensorOptional,
    const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional, const aclTensor *mmWeightOptional,
    const char *group, int64_t epWorldSize, const aclIntArray *sendCounts, const aclIntArray *recvCounts,
    bool transGmmWeight, bool transMmWeight, aclTensor *y, aclTensor *mmYOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    auto ret_param =
        CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
                    group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_counts = checkCounts("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", gmmWeight, epWorldSize,
                                  sendCountsTensorOptional, recvCountsTensorOptional, sendCounts, recvCounts);
    CHECK_RET(ret_counts == ACLNN_SUCCESS, ret_counts);

    uint64_t sendArrSize = 0U;
    uint64_t recvArrSize = 0U;
    if (sendCounts != nullptr) {
        aclGetIntArraySize(sendCounts, &sendArrSize);
    }
    if (recvCounts != nullptr) {
        aclGetIntArraySize(recvCounts, &recvArrSize);
    }
    const aclIntArray *sendCountsReal = (sendCounts != nullptr && sendArrSize == 0U) ? nullptr : sendCounts;
    const aclIntArray *recvCountsReal = (recvCounts != nullptr && recvArrSize == 0U) ? nullptr : recvCounts;

    const char *commMode = "ai_cpu";
    char *str_commMode = const_cast<char *>(commMode);
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize, sendCountsReal, recvCountsReal, transGmmWeight, transMmWeight,
        str_commMode, y, mmYOptional, workspaceSize, executor);
    OP_LOGD("GroupedMatMulAlltoAllv, aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize ret %d.", ret);
    if (*executor != nullptr) {
        uint8_t commModeType = Mc2Comm::COMM_MODE_AICPU;
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeType));
        NnopbaseSetUserHandle(*executor, args);
    }
    return ret;
}

aclnnStatus aclnnGroupedMatMulAlltoAllv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            void *arg = NnopbaseGetUserHandle(executor);
            uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
            uint8_t commMode = static_cast<uint8_t>(handleVal);
            if (commMode == Mc2Comm::COMM_MODE_AICPU) {
                OP_LOGD("Arch35 platform, use AICPU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
            } else {
                OP_LOGD("Arch35 platform, use CCU mode");
                NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
            }
        } else {
            OP_LOGD("Arch22 platform, use AICPU mode");
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
    }
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllv(workspace, workspaceSize, executor, stream);
    OP_LOGD("GroupedMatMulAlltoAllv, aclnnInnerGroupedMatMulAlltoAllv ret %d.", ret);
    return ret;
}

#ifdef __cplusplus
}
#endif
