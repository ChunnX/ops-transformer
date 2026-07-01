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

// HCCL 通信服务类型枚举：决定 AlltoAllv 通信使用的底层硬件引擎
enum class NnopbaseHcclServerType : uint32_t {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,  // AICPU 通信：通过 AI CPU 软件栈驱动，兼容性好但延迟较高
    NNOPBASE_HCCL_SERVER_TYPE_MTE,        // MTE 通信：通过 Memory Transfer Engine
    NNOPBASE_HCCL_SERVER_TYPE_CCU,        // CCU 通信：通过 Cube Communication Unit，低延迟高性能（仅 Ascend 950 支持）
    NNOPBASE_HCCL_SERVER_TYPE_END
};

// 弱符号：由运行时库提供实现，若链接的库版本不支持则为 nullptr（通过 if 判断保护）
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
extern "C" void *NnopbaseGetUserHandle(void *executor);

static bool CheckNullStatus(const aclTensor* gmmX, const aclTensor* gmmWeight,
                            const aclTensor* sendCountsTensorOptional, const aclTensor* recvCountsTensorOptional,
                            const aclTensor* mmXOptional, const aclTensor* mmWeightOptional, const char* group,
                            bool transGmmWeight, bool transMmWeight, aclTensor* y, const aclTensor* mmYOptional)
{
    (void)transGmmWeight;
    (void)transMmWeight;
    OP_CHECK_NULL(gmmX, return false);
    OP_CHECK_NULL(gmmWeight, return false);
    OP_CHECK_NULL(y, return false);
    // send/recv counts tensor 当前版本不支持，必须为 nullptr（计数通过属性传入）
    if ((sendCountsTensorOptional != nullptr) || (recvCountsTensorOptional != nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize",
            "sendCountsTensorOptional/recvCountsTensorOptional", "non-null", "The value of sendCountsTensorOptional/recvCountsTensorOptional must be nullptr.");
        return false;
    }
    if ((group == nullptr) || (strnlen(group, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "group");
        return false;
    }
    // 共享专家三参数必须同时提供或同时缺省，不允许部分提供
    if ((!((mmXOptional != nullptr) && (mmWeightOptional != nullptr) && (mmYOptional != nullptr))) &&
        (!((mmXOptional == nullptr) && (mmWeightOptional == nullptr) && (mmYOptional == nullptr)))) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize",
            "mmXOptional/mmWeightOptional/mmYOptional", "mixed null/non-null", "should all be null or all not be null");
        return false;
    }
    return true;
}

static aclnnStatus CheckSendAndRecv(const aclIntArray* sendCounts, const aclIntArray* recvCounts)
{
    if (sendCounts == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "sendCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "recvCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U;
    uint64_t sendSize = 0U;
    aclGetIntArraySize(recvCounts, &recvSize);
    aclGetIntArraySize(sendCounts, &sendSize);
    if (recvSize == 0U) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "recvCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sendSize == 0U) {
        OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "sendCounts");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

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
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAlltoAllvGetWorkspaceSize", "group", "too long",
            ("group name length must be less than " + std::to_string(HCCL_GROUP_NAME_MAX)).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

/*
 * 第一阶段接口：计算 workspace 大小并构造 executor
 *
 * V1 通信模式选择策略（与 V2 的区别）：
 *   不接收 commMode 参数，根据平台架构自动决定：
 *     - Ascend 950 (DAV_3510) -> "ccu"（低延迟高性能）
 *     - 其他平台 -> "ai_cpu"（兼容性好）
 *
 * 通信模式枚举值通过 userHandle 存入 executor，供第二阶段接口读取。
 */
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
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);

    // V1 根据平台架构自动选择通信模式（V2 由调用方通过 commMode 参数指定）
    const char *commMode = (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) ? "ccu" : "ai_cpu";
    char *str_commMode = const_cast<char *>(commMode);
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(
        gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional, mmWeightOptional,
        const_cast<char *>(group), epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight,
        str_commMode, y, mmYOptional, workspaceSize, executor);
    OP_LOGD("GroupedMatMulAlltoAllv, aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize ret %d.", ret);

    // 将通信模式枚举值编码为 void* 存入 executor 的 userHandle，
    // 第二阶段接口通过 GetUserHandle 读出，据此设置 HCCL 服务类型
    if (*executor != nullptr) {
        uint8_t commModeType = (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) ?
            Mc2Comm::COMM_MODE_CCU :
            Mc2Comm::COMM_MODE_AICPU;
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeType));
        NnopbaseSetUserHandle(*executor, args);
    }
    return ret;
}

/*
 * 第二阶段接口：执行实际计算
 *
 * 根据通信模式设置 HCCL 服务类型（AICPU/CCU），然后调用 Inner 接口执行。
 * 平台差异：
 *   - Ascend 950：根据 userHandle 中的通信模式选择 AICPU 或 CCU
 *   - 其他平台：统一使用 AICPU
 */
aclnnStatus aclnnGroupedMatMulAlltoAllv(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                        aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            // 从 executor 读出第一阶段存储的通信模式枚举值
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
