/* *
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
#include "opdev/common_types.h"
#include "aclnn_grouped_mat_mul_allto_allv_v2.h"
#include "aclnnInner_grouped_mat_mul_allto_allv.h"
#include "mc2_comm_utils.h"

#ifdef BUILD_OPEN_PROJECT
#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif
#endif

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
// 将通信模式枚举值存入 executor，供第二阶段接口读取
extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);
// 从 executor 读取第一阶段存储的通信模式枚举值
extern "C" void *NnopbaseGetUserHandle(void *executor);

/*
 * 解析通信模式字符串为枚举值
 *
 * 通信模式选择策略（仅 Ascend 950 平台）：
 *   - "ai_cpu" -> AICPU 模式
 *   - "ccu"    -> CCU 模式
 *   - ""       -> 自动选择：卡数 <= 8 用 CCU（低延迟），卡数 > 8 用 AICPU（高扩展性）
 *
 * CCU 在低卡数场景延迟更低、吞吐更高；AICPU 在高卡数场景更稳定可靠。
 * 非 Ascend 950 平台不解析，统一使用 AICPU 模式。
 */
static aclnnStatus CheckAndHandleCommMode(const char *group, const char *commModeStr, uint8_t &commModeEnum)
{
    uint32_t rankSize = 0;
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    auto getRankSizeRet = Mc2Aclnn::Mc2Context::GetMc2RankSize(group, rankSize);
    CHECK_RET(getRankSizeRet == ACLNN_SUCCESS, getRankSizeRet);
#endif
    const size_t maxLength = 6UL;
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        if (strncmp(commModeStr, "ai_cpu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_AICPU;
        } else if (strncmp(commModeStr, "ccu", maxLength) == 0) {
            commModeEnum = Mc2Comm::COMM_MODE_CCU;
        } else if (strncmp(commModeStr, "", maxLength) == 0) {
            if (rankSize <= 8) {
                commModeEnum = Mc2Comm::COMM_MODE_CCU;
            } else {
                commModeEnum = Mc2Comm::COMM_MODE_AICPU;
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "commMode only support '', 'ccu', 'ai_cpu'.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static bool CheckNullStatus(const aclTensor *gmmX, const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const char *group, bool transGmmWeight, bool transMmWeight, aclTensor *y,
    const aclTensor *mmYOptional)
{
    (void)transGmmWeight;
    (void)transMmWeight;
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

static aclnnStatus CheckSendAndRecv(const aclIntArray *sendCounts, const aclIntArray *recvCounts)
{
    if (sendCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sendCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (recvCounts == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "recvCounts should not be null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    uint64_t recvSize = 0U;
    uint64_t sendSize = 0U;
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

static aclnnStatus CheckParams(const aclTensor *gmmX, const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const char *group, int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, aclTensor *y, aclTensor *mmYOptional)
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

/*
 * 第一阶段接口：计算 workspace 大小并构造 executor
 *
 * 执行流程：
 *   1. 入参校验（CheckParams + CheckSendAndRecv）
 *   2. 解析 commMode 字符串为通信模式枚举值（CheckAndHandleCommMode）
 *   3. 调用 Inner 接口计算 workspace 大小并构造 executor
 *   4. 将通信模式枚举值编码为 void* 存入 executor 的 userHandle，
 *      供第二阶段接口读取以设置底层 HCCL 通信引擎类型
 */
aclnnStatus aclnnGroupedMatMulAlltoAllvV2GetWorkspaceSize(const aclTensor *gmmX, const aclTensor *gmmWeight,
    const aclTensor *sendCountsTensorOptional, const aclTensor *recvCountsTensorOptional, const aclTensor *mmXOptional,
    const aclTensor *mmWeightOptional, const char *group, const char *commMode, int64_t epWorldSize,
    const aclIntArray *sendCounts, const aclIntArray *recvCounts, bool transGmmWeight, bool transMmWeight, aclTensor *y,
    aclTensor *mmYOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    auto ret_param = CheckParams(gmmX, gmmWeight, sendCountsTensorOptional, recvCountsTensorOptional, mmXOptional,
        mmWeightOptional, group, epWorldSize, sendCounts, recvCounts, transGmmWeight, transMmWeight, y, mmYOptional);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);
    auto ret_send_and_recv = CheckSendAndRecv(sendCounts, recvCounts);
    CHECK_RET(ret_send_and_recv == ACLNN_SUCCESS, ret_send_and_recv);
    if (commMode == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Optional commMode name is Empty.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    char *str_commMode = const_cast<char *>(commMode);
    uint8_t commModeEnum = 0;
    aclnnStatus checkCommModeRet = CheckAndHandleCommMode(group, commMode, commModeEnum);
    CHECK_RET(checkCommModeRet == ACLNN_SUCCESS, checkCommModeRet);
    aclnnStatus ret = aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize(gmmX, gmmWeight, sendCountsTensorOptional,
        recvCountsTensorOptional, mmXOptional, mmWeightOptional, const_cast<char *>(group), epWorldSize, sendCounts,
        recvCounts, transGmmWeight, transMmWeight, str_commMode, y, mmYOptional, workspaceSize, executor);
    OP_LOGD("GroupedMatMulAlltoAllv, aclnnInnerGroupedMatMulAlltoAllvGetWorkspaceSize ret %d.", ret);
    // 将通信模式枚举值跨阶段传递：编码为 void* 存入 executor->userHandle
    if (*executor != nullptr) {
        void *args = reinterpret_cast<void *>(static_cast<uint8_t>(commModeEnum));
        NnopbaseSetUserHandle(*executor, args);
    }
    return ret;
}

/*
 * 第二阶段接口：执行实际的通信与计算
 *
 * 执行流程：
 *   1. 从 executor->userHandle 读出第一阶段存储的通信模式枚举值
 *   2. 根据平台架构和通信模式设置 HCCL 服务类型：
 *      - Ascend 950 (DAV_3510)：按用户指定的 commMode 设置 CCU 或 AICPU
 *      - 其他平台（如 Atlas A3）：统一使用 AICPU 模式
 *   3. 调用 Inner 接口执行 AlltoAllv 通信 + GroupedMatMul 计算
 *
 * 注意：NnopbaseSetHcclServerType 为弱符号，仅在运行时库支持时才调用，
 * 确保向后兼容性。
 */
aclnnStatus aclnnGroupedMatMulAlltoAllvV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            // Ascend 950 平台：从 userHandle 解码通信模式，按用户选择设置 CCU 或 AICPU
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
            // 非 Ascend 950 平台（如 Atlas A3）：统一使用 AICPU 模式
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