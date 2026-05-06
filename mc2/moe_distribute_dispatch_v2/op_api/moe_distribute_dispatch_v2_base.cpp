/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_distribute_dispatch_v2_base.cpp
 * \brief
 */

#include <algorithm>
#include "common/utils/op_mc2.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "moe_distribute_dispatch_v2_base.h"
#include "aclnnInner_moe_distribute_dispatch_v2.h"

#ifdef BUILD_OPEN_PROJECT
#include "version/hcomm_version.h"
#define HCCL_CHANNEL_SUPPORT_VERSION 90000000
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include "common/op_api/mc2_context.h"
#endif
#endif

using namespace Ops::Transformer;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern "C" void NnopbaseSetUserHandle(void *executor, void *handle);

extern "C" void *NnopbaseGetUserHandle(void *executor);

#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
extern aclnnStatus aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales,
    const aclTensor *xActiveMask, const aclTensor *expertScales, const aclTensor *elasticInfo,
    const aclTensor *performanceInfo, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t cclBufferSize, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t shareExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, const char *commAlg,
    int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum, int64_t ydtype, aclTensor *expandX,
    aclTensor *dynamicScales, aclTensor *assist_info_for_combine, aclTensor *expertTokensNums, aclTensor *epRecvCounts,
    aclTensor *tpRecvCounts, aclTensor *expandScales, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMoeDistributeDispatchV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream);

enum class CommType : uint64_t {
    AIV = 0, // AIV通信设置为0
    CCU = 1  // ccu通信设置为1
};
#endif

bool DispatchCheckNotNull(const aclTensor *x, const aclTensor *expertIds, const char *groupEp,
                          [[maybe_unused]] const char *groupTp, aclTensor *expandX,
                          [[maybe_unused]] aclTensor *dynamicScales, aclTensor *assistInfoForCombine,
                          aclTensor *expertTokensNums, aclTensor *epRecvCounts, aclTensor *tpRecvCounts)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIds, return false);
    OP_CHECK_NULL(expandX, return false);
    OP_CHECK_NULL(assistInfoForCombine, return false);
    OP_CHECK_NULL(expertTokensNums, return false);
    OP_CHECK_NULL(tpRecvCounts, return false);
    OP_CHECK_NULL(epRecvCounts, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "group groupEp name is Empty");
        return false;
    }
    return true;
}

aclnnStatus DispatchCheckParams(const aclTensor *x, const aclTensor *expertIds, const char *groupEp,
                                const char *groupTp, int64_t quantMode, aclTensor *expandX, aclTensor *dynamicScales,
                                aclTensor *assistInfoForCombine, aclTensor *expertTokensNums, aclTensor *epRecvCounts,
                                aclTensor *tpRecvCounts)
{
    CHECK_RET(DispatchCheckNotNull(x, expertIds, groupEp, groupTp, expandX, dynamicScales, assistInfoForCombine,
                                   expertTokensNums, epRecvCounts, tpRecvCounts),
              ACLNN_ERR_PARAM_NULLPTR);

    if (quantMode == DISPATCH_DYNAMIC_QUANT_MODE) {
        OP_LOGD("quantMode = 2, dynamicScales can't be null");
        CHECK_RET(dynamicScales != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupEp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required groupTp name exceeds %zu", HCCL_GROUP_NAME_MAX);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    return ACLNN_SUCCESS;
}

static void SetCommArgs(const bool is950, const bool is910B, const char *commAlg, aclOpExecutor **executor)
{
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    if (is950) {
        void *arg = reinterpret_cast<void *>(static_cast<uintptr_t>(CommType::AIV)); // 默认MTE为0
        if (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            arg = reinterpret_cast<void *>(static_cast<uintptr_t>(CommType::CCU)); // ccu为1
        }
        NnopbaseSetUserHandle(*executor, arg);
    }
#endif

    if (NnopbaseSetHcclServerType) { // 给ACLnn框架指定通讯方式。
        if (is910B) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        } else if (is950 && commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0) {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_CCU);
        } else {
            NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
        }
    }
}

aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSizeBase(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *expertScalesOptional, const aclTensor *elasticInfoOptional,
    const aclTensor *performanceInfoOptional, const char *groupEp, int64_t epWorldSize, int64_t epRankId,
    int64_t moeExpertNum, const char *groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs,
    int64_t expertTokenNumsType, const char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum,
    int64_t constExpertNum, aclTensor *expandXOut, aclTensor *dynamicScalesOut, aclTensor *assistInfoForCombineOut,
    aclTensor *expertTokenNumsOut, aclTensor *epRecvCountsOut, aclTensor *tpRecvCountsOut, aclTensor *expandScalesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("enter to the  aclnnMoeDistributeDispatchGetWorkspaceSizeBase\n");
    const static bool is910B = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B;
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = DispatchCheckParams(x, expertIds, groupEp, groupTp, quantMode, expandXOut, dynamicScalesOut,
                                        assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    const aclTensor *performanceInfoOptionalDispatchV2Temp = performanceInfoOptional;
    aclTensor *mc2Context = nullptr;
    aclnnStatus getWorkspaceSizesRes;
    const char *groupTpDispatchV2Temp = groupTp;
    if (is910B) {
        groupTpDispatchV2Temp = "";
    } else if (is950) {
        performanceInfoOptionalDispatchV2Temp = nullptr;
    }

    int64_t ydtype = expandXOut->GetDataType();
    if (!is950 || (commAlg != nullptr && std::strcmp(commAlg, "ccu") == 0)) { // ccu暂时不支持新方案
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV2GetWorkspaceSize(
            x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
            performanceInfoOptionalDispatchV2Temp, const_cast<char *>(groupEp), epWorldSize, epRankId, moeExpertNum,
            const_cast<char *>(groupTpDispatchV2Temp), tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
            sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, const_cast<char *>(commAlg), zeroExpertNum,
            copyExpertNum, constExpertNum, ydtype, expandXOut, dynamicScalesOut, assistInfoForCombineOut,
            expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut, expandScalesOut, workspaceSize, executor);
    } else {
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
        uint64_t hcclBuffSize = 0;
        const char *opName = "moe_distribute_dispatch_combine_v2";
        auto ret = Mc2Aclnn::Mc2Context::GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        getWorkspaceSizesRes = aclnnInnerMoeDistributeDispatchV3GetWorkspaceSize(
            mc2Context, x, expertIds, scalesOptional, xActiveMaskOptional, expertScalesOptional, elasticInfoOptional,
            performanceInfoOptionalDispatchV2Temp, epWorldSize, epRankId, moeExpertNum, hcclBuffSize, tpWorldSize,
            tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType,
            const_cast<char *>(commAlg), zeroExpertNum, copyExpertNum, constExpertNum, ydtype, expandXOut,
            dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut, epRecvCountsOut, tpRecvCountsOut,
            expandScalesOut, workspaceSize, executor);
#endif
    }
    SetCommArgs(is950, is910B, commAlg, executor);
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnMoeDistributeDispatchBase(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           aclrtStream stream)
{
#if defined(BUILD_OPEN_PROJECT) && HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (is950) {
        void *arg = NnopbaseGetUserHandle(executor);
        uintptr_t handleVal = reinterpret_cast<uintptr_t>(arg);
        CommType commType = static_cast<CommType>(handleVal);
        if (commType == CommType::AIV) {
            OP_LOGD("enter to the  aclnnInnerMoeDistributeDispatchV2Extend");
            return aclnnInnerMoeDistributeDispatchV3(workspace, workspaceSize, executor, stream); // mte走新模版
        }
    }
#endif
    return aclnnInnerMoeDistributeDispatchV2(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif