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
 * \file distribute_barrier_base.cpp
 * \brief
 */
#include <algorithm>

#include "aclnn_distribute_barrier_v2.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "common/utils/op_mc2_def.h"
#include "opdev/common_types.h"
#include "opdev/op_log.h"
#include "distribute_barrier_base.h"
#include "mc2_moe_context.h"
#include "common/op_api/mc2_moe_context_tensor.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
extern aclnnStatus aclnnInnerDistributeBarrier(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                               aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerDistributeBarrierGetWorkspaceSize(const aclTensor* xRef, const aclTensor* timeOut,
                                                               const aclTensor* elasticInfo, const char* group,
                                                               int64_t worldSize, uint64_t* workspaceSize,
                                                               aclOpExecutor** executor);
extern aclnnStatus aclnnInnerDistributeBarrierExtend(void* workspace, uint64_t workspaceSize,
                                                     aclOpExecutor* executor, aclrtStream stream);

extern aclnnStatus aclnnInnerDistributeBarrierExtendGetWorkspaceSize(const aclTensor* context, const aclTensor* xRef,
                                                                     const aclTensor* timeOut,
                                                                     const aclTensor* elasticInfo, const char* group,
                                                                     int64_t worldSize, uint64_t* workspaceSize,
                                                                     aclOpExecutor** executor);

// check nullptr
bool BarrierCheckNullStatus(const aclTensor* xRef, const char* group)
{
    // 检查必选入参出参为非空
    OP_CHECK_NULL(xRef, return false);
    if (group == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name is Empty.");
        return false;
    }
    return true;
}

// 入参校验
aclnnStatus BarrierCheckParams(const aclTensor* xRef, const char* group)
{
    CHECK_RET(BarrierCheckNullStatus(xRef, group), ACLNN_ERR_PARAM_NULLPTR);
    auto groupStrnLen = strnlen(group, HCCL_GROUP_NAME_MAX);
    if ((groupStrnLen >= HCCL_GROUP_NAME_MAX) || (groupStrnLen == 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Required group name length in range (0, HCCL_GROUP_NAME_MAX), but it's %zu.",
                groupStrnLen);
        return false;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDistributeBarrierGetWorkspaceSizeBase(const aclTensor* xRef, const aclTensor* timeOut,
                                                       const aclTensor* elasticInfo, const char* group,
                                                       int64_t worldSize, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    auto retParam = BarrierCheckParams(xRef, group);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    aclTensor *mc2Context = nullptr;
    aclnnStatus getWorkspaceSizesRes;
    aclnnStatus ret;

    if (!is950) {
        getWorkspaceSizesRes aclnnInnerDistributeBarrierGetWorkspaceSize(xRef, timeOut, elasticInfo,
                                                           group, worldSize, workspaceSize, executor);
    } else {
        uint64_t hcclBuffSize = 0;
        string libPath;
        auto ret = GetBuiltinLibPath(libPath);
        OP_CHECK(
        ret, OP_LOGI("Leaving func: LoadLibResource, ASCEND_Lib_PATH not config."), return ACLNN_ERR_PARAM_INVALID);
        const char* opName = "distribute_barrier_extend";
        ret = LoadBuiltinOpApi(libPath);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        ret = GetMc2ContextTensor(groupEp, opName, hcclBuffSize, mc2Context);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        getWorkspaceSizesRes = aclnnInnerDistributeBarrierExtendGetWorkspaceSize(const aclTensor* context,
                                                        const aclTensor* xRef, const aclTensor* timeOut,
                                                        const aclTensor* elasticInfo, const char* group,
                                                        int64_t worldSize, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor);
    }
    return getWorkspaceSizesRes;
}

aclnnStatus aclnnDistributeBarrierBase(void* workspace, uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream)
{
    const static bool is950 = GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510;
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(*executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    if (is950) {
        OP_LOGD("aclnn_disrtibute_barrer_extend inner start");
        return aclnnInnerDistributeBarrierExtend(workspace, workspaceSize, executor, stream);
    } else {
        return aclnnInnerDistributeBarrier(workspace, workspaceSize, executor,
                                           stream);
    }
}
#ifdef __cplusplus
}
#endif
