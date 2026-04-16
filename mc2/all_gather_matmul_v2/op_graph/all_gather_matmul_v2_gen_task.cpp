/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_matmul_v2_gen_task.cpp
 * \brief
 */
#include <vector>

#include "op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "ops_error.h"
#include "mc2_gen_task_utils.h"
#include "mc2_a5_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#endif

namespace ops {
static bool IsTargetPlatform(const char *nodeName, const std::set<std::string> &targetPlatform)
{
    fe::PlatFormInfos platform_info;
    fe::OptionalInfos optional_info;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
        ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Cannot get platform info!");
        return false;
    }
    std::string short_soc_version;
    if (!platform_info.GetPlatformRes("version", "Short_SoC_version", short_soc_version) || short_soc_version.empty()) {
        OPS_LOG_E(nodeName, "Cannot get short soc version!");
        return false;
    }
    OPS_LOG_D(nodeName, "Get soc version: %s", short_soc_version.c_str());
    return targetPlatform.count(short_soc_version) > 0;
}
#ifdef BUILD_OPEN_PROJECT
//待开源补充
#else // mc2 gen task utils
static const std::set<std::string> platform910_95 = {"Ascend910_95"};
static ge::Status AllGatherMatmulV2CalcOpParam(gert::ExeResGenerationContext *context)
{
    if (IsTargetPlatform(context->GetNodeName(), platform910_95)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU CalcParam");
        return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    OPS_LOG_E(context->GetNodeName(), "Only support A5");
    return ge::GRAPH_FAILED;
}

static ge::Status AllGatherMatmulV2GenTask(const gert::ExeResGenerationContext *context,
                                           std::vector<std::vector<uint8_t>> &tasks)
{
    if (IsTargetPlatform(context->GetNodeName(), platform910_95)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask");
        return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2A5GenTaskUtils::Mc2GenTaskCallBack910A5);
    }
    OPS_LOG_E(context->GetNodeName(), "Only support A5");
    return ge::GRAPH_FAILED;
}

IMPL_OP_CT(AllGatherMatmulV2).CalcOpParam(AllGatherMatmulV2CalcOpParam).GenerateTask(AllGatherMatmulV2GenTask);
#endif
} // namespace ops
