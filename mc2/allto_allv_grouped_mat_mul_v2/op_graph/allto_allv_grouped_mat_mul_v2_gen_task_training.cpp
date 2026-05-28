/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_v2_gen_task_training.cpp
 * \brief
 */
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"
#include "mc2_comm_utils.h"
#include "common/utils/op_mc2.h"

namespace ops {
ge::Status AlltoAllvGroupedMatMulV2CalcParamFunc(gert::ExeResGenerationContext *context)
{
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_D(context->GetNodeName(), "attrs is null, use default AICPU GenTask"),
               return ge::GRAPH_FAILED);
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);

    const char* serverType = nullptr;
    const char* streamType = nullptr;

    if (commMode == nullptr || strlen(commMode) == 0) {
        if (isArch35) {
            serverType = "ccu server";
            streamType = "ccu_stream";
            OPS_LOG_D(context->GetNodeName(), "comm_mode is empty, arch35 use CCU GenTask");
        } else {
            serverType = "aicpu kfc server";
            streamType = "kfc_stream";
            OPS_LOG_D(context->GetNodeName(), "comm_mode is empty, arch22 use AICPU GenTask");
        }
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "comm_mode is ai_cpu, use AICPU GenTask");
    } else if (strcmp(commMode, "ccu") == 0) {
        serverType = "ccu server";
        streamType = "ccu_stream";
        OPS_LOG_D(context->GetNodeName(), "comm_mode is ccu, use CCU GenTask");
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid comm_mode: %s", commMode);
        return ge::GRAPH_FAILED;
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvGroupedMatMulV2GenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        OPS_LOG_D(context->GetNodeName(), "arch22, always use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    
    auto attrs = context->GetAttrs();
    OPS_ERR_IF(attrs == nullptr,
               OPS_LOG_D(context->GetNodeName(), "attrs is null, use CCU GenTask"),
               return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks));
    const char* commMode = attrs->GetStr(static_cast<size_t>(AlltoAllvGroupedMatMulV2AttrIdx::K_COMM_MODE));
    
    if (commMode == nullptr || strlen(commMode) == 0 || strcmp(commMode, "ccu") == 0) {
        OPS_LOG_D(context->GetNodeName(), "comm_mode is %s, arch35 use CCU GenTask", 
                  commMode == nullptr || strlen(commMode) == 0 ? "empty" : commMode);
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    } else if (strcmp(commMode, "ai_cpu") == 0) {
        OPS_LOG_D(context->GetNodeName(), "comm_mode is ai_cpu, arch35 use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    } else {
        OPS_LOG_E(context->GetNodeName(), "Invalid comm_mode: %s", commMode);
        return ge::GRAPH_FAILED;
    }
}

IMPL_OP(AlltoAllvGroupedMatMulV2)
    .CalcOpParam(AlltoAllvGroupedMatMulV2CalcParamFunc)
    .GenerateTask(AlltoAllvGroupedMatMulV2GenTaskFunc);
} // namespace ops

