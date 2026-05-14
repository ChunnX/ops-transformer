/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_grouped_mat_mul_gen_task_training.cpp
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

namespace ops {
ge::Status AlltoAllvGroupedMatMulCalcParamFunc(gert::ExeResGenerationContext *context)
{
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);

    const char* serverType = nullptr;
    const char* streamType = nullptr;

    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "ENV_MC2_COMM_MODE_AICPU set, force AICPU GenTask");
    } else if (isArch35) {
        serverType = "ccu server";
        streamType = "ccu_stream";
        OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    } else {
        serverType = "aicpu kfc server";
        streamType = "kfc_stream";
        OPS_LOG_D(context->GetNodeName(), "arch22, use AICPU GenTask");
    }
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);
}

ge::Status AlltoAllvGroupedMatMulGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    bool isArch35 = IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5);
    if (!isArch35) {
        OPS_LOG_D(context->GetNodeName(), "arch22, always use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    uint8_t commMode = Mc2Comm::GetCommModeFromEnv();
    if (commMode == Mc2Comm::COMM_MODE_AICPU) {
        OPS_LOG_D(context->GetNodeName(), "arch35 with ENV_MC2_COMM_MODE_AICPU, use AICPU GenTask");
        return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "arch35, use CCU GenTask");
    return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
}

// new ver
IMPL_OP(AlltoAllvGroupedMatMul)
    .CalcOpParam(AlltoAllvGroupedMatMulCalcParamFunc)
    .GenerateTask(AlltoAllvGroupedMatMulGenTaskFunc);
} // namespace ops

