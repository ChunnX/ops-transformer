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
 * \file quant_grouped_mat_mul_allto_allv_gen_task_training.cpp
 * \brief
 */
#include <vector>

#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "mc2_platform_info.h"

namespace ops {
ge::Status QuantGroupedMatMulAlltoAllvCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU CalcParamFunc");
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3 AICPU CalcParamFunc");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status QuantGroupedMatMulAlltoAllvGenTaskFunc(const gert::ExeResGenerationContext *context,
                                             std::vector<std::vector<uint8_t>> &tasks)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTaskFunc");
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    OPS_LOG_D(context->GetNodeName(), "Do A3 AICPU GenTaskFunc");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
}

// new ver
IMPL_OP(QuantGroupedMatMulAlltoAllv)
    .CalcOpParam(QuantGroupedMatMulAlltoAllvCalcParamFunc)
    .GenerateTask(QuantGroupedMatMulAlltoAllvGenTaskFunc);
} // namespace ops
