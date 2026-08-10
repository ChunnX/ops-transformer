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
 * \file urma_p2p_benchmark_gen_task.cpp
 * \brief URMA点对点通信基准测试算子的gen_task实现
 */
#include <vector>
#include <set>
#include <string>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_moe_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"

namespace ops {

ge::Status UrmaP2pBenchmarkCalcParamFunc(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Do urma_p2p_benchmark calc param.");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status UrmaP2pBenchmarkGenTaskFunc(const gert::ExeResGenerationContext *context,
                                       std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "Do urma_p2p_benchmark gen task.");
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallbackV2(context, tasks);
}

IMPL_OP(UrmaP2pBenchmark)
    .CalcOpParam(UrmaP2pBenchmarkCalcParamFunc)
    .GenerateTask(UrmaP2pBenchmarkGenTaskFunc);

} // namespace ops