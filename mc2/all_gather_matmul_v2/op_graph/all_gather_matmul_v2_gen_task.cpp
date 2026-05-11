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
 * \file all_gather_matmul_v2_gen_task.cpp
 * \brief
 */
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"

namespace ops {
static const size_t ATTR_INDEX_RANK_SIZE_ALLGATHER_MATMUL_V2 = 6;
static const size_t ATTR_INDEX_COMM_MODE_ALLGATHER_MATMUL_V2 = 12;
static ge::Status AllGatherMatmulV2CalcOpParam(gert::ExeResGenerationContext *context)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        auto *attrs = context->GetAttrs();
        if (attrs == nullptr) {
            OPS_LOG_E(context->GetNodeName(), "Attrs pointer is null.");
            return ge::GRAPH_FAILED;
        }
        const char *commModePtr = attrs->GetStr(static_cast<size_t>(ATTR_INDEX_COMM_MODE_ALLGATHER_MATMUL_V2));
        if (commModePtr == nullptr) {
            OPS_LOG_E(context->GetNodeName(), "commMode pointer is null.");
            return ge::GRAPH_FAILED;
        }
        std::string commMode = commModePtr;
        if (commMode.empty()) {
            OPS_LOG_W(context->GetNodeName(), "commMode is empty, will use ccu");
            commMode = "ccu";
        }
        OPS_LOG_D(context->GetNodeName(), "commMode is %s.", commMode.c_str());

        const char* serverType = nullptr;
        const char* streamType = nullptr;
        if (commMode == "default") {
            OPS_LOG_D(context->GetNodeName(), "Do A5 default GenTask CalcOpParam");
            const int64_t* rankSizePtr = attrs->GetInt(static_cast<size_t>(ATTR_INDEX_RANK_SIZE_ALLGATHER_MATMUL_V2));
            if (rankSizePtr == nullptr) {
                OPS_LOG_E(context->GetNodeName(), "rankSize pointer is null.");
            }
            OPS_LOG_D(context->GetNodeName(), "===gentask rankSize %d", *rankSizePtr);
            if (*rankSizePtr <= 8) {
                serverType = "ccu server";
                streamType = "ccu_stream";
                OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask CalcOpParam");
            }else{
                serverType = "aicpu kfc server";
                streamType = "kfc_stream";
                OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTask CalcOpParam");
            }
        } else if (commMode == "ccu") {
            serverType = "ccu server";
            streamType = "ccu_stream";
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTask CalcOpParam");
        }else {
            serverType = "aicpu kfc server";
            streamType = "kfc_stream";
            OPS_LOG_D(context->GetNodeName(), "Do A5 AICPU GenTask CalcOpParam");
        } 
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, serverType, streamType);

    }
    OPS_LOG_E(context->GetNodeName(), "Only support A5");
    return ge::GRAPH_FAILED;
}

static ge::Status AllGatherMatmulV2GenTask(const gert::ExeResGenerationContext *context,
                                           std::vector<std::vector<uint8_t>> &tasks)
{
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTaskFunc");
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    OPS_LOG_E(context->GetNodeName(), "Only support A5");
    return ge::GRAPH_FAILED;
}

IMPL_OP(AllGatherMatmulV2).CalcOpParam(AllGatherMatmulV2CalcOpParam).GenerateTask(AllGatherMatmulV2GenTask);
} // namespace ops
