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
 * \file matmul_all_reduce_gen_task.cpp
 * \brief
 */
#include <vector>

#include "op_graph/mc2_gen_task_ops_utils.h"
#include "op_graph/matmul_all_reduce_gen_task_ops_utils.h" //in transformer dev
#include "op_graph/mc2_gen_task_ops_utils_arch35.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#include "mc2_platform_info.h"
#include "mc2_comm_utils.h"
#include "mc2_hcom_topo_info.h"
#include "common/utils/op_mc2.h"

namespace ops {
// uint32_t GentaskGetRankSize(const char *group) {
//     OP_LOGE("MatmulAllReduceCalcParamFunc", "Enter gentask GentaskGetRankSize");
//   uint32_t rankSize = 8;
//   if (Mc2Hcom::MC2HcomTopology::CommGetInstSizeByGroup(group, &rankSize)!=HCCL_SUCCESS) {
//       OP_LOGE("MatmulAllReduceCalcParamFunc", "Get rank size of group %s failed", group);
//       return 0;
//   }
//   return rankSize;
// }
// uint8_t GentaskGetAndConvertCommMode(const char *commModeStr, uint32_t rankSize)
// {
//     OP_LOGE("MatmulAllReduceCalcParamFunc", "Enter gentask GentaskGetAndConvertCommMode");
//     uint8_t commMode = Mc2Comm::COMM_MODE_CCU;
//     if (commModeStr == nullptr) {
//         OP_LOGI("MatmulAllReduce", "commModeStr is nullptr, commMode will be set to ccu.");
//     } else {
//         if (std::strcmp(commModeStr, "ai_cpu") == 0) {
//             commMode = Mc2Comm::COMM_MODE_AICPU;
//         } else if (std::strcmp(commModeStr, "ccu") == 0) {
//             commMode = Mc2Comm::COMM_MODE_CCU;
//         } else if (std::strcmp(commModeStr, "") == 0) {
//             if (rankSize <= 8) {
//                 commMode = Mc2Comm::COMM_MODE_CCU;
//             } else {
//                 commMode = Mc2Comm::COMM_MODE_AICPU;
//             }
//             OP_LOGI("MatmulAllReduce", "commMode is default, and rankDim is %d, will use commMode: %d.", rankSize, commMode);
//         }
//     }
//     return commMode;
// }
ge::Status MatmulAllReduceCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Failed to get attrs."),
                    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream"));
    const char *groupStr = attrs->GetAttrPointer<char>(0);
    OP_TILING_CHECK(groupStr == nullptr, OP_LOGE(context->GetNodeName(), "The input attr group is null pointer."),
                    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream"));
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        const char *commModeStr = attrs->GetAttrPointer<char>(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_COMM_MODE));    
        // uint32_t rankSize = GentaskGetRankSize(groupStr);
        // uint8_t commMode = GentaskGetAndConvertCommMode(commModeStr, rankSize);
        if (commModeStr != nullptr && (std::strcmp(commModeStr, "ccu") == 0)) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU CalcParamFunc");
            return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
        }
    }
    OPS_LOG_D(context->GetNodeName(), "Do general CalcParamFunc");
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

ge::Status MatmulAllReduceGenTaskFunc(const gert::ExeResGenerationContext *context,
                                      std::vector<std::vector<uint8_t>> &tasks)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(context->GetNodeName(), "Failed to get attrs."),
                    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream"));
    const char *groupStr = attrs->GetAttrPointer<char>(0);
    OP_TILING_CHECK(groupStr == nullptr, OP_LOGE(context->GetNodeName(), "The input attr group is null pointer."),
                    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream"));
    if (IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        const char *commModeStr = attrs->GetAttrPointer<char>(static_cast<size_t>(ops::MmAllReduceAttrIdx::K_COMM_MODE));
        // uint32_t rankSize = GentaskGetRankSize(groupStr);
        // uint8_t commMode = GentaskGetAndConvertCommMode(commModeStr, rankSize);
        if (commModeStr != nullptr && (std::strcmp(commModeStr, "ccu") == 0)) {
            OPS_LOG_D(context->GetNodeName(), "Do A5 CCU GenTaskFunc");
            return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
        }
    }
    OPS_LOG_D(context->GetNodeName(), "Do general GenTaskFunc");
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}

// new ver
IMPL_OP(MatmulAllReduce).CalcOpParam(MatmulAllReduceCalcParamFunc).GenerateTask(MatmulAllReduceGenTaskFunc);
} // namespace ops
