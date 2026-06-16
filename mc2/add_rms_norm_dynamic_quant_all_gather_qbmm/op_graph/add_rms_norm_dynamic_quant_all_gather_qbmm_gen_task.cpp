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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_gen_task.cpp
 * \brief 静态shape图下沉实现 - AddRmsNormDynamicQuantAllGatherQbmm
 */
#include <vector>
#include <string>
#include "op_mc2.h"

#include "add_rms_norm_dynamic_quant_all_gather_qbmm_gen_task.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/ascend_string.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"

namespace ops {

ge::Status Mc2GenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::CommonKFCMc2CalcParamFunc(
    const gert::ExeResGenerationContext *context, const ge::AscendString &name,
    const ge::AscendString &reuse_key)
{
    if (context == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get context.");
        return ge::GRAPH_FAILED;
    }
    gert::StreamInfo stream_info;
    std::vector<int64_t> stream_depend_value(0);
    stream_info.name = name;
    stream_info.reuse_key = reuse_key;
    stream_info.depend_value_input_indices = stream_depend_value;
    stream_info.required = true;

    std::vector<gert::StreamInfo> stream_infos;
    stream_infos.push_back(stream_info);
    const auto ret = context->SetAttachedStreamInfos(stream_infos);
    if (ret != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set attached stream infos.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2GenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::InsertHiddenInputsForAicoreTask(
    const gert::ExeResGenerationContext *context, ge::KernelLaunchInfo &aicore_task,
    size_t (*get_insert_idx)(const std::vector<ge::ArgDescInfo> &), size_t input_cnt)
{
    if (context == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get context.");
        return ge::GRAPH_FAILED;
    }
    if (get_insert_idx == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get get_insert_idx.");
        return ge::GRAPH_FAILED;
    }

    std::vector<ge::ArgDescInfo> argDescInfos;

    auto argsFormatStr = aicore_task.GetArgsFormat();
    if (argsFormatStr == nullptr) {
        OPS_LOG_E(context->GetNodeName(), "Failed to get Args Format from aicore task.");
        return ge::GRAPH_FAILED;
    }
    argDescInfos = ge::ArgsFormatSerializer::Deserialize(argsFormatStr);
    size_t insert_idx = get_insert_idx(argDescInfos);
    OPS_LOG_D(context->GetNodeName(), "Insertion position is %zu, insert inputCnt is %zu.", insert_idx, input_cnt);

    for (size_t i = 0; i < input_cnt; ++i, ++insert_idx) {
        argDescInfos.insert(argDescInfos.begin() + insert_idx,
                            ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcom));
    }

    auto argDescInfosSerialize = ge::ArgsFormatSerializer::Serialize(argDescInfos);
    if (aicore_task.SetArgsFormat(argDescInfosSerialize.GetString()) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context->GetNodeName(), "Failed to set args format for aicore task.");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_I(context->GetNodeName(), "aicore ArgsFormat: %s", argDescInfosSerialize.GetString());

    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2MoeGenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::McMoeInsertHiddenInputForAicore(
    const gert::ExeResGenerationContext *context, const int32_t groupCnt,
    std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();

    const auto getIdxFunc = [](const std::vector<ge::ArgDescInfo> &argDescInfo) {
        size_t insertIdx = 0U;
        for (; insertIdx < argDescInfo.size(); ++insertIdx) {
            if (argDescInfo[insertIdx].GetType() == ge::ArgDescType::kIrInput ||
                argDescInfo[insertIdx].GetType() == ge::ArgDescType::kInputInstance) {
                break;
            }
        }
        return insertIdx;
    };

    ge::KernelLaunchInfo aicoreTask = ge::KernelLaunchInfo::LoadFromData(context, tasks.back());
    if (Mc2GenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::InsertHiddenInputsForAicoreTask(
            context, aicoreTask, getIdxFunc, groupCnt) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Failed to insert hidden input for mix task.");
        return ge::GRAPH_FAILED;
    }
    tasks.back() = aicoreTask.Serialize();

    OPS_LOG_D(nodeName, "Modify AICore task for mc2 node successfully.");
    return ge::GRAPH_SUCCESS;
}

ge::Status Mc2MoeGenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::Mc2MoeGenTaskCallbackV2(
    const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    if (tasks.size() <= 0) {
        OPS_LOG_E(nodeName, "Failed to get task for mc2 node.");
        return ge::GRAPH_FAILED;
    }

    const char* opType = ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_OP_TYPE;
    const int32_t groupCnt = GROUP_CNT_OF_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM;

    OPS_LOG_D(nodeName, "Op [%s] get group [%d] success.", opType, groupCnt);

    if (McMoeInsertHiddenInputForAicore(context, groupCnt, tasks) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(nodeName, "Insert hidden input for [%s] failed.", opType);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::Status AddRmsNormDynamicQuantAllGatherQbmmCalcOpParamFunc(gert::ExeResGenerationContext *context)
{
    OPS_LOG_D(context->GetNodeName(), "Do general CalcParam in AddRmsNormDynamicQuantAllGatherQbmm");
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::CommonKFCMc2CalcParamFunc(
        context, name, reuseKey);
}

static ge::Status AddRmsNormDynamicQuantAllGatherQbmmGenTaskFunc(
    const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    const char *nodeName = context->GetNodeName();
    OPS_LOG_D(nodeName, "Do A3 GenTask in AddRmsNormDynamicQuantAllGatherQbmm");
    return Mc2MoeGenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm::Mc2MoeGenTaskCallbackV2(context, tasks);
}

IMPL_OP(AddRmsNormDynamicQuantAllGatherQbmm)
    .CalcOpParam(AddRmsNormDynamicQuantAllGatherQbmmCalcOpParamFunc)
    .GenerateTask(AddRmsNormDynamicQuantAllGatherQbmmGenTaskFunc);

} // namespace ops
