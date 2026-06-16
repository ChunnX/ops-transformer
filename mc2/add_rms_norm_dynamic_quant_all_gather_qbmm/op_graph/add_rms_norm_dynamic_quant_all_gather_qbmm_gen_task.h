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
 * \file add_rms_norm_dynamic_quant_all_gather_qbmm_gen_task.h
 * \brief 静态shape图下沉 - AddRmsNormDynamicQuantAllGatherQbmm
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_GEN_TASK_H
#define ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_GEN_TASK_H

#include <string>
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/ascend_string.h"
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"

namespace ops {
const char* ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_OP_TYPE = "AddRmsNormDynamicQuantAllGatherQbmm";
const int32_t GROUP_CNT_OF_ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM = 1;

class Mc2GenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm {
public:
    static ge::Status InsertHiddenInputsForAicoreTask(const gert::ExeResGenerationContext *context,
                                                      ge::KernelLaunchInfo &aicore_task,
                                                      size_t (*get_insert_idx)(const std::vector<ge::ArgDescInfo> &),
                                                      size_t input_cnt = 1U);
    static ge::Status CommonKFCMc2CalcParamFunc(const gert::ExeResGenerationContext *context,
                                                const ge::AscendString &name, const ge::AscendString &reuse_key);
};

class Mc2MoeGenTaskOpsUtilsAddRmsNormDynamicQuantAllGatherQbmm {
public:
    static ge::Status McMoeInsertHiddenInputForAicore(const gert::ExeResGenerationContext *context,
                                                      const int32_t groupCnt, std::vector<std::vector<uint8_t>> &tasks);
    static ge::Status Mc2MoeGenTaskCallbackV2(const gert::ExeResGenerationContext *context,
                                              std::vector<std::vector<uint8_t>> &tasks);
};
} // namespace ops
#endif // ADD_RMS_NORM_DYNAMIC_QUANT_ALL_GATHER_QBMM_GEN_TASK_H
