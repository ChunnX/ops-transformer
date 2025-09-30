/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_mat_mul_allto_allv_gen_task_training.cpp
 * \brief
 */
#include <vector>

#include "op_mc2.h"
#include "platform/platform_info.h"

#ifdef BUILD_OPEN_PROJECT
#include "mc2_gen_task_ops_utils.h"
#include "mc2_moe_gen_task_ops_utils.h"
#include "graph/arg_desc_info.h"
#include "graph/kernel_launch_info.h"
#include "register/op_impl_registry.h"
#include "mc2_log.h"
#else
#include "mc2_gen_task_training.h"
#include "mc2_gen_task_utils.h"
#include "register/op_ct_impl_registry.h"
#endif

namespace ops {

#ifdef BUILD_OPEN_PROJECT
ge::Status GroupedMatMulAlltoAllvCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status GroupedMatMulAlltoAllvGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2MoeGenTaskOpsUtils::Mc2MoeGenTaskCallback(context, tasks);
}

// new ver
IMPL_OP(GroupedMatMulAlltoAllv)
    .CalcOpParam(GroupedMatMulAlltoAllvCalcParamFunc)
    .GenerateTask(GroupedMatMulAlltoAllvGenTaskFunc);
#else // mc2 gen task utils
ge::Status GroupedMatMulAlltoAllvCalcParamFunc(gert::ExeResGenerationContext *context)
{
    const ge::AscendString name = "aicpu kfc server";
    const ge::AscendString reuseKey = "kfc_stream";
    return Mc2GenTaskUtils::CommonKFCMc2CalcParamFunc(context, name, reuseKey);
}

ge::Status GroupedMatMulAlltoAllvGenTaskFunc(const gert::ExeResGenerationContext *context,
                                            std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2GenTaskUtils::CommonKFCMc2GenTask(context, tasks, Mc2GenTaskTraining::Mc2TrainingGenTaskCallback);
}

IMPL_OP_CT(GroupedMatMulAlltoAllv)
    .CalcOpParam(GroupedMatMulAlltoAllvCalcParamFunc)
    .GenerateTask(GroupedMatMulAlltoAllvGenTaskFunc);
#endif
} // namespace ops

