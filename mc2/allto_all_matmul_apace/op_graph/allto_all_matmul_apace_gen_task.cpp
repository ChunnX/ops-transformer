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
 * \file allto_all_matmul_apace_gen_task.cpp
 * \brief GenTask callback for AlltoAllMatmulApace (graph execution path)
 */

#include <vector>

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "op_graph/mc2_gen_task_ops_utils.h"
#include "register/op_impl_registry.h"

namespace ops {

static ge::Status AlltoAllMatmulApaceCalcOpParamFunc(gert::ExeResGenerationContext *context)
{
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}

static ge::Status AlltoAllMatmulApaceGenTaskFunc(const gert::ExeResGenerationContext *context,
                                                 std::vector<std::vector<uint8_t>> &tasks)
{
    return Mc2GenTaskOpsUtils::CommonKFCMc2GenTask(context, tasks);
}

IMPL_OP(AlltoAllMatmulApace)
    .CalcOpParam(AlltoAllMatmulApaceCalcOpParamFunc)
    .GenerateTask(AlltoAllMatmulApaceGenTaskFunc);

} // namespace ops
