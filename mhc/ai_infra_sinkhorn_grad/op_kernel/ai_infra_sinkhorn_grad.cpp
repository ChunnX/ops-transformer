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

\file ai_infra_sinkhorn_grad.cpp
\brief
*/
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ai_infra_sinkhorn_grad_generalized.h"

using namespace AscendC;
using namespace AiInfraSinkhornGrad;

extern "C" __global__ __aicore__ void ai_infra_sinkhorn_grad(GM_ADDR gradOutput, GM_ADDR normOut, GM_ADDR sumOut,
                                            GM_ADDR gradInput, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (g_coreType == AIC)
    {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);

    if (workspace == nullptr)
    {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr)
    {
        return;
    }

    TPipe tPipe;
    AiInfraSinkhornGradGeneralized op;
    op.Init(gradOutput, normOut, sumOut, gradInput, userWS, &tilingData, &tPipe);
    op.Process();
    tPipe.Destroy();
}