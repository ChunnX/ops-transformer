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
 * \file ai_infra_mhc_post_grad.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ai_infra_mhc_post_grad.h"

using namespace AscendC;
using namespace AiInfraMhcPostGrad;

extern "C" __global__ __aicore__ void ai_infra_mhc_post_grad(
    GM_ADDR gradOutput, GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR gradX, GM_ADDR gradHRes, GM_ADDR gradHOut, GM_ADDR gradHPost,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    GET_TILING_DATA(tilingData, tiling);

    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;

    // Template instantiation based on alignment flags from tiling data
    // Note: Template parameters must be compile-time constants
    uint32_t isNAligned = tilingData.isNAligned;
    uint32_t isNNAligned = tilingData.isNNAligned;
    uint32_t isDAligned = tilingData.isDAligned;

    if (isNAligned == 0 && isNNAligned == 0 && isDAligned == 0) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 0, 0, 0> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 0 && isNNAligned == 0 && isDAligned == 1) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 0, 0, 1> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 0 && isNNAligned == 1 && isDAligned == 0) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 0, 1, 0> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 0 && isNNAligned == 1 && isDAligned == 1) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 0, 1, 1> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 1 && isNNAligned == 0 && isDAligned == 0) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 1, 0, 0> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 1 && isNNAligned == 0 && isDAligned == 1) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 1, 0, 1> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 1 && isNNAligned == 1 && isDAligned == 0) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 1, 1, 0> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else if (isNAligned == 1 && isNNAligned == 1 && isDAligned == 1) {
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 1, 1, 1> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    } else {
        // Fallback to slowest path
        AiInfraMhcPostGradKernel<DTYPE_GRAD_OUTPUT, 0, 0, 0> op;
        op.Init(gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, userWS, &tilingData, &tPipe);
        op.Process();
    }

    tPipe.Destroy();
}
