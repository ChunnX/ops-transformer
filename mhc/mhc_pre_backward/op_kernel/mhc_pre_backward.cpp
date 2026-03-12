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
 * \file mhc_pre_backward.cpp
 * \brief
 */

#include "arch35/mhc_pre_backward.h"

using namespace AscendC;
using namespace matmul;
using namespace MhcPreBackward;

extern "C" __global__ __aicore__ void mhc_pre_backward(
    GM_ADDR x, GM_ADDR phi, GM_ADDR alpha,
    GM_ADDR h_in_grad, GM_ADDR h_post_grad, GM_ADDR h_comb_before_grad,
    GM_ADDR inv_rms, GM_ADDR mm_res, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR gamma, GM_ADDR x_grad,
    GM_ADDR hc_weight_grad, GM_ADDR alpha_grad, GM_ADDR bias_post_grad, GM_ADDR gamma_grad, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    InitParams initParams{x, phi, alpha,
        h_in_grad, h_post_grad, h_comb_before_grad,
        inv_rms, mm_res, h_pre, h_post, gamma,
        x_grad, hc_weight_grad, alpha_grad, bias_post_grad,
        gamma_grad, user, &pipe, &tilingData};
    if (TILING_KEY_IS(0UL)) {
        MT_C0 mm0;
        MT_C1 mm1;
        mm0.Init(&tilingData.matmulTilingC0, &pipe);
        mm1.Init(&tilingData.matmulTilingC1, &pipe);
        MhcPreBackwardKernel<DTYPE_X, float32_t> op(mm0, mm1);
        op.Init(initParams);
        op.Process();
    }
}
