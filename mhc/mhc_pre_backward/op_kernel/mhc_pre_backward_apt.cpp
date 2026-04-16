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
 * \file mhc_pre_backward.cpp
 * \brief
 */

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#include "arch35/mhc_pre_backward_kernel.h"
#include "arch35/mhc_pre_backward_tiling_key.h"

using namespace AscendC;
using namespace matmul;
using namespace MhcPreBackward;

template <int8_t TILING_MODE>
__global__ __aicore__ void mhc_pre_backward(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR grad_h_in,
                                                       GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR inv_rms,
                                                       GM_ADDR h_mix, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR gamma,
                                                       GM_ADDR grad_x_post_optional, GM_ADDR grad_x, GM_ADDR grad_phi,
                                                       GM_ADDR grad_alpha, GM_ADDR grad_bias, GM_ADDR grad_gamma,
                                                       GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    if constexpr (TILING_MODE == MHC_PRE_BACKWARD_DEFAULT) {
        InitParams initParams{
            x,          phi,      alpha, grad_h_in,   grad_h_post, grad_h_res, inv_rms,  h_mix,
            h_pre,      h_post,   gamma, grad_x_post_optional, grad_x, grad_phi, grad_alpha, grad_bias,
            grad_gamma, user,     &pipe, &tilingData};
        MT_C0 mm0;
        MT_C1 mm1;
        mm0.Init(&tilingData.matmulTilingC0, &pipe);
        mm1.Init(&tilingData.matmulTilingC1, &pipe);
        MhcPreBackwardKernel<DTYPE_X, float32_t> op(mm0, mm1);
        op.Init(initParams);
        op.Process();
    }
}
#endif
