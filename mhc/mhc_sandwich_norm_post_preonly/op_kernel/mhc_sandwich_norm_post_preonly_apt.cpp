/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

// ========================================================================
// R9: MhcSandwichNormPostPreonly — entry point
//
// File structure:
//   mhc_sandwich_norm_post_preonly_common.h       — constants, type traits
//   mhc_sandwich_norm_post_preonly_kernel.h       — class: Init, Process, shared building blocks
//   mhc_sandwich_norm_post_preonly_dualcore.h     — ProcessTokenDualCore (single-tile, D<=2560)
//   mhc_sandwich_norm_post_preonly_dualcore_mt.h  — ProcessTokenDualCoreMultiTile (multi-tile, D=5120)
//   mhc_sandwich_norm_post_preonly_singlecore.h   — ProcessToken + PhaseAB/C/D (fallback)
//   mhc_sandwich_norm_post_preonly.cpp            — this file (kernel entry point)
// ========================================================================

#include "arch35/mhc_sandwich_norm_post_preonly_kernel.h"

extern "C" __global__ __aicore__ void
mhc_sandwich_norm_post_preonly(GM_ADDR h_out, GM_ADDR residual, GM_ADDR h_post, GM_ADDR h_res, GM_ADDR phi,
                                        GM_ADDR alpha_0, GM_ADDR bias, GM_ADDR gamma_0, GM_ADDR gamma_1,
                                        GM_ADDR gamma_2, GM_ADDR h_in_prime, GM_ADDR x_2_out, GM_ADDR h_in_f32,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (workspace == nullptr) {
        return;
    }
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

    KernelMhcSandwichNormPostPreonly<DTYPE_H_OUT> op;
    op.Init(h_out, residual, h_post, h_res, phi, alpha_0, bias, gamma_0, gamma_1, gamma_2, h_in_prime, x_2_out,
            h_in_f32, usrWorkspace, tiling);
    op.Process();
}
