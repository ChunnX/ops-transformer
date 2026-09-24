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
// MhcNormPostPreonly A5入口
//
// 文件结构：
//   mhc_norm_post_preonly_common.h       — 常量和类型特征
//   mhc_norm_post_preonly_kernel.h       — 类定义、初始化、执行和共享计算模块
//   mhc_norm_post_preonly_kernel_compute.h — 单双AIV共用的计算、IO和同步辅助函数
//   mhc_norm_post_preonly_single_aiv.h   — 已验证的单AIV回退路径
//   mhc_norm_post_preonly_dual_sibling_aiv.h — A5双sibling AIV路径
//   mhc_norm_post_preonly_apt.cpp        — Kernel入口
// ========================================================================

#include "arch35/mhc_norm_post_preonly_kernel.h"

extern "C" __global__ __aicore__ void mhc_norm_post_preonly(GM_ADDR h_out, GM_ADDR residual, GM_ADDR h_post,
                                                            GM_ADDR h_res, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias,
                                                            GM_ADDR pre_mix, GM_ADDR gamma_0, GM_ADDR gamma_1,
                                                            GM_ADDR gamma_2, GM_ADDR h_in_prime, GM_ADDR x_2_out,
                                                            GM_ADDR h_in_f32, GM_ADDR h_pre, GM_ADDR workspace,
                                                            GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // bit0和bit1选择可选语义分支，bit2保留给双sibling AIV执行方案。
    // 枚举四个单AIV键，并把两个可选语义变为编译期常量，
    // 使每个二进制只保留实际需要的执行分支。
    if (TILING_KEY_IS(0)) {
        KernelMhcNormPostPreonly<DTYPE_H_OUT, false, false> op;
        op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                x_2_out, h_in_f32, h_pre, nullptr, tiling);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelMhcNormPostPreonly<DTYPE_H_OUT, true, false> op;
        op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                x_2_out, h_in_f32, h_pre, nullptr, tiling);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelMhcNormPostPreonly<DTYPE_H_OUT, false, true> op;
        op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                x_2_out, h_in_f32, h_pre, nullptr, tiling);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelMhcNormPostPreonly<DTYPE_H_OUT, true, true> op;
        op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                x_2_out, h_in_f32, h_pre, nullptr, tiling);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KERNEL_TASK_TYPE(4, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIV {
            KernelMhcNormPostPreonly<DTYPE_H_OUT, false, false, true> op;
            op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                    x_2_out, h_in_f32, h_pre, GetUserWorkspace(workspace), tiling);
            op.Process();
        }
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIV {
            KernelMhcNormPostPreonly<DTYPE_H_OUT, true, false, true> op;
            op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                    x_2_out, h_in_f32, h_pre, GetUserWorkspace(workspace), tiling);
            op.Process();
        }
    } else if (TILING_KEY_IS(6)) {
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIV {
            KernelMhcNormPostPreonly<DTYPE_H_OUT, false, true, true> op;
            op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                    x_2_out, h_in_f32, h_pre, GetUserWorkspace(workspace), tiling);
            op.Process();
        }
    } else if (TILING_KEY_IS(7)) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
        if ASCEND_IS_AIV {
            KernelMhcNormPostPreonly<DTYPE_H_OUT, true, true, true> op;
            op.Init(h_out, residual, h_post, h_res, phi, alpha, bias, pre_mix, gamma_0, gamma_1, gamma_2, h_in_prime,
                    x_2_out, h_in_f32, h_pre, GetUserWorkspace(workspace), tiling);
            op.Process();
        }
    }
}
