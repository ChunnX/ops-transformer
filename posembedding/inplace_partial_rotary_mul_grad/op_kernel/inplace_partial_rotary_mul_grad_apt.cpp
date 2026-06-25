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
 * \file inplace_partial_rotary_mul_grad_apt.cpp
 * \brief InplacePartialRotaryMulGrad AscendC kernel entry for arch35 / ascend950.
 *
 * The (dy, cos, sin) dtype combination is selected at compile time via DTYPE_DY / DTYPE_COS macros
 * injected by the build system from the JSON binary config entries.
 * Note: sin always shares DTYPE_COS (cos and sin have identical dtypes per the op constraint).
 */

#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_grad_common.h"
#include "arch35/inplace_partial_rotary_mul_grad_bab.h"

#define TILING_KEY_ABA 201
#define TILING_KEY_BA 202
#define TILING_KEY_BAB 203
#define TILING_KEY_AB 204
#define TILING_KEY_A 205
#define TILING_KEY_B 206

using namespace AscendC;
using namespace InplacePartialRotaryMulGrad;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
                                                                      GM_ADDR dx, GM_ADDR dcos, GM_ADDR dsin,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::printf("enter inplace_partial_rotary_mul_grad\n");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)x;
    (void)dcos;
    (void)dsin;
    (void)workspace;

    AscendC::printf("before GET_CORE_TYPE\n");

    if (g_coreType == AIC) {
        return;
    }

    AscendC::printf("after GET_CORE_TYPE\n");

    // AscendC::printf("before REGISTER_TILING_DEFAULT\n");

    // REGISTER_TILING_DEFAULT(InplacePartialRotaryMulGradTilingData);

    // AscendC::printf("after REGISTER_TILING_DEFAULT\n");

    // AscendC::printf("before GET_TILING_DATA_WITH_STRUCT\n");
    // GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, tilingData, tiling);

    // AscendC::printf("after GET_TILING_DATA_WITH_STRUCT\n");
    AscendC::printf("before TILING_KEY_IS\n");
    if (TILING_KEY_IS(TILING_KEY_BAB)) {
        AscendC::printf("before GET_TILING_DATA_WITH_STRUCT\n");
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradRegbaseTilingData, tilingDataBab, tiling);
        const InplacePartialRotaryMulGradRegbaseTilingData *__restrict tilingData = &tilingDataBab;
        AscendC::printf("after GET_TILING_DATA_WITH_STRUCT\n");
        AscendC::printf("if TILING_KEY_IS(TILING_KEY_BAB)\n");
    }
    AscendC::printf("after TILING_KEY_IS\n");
    AscendC::printf("leave inplace_partial_rotary_mul_grad\n");
    // // No-op: empty slice
    // if (tilingData.sliceLength == 0) {
    //     return;
    // }

    // TPipe pipe;

    // if (TILING_KEY_IS(TILING_KEY_ABA)) {
    //     const InplacePartialRotaryMulGradRegbaseParams &params = tilingData.inplacePartialRotaryMulGradParams;
    //     InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, false> op;
    //     op.Init(&pipe, &params, dy, cos, sin, dx);
    //     op.Process();
    // } else if (TILING_KEY_IS(TILING_KEY_BA)) {
    //     const InplacePartialRotaryMulGradRegbaseParams &params = tilingData.inplacePartialRotaryMulGradParams;
    //     InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, true> op;
    //     op.Init(&pipe, &params, dy, cos, sin, dx);
    //     op.Process();
    // } else if (TILING_KEY_IS(TILING_KEY_BAB)) {
    //     AscendC::printf("if TILING_KEY_IS(TILING_KEY_BAB)\n");
    //     const InplacePartialRotaryMulGradRegbaseParams &params = tilingData.inplacePartialRotaryMulGradParams;
    //     InplacePartialRotaryMulGradBAB<DTYPE_DY, DTYPE_COS> op;
    //     op.Init(&pipe, &params, dy, cos, sin, dx);
    //     op.Process();
    // } else if (TILING_KEY_IS(TILING_KEY_AB)) {
    //     const InplacePartialRotaryMulGradRegbaseABParams &params = tilingData.inplacePartialRotaryMulGradABParams;
    //     InplacePartialRotaryMulGradAB<DTYPE_DY, DTYPE_COS> op;
    //     op.Init(&pipe, &params, dy, cos, sin, dx);
    //     op.Process();
    // } else if (TILING_KEY_IS(TILING_KEY_A)) {
    //     const InplacePartialRotaryMulGradRegbaseParams &params = tilingData.inplacePartialRotaryMulGradParams;
    //     InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, false> op;
    //     op.Init(&pipe, &params, dy, cos, sin, dx);
    //     op.Process();
    // } else if (TILING_KEY_IS(TILING_KEY_B)) {
    //     const InplacePartialRotaryMulGradRegbaseParams &params = tilingData.inplacePartialRotaryMulGradParams;
    //     InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, true> op;
    //     op.Init(&pipe, &params, dy, cos, sin, dx);
    //     op.Process();
    // }
}
