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
 * \brief InplacePartialRotaryMulGrad AscendC (RegBase) kernel entry for arch35 / ascend950.
 *
 * Only the x == nullptr scenario is supported (dcos / dsin are not computed). The kernel dispatches on the
 * (dy, cos) dtype combination; rotary_mode is selected at runtime inside the kernel class.
 */

#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_grad_tiling_data.h"
#include "arch35/inplace_partial_rotary_mul_grad.h"

using namespace AscendC;
using namespace InplacePartialRotaryMulGrad;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(
    GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
    GM_ADDR dyOut, GM_ADDR dcos, GM_ADDR dsin,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)x;
    (void)dcos;
    (void)dsin;
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, tilingDataIn, tiling);
    const InplacePartialRotaryMulGradTilingData* __restrict tilingData = &tilingDataIn;

    // start == end: nothing to compute.
    if (tilingData->sliceLen <= 0) {
        return;
    }
    if (GetBlockIdx() >= tilingData->usedCoreNum) {
        return;
    }

    AscendC::TPipe pipe;
    int64_t dyDtype = tilingData->dyDtypeCode;
    int64_t csDtype = tilingData->cosDtypeCode;

#define DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(TDY, TCS)                                          \
    do {                                                                                            \
        InplacePartialRotaryMulGrad<TDY, TCS> op;                                                   \
        op.Init(dy, cos, sin, dyOut, tilingData, &pipe);                                           \
        op.Process();                                                                              \
    } while (0)

    if (dyDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP32) {
        if (csDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP32) {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(float, float);
        } else if (csDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP16) {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(float, half);
        } else {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(float, bfloat16_t);
        }
    } else if (dyDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP16) {
        if (csDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP32) {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(half, float);
        } else if (csDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP16) {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(half, half);
        } else {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(half, bfloat16_t);
        }
    } else {
        if (csDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP32) {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(bfloat16_t, float);
        } else if (csDtype == INPLACE_PARTIAL_ROTARY_DTYPE_FP16) {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(bfloat16_t, half);
        } else {
            DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD(bfloat16_t, bfloat16_t);
        }
    }
#undef DISPATCH_INPLACE_PARTIAL_ROTARY_MUL_GRAD
}
