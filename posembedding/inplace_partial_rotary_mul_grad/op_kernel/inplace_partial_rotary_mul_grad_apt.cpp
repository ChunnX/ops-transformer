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
 * The (dy, cos, sin) dtype combination is selected at compile time via DTYPE_DY / DTYPE_COS / DTYPE_SIN
 * macros injected by the build system from the JSON binary config entries.
 */

#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_grad_tiling_data.h"
#include "arch35/inplace_partial_rotary_mul_grad_tiling_key.h"
#include "arch35/inplace_partial_rotary_mul_grad_common.h"
#include "arch35/inplace_partial_rotary_mul_grad_ab.h"
#include "arch35/inplace_partial_rotary_mul_grad_a_and_b.h"
#include "arch35/inplace_partial_rotary_mul_grad_aba_and_ba.h"
#include "arch35/inplace_partial_rotary_mul_grad_bab.h"

using namespace AscendC;
using namespace InplacePartialRotaryMulGrad;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
                                                                      GM_ADDR dyOut, GM_ADDR dcos, GM_ADDR dsin,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)x;
    (void)dcos;
    (void)dsin;
    (void)workspace;

    if (g_coreType == AIC) {
        return;
    }

    // No-op: empty slice
    {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _noop, tiling);
        if (_noop.sliceLength == 0) {
            return;
        }
    }

    TPipe pipe;

    if (TILING_KEY_IS(InplacePartialRotaryMulGradTilingKey::TILING_KEY_ABA)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _tiling, tiling);
        const InplacePartialRotaryMulGradRegbaseParams *__restrict__ tilingData =
            &_tiling.inplacePartialRotaryMulGradParams;
        InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, DTYPE_SIN, false> op;
        op.Init(&pipe, tilingData, dy, cos, sin, dyOut);
        op.Process();
    } else if (TILING_KEY_IS(InplacePartialRotaryMulGradTilingKey::TILING_KEY_BA)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _tiling, tiling);
        const InplacePartialRotaryMulGradRegbaseParams *__restrict__ tilingData =
            &_tiling.inplacePartialRotaryMulGradParams;
        InplacePartialRotaryMulGradABAAndBA<DTYPE_DY, DTYPE_COS, DTYPE_SIN, true> op;
        op.Init(&pipe, tilingData, dy, cos, sin, dyOut);
        op.Process();
    } else if (TILING_KEY_IS(InplacePartialRotaryMulGradTilingKey::TILING_KEY_BAB)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _tiling, tiling);
        const InplacePartialRotaryMulGradRegbaseParams *__restrict__ tilingData =
            &_tiling.inplacePartialRotaryMulGradParams;
        InplacePartialRotaryMulGradBAB<DTYPE_DY, DTYPE_COS, DTYPE_SIN> op;
        op.Init(&pipe, tilingData, dy, cos, sin, dyOut);
        op.Process();
    } else if (TILING_KEY_IS(InplacePartialRotaryMulGradTilingKey::TILING_KEY_AB)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _tiling, tiling);
        const InplacePartialRotaryMulGradRegbaseABParams *__restrict__ tilingData =
            &_tiling.inplacePartialRotaryMulGradABParams;
        InplacePartialRotaryMulGradAB<DTYPE_DY, DTYPE_COS, DTYPE_SIN> op;
        op.Init(&pipe, tilingData, dy, cos, sin, dyOut);
        op.Process();
    } else if (TILING_KEY_IS(InplacePartialRotaryMulGradTilingKey::TILING_KEY_A)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _tiling, tiling);
        const InplacePartialRotaryMulGradRegbaseParams *__restrict__ tilingData =
            &_tiling.inplacePartialRotaryMulGradParams;
        InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, DTYPE_SIN, false> op;
        op.Init(&pipe, tilingData, dy, cos, sin, dyOut);
        op.Process();
    } else if (TILING_KEY_IS(InplacePartialRotaryMulGradTilingKey::TILING_KEY_B)) {
        GET_TILING_DATA_WITH_STRUCT(InplacePartialRotaryMulGradTilingData, _tiling, tiling);
        const InplacePartialRotaryMulGradRegbaseParams *__restrict__ tilingData =
            &_tiling.inplacePartialRotaryMulGradParams;
        InplacePartialRotaryMulGradAAndB<DTYPE_DY, DTYPE_COS, DTYPE_SIN, true> op;
        op.Init(&pipe, tilingData, dy, cos, sin, dyOut);
        op.Process();
    }
}
