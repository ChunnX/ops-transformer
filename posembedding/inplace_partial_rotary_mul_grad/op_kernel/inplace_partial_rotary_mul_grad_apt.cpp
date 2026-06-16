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
 * Only the x == nullptr scenario is supported (dcos / dsin are not computed).
 * The (dy, cos) dtype combination is selected at compile time via DTYPE_DY / DTYPE_COS macros
 * injected by the build system from the JSON binary config entries.
 */

#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_grad_tiling_data.h"
#include "arch35/inplace_partial_rotary_mul_grad.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
                                                                      GM_ADDR dyOut, GM_ADDR dcos, GM_ADDR dsin,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)x;
    (void)dcos;
    (void)dsin;
    (void)workspace;
}
