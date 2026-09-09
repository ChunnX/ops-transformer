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
 * \file moe_tutel_combine_x_apt.cpp
 * \brief Kernel entry of MoeTutelCombineX for arch35 (Ascend 950).
 *
 * The tiling key decides among three paths:
 *   TILING_KEY_EMPTY           - empty tensors: zero the x_grad output
 *   TILING_KEY_NORMAL_NO_BATCH - normal, y_grad 2D
 *   TILING_KEY_NORMAL_BATCH    - normal, y_grad 3D
 */
#include "kernel_operator.h"
#include "arch35/moe_tutel_combine_x_kernel.h"
#include "arch35/moe_tutel_combine_x_tiling_struct.h"

extern "C" __global__ __aicore__ void moe_tutel_combine_x(
    GM_ADDR y_grad, GM_ADDR gates, GM_ADDR indices, GM_ADDR locations,
    GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(MoetutelcombinexTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoetutelcombinexTilingData, t, tiling);
    AscendC::TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
        // GET_TILING_DATA_WITH_STRUCT yields a const tiling reference under the
        // static-shape JIT codegen; the empty-path Init takes a mutable pointer
        // but only reads the shape fields, so copy to a mutable local.
        MoetutelcombinexTilingData tdMutable = t;
        ScatterAddEmptyKernel<DTYPE_Y_GRAD> op;
        op.Init(y_grad, gates, indices, locations, x_grad, &tdMutable, &pipe);
        op.Process();
    } else {
        const bool hasBatch = TILING_KEY_IS(TILING_KEY_NORMAL_BATCH);
        GM_ADDR ins[kMaxInputSlots] = {y_grad, gates, indices, locations};
        GM_ADDR outs[kMaxOutputSlots] = {x_grad};
        MoeTutelCombineXKernel<DTYPE_Y_GRAD> kernel;
        kernel.Init(ins, outs, &t);
        kernel.Process(hasBatch);
    }
}
