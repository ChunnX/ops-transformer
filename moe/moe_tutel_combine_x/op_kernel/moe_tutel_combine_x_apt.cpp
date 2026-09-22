/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * op_kernel/moe_tutel_combine_x_apt.cpp
 * =============================================================================
 * Role: Ascend C kernel entry point for MoeTutelCombineX on arch35 (Ascend 950).
 *        Runtime tiling-key dispatch (repo convention — no ASCENDC_TPL):
 *          key=0 (TILING_KEY_NORMAL_NO_BATCH): normal no-batch — y_grad [EC,H],
 *                metadata [S]
 *          key=1 (TILING_KEY_EMPTY):           empty / all-zero short-circuit
 *          key=2 (TILING_KEY_NORMAL_BATCH):    normal batch — y_grad [B,EC,H],
 *                metadata [B,S]
 *
 *        The host tiling selects the sub-kernel via SetTilingKey(0/1/2); the
 *        compile codegen instantiates one sub-kernel per TILING_KEY_IS branch
 *        per dtype (DTYPE_Y_GRAD macro binds float / half / bfloat16_t).
 *
 * Entry form (arch35 TPL-free entry paradigm, same shape as the repo arch35
 *   operators, e.g. moe_finalize_routing_v2_apt.cpp):
 *   KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) (pure-Vector operator,
 *   no Cube) + REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT.
 *
 * workspace: reserved placeholder, never consumed (no cross-core reduction /
 *   intermediate spill). The host tiling declares the framework-reserved
 *   16 MiB sys workspace so the kernel's parameter table stays aligned.
 *
 * Operator name variants:
 *   - PascalCase:   MoeTutelCombineX
 *   - snake_case:   moe_tutel_combine_x  <- device kernel symbol name
 *   - UPPER_SNAKE:  MOE_TUTEL_COMBINE_X
 * =============================================================================
 */

// kernel_operator.h: Ascend C kernel framework (AscendC:: namespace, tiling macros).
#include "kernel_operator.h"
// arch35 kernel classes: normal path (key=0/2) and empty path (key=1).
#include "arch35/moe_tutel_combine_x_kernel.h"
#include "arch35/moe_tutel_combine_x_empty.h"
// TilingData struct shared between host tiling code and this device entry.
#include "arch35/moe_tutel_combine_x_tiling_struct.h"

// TilingKey 传递: host tiling SetTilingKey(0/1/2) -> TILING_KEY_IS 运行时分发
// DTYPE 传递: OpDef .DataType 列 -> DTYPE_Y_GRAD 宏注入
extern "C" __global__ __aicore__ void moe_tutel_combine_x(GM_ADDR y_grad, GM_ADDR gates, GM_ADDR indices,
                                                          GM_ADDR locations, GM_ADDR x_grad, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); // 纯 Vector 算子（无 Cube）
    (void)workspace; // 形参保留占位不消费（无跨核归约 / 中间结果落盘）
    REGISTER_TILING_DEFAULT(MoetutelcombinexTilingData);
    GET_TILING_DATA_WITH_STRUCT(MoetutelcombinexTilingData, t, tiling);
    AscendC::TPipe pipe;

    if (TILING_KEY_IS(TILING_KEY_EMPTY)) {
        // GET_TILING_DATA_WITH_STRUCT yields a const tiling reference under the
        // static-shape JIT codegen; the empty-path Init takes a mutable pointer
        // but only reads the shape fields, so copy to a mutable local.
        MoetutelcombinexTilingData tdMutable = t;
        MoeTutelCombineXEmptyKernel<DTYPE_Y_GRAD> op; // 空 / 全零输出路径（key=1）
        op.Init(y_grad, gates, indices, locations, x_grad, &tdMutable, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_NORMAL_BATCH)) {
        GM_ADDR ins[kMaxInputSlots] = {y_grad, gates, indices, locations};
        GM_ADDR outs[kMaxOutputSlots] = {x_grad};
        MoeTutelCombineXKernel<DTYPE_Y_GRAD, false, true> op; // 主路径带 batch（key=2）
        op.Init(ins, outs, &t);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_NORMAL_NO_BATCH)) {
        // Every branch must be an explicit TILING_KEY_IS(...): the compile
        // codegen instantiates one loadable sub-kernel symbol per explicit
        // branch, so a bare `else` fallback leaves key=0 without a
        // `<kernel>_0` symbol and the runtime binary registration fails.
        GM_ADDR ins[kMaxInputSlots] = {y_grad, gates, indices, locations};
        GM_ADDR outs[kMaxOutputSlots] = {x_grad};
        MoeTutelCombineXKernel<DTYPE_Y_GRAD, false, false> op; // 主路径无 batch（key=0）
        op.Init(ins, outs, &t);
        op.Process();
    } else {
        // Unreachable: the host tiling only sets tiling keys 0/1/2.
        return;
    }
}
