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
 * \file moe_tutel_combine_gates_apt.cpp
 * \brief MoeTutelCombineGates Ascend C kernel entry.
 */

#include "kernel_operator.h"                                // Ascend C kernel framework (AscendC:: namespace)
#include "./arch35/moe_tutel_combine_gates_struct.h"        // ASCENDC_TPL_ARGS_DECL / ASCENDC_TPL_SEL (DTYPE_MODE axis)
#include "./arch35/moe_tutel_combine_gates_tiling_struct.h" // MoeTutelCombineGatesTilingData struct
#include "./arch35/moe_tutel_combine_gates_16b.h" // Branch-0 / Branch-2 kernel class (MoeTutelCombineGatesB16)
#include "./arch35/moe_tutel_combine_gates_32b.h" // Branch-1 kernel class (MoeTutelCombineGatesB32)

// NOTE: the ASCENDC_TPL declarations must stay visible in this translation unit —
// the kernel codegen reads them to instantiate the entry for every TPL combination
// (`moe_tutel_combine_gates<0>(...)` / `<1>` / `<2>`). Dropping the include makes the
// generated caller emit a template-argument-less call, which fails to compile.

// The TilingData struct is NOT templated on the key: a single
// MoeTutelCombineGatesTilingData (7 × int32_t,
// TilingData.md §1) serves every tilingKey branch, so no per-branch type alias exists
// and all three arms call the same GET_TILING_DATA_WITH_STRUCT.

// ===========================================================================
// R4-4: TilingData ABI guard (the host side carries the same assert —
// op_host/moe_tutel_combine_gates_tiling_arch35.cpp, right below the
// platform constants).
//
// REGISTER_TILINGDATA_SIZE(...) is sizeof(tiling_struct) and GET_TILING_DATA_WITH_STRUCT
// below copies exactly that many bytes: the tiling buffer has no length field and no
// version field, so the field set IS the ABI. The mismatch is asymmetric — a new host
// with an old kernel is harmless (8 extra bytes nobody reads) while a NEW KERNEL over
// an OLD host .so reads 8 bytes of stale memory into ubRowElems / ubFp32Elems, and
// those two values are used verbatim as TPipe::InitBuffer sizes (a huge allocation /
// hang, or an under-sized route). The assert turns a one-sided field change into a
// compile error; the runtime backstop for an already-built pair lives in the Init() of
// arch35/moe_tutel_combine_gates_{16b,32b}.h, and the deployment rule (host .so and
// kernel .o must be replaced together) is written down in
// the TilingData ABI note in op_host/moe_tutel_combine_gates_tiling_arch35.cpp.
// ===========================================================================
static_assert(sizeof(MoeTutelCombineGatesTilingData) == 7 * sizeof(int32_t),
              "MoeTutelCombineGates TilingData ABI changed: host .so and kernel .o must be rebuilt and "
              "replaced together (see the TilingData ABI note in op_host/moe_tutel_combine_gates_tiling_arch35.cpp)");

// ===========================================================================
// template<int DTYPE_MODE> __global__ __aicore__ void moe_tutel_combine_gates(
//     GM_ADDR x, GM_ADDR y_grad, GM_ADDR indices, GM_ADDR locations,
//     GM_ADDR gates_grad, GM_ADDR workspace, GM_ADDR tiling)
//
// Kernel entry for MoeTutelCombineGates. Each AIV core executes this once per launch.
//
// Template parameter DTYPE_MODE (0/1/2) — the only TPL axis, declared in
// arch35/moe_tutel_combine_gates_struct.h and selected from the host-side tiling key
// (TilingKey.md, REQUIREMENTS §7.5 A10).
//
// Parameters (all GM_ADDR = global memory pointers on device), in prototype order:
//   x          — input token embedding matrix [S, H]
//   y_grad     — expert row buffer [B, C, H] (rows gathered via indices/locations)
//   indices    — expert id per task, int32
//   locations  — slot position per task, int32
//   gates_grad — output gate gradient, shape = indices.shape
//   workspace  — ignored: the operator uses no workspace (A12)
//   tiling     — tiling data buffer (contains MoeTutelCombineGatesTilingData)
//
// Behaviour:
//   1. REGISTER_NONE_TILING: register the kernel task with the AICore runtime
//   2. KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY): pure Vector operator (B5)
//   3. one AscendC::TPipe, created here and passed by pointer into the kernel class
//   4. dispatch on DTYPE_MODE and run the branch's kernel class
// ===========================================================================
template <int DTYPE_MODE>
__global__ __aicore__ void moe_tutel_combine_gates(GM_ADDR x, GM_ADDR y_grad, GM_ADDR indices, GM_ADDR locations,
                                                   GM_ADDR gates_grad, GM_ADDR workspace, GM_ADDR tiling)
{
    // REGISTER_NONE_TILING: registers the kernel with the AICore runtime.
    // "NONE_TILING" means the tiling data is read manually via GET_TILING_DATA_WITH_STRUCT
    // rather than being auto-tiled by the framework. It must precede that macro, which
    // otherwise does not know the TilingData type name.
    REGISTER_NONE_TILING;

    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY):
    //   Sets this kernel to run on AIV (AI Vector) cores only.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // TPipe is created once here (Kernel.md §1) and handed to the kernel class by
    // pointer; the class owns no TPipe of its own.
    AscendC::TPipe pipe;

    if constexpr (DTYPE_MODE == MOE_TUTEL_COMBINE_GATES_FP32_MODE) {
        // ---- tilingKey 1 (fp32 branch, DESIGN-BRANCH-1.md) ----
        // The whole numeric chain runs in fp32 (accumulator dtype == input dtype),
        // so this branch carries no Cast at all (P = 9 UB routes, §4).
        GET_TILING_DATA_WITH_STRUCT(MoeTutelCombineGatesTilingData, tilingData, tiling);
        MoeTutelCombineGatesB32<float> op;
        op.Init(x, y_grad, indices, locations, gates_grad, &tilingData, &pipe);
        op.Process();
    } else if constexpr (DTYPE_MODE == MOE_TUTEL_COMBINE_GATES_FP16_MODE) {
        // ---- tilingKey 2 (fp16 branch, DESIGN-BRANCH-2.md) ----
        // Shares the _16b.h template class with Branch-0 and differs only in `T`
        // (DESIGN-BRANCH-0.md §0): all of the fp16 branch's behaviour below the
        // entry point — the H == 0 early short circuit, the two independent
        // barriers around the task-level DMAs (#2 / #3), the four-condition
        // out-of-range zero mask (e<0 / e>=B / p<0 / p>=C) and the faithful
        // NaN/±Inf propagation — is carried by the shared class, so this arm only
        // instantiates it with `half` (same dispatch shape as the reference
        // implementation's `MoeTutelCombineGates<half>` arm).
        GET_TILING_DATA_WITH_STRUCT(MoeTutelCombineGatesTilingData, tilingData, tiling);
        MoeTutelCombineGatesB16<half> op;
        op.Init(x, y_grad, indices, locations, gates_grad, &tilingData, &pipe);
        op.Process();
    } else {
        // ---- tilingKey 0 (bf16 branch, DESIGN-BRANCH-0.md; the A10 default) ----
        GET_TILING_DATA_WITH_STRUCT(MoeTutelCombineGatesTilingData, tilingData, tiling);
        MoeTutelCombineGatesB16<bfloat16_t> op;
        op.Init(x, y_grad, indices, locations, gates_grad, &tilingData, &pipe);
        op.Process();
    }
    (void)workspace; // A12: the operator has no workspace at all
}
