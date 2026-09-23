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
 * \file moe_tutel_combine_gates_def.cpp
 * \brief MoeTutelCombineGates operator definition (OpDef).
 */

#include "register/op_def_registry.h" // OpDef base class, OP_ADD macro

namespace ops {

// ---------------------------------------------------------------------------
// class MoeTutelCombineGates : public OpDef
//
// Operator definition class. The constructor chain-calls methods to define
// inputs and outputs using a builder pattern.
//
// Constructor parameter: name — operator instance name (for graph construction)
//   This is NOT the operator type name; it's the instance name like "moe_tutel_combine_gates_1".
//
// KEY METHODS USED:
//   this->Input("name")   — begins defining a named input tensor
//     .ParamType(REQUIRED/OPTIONAL)  — whether the input must be provided
//     .DataType({...})              — list of supported data types
//     .Format({...})                — list of supported tensor formats per dtype
//     .UnknownShapeFormat({...})    — format for dynamic (unknown) shapes
//     .AutoContiguous()             — non-contiguous tensors are made contiguous by
//                                     the framework before the kernel runs
//
//   this->Output("name")  — begins defining a named output tensor (same API as Input)
//
//   OpAICoreConfig        — configuration for AICore kernel compilation
//     .DynamicCompileStaticFlag(true)   — enable dynamic compile for static shapes
//     .DynamicFormatFlag(false)         — disable dynamic format selection
//     .DynamicRankSupportFlag(true)     — enable dynamic rank support
//     .DynamicShapeSupportFlag(true)    — enable dynamic shape support
//     .NeedCheckSupportFlag(false)      — skip support check at compile time
//     (PrecisionReduceFlag is deliberately not set — see Interface.md OpAICoreConfig section)
//     .ExtendCfgInfo(key, value)        — set extended configuration
//
//   this->AICore().SetTiling(optiling::TilingFunc).AddConfig("chipname", config)
//     — bind the tiling callback and associate the AICore config with a chip
// ---------------------------------------------------------------------------
class MoeTutelCombineGates : public OpDef {
public:
    explicit MoeTutelCombineGates(const char *name)
        : OpDef(name)
    {
        // --- Input: x ---
        // Token embedding matrix [S, H]. REQUIRED.
        // 3 dtypes: BF16, FP16, FP32 — each with FORMAT_ND (N-dimensional) layout.
        // .AutoContiguous()：算子按连续语义取行，声明「进入 kernel 的输入必须连续」。
        // 本算子仅保留 GE 图模式（aclnn 调用方式及其自动生成桩已删除）：图输入由
        // 框架喂入 / 分配，`.AutoContiguous()` 作为契约声明保留。
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Input: y_grad ---
        // Expert row buffer [B, C, H] whose rows are gathered by indices/locations.
        // REQUIRED. Same dtype list and format as x.
        this->Input("y_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Input: indices ---
        // Expert id of every (batch, sample) task. REQUIRED, int32 only.
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Input: locations ---
        // Slot position inside the expert of every (batch, sample) task.
        // REQUIRED, int32 only, same shape as indices.
        this->Input("locations")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Output: gates_grad ---
        // Per-task gate gradient, shape = indices.shape (proto.md §4).
        // REQUIRED. dtype always equals the floating-point input dtype.
        //
        // .AutoContiguous()：本算子仅保留 GE 图模式（aclnn 调用方式及其自动生成桩
        // 已删除）。图模式下输出 gates_grad 由 GE 按 InferShape 推导出的 shape 分配，
        // 天然连续，调用方不自行分配输出内存；kernel / tiling 层不感知输出 stride，
        // 按连续语义写回。该声明与四输入的声明形式保持一致，作为「输出按连续语义
        // 写回」的契约登记。
        this->Output("gates_grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // --- Attributes: none ---
        // proto.md §1: 本算子无属性：官方 IR 无 .ATTR，参考实现 OpDef 无 .Attr() 调用；
        // samples / hidden / batch / capacity 全部由输入 shape 推出，不得新增 .Attr(...)。

        // --- Inference registration (proto.md §1) ---
        // proto.md declares: this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        // This CANN release ships no ge::InferShape / ge::InferDataType symbols
        // (register/op_def.h accepts only gert::OpImplRegisterV2 kernels), so the two
        // callbacks are registered through the operator-impl macros instead — same
        // functions, same signatures, different registration entry point:
        // register/op_def.h accepts only gert::OpImplRegisterV2 kernels), so the two
        // callbacks are registered through the operator-impl macros instead — same
        // functions, same signatures, different registration entry point:
        //   InferShape    → IMPL_OP_INFERSHAPE(MoeTutelCombineGates).InferShape(...)
        //                   in op_host/moe_tutel_combine_gates_infershape.cpp
        //   InferDataType → IMPL_OP_INFERSHAPE(MoeTutelCombineGates).InferDataType(...)
        //                   in op_host/moe_tutel_combine_gates_infershape.cpp
        // 两个回调挂在同一个 OpImplRegisterV2 上，即「infershape 双挂」：只挂 InferShape 会让
        // GE 在本注册面上拿不到输出 dtype 推导。

        // --- AICore Configuration for Ascend 950 ---
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig
            .DynamicCompileStaticFlag(true) // JIT compile for static shapes
            .DynamicFormatFlag(false)       // Fixed format (no dynamic selection)
            .DynamicRankSupportFlag(true)   // Support variable ranks at runtime
            .DynamicShapeSupportFlag(true)  // Support variable shapes at runtime
            .NeedCheckSupportFlag(false)    // Skip chip support check
            // PrecisionReduceFlag is deliberately NOT set (Interface.md OpAICoreConfig section):
            // it lets the compiler reduce intermediate precision (fp32 -> fp16), which is
            // opposite to the A9 fp32-intermediate-accumulation requirement; no trusted source
            // confirms its effect on this operator, and the reference implementation never sets
            // it either. A9 is guaranteed by the kernel's explicit Cast -> Mul(fp32) ->
            // compensated-reduction chain (Kernel.md §9.1/§9.2/§9.4), not by this flag.
            .ExtendCfgInfo("opFile.value",
                           "moe_tutel_combine_gates_apt"); // Links to kernel file moe_tutel_combine_gates_apt.cpp
        // AddConfig associates the AICore config with the "ascend950" chip.
        // The tiling callbacks are registered by IMPL_OP_OPTILING(MoeTutelCombineGates)
        // in op_host/moe_tutel_combine_gates_tiling_arch35.cpp.
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

// OP_ADD(MoeTutelCombineGates):
//   Registers the MoeTutelCombineGates OpDef class with the CANN operator definition registry.
//   This makes the operator visible to op_build and the rest of the CANN toolchain.
OP_ADD(MoeTutelCombineGates);
} // namespace ops
