# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    from typing import Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False],
        [False, False, True, True, True, True, True],
    )
    def MoeFinalizeRouting(
        expanded_x: Tensor,
        expanded_row_idx: Tensor,
        x1: Optional[Tensor],
        x2: Optional[Tensor],
        bias: Optional[Tensor],
        scales: Optional[Tensor],
        expert_idx: Optional[Tensor],
        *,
        drop_pad_mode: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeFinalizeRoutingV2)\n
        .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .INPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(expert_idx, TensorType({DT_INT32}))\n
        .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .ATTR(drop_pad_mode, Int, 0)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "expanded_x": expanded_x,
            "expanded_row_idx": expanded_row_idx,
            "x1": x1,
            "x2": x2,
            "bias": bias,
            "scales": scales,
            "expert_idx": expert_idx,
        }

        attrs = {
            "drop_pad_mode": attr.Int(drop_pad_mode),
        }

        outputs = ["y"]

        return ge_op(
            op_type="MoeFinalizeRoutingV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MoeFinalizeRoutingV2")
            .input("expanded_x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .input("expanded_row_idx", "DT_INT32")
            .optional_input("x1", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("x2", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("bias", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("scales", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("expert_idx", "DT_INT32")
            .attr("drop_pad_mode", attr.Int(0))
            .output("y", "DT_FLOAT, DT_FLOAT16, DT_BF16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_finalize_routing.default
    )
    def convert_moe_finalize_routing(
        expanded_x: Tensor,
        expanded_row_idx: Tensor,
        x1: Optional[Tensor],
        x2: Optional[Tensor],
        bias: Optional[Tensor],
        scales: Optional[Tensor],
        expert_idx: Optional[Tensor],
        drop_pad_mode: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        return MoeFinalizeRouting(
            expanded_x=expanded_x,
            expanded_row_idx=expanded_row_idx,
            x1=x1,
            x2=x2,
            bias=bias,
            scales=scales,
            expert_idx=expert_idx,
            drop_pad_mode=drop_pad_mode,
        )
