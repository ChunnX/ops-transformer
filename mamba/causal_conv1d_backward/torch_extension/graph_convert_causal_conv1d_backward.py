# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
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


_ACTIVATION_MAP = {
    "silu": "silu",
    "none": "none",
}


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False],
        [False, False, False, True],
    )
    def CausalConv1dBackward(
        grad_y: Tensor,
        x: Tensor,
        weight: Tensor,
        query_start_loc: Optional[Tensor] = None,
        *,
        activation_mode: str = "silu",
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(CausalConv1dBackward)\n
        .INPUT(grad_y, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT32}))\n
        .ATTR(activation_mode, String, "silu")\n
        .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(grad_weight, TensorType({DT_BF16, DT_FLOAT16}))\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "grad_y": grad_y,
            "x": x,
            "weight": weight,
            "query_start_loc": query_start_loc,
        }
        attrs = {
            "activation_mode": attr.Str(activation_mode),
        }
        outputs = ["grad_x", "grad_weight"]

        return ge_op(
            op_type="CausalConv1dBackward",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("CausalConv1dBackward")
            .input("grad_y", "DT_BF16, DT_FLOAT16")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("weight", "DT_BF16, DT_FLOAT16")
            .optional_input("query_start_loc", "DT_INT32")
            .attr("activation_mode", attr.Str("silu"))
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_weight", "DT_BF16, DT_FLOAT16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.causal_conv1d_backward.default
    )
    def convert_causal_conv1d_backward(
        grad_y: Tensor,
        x: Tensor,
        weight: Tensor,
        query_start_loc: Optional[Tensor] = None,
        *,
        activation: str = "silu",
        meta_outputs: TensorSpec = None,
    ):
        activation_mode = _ACTIVATION_MAP.get(activation, activation)
        grad_x, grad_weight = CausalConv1dBackward(
            grad_y=grad_y,
            x=x,
            weight=weight,
            query_start_loc=query_start_loc,
            activation_mode=activation_mode,
        )
        return grad_x, grad_weight

else:

    def convert_causal_conv1d_backward(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
