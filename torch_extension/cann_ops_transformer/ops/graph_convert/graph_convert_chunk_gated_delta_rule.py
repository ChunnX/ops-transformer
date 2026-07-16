# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
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


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, True],
        [False, False, False, False, False, False, True],
    )
    def ChunkGatedDeltaRule(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        actual_seq_lengths: Tensor,
        g: Optional[Tensor] = None,
        *,
        scale_value: float = 1.0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(ChunkGatedDeltaRule)
        .INPUT(query, TensorType({DT_BF16}))
        .INPUT(key, TensorType({DT_BF16}))
        .INPUT(value, TensorType({DT_BF16}))
        .INPUT(beta, TensorType({DT_BF16}))
        .INPUT(initial_state, TensorType({DT_BF16}))
        .INPUT(actual_seq_lengths, TensorType({DT_INT32}))
        .OPTIONAL_INPUT(g, TensorType({DT_FLOAT}))
        .OUTPUT(out, TensorType({DT_BF16}))
        .OUTPUT(final_state, TensorType({DT_BF16}))
        .ATTR(scale_value, Float, 1.0)
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "beta": beta,
            "initial_state": initial_state,
            "actual_seq_lengths": actual_seq_lengths,
            "g": g,
        }
        attrs = {
            "scale_value": attr.Float(scale_value),
        }
        outputs = ["out", "final_state"]

        return ge_op(
            op_type="ChunkGatedDeltaRule",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("ChunkGatedDeltaRule")
            .input("query", "DT_BF16")
            .input("key", "DT_BF16")
            .input("value", "DT_BF16")
            .input("beta", "DT_BF16")
            .input("initial_state", "DT_BF16")
            .input("actual_seq_lengths", "DT_INT32")
            .optional_input("g", "DT_FLOAT")
            .attr("scale_value", attr.Float(1.0))
            .output("out", "DT_BF16")
            .output("final_state", "DT_BF16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.chunk_gated_delta_rule.default
    )
    def convert_chunk_gated_delta_rule(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        beta: Tensor,
        initial_state: Tensor,
        actual_seq_lengths: Tensor,
        *,
        g: Optional[Tensor] = None,
        scale: float = 1.0,
        meta_outputs: TensorSpec = None,
    ):
        out, final_state = ChunkGatedDeltaRule(
            query=query,
            key=key,
            value=value,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            g=g,
            scale_value=scale,
        )
        return out, final_state

else:

    def convert_chunk_gated_delta_rule(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
