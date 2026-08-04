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
        [False, False, False, False, False, False, False],
        [False, False, False, True, False, False, False],
    )
    def MsaIndexScore(
        query: Tensor,
        key_cache: Tensor,
        block_table: Tensor,
        scale: Optional[Tensor],
        actual_seq_lengths_query: Tensor,
        actual_seq_lengths_key: Tensor,
        start_loc: Tensor,
        *,
        block_size: int = 128,
        fill_invalid_tail: bool = True,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MsaIndexScore)\n
        .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))\n
        .INPUT(key_cache, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))\n
        .INPUT(block_table, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))\n
        .INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))\n
        .INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))\n
        .INPUT(start_loc, TensorType({DT_INT32}))\n
        .OUTPUT(score, TensorType({DT_FLOAT}))\n
        .ATTR(block_size, Int, 128)\n
        .ATTR(fill_invalid_tail, Bool, true)
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key_cache": key_cache,
            "block_table": block_table,
            "scale": scale,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_key": actual_seq_lengths_key,
            "start_loc": start_loc,
        }
        attrs = {
            "block_size": attr.Int(block_size),
            "fill_invalid_tail": attr.Bool(fill_invalid_tail),
        }
        outputs = ["score"]
        return ge_op(
            op_type="MsaIndexScore",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MsaIndexScore")
            .input("query", "DT_FLOAT16, DT_BF16")
            .input("key_cache", "DT_FLOAT16, DT_BF16, DT_INT8")
            .input("block_table", "DT_INT32")
            .optional_input("scale", "DT_FLOAT")
            .input("actual_seq_lengths_query", "DT_INT32")
            .input("actual_seq_lengths_key", "DT_INT32")
            .input("start_loc", "DT_INT32")
            .attr("block_size", attr.Int(128))
            .attr("fill_invalid_tail", attr.Bool(True))
            .output("score", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.msa_index_score.default
    )
    def convert_msa_index_score(
        query: Tensor,
        key_cache: Tensor,
        block_table: Tensor,
        actual_seq_lengths_query: Tensor,
        actual_seq_lengths_key: Tensor,
        start_loc: Tensor,
        *,
        scale: Optional[Tensor] = None,
        block_size: int = 128,
        fill_invalid_tail: bool = True,
        meta_outputs: TensorSpec = None,
    ):
        """GE converter for msa_index_score."""
        return MsaIndexScore(
            query=query,
            key_cache=key_cache,
            block_table=block_table,
            scale=scale,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            start_loc=start_loc,
            block_size=block_size,
            fill_invalid_tail=fill_invalid_tail,
        )

else:

    def convert_msa_index_score(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
