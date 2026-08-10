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
        [False, False, False, False, False, False, False, False],
        [False, False, True, True, True, True, True, False],
    )
    def MsaIndexScore(
        query: Tensor,
        key: Tensor,
        block_table: Optional[Tensor],
        scale: Optional[Tensor],
        atten_mask: Optional[Tensor],
        actual_seq_lengths_query: Optional[Tensor],
        actual_seq_lengths_key: Optional[Tensor],
        start_loc: Tensor,
        *,
        sparse_mode: int = 3,
        dependencies=None,
        node_name=None,
    ):
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key": key,
            "block_table": block_table,
            "scale": scale,
            "atten_mask": atten_mask,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_key": actual_seq_lengths_key,
            "start_loc": start_loc,
        }
        attrs = {
            "sparse_mode": attr.Int(sparse_mode),
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
            .input("key", "DT_FLOAT16, DT_BF16, DT_INT8")
            .optional_input("block_table", "DT_INT32")
            .optional_input("scale", "DT_FLOAT")
            .optional_input("atten_mask", "DT_INT8")
            .optional_input("actual_seq_lengths_query", "DT_INT32")
            .optional_input("actual_seq_lengths_key", "DT_INT32")
            .input("start_loc", "DT_INT32")
            .attr("sparse_mode", attr.Int(3))
            .output("score", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.msa_index_score.default
    )
    def convert_msa_index_score(
        query: Tensor,
        key: Tensor,
        block_table: Optional[Tensor],
        scale: Optional[Tensor],
        atten_mask: Optional[Tensor],
        actual_seq_lengths_query: Optional[Tensor],
        actual_seq_lengths_key: Optional[Tensor],
        start_loc: Tensor,
        *,
        sparse_mode: int = 3,
        meta_outputs: TensorSpec = None,
    ):
        return MsaIndexScore(
            query=query,
            key=key,
            block_table=block_table,
            scale=scale,
            atten_mask=atten_mask,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            start_loc=start_loc,
            sparse_mode=sparse_mode,
        )

else:

    def convert_msa_index_score(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
