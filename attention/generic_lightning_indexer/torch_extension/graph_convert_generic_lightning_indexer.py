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
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr
    from typing import Optional, List

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    def _stride0(tensor) -> int:
        if tensor is None:
            return -1
        for accessor in (
            lambda t: t.stride(0),
            lambda t: t.stride[0],
            lambda t: t.tensor.stride(0),
        ):
            try:
                val = accessor(tensor)
                if callable(val):
                    val = val()
                return int(val)
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
        return -1

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.generic_lightning_indexer.default
    )
    def convert_generic_lightning_indexer(
        query: Tensor,
        key: Tensor,
        *,
        weights: Optional[Tensor] = None,
        actual_seq_q: Optional[Tensor] = None,
        actual_seq_k: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        q_descale: Optional[Tensor] = None,
        k_descale: Optional[Tensor] = None,
        output_idx_offset: Optional[Tensor] = None,
        metadata: Optional[Tensor] = None,
        start_loc: Optional[Tensor] = None,
        topk: int = 2048,
        cmp_ratio: int = 1,
        layout_q: str = "BSND",
        layout_k: str = "BSND",
        sparse_mode: int = 0,
        act_type: int = 0,
        has_weighted_sum: bool = True,
        expand_indices: bool = False,
        append_tail: bool = False,
        quant_mode: int = -1,
        return_value: bool = False,
        select_mode: int = 0,
        key_reduce_mode: int = 0,
        score_scale: float = 1.0,
        key_stride0: int = -1,
        k_descale_stride0: int = -1,
        max_seqlen_q: int = -1,
        init_blocks: int = 0,
        local_blocks: int = 0,
        meta_outputs: List[TensorSpec] = None,
    ):
        ks0 = key_stride0 if key_stride0 >= 0 else _stride0(key)
        kds0 = k_descale_stride0 if k_descale_stride0 >= 0 else _stride0(k_descale)
        return torchair.ge.custom_op(
            "GenericLightningIndexer",
            inputs={
                "query": query,
                "key": key,
                "weights": weights,
                "actual_seq_q": actual_seq_q,
                "actual_seq_k": actual_seq_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "block_table": block_table,
                "q_descale": q_descale,
                "k_descale": k_descale,
                "output_idx_offset": output_idx_offset,
                "metadata": metadata,
                "start_loc": start_loc,
            },
            attrs={
                "topk": attr.Int(topk),
                "cmp_ratio": attr.Int(cmp_ratio),
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "sparse_mode": attr.Int(sparse_mode),
                "act_type": attr.Int(act_type),
                "has_weighted_sum": attr.Bool(has_weighted_sum),
                "expand_indices": attr.Bool(expand_indices),
                "append_tail": attr.Bool(append_tail),
                "quant_mode": attr.Int(quant_mode),
                "return_value": attr.Bool(return_value),
                "select_mode": attr.Int(select_mode),
                "key_reduce_mode": attr.Int(key_reduce_mode),
                "score_scale": attr.Float(score_scale),
                "key_stride0": attr.Int(ks0),
                "k_descale_stride0": attr.Int(kds0),
                "max_seqlen_q": attr.Int(max_seqlen_q),
                "init_blocks": attr.Int(init_blocks),
                "local_blocks": attr.Int(local_blocks),
            },
            outputs=["sparse_indices_out", "sparse_values_out"],
        )
