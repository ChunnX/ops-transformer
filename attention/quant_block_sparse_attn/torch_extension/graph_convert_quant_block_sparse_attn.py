# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import List, Optional

try:
    import torch
    import torch_npu
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    def QuantBlockSparseAttn(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        v_descale: Tensor,
        sparse_indices: Tensor,
        sparse_seq_len: Tensor,
        *,
        p_scale: Optional[Tensor] = None,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_kv: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        atten_mask: Optional[Tensor] = None,
        metadata: Optional[Tensor] = None,
        softmax_scale: float = 1.0,
        sparse_q_block_size: int = 128,
        sparse_kv_block_size: int = 128,
        layout_kv: str = "PA_BNBD",
        layout_q: str = "TND",
        layout_sparse_indices: str = "B_N_Qb_Kb",
        layout_out: str = "TND",
        quant_mode: int = 1,
        mask_mode: int = 3,
        return_softmax_lse: bool = False,
    ):
        """Build the QuantBlockSparseAttn GE node using its registered input/attribute contract."""
        return ge_op(
            op_type="QuantBlockSparseAttn",
            inputs={
                "query": query,
                "key": key,
                "value": value,
                "q_descale": q_descale,
                "k_descale": k_descale,
                "v_descale": v_descale,
                "sparse_indices": sparse_indices,
                "sparse_seq_len": sparse_seq_len,
                "p_scale": p_scale,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_kv": cu_seqlens_kv,
                "seqused_q": seqused_q,
                "seqused_kv": seqused_kv,
                "block_table": block_table,
                "atten_mask": atten_mask,
                "metadata": metadata,
            },
            attrs={
                "softmax_scale": attr.Float(softmax_scale),
                "sparse_q_block_size": attr.Int(sparse_q_block_size),
                "sparse_kv_block_size": attr.Int(sparse_kv_block_size),
                "layout_kv": attr.Str(layout_kv),
                "layout_q": attr.Str(layout_q),
                "layout_sparse_indices": attr.Str(layout_sparse_indices),
                "layout_out": attr.Str(layout_out),
                "quant_mode": attr.Int(quant_mode),
                "mask_mode": attr.Int(mask_mode),
                "return_softmax_lse": attr.Bool(return_softmax_lse),
            },
            outputs=["attention_out", "softmax_lse"],
            ir=IrDef("QuantBlockSparseAttn")
            .input("query", "DT_FLOAT8_E4M3FN")
            .input("key", "DT_FLOAT8_E4M3FN")
            .input("value", "DT_FLOAT8_E4M3FN")
            .input("q_descale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .input("k_descale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .input("v_descale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .input("sparse_indices", "DT_INT32")
            .input("sparse_seq_len", "DT_INT32")
            .optional_input("p_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("cu_seqlens_q", "DT_INT32")
            .optional_input("cu_seqlens_kv", "DT_INT32")
            .optional_input("seqused_q", "DT_INT32")
            .optional_input("seqused_kv", "DT_INT32")
            .optional_input("block_table", "DT_INT32")
            .optional_input("atten_mask", "DT_UINT8")
            .optional_input("metadata", "DT_INT32")
            .attr("softmax_scale", attr.Float(1.0))
            .attr("sparse_q_block_size", attr.Int(128))
            .attr("sparse_kv_block_size", attr.Int(128))
            .attr("layout_kv", attr.Str("PA_BNBD"))
            .attr("layout_q", attr.Str("TND"))
            .attr("layout_sparse_indices", attr.Str("B_N_Qb_Kb"))
            .attr("layout_out", attr.Str("TND"))
            .attr("quant_mode", attr.Int(1))
            .attr("mask_mode", attr.Int(3))
            .attr("return_softmax_lse", attr.Bool(False))
            .output("attention_out", "DT_BF16")
            .output("softmax_lse", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.npu_quant_block_sparse_attn.default
    )
    def convert_npu_quant_block_sparse_attn(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_descale: Tensor,
        k_descale: Tensor,
        v_descale: Tensor,
        sparse_indices: Tensor,
        sparse_seq_len: Tensor,
        *,
        p_scale: Optional[Tensor] = None,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_kv: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        atten_mask: Optional[Tensor] = None,
        metadata: Optional[Tensor] = None,
        softmax_scale: float = 1.0,
        sparse_q_block_size: int = 128,
        sparse_kv_block_size: int = 128,
        layout_kv: str = "PA_BNBD",
        layout_q: str = "TND",
        layout_sparse_indices: str = "B_N_Qb_Kb",
        layout_out: str = "TND",
        quant_mode: int = 1,
        mask_mode: int = 3,
        return_softmax_lse: bool = False,
        meta_outputs: List[TensorSpec] = None,
    ):
        return QuantBlockSparseAttn(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            sparse_indices,
            sparse_seq_len,
            p_scale=p_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            block_table=block_table,
            atten_mask=atten_mask,
            metadata=metadata,
            softmax_scale=softmax_scale,
            sparse_q_block_size=sparse_q_block_size,
            sparse_kv_block_size=sparse_kv_block_size,
            layout_kv=layout_kv,
            layout_q=layout_q,
            layout_sparse_indices=layout_sparse_indices,
            layout_out=layout_out,
            quant_mode=quant_mode,
            mask_mode=mask_mode,
            return_softmax_lse=return_softmax_lse,
        )

    def QuantBlockSparseAttnMetadata(
        sparse_seq_len: Tensor,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_kv: Optional[Tensor] = None,
        batch_size: int = 0,
        sparse_block_size_q: int = 128,
        sparse_block_size_k: int = 128,
        quant_mode: int = 1,
        mask_mode: int = 3,
        layout_q: str = "TND",
        layout_kv: str = "PA_BNBD",
        layout_sparse_indices: str = "B_N_Qb_Kb",
    ):
        """Build the QuantBlockSparseAttnMetadata GE node using its registered input/attribute contract."""
        stream_info = torch.npu.get_stream_limit(torch.npu.current_stream())
        aic_core_num = stream_info.get("cube_core_num")
        aiv_core_num = stream_info.get("vector_core_num")
        soc_version = torch.npu.get_device_properties().name
        return ge_op(
            op_type="QuantBlockSparseAttnMetadata",
            inputs={
                "sparse_seq_len": sparse_seq_len,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_kv": cu_seqlens_kv,
                "seqused_q": seqused_q,
                "seqused_kv": seqused_kv,
            },
            attrs={
                "batch_size": attr.Int(batch_size),
                "num_heads_q": attr.Int(num_heads_q),
                "num_heads_kv": attr.Int(num_heads_kv),
                "head_dim": attr.Int(head_dim),
                "sparse_block_size_q": attr.Int(sparse_block_size_q),
                "sparse_block_size_k": attr.Int(sparse_block_size_k),
                "quant_mode": attr.Int(quant_mode),
                "mask_mode": attr.Int(mask_mode),
                "layout_q": attr.Str(layout_q),
                "layout_kv": attr.Str(layout_kv),
                "layout_sparse_indices": attr.Str(layout_sparse_indices),
                "soc_version": attr.Str(soc_version),
                "aic_core_num": attr.Int(aic_core_num),
                "aiv_core_num": attr.Int(aiv_core_num),
            },
            outputs=["metadata"],
            ir=IrDef("QuantBlockSparseAttnMetadata")
            .input("sparse_seq_len", "DT_INT32")
            .optional_input("cu_seqlens_q", "DT_INT32")
            .optional_input("cu_seqlens_kv", "DT_INT32")
            .optional_input("seqused_q", "DT_INT32")
            .optional_input("seqused_kv", "DT_INT32")
            .required_attr("batch_size", attr.Int)
            .required_attr("num_heads_q", attr.Int)
            .required_attr("num_heads_kv", attr.Int)
            .required_attr("head_dim", attr.Int)
            .attr("sparse_block_size_q", attr.Int(128))
            .attr("sparse_block_size_k", attr.Int(128))
            .attr("quant_mode", attr.Int(1))
            .attr("mask_mode", attr.Int(3))
            .attr("layout_q", attr.Str("TND"))
            .attr("layout_kv", attr.Str("PA_BNBD"))
            .attr("layout_sparse_indices", attr.Str("B_N_Qb_Kb"))
            .required_attr("soc_version", attr.Str)
            .required_attr("aic_core_num", attr.Int)
            .required_attr("aiv_core_num", attr.Int)
            .output("metadata", "DT_INT32"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.npu_quant_block_sparse_attn_metadata.default
    )
    def convert_npu_quant_block_sparse_attn_metadata(
        sparse_seq_len: Tensor,
        num_heads_q: int,
        num_heads_kv: int,
        head_dim: int,
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_kv: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_kv: Optional[Tensor] = None,
        batch_size: int = 0,
        sparse_block_size_q: int = 128,
        sparse_block_size_k: int = 128,
        quant_mode: int = 1,
        mask_mode: int = 3,
        layout_q: str = "TND",
        layout_kv: str = "PA_BNBD",
        layout_sparse_indices: str = "B_N_Qb_Kb",
        meta_outputs: List[TensorSpec] = None,
    ):
        return QuantBlockSparseAttnMetadata(
            sparse_seq_len,
            num_heads_q,
            num_heads_kv,
            head_dim,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_k=sparse_block_size_k,
            quant_mode=quant_mode,
            mask_mode=mask_mode,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
        )
