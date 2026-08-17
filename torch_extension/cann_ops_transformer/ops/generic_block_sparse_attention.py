# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


class GenericBlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_generic_block_sparse_attention")

    def sources(self):
        return ["ops/csrc/generic_block_sparse_attention.cpp"]

    def schema(self):
        return (
            "npu_generic_block_sparse_attention("
            "Tensor query, Tensor key, Tensor value, Tensor sparse_block_idx, Tensor sparse_block_count, "
            "int[] block_shape, *, "
            "Tensor? metadata=None, Tensor? atten_mask=None, "
            "Tensor? q_dequant_scale=None, Tensor? k_dequant_scale=None, Tensor? v_dequant_scale=None, "
            "Tensor? p_quant_scale=None, Tensor? cu_seq_lengths_q=None, Tensor? cu_seq_lengths_kv=None, "
            "Tensor? seq_used_q=None, Tensor? seq_used_kv=None, Tensor? block_table=None, "
            "int is_packed_gqa=1, str layout_q=\"TND\", str layout_kv=\"PAGED_BBND\", "
            "float scale_value=0.0, int mask_type=1, int quant_type=0, float dst_type_max=0.0, "
            "int softmax_precision=0, int win_left=-1, int win_right=-1, "
            "int return_softmax_lse=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_generic_block_sparse_attention_meta(
            query,
            key,
            value,
            sparse_block_idx,
            sparse_block_count,
            block_shape,
            *,
            metadata=None,
            atten_mask=None,
            q_dequant_scale=None,
            k_dequant_scale=None,
            v_dequant_scale=None,
            p_quant_scale=None,
            cu_seq_lengths_q=None,
            cu_seq_lengths_kv=None,
            seq_used_q=None,
            seq_used_kv=None,
            block_table=None,
            is_packed_gqa=1,
            layout_q="TND",
            layout_kv="PAGED_BBND",
            scale_value=0.0,
            mask_type=1,
            quant_type=0,
            dst_type_max=0.0,
            softmax_precision=0,
            win_left=-1,
            win_right=-1,
            return_softmax_lse=0,
        ):
            attention_out = torch.empty_like(query, device="meta")
            if return_softmax_lse:
                if query.dim() == 3:
                    softmax_lse = torch.empty(
                        (query.shape[0], query.shape[1], 1), dtype=torch.float32, device="meta"
                    )
                elif layout_q == "BNSD":
                    softmax_lse = torch.empty(
                        (query.shape[0], query.shape[1], query.shape[2], 1),
                        dtype=torch.float32,
                        device="meta",
                    )
                else:
                    softmax_lse = torch.empty(
                        (query.shape[0], query.shape[2], query.shape[1], 1),
                        dtype=torch.float32,
                        device="meta",
                    )
            else:
                softmax_lse = torch.empty((0,), dtype=torch.float32, device="meta")
            return attention_out, softmax_lse


if getattr(torch_npu, "_ops_transformer_generic_block_sparse_attention_registered", False):
    npu_generic_block_sparse_attention = torch_npu.npu_generic_block_sparse_attention
else:
    _op_builder = GenericBlockSparseAttentionOpBuilder()

    def _load_op_module():
        return _op_builder.load()

    @impl(AS_LIBRARY, _op_builder.name, "PrivateUse1")
    def npu_generic_block_sparse_attention(
        query,
        key,
        value,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
        *,
        metadata=None,
        atten_mask=None,
        q_dequant_scale=None,
        k_dequant_scale=None,
        v_dequant_scale=None,
        p_quant_scale=None,
        cu_seq_lengths_q=None,
        cu_seq_lengths_kv=None,
        seq_used_q=None,
        seq_used_kv=None,
        block_table=None,
        is_packed_gqa=1,
        layout_q="TND",
        layout_kv="PAGED_BBND",
        scale_value=0.0,
        mask_type=1,
        quant_type=0,
        dst_type_max=0.0,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
        return_softmax_lse=0,
    ):
        return _load_op_module().npu_generic_block_sparse_attention(
            query,
            key,
            value,
            sparse_block_idx,
            sparse_block_count,
            metadata,
            atten_mask,
            q_dequant_scale,
            k_dequant_scale,
            v_dequant_scale,
            p_quant_scale,
            cu_seq_lengths_q,
            cu_seq_lengths_kv,
            seq_used_q,
            seq_used_kv,
            block_table,
            block_shape,
            is_packed_gqa,
            layout_q,
            layout_kv,
            scale_value,
            mask_type,
            quant_type,
            dst_type_max,
            softmax_precision,
            win_left,
            win_right,
            return_softmax_lse,
        )

    torch_npu.npu_generic_block_sparse_attention = npu_generic_block_sparse_attention
    torch_npu._ops_transformer_generic_block_sparse_attention_registered = True


torch.compiler.allow_in_graph(npu_generic_block_sparse_attention)
