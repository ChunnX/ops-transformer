# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
"""PyTorch registration for GenericSparseAttentionGrad."""

from typing import List, Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY, OpBuilder

GSAG_METADATA_SIZE = 4096
GSAG_METADATA_OP_NAME = "generic_sparse_attention_grad_metadata"


class GenericSparseAttentionGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("generic_sparse_attention_grad")

    def sources(self):
        return ["ops/csrc/generic_sparse_attention_grad.cpp"]

    def schema(self):
        return [
            "generic_sparse_attention_grad_metadata("
            "Tensor rsvd_block_idx, Tensor rsvd_block_count, "
            "Tensor? cu_seq_lengths=None, Tensor? cu_seq_lengths_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int max_q_seqlen=0, int max_kv_seqlen=0, int num_q_heads=0, int num_kv_heads=0, int head_dim=128, "
            "int[] block_shape=[1, 128], int is_packed_gqa=1, "
            'str q_input_layout="TND", str kv_input_layout="TND", '
            "int mask_type=0, int softmax_precision=0, "
            "int window_size_left=-1, int window_size_right=-1) -> Tensor",
            "generic_sparse_attention_grad("
            "Tensor query, Tensor key, Tensor value, Tensor dout, Tensor out, Tensor lse, "
            "Tensor rsvd_block_idx, Tensor rsvd_block_count, Tensor metadata, "
            "Tensor? atten_mask=None, Tensor? cu_seq_lengths=None, Tensor? cu_seq_lengths_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int[] block_shape=[0], int is_packed_gqa=1, "
            'str q_input_layout="TND", str kv_input_layout="TND", '
            "float scale_value=1.0, int mask_type=0, int softmax_precision=0, "
            "int window_size_left=-1, int window_size_right=-1) -> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + GSAG_METADATA_OP_NAME)
        def generic_sparse_attention_grad_metadata_meta(
            rsvd_block_idx: torch.Tensor,
            rsvd_block_count: torch.Tensor,
            cu_seq_lengths: Optional[torch.Tensor] = None,
            cu_seq_lengths_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            max_q_seqlen: int = 0,
            max_kv_seqlen: int = 0,
            num_q_heads: int = 0,
            num_kv_heads: int = 0,
            head_dim: int = 128,
            block_shape: Optional[List[int]] = None,
            is_packed_gqa: int = 1,
            q_input_layout: str = "TND",
            kv_input_layout: str = "TND",
            mask_type: int = 0,
            softmax_precision: int = 0,
            window_size_left: int = -1,
            window_size_right: int = -1,
        ) -> torch.Tensor:
            del (
                rsvd_block_count,
                cu_seq_lengths,
                cu_seq_lengths_kv,
                seqused_q,
                seqused_kv,
                max_q_seqlen,
                max_kv_seqlen,
                num_q_heads,
                num_kv_heads,
                head_dim,
                block_shape,
                is_packed_gqa,
                q_input_layout,
                kv_input_layout,
                mask_type,
                softmax_precision,
                window_size_left,
                window_size_right,
            )
            return torch.empty((GSAG_METADATA_SIZE,), dtype=torch.int64, device=rsvd_block_idx.device)

        @torch.library.register_fake("cann_ops_transformer::generic_sparse_attention_grad")
        def generic_sparse_attention_grad_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            dout: torch.Tensor,
            out: torch.Tensor,
            lse: torch.Tensor,
            rsvd_block_idx: torch.Tensor,
            rsvd_block_count: torch.Tensor,
            metadata: torch.Tensor,
            atten_mask: Optional[torch.Tensor] = None,
            cu_seq_lengths: Optional[torch.Tensor] = None,
            cu_seq_lengths_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            block_shape: Optional[List[int]] = None,
            is_packed_gqa: int = 1,
            q_input_layout: str = "TND",
            kv_input_layout: str = "TND",
            scale_value: float = 1.0,
            mask_type: int = 0,
            softmax_precision: int = 0,
            window_size_left: int = -1,
            window_size_right: int = -1,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del (
                dout,
                out,
                lse,
                rsvd_block_idx,
                rsvd_block_count,
                metadata,
                atten_mask,
                cu_seq_lengths,
                cu_seq_lengths_kv,
                seqused_q,
                seqused_kv,
                block_shape,
                is_packed_gqa,
                q_input_layout,
                kv_input_layout,
                scale_value,
                mask_type,
                softmax_precision,
                window_size_left,
                window_size_right,
            )
            return (
                torch.empty_like(query, device="meta"),
                torch.empty_like(key, device="meta"),
                torch.empty_like(value, device="meta"),
            )


generic_sparse_attention_grad_op_builder = GenericSparseAttentionGradOpBuilder()


@impl(AS_LIBRARY, GSAG_METADATA_OP_NAME, "PrivateUse1")
def generic_sparse_attention_grad_metadata(
    rsvd_block_idx: torch.Tensor,
    rsvd_block_count: torch.Tensor,
    cu_seq_lengths: Optional[torch.Tensor] = None,
    cu_seq_lengths_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    max_q_seqlen: int = 0,
    max_kv_seqlen: int = 0,
    num_q_heads: int = 0,
    num_kv_heads: int = 0,
    head_dim: int = 128,
    block_shape: Optional[List[int]] = None,
    is_packed_gqa: int = 1,
    q_input_layout: str = "TND",
    kv_input_layout: str = "TND",
    mask_type: int = 0,
    softmax_precision: int = 0,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> torch.Tensor:
    if block_shape is None:
        block_shape = [1, 128]
    op_module = generic_sparse_attention_grad_op_builder.load()
    return op_module.generic_sparse_attention_grad_metadata(
        rsvd_block_idx,
        rsvd_block_count,
        cu_seq_lengths,
        cu_seq_lengths_kv,
        seqused_q,
        seqused_kv,
        max_q_seqlen,
        max_kv_seqlen,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_shape,
        is_packed_gqa,
        q_input_layout,
        kv_input_layout,
        mask_type,
        softmax_precision,
        window_size_left,
        window_size_right,
    )


@impl(AS_LIBRARY, "generic_sparse_attention_grad", "PrivateUse1")
def generic_sparse_attention_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dout: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    rsvd_block_idx: torch.Tensor,
    rsvd_block_count: torch.Tensor,
    metadata: torch.Tensor,
    atten_mask: Optional[torch.Tensor] = None,
    cu_seq_lengths: Optional[torch.Tensor] = None,
    cu_seq_lengths_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    is_packed_gqa: int = 1,
    q_input_layout: str = "TND",
    kv_input_layout: str = "TND",
    scale_value: float = 1.0,
    mask_type: int = 0,
    softmax_precision: int = 0,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_shape is None:
        block_shape = [0]
    op_module = generic_sparse_attention_grad_op_builder.load()
    return op_module.generic_sparse_attention_grad(
        query,
        key,
        value,
        dout,
        out,
        lse,
        rsvd_block_idx,
        rsvd_block_count,
        metadata,
        atten_mask,
        cu_seq_lengths,
        cu_seq_lengths_kv,
        seqused_q,
        seqused_kv,
        block_shape,
        is_packed_gqa,
        q_input_layout,
        kv_input_layout,
        scale_value,
        mask_type,
        softmax_precision,
        window_size_left,
        window_size_right,
    )


@torch.library.register_kernel("cann_ops_transformer::generic_sparse_attention_grad", None)
def generic_sparse_attention_grad_fallback(*args, **kwargs):
    return generic_sparse_attention_grad(*args, **kwargs)


@torch.library.register_kernel("cann_ops_transformer::" + GSAG_METADATA_OP_NAME, None)
def generic_sparse_attention_grad_metadata_fallback(*args, **kwargs):
    return generic_sparse_attention_grad_metadata(*args, **kwargs)


torch.compiler.allow_in_graph(generic_sparse_attention_grad_metadata)
torch.compiler.allow_in_graph(generic_sparse_attention_grad)
