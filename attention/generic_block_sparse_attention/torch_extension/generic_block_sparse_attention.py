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

import torch
import torch_npu  # noqa: F401
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

GBSA_METADATA_SIZE = 1024
GBSA_METADATA_OP_NAME = "generic_block_sparse_attention_metadata"


class GenericBlockSparseAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("generic_block_sparse_attention", category="attention")

    def sources(self):
        return ["csrc/attention/generic_block_sparse_attention.cpp"]

    def schema(self):
        return [
            (
                "generic_block_sparse_attention_metadata("
                "Tensor sparse_block_idx, Tensor sparse_block_count, "
                "int max_q_seq_len, int max_kv_seq_len, int num_q_heads, int num_kv_heads, int head_dim, "
                "int[] block_shape, *, Tensor? cu_seq_lengths=None, Tensor? cu_seq_lengths_kv=None, "
                "Tensor? seq_used_q=None, Tensor? seq_used_kv=None, int is_packed_gqa=1, "
                'str q_input_layout="TND", str kv_input_layout="PAGED_BBND", '
                "int mask_type=1, int quant_type=0, int softmax_precision=1, "
                "int window_size_left=-1, int window_size_right=-1) -> Tensor"
            ),
            (
                "generic_block_sparse_attention("
                "Tensor query, Tensor key, Tensor value, "
                "Tensor sparse_block_idx, Tensor sparse_block_count, "
                "int[] block_shape, *, "
                "Tensor? metadata=None, "
                "Tensor? atten_mask=None, "
                "Tensor? q_dequant_scale=None, "
                "Tensor? k_dequant_scale=None, "
                "Tensor? v_dequant_scale=None, "
                "Tensor? p_quant_scale=None, "
                "Tensor? cu_seq_lengths_q=None, "
                "Tensor? cu_seq_lengths_kv=None, "
                "Tensor? seq_used_q=None, "
                "Tensor? seq_used_kv=None, "
                "Tensor? block_table=None, "
                "int is_packed_gqa=1, "
                'str q_input_layout="TND", '
                'str kv_input_layout="PAGED_BBND", '
                "float scale_value=0.0, "
                "int mask_type=1, "
                "int quant_type=0, "
                "float dst_type_max=0.0, "
                "int softmax_precision=1, "
                "int window_size_left=-1, "
                "int window_size_right=-1, "
                "bool return_softmax_lse=False, "
                "ScalarType? attention_out_dtype=None"
                ") -> (Tensor, Tensor)"
            ),
        ]

    def register_meta(self):
        @torch.library.register_fake(
            "cann_ops_transformer::" + GBSA_METADATA_OP_NAME
        )
        def generic_block_sparse_attention_metadata_meta(
            sparse_block_idx,
            sparse_block_count,
            max_q_seq_len,
            max_kv_seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_shape,
            *,
            cu_seq_lengths=None,
            cu_seq_lengths_kv=None,
            seq_used_q=None,
            seq_used_kv=None,
            is_packed_gqa=1,
            q_input_layout="TND",
            kv_input_layout="PAGED_BBND",
            mask_type=1,
            quant_type=0,
            softmax_precision=1,
            window_size_left=-1,
            window_size_right=-1,
        ):
            return torch.empty(
                (GBSA_METADATA_SIZE,), dtype=torch.int32, device="meta"
            )

        @impl(get_as_library(), self.name, "Meta")
        def generic_block_sparse_attention_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            sparse_block_idx: torch.Tensor,
            sparse_block_count: torch.Tensor,
            block_shape: List[int],
            *,
            metadata: Optional[torch.Tensor] = None,
            atten_mask: Optional[torch.Tensor] = None,
            q_dequant_scale: Optional[torch.Tensor] = None,
            k_dequant_scale: Optional[torch.Tensor] = None,
            v_dequant_scale: Optional[torch.Tensor] = None,
            p_quant_scale: Optional[torch.Tensor] = None,
            cu_seq_lengths_q: Optional[torch.Tensor] = None,
            cu_seq_lengths_kv: Optional[torch.Tensor] = None,
            seq_used_q: Optional[torch.Tensor] = None,
            seq_used_kv: Optional[torch.Tensor] = None,
            block_table: Optional[torch.Tensor] = None,
            is_packed_gqa: int = 1,
            q_input_layout: str = "TND",
            kv_input_layout: str = "PAGED_BBND",
            scale_value: float = 0.0,
            mask_type: int = 1,
            quant_type: int = 0,
            dst_type_max: float = 0.0,
            softmax_precision: int = 1,
            window_size_left: int = -1,
            window_size_right: int = -1,
            return_softmax_lse: bool = False,
            attention_out_dtype: Optional[torch.dtype] = None,
        ):
            if query.dtype == torch.float8_e4m3fn:
                if attention_out_dtype is None:
                    attention_out_dtype = torch.bfloat16
                if attention_out_dtype not in (torch.half, torch.bfloat16):
                    raise ValueError(
                        "attention_out_dtype must be half or bfloat16 when query dtype is float8_e4m3fn"
                    )
                out_dtype = attention_out_dtype
            elif attention_out_dtype is not None:
                out_dtype = attention_out_dtype
            else:
                out_dtype = query.dtype

            attn = query.new_empty(query.shape, dtype=out_dtype)
            if return_softmax_lse:
                if query.dim() == 3:
                    lse_shape = (query.size(0), query.size(1), 1)
                elif q_input_layout == "BNSD":
                    lse_shape = (query.size(0), query.size(1), query.size(2), 1)
                else:
                    lse_shape = (query.size(0), query.size(2), query.size(1), 1)
                lse = query.new_empty(lse_shape, dtype=torch.float32)
            else:
                lse = query.new_empty((0,), dtype=torch.float32)
            return attn, lse


generic_block_sparse_attention_op_builder = GenericBlockSparseAttentionOpBuilder()
generic_block_sparse_attention_op_builder._ensure_initialized()


@impl(get_as_library(), GBSA_METADATA_OP_NAME, "PrivateUse1")
def generic_block_sparse_attention_metadata(
    sparse_block_idx,
    sparse_block_count,
    max_q_seq_len,
    max_kv_seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_shape,
    *,
    cu_seq_lengths=None,
    cu_seq_lengths_kv=None,
    seq_used_q=None,
    seq_used_kv=None,
    is_packed_gqa=1,
    q_input_layout="TND",
    kv_input_layout="PAGED_BBND",
    mask_type=1,
    quant_type=0,
    softmax_precision=1,
    window_size_left=-1,
    window_size_right=-1,
):
    op_module = generic_block_sparse_attention_op_builder.load()
    return op_module.generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        cu_seq_lengths,
        cu_seq_lengths_kv,
        seq_used_q,
        seq_used_kv,
        max_q_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_shape,
        is_packed_gqa,
        q_input_layout,
        kv_input_layout,
        mask_type,
        quant_type,
        softmax_precision,
        window_size_left,
        window_size_right,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::" + GBSA_METADATA_OP_NAME, None
)
def generic_block_sparse_attention_metadata_fallback(
    sparse_block_idx,
    sparse_block_count,
    max_q_seq_len,
    max_kv_seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    block_shape,
    *,
    cu_seq_lengths=None,
    cu_seq_lengths_kv=None,
    seq_used_q=None,
    seq_used_kv=None,
    is_packed_gqa=1,
    q_input_layout="TND",
    kv_input_layout="PAGED_BBND",
    mask_type=1,
    quant_type=0,
    softmax_precision=1,
    window_size_left=-1,
    window_size_right=-1,
):
    return generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        max_q_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_shape,
        cu_seq_lengths=cu_seq_lengths,
        cu_seq_lengths_kv=cu_seq_lengths_kv,
        seq_used_q=seq_used_q,
        seq_used_kv=seq_used_kv,
        is_packed_gqa=is_packed_gqa,
        q_input_layout=q_input_layout,
        kv_input_layout=kv_input_layout,
        mask_type=mask_type,
        quant_type=quant_type,
        softmax_precision=softmax_precision,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )


torch.compiler.allow_in_graph(generic_block_sparse_attention_metadata)


@impl(get_as_library(), generic_block_sparse_attention_op_builder.name, "PrivateUse1")
def generic_block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    block_shape: List[int],
    *,
    metadata: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    q_dequant_scale: Optional[torch.Tensor] = None,
    k_dequant_scale: Optional[torch.Tensor] = None,
    v_dequant_scale: Optional[torch.Tensor] = None,
    p_quant_scale: Optional[torch.Tensor] = None,
    cu_seq_lengths_q: Optional[torch.Tensor] = None,
    cu_seq_lengths_kv: Optional[torch.Tensor] = None,
    seq_used_q: Optional[torch.Tensor] = None,
    seq_used_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    is_packed_gqa: int = 1,
    q_input_layout: str = "TND",
    kv_input_layout: str = "PAGED_BBND",
    scale_value: float = 0.0,
    mask_type: int = 1,
    quant_type: int = 0,
    dst_type_max: float = 0.0,
    softmax_precision: int = 1,
    window_size_left: int = -1,
    window_size_right: int = -1,
    return_softmax_lse: bool = False,
    attention_out_dtype: Optional[torch.dtype] = None,
):
    if query.dtype == torch.float8_e4m3fn and attention_out_dtype is None:
        attention_out_dtype = torch.bfloat16

    op_module = generic_block_sparse_attention_op_builder.load()
    return op_module.generic_block_sparse_attention(
        query,
        key,
        value,
        sparse_block_idx,
        sparse_block_count,
        block_shape,
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
        is_packed_gqa,
        q_input_layout,
        kv_input_layout,
        scale_value,
        mask_type,
        quant_type,
        dst_type_max,
        softmax_precision,
        window_size_left,
        window_size_right,
        return_softmax_lse,
        attention_out_dtype,
    )


generic_block_sparse_attention = (
    torch.ops.cann_ops_transformer.generic_block_sparse_attention
)
generic_block_sparse_attention_metadata = (
    torch.ops.cann_ops_transformer.generic_block_sparse_attention_metadata
)

if not hasattr(torch_npu, "npu_generic_block_sparse_attention_metadata"):
    torch_npu.npu_generic_block_sparse_attention_metadata = (
        generic_block_sparse_attention_metadata
    )

torch.compiler.allow_in_graph(generic_block_sparse_attention)
