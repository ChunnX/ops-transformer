# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

# Match the arch35 metadata layout and its existing ACLNN wrapper.
_HEAD_METADATA_SIZE = 8
_AIC_CORE_NUM = 36
_AIV_CORE_NUM = 72
_METADATA_SIZE = 8
_FD_METADATA_SIZE = 8


class _QuantBlockSparseAttnOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_quant_block_sparse_attn", category="attention")

    def sources(self):
        return ["csrc/attention/quant_block_sparse_attn.cpp"]

    def schema(self):
        return [
            "npu_quant_block_sparse_attn(Tensor query, Tensor key, Tensor value, Tensor q_descale, "
            "Tensor k_descale, Tensor v_descale, Tensor sparse_indices, Tensor sparse_seq_len, *, "
            "Tensor? p_scale=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, Tensor? block_table=None, Tensor? atten_mask=None, Tensor? metadata=None, "
            "float softmax_scale=1.0, int sparse_q_block_size=128, int sparse_kv_block_size=128, str layout_kv='PA_BNBD', "
            "str layout_q='TND', str layout_sparse_indices='B_N_Qb_Kb', str layout_out='TND', int quant_mode=1, "
            "int mask_mode=3, bool return_softmax_lse=False) -> (Tensor, Tensor)",
            "npu_quant_block_sparse_attn_metadata(Tensor sparse_seq_len, int num_heads_q, "
            "int num_heads_kv, int head_dim, *, Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int batch_size=0, int sparse_block_size_q=128, int sparse_block_size_k=128, "
            "int quant_mode=1, int mask_mode=3, str layout_q='TND', str layout_kv='PA_BNBD', "
            "str layout_sparse_indices='B_N_Qb_Kb') -> Tensor",
        ]

    def register_meta(self):
        @impl(get_as_library(), "npu_quant_block_sparse_attn", "Meta")
        def npu_quant_block_sparse_attn_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            v_descale: torch.Tensor,
            sparse_indices: torch.Tensor,
            sparse_seq_len: torch.Tensor,
            *,
            p_scale: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            block_table: Optional[torch.Tensor] = None,
            atten_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
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
            torch._check(query.dim() == 3, lambda: "query must be 3D")
            torch._check(
                layout_q in ("TND", "NTD"), lambda: "layout_q must be TND or NTD"
            )
            torch._check(layout_out == "TND", lambda: "layout_out must be TND")
            torch._check(layout_kv == "PA_BNBD", lambda: "layout_kv must be PA_BNBD")
            torch._check(quant_mode in (1, 2), lambda: "quant_mode must be 1 or 2")
            torch._check(
                quant_mode != 2 or layout_q == "TND",
                lambda: "MXFP8 requires layout_q=TND",
            )
            t_size = query.size(1) if layout_q == "NTD" else query.size(0)
            n_size = query.size(0) if layout_q == "NTD" else query.size(1)
            lse_shape = (t_size, n_size) if quant_mode == 2 else (n_size, t_size)
            return (
                query.new_empty((t_size, n_size, query.size(2)), dtype=torch.bfloat16),
                query.new_empty(
                    lse_shape if return_softmax_lse else (), dtype=torch.float32
                ),
            )

        @impl(get_as_library(), "npu_quant_block_sparse_attn_metadata", "Meta")
        def npu_quant_block_sparse_attn_metadata_meta(
            sparse_seq_len: torch.Tensor,
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            batch_size: int = 0,
            sparse_block_size_q: int = 128,
            sparse_block_size_k: int = 128,
            quant_mode: int = 1,
            mask_mode: int = 3,
            layout_q: str = "TND",
            layout_kv: str = "PA_BNBD",
            layout_sparse_indices: str = "B_N_Qb_Kb",
        ):
            torch._check(batch_size > 0, lambda: "batch_size must be positive")
            torch._check(
                num_heads_q > 0 and num_heads_kv > 0 and (head_dim > 0),
                lambda: "num_heads_q, num_heads_kv and head_dim must be positive",
            )
            metadata_size = (
                _HEAD_METADATA_SIZE
                + batch_size * num_heads_q * _AIC_CORE_NUM * _METADATA_SIZE
                + _AIV_CORE_NUM * _FD_METADATA_SIZE
            )
            return sparse_seq_len.new_empty((metadata_size,), dtype=torch.int32)


_qbsa_op_builder = _QuantBlockSparseAttnOpBuilder()
_qbsa_op_builder._ensure_initialized()


@impl(get_as_library(), "npu_quant_block_sparse_attn", "PrivateUse1")
def _npu_quant_block_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_seq_len: torch.Tensor,
    *,
    p_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
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
    _op_module = _qbsa_op_builder.load()
    return _op_module.npu_quant_block_sparse_attn(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        sparse_indices,
        sparse_seq_len,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        atten_mask,
        metadata,
        softmax_scale,
        sparse_q_block_size,
        sparse_kv_block_size,
        layout_kv,
        layout_q,
        layout_sparse_indices,
        layout_out,
        quant_mode,
        mask_mode,
        return_softmax_lse,
    )


@impl(get_as_library(), "npu_quant_block_sparse_attn_metadata", "PrivateUse1")
def _npu_quant_block_sparse_attn_metadata(
    sparse_seq_len: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_k: int = 128,
    quant_mode: int = 1,
    mask_mode: int = 3,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNBD",
    layout_sparse_indices: str = "B_N_Qb_Kb",
):
    _op_module = _qbsa_op_builder.load()
    return _op_module.npu_quant_block_sparse_attn_metadata(
        sparse_seq_len,
        num_heads_q,
        num_heads_kv,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        sparse_block_size_q,
        sparse_block_size_k,
        quant_mode,
        mask_mode,
        layout_q,
        layout_kv,
        layout_sparse_indices,
    )


def quant_block_sparse_attn(
    # Required tensor inputs.
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_seq_len: torch.Tensor,
    *,
    # Optional tensor inputs.
    p_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    # Optional attributes.
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
    """Compute FP8/MXFP8 quantized block sparse attention through ACLNN.

    query uses TND (or NTD in quant_mode=1); key/value use PA_BNBD.
    q_descale, k_descale and v_descale are FP32 for FP8 or E8M0 for MXFP8.
    p_scale may be None, meaning 1. sparse_indices and sparse_seq_len are
    int32 tensors in layout_sparse_indices. atten_mask is optional uint8.
    softmax_scale defaults to 1.0 and scales QK; it is not inferred from D.
    sparse_q_block_size/sparse_kv_block_size default to 128. Optional int32
    sequence tensors, block_table and metadata describe the packed sequences
    and paged cache. metadata is
    produced by quant_block_sparse_attn_metadata before graph capture.
    layout_out is TND; quant_mode is 1 (FP8) or 2 (MXFP8); mask_mode is 0 or 3.
    All supplied tensors must be on the same NPU device.

    Returns BF16 attention_out [T,N,D] and FP32 softmax_lse. With
    return_softmax_lse=True, LSE is [N,T] for FP8 and [T,N] for MXFP8.
    Otherwise LSE is an unused scalar, preserving the existing QBSA contract."""
    return torch.ops.cann_ops_transformer.npu_quant_block_sparse_attn(
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


def quant_block_sparse_attn_metadata(
    sparse_seq_len: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_k: int = 128,
    quant_mode: int = 1,
    mask_mode: int = 3,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNBD",
    layout_sparse_indices: str = "B_N_Qb_Kb",
):
    """Build the int32 metadata consumed by quant_block_sparse_attn.

    sparse_seq_len is an int32 NPU tensor; num_heads_q/num_heads_kv/head_dim
    and batch_size are positive dimensions. Optional int32 sequence tensors
    must use the same device. sparse_block_size_q/sparse_block_size_k define
    the sparse tiles; quant_mode is 1 (FP8) or 2 (MXFP8), mask_mode is 0 or 3,
    and layout_q/layout_kv/layout_sparse_indices describe the main inputs.
    Returns a 1D int32 NPU tensor. Build metadata before ACLGraph capture."""
    return torch.ops.cann_ops_transformer.npu_quant_block_sparse_attn_metadata(
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
