# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific language governing permissions and limitations under
# the License.
# -----------------------------------------------------------------------------------------------------------
"""PyTorch frontend for the Generic block-sparse attention backward operator."""

from typing import Optional, Sequence, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY, OpBuilder


class GenericBlockSparseAttentionGradOpBuilder(OpBuilder):
    """JIT builder for the ACLNN-backed sparse attention backward wrapper."""

    def __init__(self):
        super().__init__("generic_block_sparse_attention_grad")

    def sources(self):
        return ["ops/csrc/generic_block_sparse_attention_grad.cpp"]

    def schema(self) -> str:
        # Keep this signature aligned with torch_npu's all_version API:
        # npu_block_sparse_attention_backward(...).
        return (
            "generic_block_sparse_attention_grad("
            "Tensor d_out, Tensor query, Tensor key, Tensor value, Tensor attention_out, "
            "Tensor softmax_lse, Tensor rsvd_block_idx, Tensor rsvd_block_count, Tensor? metadata, "
            "Tensor? atten_mask, int[]? block_shape, Tensor? cu_seq_lengths, Tensor? cu_seq_lengths_kv, "
            "Tensor? actual_seq_lengths, Tensor? actual_seq_lengths_kv, int is_packed_gqa, "
            "str q_input_layout, str kv_input_layout, float scale_value, int mask_type, "
            "int softmax_precision, int win_left, int win_right) "
            "-> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def generic_block_sparse_attention_grad_meta(
            d_out: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_out: torch.Tensor,
            softmax_lse: torch.Tensor,
            rsvd_block_idx: torch.Tensor,
            rsvd_block_count: torch.Tensor,
            metadata: Optional[torch.Tensor],
            atten_mask: Optional[torch.Tensor],
            block_shape: Optional[Sequence[int]],
            cu_seq_lengths: Optional[torch.Tensor],
            cu_seq_lengths_kv: Optional[torch.Tensor],
            actual_seq_lengths: Optional[torch.Tensor],
            actual_seq_lengths_kv: Optional[torch.Tensor],
            is_packed_gqa: int,
            q_input_layout: str,
            kv_input_layout: str,
            scale_value: float,
            mask_type: int,
            softmax_precision: int,
            win_left: int,
            win_right: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # The backward outputs have exactly the layouts, shapes and dtypes of Q/K/V.
            return (
                query.new_empty(query.shape),
                key.new_empty(key.shape),
                value.new_empty(value.shape),
            )


generic_block_sparse_attention_grad_op_builder = (
    GenericBlockSparseAttentionGradOpBuilder()
)


@impl(AS_LIBRARY, generic_block_sparse_attention_grad_op_builder.name, "PrivateUse1")
def _generic_block_sparse_attention_grad_npu(
    d_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    rsvd_block_idx: torch.Tensor,
    rsvd_block_count: torch.Tensor,
    metadata: Optional[torch.Tensor],
    atten_mask: Optional[torch.Tensor],
    block_shape: Optional[Sequence[int]],
    cu_seq_lengths: Optional[torch.Tensor],
    cu_seq_lengths_kv: Optional[torch.Tensor],
    actual_seq_lengths: Optional[torch.Tensor],
    actual_seq_lengths_kv: Optional[torch.Tensor],
    is_packed_gqa: int,
    q_input_layout: str,
    kv_input_layout: str,
    scale_value: float,
    mask_type: int,
    softmax_precision: int,
    win_left: int,
    win_right: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch the NPU call to the lazily compiled C++ wrapper."""
    op_module = generic_block_sparse_attention_grad_op_builder.load()
    bsnd_output = q_input_layout.upper() == "BSND"
    if bsnd_output:
        # ARC22 consumes BNSD strides internally; transpose only the public BSND view.
        d_out = d_out.permute(0, 2, 1, 3).contiguous()
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()
        attention_out = attention_out.permute(0, 2, 1, 3).contiguous()
        if softmax_lse.dim() == 4:
            softmax_lse = softmax_lse.permute(0, 2, 1, 3).contiguous()
        elif softmax_lse.dim() == 3:
            softmax_lse = softmax_lse.permute(0, 2, 1).contiguous()
        q_input_layout = "BNSD"
        kv_input_layout = "BNSD"
    result = op_module.sparse_attention_score_backward(
        d_out,
        query,
        key,
        value,
        attention_out,
        softmax_lse,
        rsvd_block_idx,
        rsvd_block_count,
        metadata,
        atten_mask,
        block_shape,
        cu_seq_lengths,
        cu_seq_lengths_kv,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        is_packed_gqa,
        q_input_layout,
        kv_input_layout,
        scale_value,
        mask_type,
        softmax_precision,
        win_left,
        win_right,
    )
    if bsnd_output:
        return tuple(t.permute(0, 2, 1, 3).contiguous() for t in result)
    return result


def generic_block_sparse_attention_grad(
    d_out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    rsvd_block_idx: torch.Tensor,
    rsvd_block_count: torch.Tensor,
    metadata: Optional[torch.Tensor],
    atten_mask: Optional[torch.Tensor],
    block_shape: Optional[Sequence[int]],
    cu_seq_lengths: Optional[torch.Tensor],
    cu_seq_lengths_kv: Optional[torch.Tensor],
    actual_seq_lengths: Optional[torch.Tensor],
    actual_seq_lengths_kv: Optional[torch.Tensor],
    is_packed_gqa: int,
    q_input_layout: str,
    kv_input_layout: str,
    scale_value: float,
    mask_type: int,
    softmax_precision: int,
    win_left: int,
    win_right: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gradients of block-sparse attention with the Generic operator.

    This interface follows ``torch_npu.npu_block_sparse_attention_backward``.
    ``query``, ``key`` and ``value`` must have the same dtype. The supported
    input layouts are ``BNSD`` and ``TND``; for ``TND`` inputs, the corresponding
    actual sequence-length list must be provided. The three returned tensors
    have the same shape and dtype as ``query``, ``key`` and ``value``.

    Args:
        d_out: Gradient of the attention output.
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_out: Forward attention output.
        softmax_lse: Forward softmax log-sum-exp tensor.
        rsvd_block_idx: Ordered KV2Q sparse block indices (INT32).
        rsvd_block_count: Valid index count for each KV2Q row (INT32).
        block_shape: Two-element ``[q_block, kv_block]`` block shape, or None
            to use the backend default.
        cu_seq_lengths: Optional Q prefix sums for TND.
        cu_seq_lengths_kv: Optional KV prefix sums for TND.
        actual_seq_lengths: Per-batch Q sequence lengths, or None.
        actual_seq_lengths_kv: Per-batch KV sequence lengths, or None.
        q_input_layout: Q layout, ``BNSD`` or ``TND``.
        kv_input_layout: KV layout, ``BNSD`` or ``TND``.
        is_packed_gqa: Whether grouped Q heads share sparse patterns.
        scale_value: Attention score scaling factor.
        mask_type: 0 for non-causal, 1 for causal.

    Returns:
        ``(d_query, d_key, d_value)`` with shapes and dtypes matching Q/K/V.
    """
    return torch.ops.cann_ops_transformer.generic_block_sparse_attention_grad(
        d_out,
        query,
        key,
        value,
        attention_out,
        softmax_lse,
        rsvd_block_idx,
        rsvd_block_count,
        metadata,
        atten_mask,
        block_shape,
        cu_seq_lengths,
        cu_seq_lengths_kv,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        is_packed_gqa,
        q_input_layout,
        kv_input_layout,
        scale_value,
        mask_type,
        softmax_precision,
        win_left,
        win_right,
    )


# Compatibility alias for callers that already import the frontend module.
sparse_attention_score_backward = generic_block_sparse_attention_grad
