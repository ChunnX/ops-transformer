"""PyTorch frontend for the metadata-only GBSAG scheduler."""

from typing import Optional, Sequence

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY, OpBuilder


class GenericBlockSparseAttentionGradMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("generic_block_sparse_attention_grad_metadata")

    def sources(self):
        return ["ops/csrc/generic_block_sparse_attention_grad_metadata.cpp"]

    def schema(self) -> str:
        return (
            "generic_block_sparse_attention_grad_metadata("
            "Tensor sparse_block_idx, Tensor sparse_block_count, Tensor? cu_seq_lengths_q, "
            "Tensor? cu_seq_lengths_kv, Tensor? seqused_q, Tensor? seqused_kv, "
            "int max_q_seqlen, int max_kv_seqlen, int num_q_heads, int num_kv_heads, int head_dim, "
            "int[]? block_shape, int is_packed_gqa, str layout_q, str layout_kv, int mask_type, "
            "int softmax_precision, int win_left, int win_right) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def metadata_meta(
            sparse_block_idx: torch.Tensor,
            sparse_block_count: torch.Tensor,
            cu_seq_lengths_q: Optional[torch.Tensor],
            cu_seq_lengths_kv: Optional[torch.Tensor],
            seqused_q: Optional[torch.Tensor],
            seqused_kv: Optional[torch.Tensor],
            max_q_seqlen: int,
            max_kv_seqlen: int,
            num_q_heads: int,
            num_kv_heads: int,
            head_dim: int,
            block_shape: Optional[Sequence[int]],
            is_packed_gqa: int,
            layout_q: str,
            layout_kv: str,
            mask_type: int,
            softmax_precision: int,
            win_left: int,
            win_right: int,
        ) -> torch.Tensor:
            return sparse_block_idx.new_empty((32 + 64 * 8,), dtype=torch.int32)


generic_block_sparse_attention_grad_metadata_op_builder = (
    GenericBlockSparseAttentionGradMetadataOpBuilder()
)


@impl(
    AS_LIBRARY,
    generic_block_sparse_attention_grad_metadata_op_builder.name,
    "PrivateUse1",
)
def _metadata_npu(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    cu_seq_lengths_q: Optional[torch.Tensor],
    cu_seq_lengths_kv: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    seqused_kv: Optional[torch.Tensor],
    max_q_seqlen: int,
    max_kv_seqlen: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_shape: Optional[Sequence[int]],
    is_packed_gqa: int,
    layout_q: str,
    layout_kv: str,
    mask_type: int,
    softmax_precision: int,
    win_left: int,
    win_right: int,
) -> torch.Tensor:
    module = generic_block_sparse_attention_grad_metadata_op_builder.load()
    return module.generic_block_sparse_attention_grad_metadata(
        sparse_block_idx,
        sparse_block_count,
        cu_seq_lengths_q,
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
        layout_q,
        layout_kv,
        mask_type,
        softmax_precision,
        win_left,
        win_right,
    )


def generic_block_sparse_attention_grad_metadata(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    cu_seq_lengths_q: Optional[torch.Tensor] = None,
    cu_seq_lengths_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    *,
    max_q_seqlen: int,
    max_kv_seqlen: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int = 128,
    block_shape: Sequence[int] = (1, 128),
    is_packed_gqa: int = 1,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    mask_type: int = 1,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
) -> torch.Tensor:
    """Build scheduler metadata; tensor values are consumed by the AICPU kernel."""
    return torch.ops.cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
        sparse_block_idx,
        sparse_block_count,
        cu_seq_lengths_q,
        cu_seq_lengths_kv,
        seqused_q,
        seqused_kv,
        max_q_seqlen,
        max_kv_seqlen,
        num_q_heads,
        num_kv_heads,
        head_dim,
        None if block_shape is None else list(block_shape),
        is_packed_gqa,
        layout_q,
        layout_kv,
        mask_type,
        softmax_precision,
        win_left,
        win_right,
    )
