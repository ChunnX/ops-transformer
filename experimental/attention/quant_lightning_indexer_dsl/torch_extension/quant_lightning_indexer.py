"""Torch registration for the separately packaged QLI DSL implementation."""

import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class QuantLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("ds41.quant_lightning_indexer", category="attention")

    def sources(self):
        return []

    def schema(self):
        return [
            'ds41.quant_lightning_indexer(Tensor q, Tensor k, Tensor w, '
            'Tensor q_descale, Tensor k_descale, int topk, int quant_mode, *, '
            'Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, '
            'Tensor? seqused_q=None, Tensor? seqused_k=None, '
            'Tensor? cmp_residual_k=None, Tensor? block_table=None, '
            'Tensor? output_idx_offset=None, Tensor? metadata=None, '
            'int max_seqlen_q=-1, int mask_mode=0, int cmp_ratio=1, '
            'str layout_q="TND", str layout_kv="PA_BBND", bool return_value=False, '
            'int candidate_topk_blocks=-1, int candidate_block_size=-1) '
            '-> (Tensor, Tensor, Tensor, Tensor)',
            'ds41.quant_lightning_indexer_metadata(Tensor? cu_seqlens_q=None, '
            'Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, '
            'Tensor? cmp_residual_k=None, *, int? batch_size=None, '
            'int max_seqlen_q=-1, int max_seqlen_k=-1, int num_heads_q, '
            'int num_heads_k, int head_dim, int topk, int mask_mode=0, int cmp_ratio=1, '
            'str layout_q="TND", str layout_kv="TND", int candidate_topk_blocks=-1, '
            'int candidate_block_size=-1) -> Tensor',
        ]

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def quant_lightning_indexer_meta(
            q, k, w, q_descale, k_descale, topk, quant_mode, *,
            cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
            cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None,
            max_seqlen_q=-1, mask_mode=0, cmp_ratio=1, layout_q="TND",
            layout_kv="PA_BBND", return_value=False, candidate_topk_blocks=-1,
            candidate_block_size=-1,
        ):
            heads = k.shape[1] if layout_kv == "TND" else k.shape[2]
            shape = (q.shape[0], heads, topk)
            candidate_shape = (q.shape[0], heads, candidate_topk_blocks) if candidate_topk_blocks > 0 else (0,)
            length_shape = (q.shape[0], heads) if candidate_topk_blocks > 0 else (0,)
            return (
                torch.empty(shape, dtype=torch.int32, device="meta"),
                torch.empty(shape if return_value else (0,), dtype=torch.bfloat16, device="meta"),
                torch.empty(candidate_shape, dtype=torch.int32, device="meta"),
                torch.empty(length_shape, dtype=torch.int32, device="meta"),
            )

        @torch.library.register_fake("cann_ops_transformer::ds41.quant_lightning_indexer_metadata")
        def quant_lightning_indexer_metadata_meta(
            cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
            cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1,
            num_heads_q, num_heads_k, head_dim, topk, mask_mode=0, cmp_ratio=1,
            layout_q="TND", layout_kv="TND", candidate_topk_blocks=-1,
            candidate_block_size=-1,
        ):
            tensors = (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k)
            device = next((tensor.device for tensor in tensors if tensor is not None), None)
            if device is None:
                device = torch.device("npu", torch.npu.current_device())
            return torch.empty((1024,), dtype=torch.int32, device=device)


quant_lightning_indexer_op_builder = QuantLightningIndexerOpBuilder()
quant_lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), "ds41.quant_lightning_indexer_metadata", "PrivateUse1")
def quant_lightning_indexer_metadata(
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    cmp_residual_k=None, *, batch_size=None, max_seqlen_q=-1, max_seqlen_k=-1,
    num_heads_q, num_heads_k, head_dim, topk, mask_mode=0, cmp_ratio=1,
    layout_q="TND", layout_kv="TND", candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    from ops.quant_lightning_indexer_metadata_dsl import quant_lightning_indexer_metadata as dsl_metadata

    return dsl_metadata(
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        batch_size=batch_size, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q, num_heads_k=num_heads_k, head_dim=head_dim,
        topk=topk, mask_mode=mask_mode, cmp_ratio=cmp_ratio,
        layout_q=layout_q, layout_k=layout_kv,
        candidate_topk_blocks=candidate_topk_blocks, candidate_block_size=candidate_block_size,
    )


torch.library.register_kernel(
    "cann_ops_transformer::ds41.quant_lightning_indexer_metadata", None,
)(quant_lightning_indexer_metadata)


@impl(get_as_library(), quant_lightning_indexer_op_builder.name, "PrivateUse1")
def quant_lightning_indexer(
    q, k, w, q_descale, k_descale, topk, quant_mode, *,
    cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None, seqused_k=None,
    cmp_residual_k=None, block_table=None, output_idx_offset=None, metadata=None,
    max_seqlen_q=-1, mask_mode=0, cmp_ratio=1, layout_q="TND",
    layout_kv="PA_BBND", return_value=False, candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    from ops.quant_lightning_indexer_dsl import quant_lightning_indexer as dsl_operator

    if metadata is None:
        metadata = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=k.shape[0] if layout_kv == "TND" else k.shape[0] * k.shape[1],
            num_heads_q=q.shape[1], num_heads_k=1, head_dim=128, topk=topk,
            mask_mode=mask_mode, cmp_ratio=cmp_ratio, layout_q=layout_q, layout_kv=layout_kv,
            candidate_topk_blocks=candidate_topk_blocks, candidate_block_size=candidate_block_size,
        )
    return dsl_operator(
        q, k, w, q_descale, k_descale, cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k, seqused_q=seqused_q, seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k, block_table=block_table,
        output_idx_offset=output_idx_offset, metadata=metadata, topk=topk, quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q, mask_mode=mask_mode, cmp_ratio=cmp_ratio,
        layout_q=layout_q, layout_k=layout_kv, return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks, candidate_block_size=candidate_block_size,
    )
