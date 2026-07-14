#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Input customization for LightningIndexer V2 TTK cases."""

import hashlib
import math

import torch
import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def fill_tensor_from_value(tensor, value):
    if tensor is None or value is None or not torch.is_tensor(tensor):
        return
    data = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(data.reshape(tensor.shape))


def fill_random_block_table(block_table, seqused_k, block_size, block_num, seed=0):
    if block_table is None or not torch.is_tensor(block_table):
        return
    actual = to_list(seqused_k)
    if not actual:
        return
    shape = tuple(block_table.shape)
    used_blocks = [math.ceil(max(x, 0) / max(int(block_size), 1)) for x in actual]
    required = sum(used_blocks)
    if required > int(block_num):
        raise ValueError(f"block_table requires {required} blocks, but block_num is {block_num}")
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed or 0))
    ids = torch.randperm(int(block_num), dtype=torch.int32, generator=gen)
    table = torch.full(shape, -1, dtype=torch.int32)
    cursor = 0
    for b_idx, blocks in enumerate(used_blocks):
        take = min(blocks, shape[1])
        table[b_idx, :take] = ids[cursor:cursor + take]
        cursor += blocks
    block_table.copy_(table.to(dtype=block_table.dtype, device=block_table.device))


def fill_output_offsets(output_idx_offset, value_range, seed=0):
    if output_idx_offset is None or not torch.is_tensor(output_idx_offset) or value_range is None:
        return
    low, high = [int(x) for x in value_range]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed or 0) + 17)
    values = torch.randint(low, high + 1, output_idx_offset.shape, dtype=torch.int32, generator=gen)
    output_idx_offset.copy_(values.to(dtype=output_idx_offset.dtype, device=output_idx_offset.device))


def _max_seq(cu_seqlens, seqused, fallback):
    if seqused is not None and seqused.numel() > 0:
        return int(seqused.max().item())
    if cu_seqlens is not None and cu_seqlens.numel() > 1:
        return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return int(fallback)


def _batch_size(q, layout_q, cu_seqlens_q, seqused_q):
    if layout_q == "BSND":
        return int(q.shape[0])
    if seqused_q is not None:
        return int(seqused_q.numel())
    if cu_seqlens_q is not None:
        return max(int(cu_seqlens_q.numel()) - 1, 0)
    return 0


def _fill_metadata(metadata, generated):
    if metadata is None:
        raise ValueError("metadata input must be declared in the CSV")
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            f"metadata shape mismatch: CSV={tuple(metadata.shape)} generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))


def generate_li_v2_inputs(q, k, w, *, cu_seqlens_q=None, cu_seqlens_k=None,
                          seqused_q=None, seqused_k=None, cmp_residual_k=None,
                          block_table=None, output_idx_offset=None, metadata=None,
                          topk=0,
                          layout_q="BSND", layout_k="BSND", mask_mode=0, cmp_ratio=1,
                          output_idx_offset_range=None, testcase_name="", **kwargs):
    """Fill regular inputs and precompute the main op metadata tensor."""
    del w
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("seqused_q", seqused_q),
        ("seqused_k", seqused_k),
        ("cmp_residual_k", cmp_residual_k),
    ):
        fill_tensor_from_value(tensor, kwargs.get(f"{name}_values"))
    seed = int(hashlib.sha256(str(testcase_name).encode("utf-8")).hexdigest()[:8], 16)
    if layout_k == "PA_BBND":
        fill_random_block_table(
            block_table,
            seqused_k,
            int(k.shape[1]),
            int(k.shape[0]),
            seed=seed,
        )
    fill_output_offsets(output_idx_offset, output_idx_offset_range, seed=seed)

    q_seq = int(q.shape[1]) if layout_q == "BSND" else int(q.shape[0])
    k_seq = int(k.shape[1]) if layout_k == "BSND" else int(k.shape[0])
    q_heads = int(q.shape[2]) if layout_q == "BSND" else int(q.shape[1])
    k_heads = int(k.shape[1]) if layout_k == "TND" else int(k.shape[2])
    generated = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        num_heads_q=q_heads,
        num_heads_k=k_heads,
        head_dim=int(q.shape[-1]),
        topk=int(topk),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=_batch_size(q, layout_q, cu_seqlens_q, seqused_q),
        max_seqlen_q=_max_seq(cu_seqlens_q, seqused_q, q_seq),
        max_seqlen_k=_max_seq(cu_seqlens_k, seqused_k, k_seq),
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=int(mask_mode),
        cmp_ratio=int(cmp_ratio),
    )
    _fill_metadata(metadata, generated)
