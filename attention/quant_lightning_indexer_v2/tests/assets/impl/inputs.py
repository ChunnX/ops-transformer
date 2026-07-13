#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Input customization for QuantLightningIndexer V2 TTK cases."""

import hashlib
import math

import torch


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
    gen.manual_seed(int(seed or 0) + 31)
    values = torch.randint(low, high + 1, output_idx_offset.shape, dtype=torch.int32, generator=gen)
    output_idx_offset.copy_(values.to(dtype=output_idx_offset.dtype, device=output_idx_offset.device))


def generate_qli_v2_inputs(query, key, weights, query_dequant_scale, key_dequant_scale, *,
                           cu_seqlens_q=None, cu_seqlens_k=None, seqused_q=None,
                           seqused_k=None, cmp_residual_k=None, block_table=None,
                           output_idx_offset=None,
                           layout_key="BSND",
                           output_idx_offset_range=None, testcase_name="", **kwargs):
    """Fill PA block table and output index offsets for QLI V2."""
    del query, weights, query_dequant_scale, key_dequant_scale
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
        ("seqused_q", seqused_q),
        ("seqused_k", seqused_k),
        ("cmp_residual_k", cmp_residual_k),
    ):
        fill_tensor_from_value(tensor, kwargs.get(f"{name}_values"))
    seed = int(hashlib.sha256(str(testcase_name).encode("utf-8")).hexdigest()[:8], 16)
    if layout_key == "PA_BBND":
        fill_random_block_table(
            block_table,
            seqused_k,
            int(key.shape[1]),
            int(key.shape[0]),
            seed=seed,
        )
    fill_output_offsets(output_idx_offset, output_idx_offset_range, seed=seed)
