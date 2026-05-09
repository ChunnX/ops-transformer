#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Shared helpers for scenario builders."""

import math
import random

import torch


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "float32": torch.float32,
}


SUPPORTED_LAYOUTS = {"BNSD", "BSND", "BSH", "BNSD_BSND"}


def deterministic_act_seq_len_kv(batch_size, kv_seq):
    """Slightly varied per-batch valid KV lengths, deterministic across launches."""
    return [min(kv_seq, max(1, kv_seq - i * 7)) for i in range(batch_size)]


def build_block_table(batch_size, act_seq_len_kv, block_size):
    """Random block_table allocating enough blocks for each batch's KV length."""
    valid_block_num = [math.ceil(s / block_size) for s in act_seq_len_kv]
    valid_block_num_sum = sum(valid_block_num)
    max_valid_block_num = max(valid_block_num)
    block_table = -torch.ones(batch_size, max_valid_block_num, dtype=torch.int32)
    block_idx = list(range(valid_block_num_sum))
    random.shuffle(block_idx)
    start = 0
    for i in range(batch_size):
        block_table[i][: valid_block_num[i]] = torch.tensor(
            block_idx[start: start + valid_block_num[i]], dtype=torch.int32)
        start += valid_block_num[i]
    return block_table, valid_block_num_sum


def make_query(batch_size, q_head_num, q_seq, head_dim, dtype, in_layout, head_dim_kv=None):
    """Allocate a randn query tensor on NPU per layout convention.

    head_dim_kv: only used for MLA mixed-D (BNSD with head_dim_q != head_dim_kv).
    """
    d = head_dim
    if in_layout == "BNSD" or in_layout == "BNSD_BSND":
        return torch.randn(batch_size, q_head_num, q_seq, d).to(dtype).npu()
    if in_layout == "BSND":
        return torch.randn(batch_size, q_seq, q_head_num, d).to(dtype).npu()
    if in_layout == "BSH":
        return torch.randn(batch_size, q_seq, q_head_num * d).to(dtype).npu()
    raise ValueError(f"unsupported layout for query: {in_layout}")


def make_kv_paged(total_blocks, block_size, kv_head_num, head_dim, dtype, cache_layout="BBH"):
    """Allocate paged K/V cache tensors. cache_layout=BBH → (blocknum, blocksize, H)."""
    if cache_layout == "BNBD":
        k = torch.randn(total_blocks, kv_head_num, block_size, head_dim).to(dtype).npu()
        v = torch.randn(total_blocks, kv_head_num, block_size, head_dim).to(dtype).npu()
    else:
        k = torch.randn(total_blocks, block_size, kv_head_num * head_dim).to(dtype).npu()
        v = torch.randn(total_blocks, block_size, kv_head_num * head_dim).to(dtype).npu()
    return k, v


def make_kv_contiguous(batch_size, kv_head_num, kv_seq, head_dim, dtype, in_layout):
    """Allocate full contiguous K/V tensors per layout (no block_table)."""
    if in_layout in ("BNSD", "BNSD_BSND"):
        k = torch.randn(batch_size, kv_head_num, kv_seq, head_dim).to(dtype).npu()
        v = torch.randn(batch_size, kv_head_num, kv_seq, head_dim).to(dtype).npu()
    elif in_layout == "BSND":
        k = torch.randn(batch_size, kv_seq, kv_head_num, head_dim).to(dtype).npu()
        v = torch.randn(batch_size, kv_seq, kv_head_num, head_dim).to(dtype).npu()
    elif in_layout == "BSH":
        k = torch.randn(batch_size, kv_seq, kv_head_num * head_dim).to(dtype).npu()
        v = torch.randn(batch_size, kv_seq, kv_head_num * head_dim).to(dtype).npu()
    else:
        raise ValueError(f"unsupported layout for K/V: {in_layout}")
    return k, v


def base_validate(case):
    """Common shape constraints applicable to all scenarios."""
    b = case["batch_size"]
    nq = case["q_head_num"]
    nkv = case["kv_head_num"]
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    d = case["head_dim"]
    layout = case.get("in_layout", "BNSD")

    if b < 1 or b > 65536:
        raise ValueError(f"batch_size out of range: {b}")
    if nq > 256 or nkv > 256 or nq < 1 or nkv < 1:
        raise ValueError(f"head_num out of range: q={nq} kv={nkv}")
    if nq % nkv:
        raise ValueError(f"q_head_num ({nq}) not multiple of kv_head_num ({nkv})")
    if nq // nkv > 64:
        raise ValueError(f"GQA ratio q/kv > 64: {nq}/{nkv}")
    if d > 512 or d < 1:
        raise ValueError(f"head_dim out of range: {d}")
    if qs < 1 or kvs < 1:
        raise ValueError(f"seq_len must be >= 1: q_seq={qs} kv_seq={kvs}")
    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(f"unsupported in_layout: {layout}")
    # BSH layout: H = N * D, must be aligned to 16 bytes for FP16/BF16.
    if layout == "BSH":
        h = nq * d
        if h * 2 % 32 != 0:  # 16 fp16/bf16 = 32 bytes
            raise ValueError(f"BSH H={h} not 32-byte aligned for fp16/bf16")
