#!/usr/bin/python
# -*- coding: utf-8 -*-
"""no_quant scenario: BNSD / BSND / BSH / BNSD_BSND, FP16 / BF16, paged KV.

This subsumes Phase 1's inline tensor construction.
"""

import math

import torch

from . import _common


def validate(case):
    _common.base_validate(case)
    layout = case.get("in_layout", "BNSD")
    qs = case["q_seq"]
    if layout == "BNSD_BSND" and qs == 1:
        raise ValueError("BNSD_BSND requires q_seq > 1")
    dtype = case.get("dtype", "float16")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError(f"no_quant only supports float16/bfloat16; got {dtype}")
    block_size = case.get("block_size", 128)
    if block_size <= 0 or block_size % 16:
        raise ValueError(f"block_size must be positive multiple of 16; got {block_size}")


def build(case):
    validate(case)
    b = case["batch_size"]
    nq = case["q_head_num"]
    nkv = case["kv_head_num"]
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    d = case["head_dim"]
    dtype = _common.DTYPE_MAP[case["dtype"]]
    layout = case["in_layout"]
    block_size = case["block_size"]
    cache_layout = case.get("cache_layout", "BBH")

    act_seq_len = [qs] * b
    act_seq_len_kv = _common.deterministic_act_seq_len_kv(b, kvs)

    block_table, total_blocks = _common.build_block_table(b, act_seq_len_kv, block_size)
    block_table = block_table.npu()
    act_seq_len_kv_t = torch.tensor(act_seq_len_kv, dtype=torch.int32)

    query = _common.make_query(b, nq, qs, d, dtype, layout)
    k_cache, v_cache = _common.make_kv_paged(total_blocks, block_size, nkv, d, dtype, cache_layout)

    scale = 1.0 / math.sqrt(d)

    pos = (query, k_cache, v_cache)
    kw = dict(
        actual_seq_lengths=act_seq_len,
        actual_seq_lengths_kv=act_seq_len_kv_t,
        num_heads=nq,
        num_key_value_heads=nkv,
        input_layout=layout,
        scale=scale,
        pre_tokens=65535,
        next_tokens=65535,
        block_table=block_table,
        block_size=block_size,
    )
    return pos, kw
