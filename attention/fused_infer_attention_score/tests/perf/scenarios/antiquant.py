#!/usr/bin/python
# -*- coding: utf-8 -*-
"""antiquant scenario: KV-cache pseudo-quantization.

Sub-modes (case['antiquant_mode']):
  0 = per-channel    scale shape: (1, N_kv, 1, D) or (N_kv, D)
  1 = per-token      scale shape: (1, B, KV_S)         dtype=FLOAT32
  3 = per-token+per-head  scale shape: (B, N_kv, KV_S) dtype=FLOAT32

K/V dtype = INT8; Q dtype follows case['dtype'] (FP16 / BF16).
For Q_S>=2 + mode 0/1, antiquant_scale dtype must be FP16 (Q dtype constraint).
"""

import math

import torch

from . import _common


def validate(case):
    _common.base_validate(case)
    mode = case.get("antiquant_mode", 0)
    if mode not in (0, 1, 3):
        raise ValueError(f"antiquant supports modes 0/1/3 in this PR; got {mode}")
    layout = case.get("in_layout", "BNSD")
    if layout not in ("BNSD", "BSND"):
        raise ValueError(f"antiquant in this PR supports BNSD/BSND only; got {layout}")
    qs = case["q_seq"]
    if mode == 3 and qs < 2:
        raise ValueError("antiquant mode 3 requires q_seq >= 2")
    dtype = case.get("dtype", "float16")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError(f"antiquant Q dtype must be float16/bfloat16; got {dtype}")
    block_size = case.get("block_size", 128)
    if block_size <= 0 or block_size % 32:
        raise ValueError(f"antiquant requires block_size multiple of 32; got {block_size}")


def _scale_dtype(case):
    """Per-token / per-token+per-head modes require FLOAT32; per-channel uses Q dtype."""
    mode = case.get("antiquant_mode", 0)
    if mode in (1, 3):
        return torch.float32
    return _common.DTYPE_MAP[case["dtype"]]


def _build_scale_offset(mode, b, nkv, kv_seq, head_dim, scale_dtype):
    if mode == 0:
        # per-channel: shape (1, N_kv, 1, D)
        scale_shape = (1, nkv, 1, head_dim)
    elif mode == 1:
        # per-token: shape (1, B, kv_seq)
        scale_shape = (1, b, kv_seq)
    elif mode == 3:
        # per-token + per-head: shape (B, N_kv, kv_seq)
        scale_shape = (b, nkv, kv_seq)
    else:
        raise ValueError(f"unsupported mode {mode}")
    scale = torch.randn(*scale_shape).to(scale_dtype).npu().abs() + 1e-3
    offset = torch.randn(*scale_shape).to(scale_dtype).npu()
    return scale, offset


def build(case):
    validate(case)
    b = case["batch_size"]
    nq = case["q_head_num"]
    nkv = case["kv_head_num"]
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    d = case["head_dim"]
    q_dtype = _common.DTYPE_MAP[case["dtype"]]
    layout = case["in_layout"]
    block_size = case["block_size"]
    cache_layout = case.get("cache_layout", "BBH")
    mode = case.get("antiquant_mode", 0)

    act_seq_len = [qs] * b
    act_seq_len_kv = _common.deterministic_act_seq_len_kv(b, kvs)
    block_table, total_blocks = _common.build_block_table(b, act_seq_len_kv, block_size)
    block_table = block_table.npu()
    act_seq_len_kv_t = torch.tensor(act_seq_len_kv, dtype=torch.int32)

    # Query: FP16 / BF16
    query = _common.make_query(b, nq, qs, d, q_dtype, layout)
    # K/V: INT8 paged
    k_cache, v_cache = _common.make_kv_paged(
        total_blocks, block_size, nkv, d, torch.int8, cache_layout)

    s_dtype = _scale_dtype(case)
    key_scale, key_offset = _build_scale_offset(mode, b, nkv, kvs, d, s_dtype)
    value_scale, value_offset = _build_scale_offset(mode, b, nkv, kvs, d, s_dtype)

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
        key_antiquant_scale=key_scale,
        key_antiquant_offset=key_offset,
        value_antiquant_scale=value_scale,
        value_antiquant_offset=value_offset,
        key_antiquant_mode=mode,
        value_antiquant_mode=mode,
    )
    return pos, kw
