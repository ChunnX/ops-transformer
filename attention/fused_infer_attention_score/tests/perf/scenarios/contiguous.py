#!/usr/bin/python
# -*- coding: utf-8 -*-
"""contiguous scenario: full K/V tensors per batch (no paged block_table).

block_table is omitted (None); K/V have full layout-dependent shapes:
  BNSD       → (B, N_kv, KV_S, D)
  BSND       → (B, KV_S, N_kv, D)
  BSH        → (B, KV_S, H_kv) where H_kv = N_kv * D
  BNSD_BSND  → Q is BNSD, output is BSND; K/V follow BNSD format
"""

import math

import torch

from . import _common


def validate(case):
    _common.base_validate(case)
    layout = case.get("in_layout", "BNSD")
    if layout not in _common.SUPPORTED_LAYOUTS:
        raise ValueError(f"contiguous unsupported layout: {layout}")
    if layout == "BNSD_BSND" and case["q_seq"] == 1:
        raise ValueError("BNSD_BSND requires q_seq > 1")
    dtype = case.get("dtype", "float16")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError(f"contiguous: dtype must be fp16/bf16; got {dtype}")


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

    act_seq_len = [qs] * b
    act_seq_len_kv = _common.deterministic_act_seq_len_kv(b, kvs)
    act_seq_len_kv_t = torch.tensor(act_seq_len_kv, dtype=torch.int32)

    query = _common.make_query(b, nq, qs, d, dtype, layout)
    k_cache, v_cache = _common.make_kv_contiguous(b, nkv, kvs, d, dtype, layout)

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
    )
    return pos, kw
