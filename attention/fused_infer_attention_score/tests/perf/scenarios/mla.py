#!/usr/bin/python
# -*- coding: utf-8 -*-
"""MLA scenario: 3 sub-modes selected by case['scenario'].

Sub-modes:
  - mla_decode:           Q_S=1, D=512, separate query_rope/key_rope of D=64
  - mla_prefill_split:    Q_S>1, D=128, separate query_rope/key_rope of D=64
  - mla_prefill_combine:  Q_S>1, Q/K D=192 (128+64 merged), V D=128, no separate rope
"""

import math

import torch

from . import _common


ROPE_D = 64
DECODE_D = 512
SPLIT_D = 128
COMBINE_QK_D = 192
COMBINE_V_D = 128


def _is_decode(scenario):
    return scenario == "mla_decode"


def _is_split(scenario):
    return scenario == "mla_prefill_split"


def _is_combine(scenario):
    return scenario == "mla_prefill_combine"


def validate(case):
    scenario = case["scenario"]
    if scenario not in ("mla_decode", "mla_prefill_split", "mla_prefill_combine"):
        raise ValueError(f"mla.validate received non-MLA scenario: {scenario}")

    b = case["batch_size"]
    nq = case["q_head_num"]
    nkv = case["kv_head_num"]
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    layout = case.get("in_layout", "BNSD")
    dtype = case.get("dtype", "bfloat16")

    if b < 1 or b > 256:
        raise ValueError(f"MLA batch_size out of range: {b}")
    if nq % nkv:
        raise ValueError(f"q_head_num ({nq}) not multiple of kv_head_num ({nkv})")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError(f"MLA scenarios use float16/bfloat16; got {dtype}")
    if layout not in ("BNSD", "BSND"):
        raise ValueError(f"MLA layout must be BNSD or BSND; got {layout}")
    if kvs < qs:
        raise ValueError(f"MLA kv_seq < q_seq invalid: {kvs} < {qs}")

    if _is_decode(scenario):
        if qs != 1:
            raise ValueError(f"mla_decode requires q_seq=1; got {qs}")
        if case["head_dim"] != DECODE_D:
            raise ValueError(f"mla_decode requires head_dim=512; got {case['head_dim']}")
        # MLA decode allowed q/kv ratios
        if (nq // nkv) not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError(f"mla_decode q/kv ratio must ∈ {{1,2,...,128}}; got {nq}/{nkv}")
    elif _is_split(scenario):
        if qs <= 1:
            raise ValueError("mla_prefill_split requires q_seq > 1")
        if case["head_dim"] != SPLIT_D:
            raise ValueError(f"mla_prefill_split requires head_dim=128; got {case['head_dim']}")
    else:  # combine
        if qs <= 1:
            raise ValueError("mla_prefill_combine requires q_seq > 1")
        # head_dim represents Q/K combined dim; V is separate.
        if case["head_dim"] != COMBINE_QK_D:
            raise ValueError(f"mla_prefill_combine requires head_dim=192 (Q/K); got {case['head_dim']}")


def _make_kv_paged_mla(total_blocks, block_size, nkv, d_kv, dtype):
    """MLA paged K/V cache. Use BNBD layout for clarity in MLA paths."""
    # Layout (blocknum, kv_head_num, blocksize, D)
    k = torch.randn(total_blocks, nkv, block_size, d_kv).to(dtype).npu()
    v = torch.randn(total_blocks, nkv, block_size, d_kv).to(dtype).npu()
    return k, v


def build(case):
    validate(case)
    scenario = case["scenario"]

    b = case["batch_size"]
    nq = case["q_head_num"]
    nkv = case["kv_head_num"]
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    layout = case["in_layout"]
    dtype = _common.DTYPE_MAP[case["dtype"]]
    block_size = case.get("block_size", 128)

    act_seq_len = [qs] * b
    act_seq_len_kv = _common.deterministic_act_seq_len_kv(b, kvs)
    block_table, total_blocks = _common.build_block_table(b, act_seq_len_kv, block_size)
    block_table = block_table.npu()
    act_seq_len_kv_t = torch.tensor(act_seq_len_kv, dtype=torch.int32)

    if _is_decode(scenario):
        d_qk = DECODE_D
        d_v = DECODE_D
        # Query main: D=512
        query = _common.make_query(b, nq, qs, d_qk, dtype, layout)
        # K/V cache main: D=512 paged
        k_cache, v_cache = _make_kv_paged_mla(total_blocks, block_size, nkv, d_v, dtype)
        # Rope tensors: separate, D=64
        if layout == "BNSD":
            query_rope = torch.randn(b, nq, qs, ROPE_D).to(dtype).npu()
        else:
            query_rope = torch.randn(b, qs, nq, ROPE_D).to(dtype).npu()
        key_rope = torch.randn(total_blocks, nkv, block_size, ROPE_D).to(dtype).npu()
    elif _is_split(scenario):
        d_qk = SPLIT_D
        d_v = SPLIT_D
        query = _common.make_query(b, nq, qs, d_qk, dtype, layout)
        k_cache, v_cache = _make_kv_paged_mla(total_blocks, block_size, nkv, d_v, dtype)
        if layout == "BNSD":
            query_rope = torch.randn(b, nq, qs, ROPE_D).to(dtype).npu()
        else:
            query_rope = torch.randn(b, qs, nq, ROPE_D).to(dtype).npu()
        key_rope = torch.randn(total_blocks, nkv, block_size, ROPE_D).to(dtype).npu()
    else:  # combine: Q/K=192, V=128, no separate rope
        d_qk = COMBINE_QK_D
        d_v = COMBINE_V_D
        query = _common.make_query(b, nq, qs, d_qk, dtype, layout)
        # K cache D=192, V cache D=128 — different last dim
        k_cache = torch.randn(total_blocks, nkv, block_size, d_qk).to(dtype).npu()
        v_cache = torch.randn(total_blocks, nkv, block_size, d_v).to(dtype).npu()
        query_rope = None
        key_rope = None

    scale = 1.0 / math.sqrt(d_qk)

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
    if query_rope is not None:
        kw["query_rope"] = query_rope
    if key_rope is not None:
        kw["key_rope"] = key_rope
    return pos, kw
