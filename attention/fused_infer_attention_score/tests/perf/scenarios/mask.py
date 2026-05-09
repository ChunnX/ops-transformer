#!/usr/bin/python
# -*- coding: utf-8 -*-
"""mask scenario: sparse_mode + atten_mask + (optional) ALiBi pse_shift.

Sub-tags (case['mask_subtype']):
  - mask_no:          sparse_mode=0, no atten_mask, no pse_shift
  - mask_full:        sparse_mode=1, atten_mask shape (Q_S, KV_S)
  - mask_leftup:      sparse_mode=2, atten_mask shape (2048, 2048)
  - mask_rightdown:   sparse_mode=3, atten_mask shape (2048, 2048)
  - mask_band:        sparse_mode=4, atten_mask shape (2048, 2048),
                      preTokens / nextTokens explicit
  - mask_pse_alibi:   sparse_mode=0, no atten_mask, pse_shift shape (1, Q_N, Q_S, KV_S)

Builds on the no_quant paged-KV path; only the mask/pse parameters differ.
"""

import math

import torch

from . import _common


SUBTYPES = {
    "mask_no", "mask_full", "mask_leftup", "mask_rightdown", "mask_band", "mask_pse_alibi",
}


def validate(case):
    _common.base_validate(case)
    sub = case.get("mask_subtype")
    if sub not in SUBTYPES:
        raise ValueError(f"mask_subtype must be one of {sorted(SUBTYPES)}; got {sub}")
    layout = case.get("in_layout", "BNSD")
    if layout not in ("BNSD", "BSND", "BSH"):
        raise ValueError(f"mask scenario: layout must be BNSD/BSND/BSH; got {layout}")
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    if sub in ("mask_leftup", "mask_rightdown"):
        if qs != kvs:
            raise ValueError(f"{sub} requires q_seq == kv_seq")
    if sub == "mask_band":
        # Caller may override; defaults validated at build time.
        pre_tok = case.get("pre_tokens", qs // 2)
        next_tok = case.get("next_tokens", 0)
        if pre_tok < 0 or next_tok < 0:
            raise ValueError("mask_band requires pre_tokens, next_tokens >= 0")
    if sub == "mask_pse_alibi":
        dtype = case.get("dtype", "float16")
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"mask_pse_alibi: dtype must be fp16/bf16; got {dtype}")


def _make_atten_mask(sub, qs, kvs):
    """Mask is a bool tensor; True positions are masked-out (per FIA convention)."""
    if sub == "mask_no" or sub == "mask_pse_alibi":
        return None
    if sub == "mask_full":
        return torch.zeros(qs, kvs, dtype=torch.bool).npu()
    # 2048×2048 optimized mask for sparse_mode 2/3/4
    return torch.zeros(2048, 2048, dtype=torch.bool).npu()


def _sparse_mode_for(sub):
    return {
        "mask_no": 0, "mask_pse_alibi": 0,
        "mask_full": 1,
        "mask_leftup": 2,
        "mask_rightdown": 3,
        "mask_band": 4,
    }[sub]


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
    block_size = case.get("block_size", 128)
    cache_layout = case.get("cache_layout", "BBH")
    sub = case["mask_subtype"]

    act_seq_len = [qs] * b
    act_seq_len_kv = _common.deterministic_act_seq_len_kv(b, kvs)
    block_table, total_blocks = _common.build_block_table(b, act_seq_len_kv, block_size)
    block_table = block_table.npu()
    act_seq_len_kv_t = torch.tensor(act_seq_len_kv, dtype=torch.int32)

    query = _common.make_query(b, nq, qs, d, dtype, layout)
    k_cache, v_cache = _common.make_kv_paged(total_blocks, block_size, nkv, d, dtype, cache_layout)

    atten_mask = _make_atten_mask(sub, qs, kvs)

    pse_shift = None
    if sub == "mask_pse_alibi":
        pse_shift = torch.randn(1, nq, qs, kvs).to(dtype).npu()

    scale = 1.0 / math.sqrt(d)
    sparse_mode = _sparse_mode_for(sub)

    pre_tokens = case.get("pre_tokens", 65535)
    next_tokens = case.get("next_tokens", 65535)
    if sub == "mask_band":
        pre_tokens = case.get("pre_tokens", qs // 2)
        next_tokens = case.get("next_tokens", 0)

    pos = (query, k_cache, v_cache)
    kw = dict(
        actual_seq_lengths=act_seq_len,
        actual_seq_lengths_kv=act_seq_len_kv_t,
        num_heads=nq,
        num_key_value_heads=nkv,
        input_layout=layout,
        scale=scale,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        block_table=block_table,
        block_size=block_size,
        sparse_mode=sparse_mode,
    )
    if atten_mask is not None:
        kw["atten_mask"] = atten_mask
    if pse_shift is not None:
        kw["pse_shift"] = pse_shift
    return pos, kw
