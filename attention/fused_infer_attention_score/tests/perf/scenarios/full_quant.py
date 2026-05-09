#!/usr/bin/python
# -*- coding: utf-8 -*-
"""full_quant scenario: per-tensor full quantization (Q/K/V all INT8).

Active parameters (per V5):
  - dequant_scale1  (FP32 or UINT64), shape (1,)   — BMM1 dequant
  - quant_scale1    (FP32),           shape (1,)   — BMM2 pre-quant
  - dequant_scale2  (FP32 or UINT64), shape (1,)   — BMM2 dequant
  - quant_scale2    (FP32 or BF16)                  — output quant; per-tensor or per-channel
  - quant_offset2   (FP32 or BF16) optional         — output quant offset

case['quant_scale2_mode'] ∈ {"per_tensor", "per_channel"} controls quant_scale2 shape.
case['out_dtype'] ∈ {"float16", "bfloat16", "int8"} — output dtype only changes quant_scale2 dtype.

Constraints (from V5 docs and the torch_npu signature):
  - q_seq must be > 1 (V5: per-tensor full quant 不支持 Q_S=1 / decode)
  - no PagedAttention, no alibi/RoPE, layout ∈ {BNSD, BSND, BSH}
"""

import math

import torch

from . import _common


def validate(case):
    _common.base_validate(case)
    qs = case["q_seq"]
    if qs <= 1:
        raise ValueError("full_quant per-tensor requires q_seq > 1 (Q_S != 1)")
    layout = case.get("in_layout", "BNSD")
    if layout not in ("BNSD", "BSND", "BSH"):
        raise ValueError(f"full_quant requires layout ∈ {{BNSD,BSND,BSH}}; got {layout}")
    out_dtype = case.get("out_dtype", "float16")
    if out_dtype not in ("float16", "bfloat16", "int8"):
        raise ValueError(f"full_quant out_dtype must be fp16/bf16/int8; got {out_dtype}")
    mode = case.get("quant_scale2_mode", "per_tensor")
    if mode not in ("per_tensor", "per_channel"):
        raise ValueError(f"quant_scale2_mode must be per_tensor or per_channel; got {mode}")


def _quant_scale2_shape(layout, nq, d, mode):
    if mode == "per_tensor":
        return (1,)
    # per-channel: BSH→(H,); BNSD→(N,D); BSND→(N,D)
    if layout == "BSH":
        return (nq * d,)
    return (nq, d)


def build(case):
    validate(case)
    b = case["batch_size"]
    nq = case["q_head_num"]
    nkv = case["kv_head_num"]
    qs = case["q_seq"]
    kvs = case["kv_seq"]
    d = case["head_dim"]
    layout = case["in_layout"]
    out_dtype_name = case.get("out_dtype", "float16")
    out_dtype = _common.DTYPE_MAP[out_dtype_name]
    qs2_mode = case.get("quant_scale2_mode", "per_tensor")

    act_seq_len = [qs] * b
    act_seq_len_kv = _common.deterministic_act_seq_len_kv(b, kvs)
    act_seq_len_kv_t = torch.tensor(act_seq_len_kv, dtype=torch.int32)

    # INT8 Q/K/V; full contiguous (no block_table) — full_quant 不支持 PA。
    query = _common.make_query(b, nq, qs, d, torch.int8, layout)
    k_cache, v_cache = _common.make_kv_contiguous(b, nkv, kvs, d, torch.int8, layout)

    # Per-tensor quant/dequant scalars (always shape (1,))
    dequant_scale1 = (torch.randn(1).abs().to(torch.float32) + 1e-3).npu()
    quant_scale1   = (torch.randn(1).abs().to(torch.float32) + 1e-3).npu()
    dequant_scale2 = (torch.randn(1).abs().to(torch.float32) + 1e-3).npu()

    qs2_shape = _quant_scale2_shape(layout, nq, d, qs2_mode)
    qs2_dtype = torch.bfloat16 if out_dtype_name == "bfloat16" else torch.float32
    quant_scale2  = (torch.randn(*qs2_shape).abs().to(qs2_dtype) + 1e-3).npu()
    quant_offset2 = torch.randn(*qs2_shape).to(qs2_dtype).npu()

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
        dequant_scale1=dequant_scale1,
        quant_scale1=quant_scale1,
        dequant_scale2=dequant_scale2,
        quant_scale2=quant_scale2,
        quant_offset2=quant_offset2,
    )
    return pos, kw
