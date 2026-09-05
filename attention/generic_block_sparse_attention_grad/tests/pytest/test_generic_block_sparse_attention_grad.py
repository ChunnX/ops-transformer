# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Reference and precision harness for generic_block_sparse_attention_grad.

Flow:
  1. Build KV2Q sparse table (CAUSAL + random Q2KV block pick).
  2. CPU golden: online-merge BlockY softmax -> sparse backward -> dq/dk/dv.
  3. NPU: metadata + GenericBlockSparseAttentionGrad via torch.ops; compare vs golden.

Usage:
  python3 test_generic_block_sparse_attention_grad.py --device-id 5
  python3 test_generic_block_sparse_attention_grad.py --device-id 5 --case tnd_tiny_by192
  pytest test_generic_block_sparse_attention_grad.py -q
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch_npu

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


def find_repo_root():
    for path in Path(__file__).resolve().parents:
        if (path / "MindSpeed").exists() or (path / "build.sh").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()

MINDSPEED_PATH = REPO_ROOT / "MindSpeed"
if MINDSPEED_PATH.exists():
    sys.path.insert(0, str(MINDSPEED_PATH))

import cann_ops_transformer  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import


GTYPE = torch.float32
MAX_KV_BLOCKS_PER_Q = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dtype_from_name(name: str):
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def compare_thresholds(input_dtype):
    if input_dtype == torch.bfloat16:
        return 2**-6, 2**-6
    return 0.001, 0.001


def build_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i, s in enumerate(seqlens_list, start=1):
        cu[i] = cu[i - 1] + int(s)
    return cu


def is_right_down_causal_valid(q: int, k: int, act_s1: int, act_s2: int) -> bool:
    return k <= act_s2 - act_s1 + q


def kv_block_has_valid_tokens(q_tok, s2_start, s2_end, act_s1, act_s2, mask_mode) -> bool:
    if mask_mode != 1:
        return True
    for k in range(s2_start, s2_end):
        if is_right_down_causal_valid(q_tok, k, act_s1, act_s2):
            return True
    return False


def q_row_valid_kv_blocks(q_tok, act_s2, block_y, act_s1, mask_mode):
    blocks = []
    for j_idx in range(math.ceil(act_s2 / block_y)):
        s2_start = j_idx * block_y
        s2_end = min(s2_start + block_y, act_s2)
        if kv_block_has_valid_tokens(q_tok, s2_start, s2_end, act_s1, act_s2, mask_mode):
            blocks.append(j_idx)
    return blocks


def build_sparse_table(batch_size, n2, max_s1, max_s2, block_y, seqlens_q, seqlens_kv, mask_mode, seed=42):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    j = math.ceil(max_s2 / block_y)
    idx = torch.full((batch_size, n2, j, max_s1), -1, dtype=torch.int32)
    cnt = torch.zeros((batch_size, n2, j), dtype=torch.int32)
    for b in range(batch_size):
        act_s1, act_s2 = seqlens_q[b], seqlens_kv[b]
        for n2_idx in range(n2):
            j_to_qs = {}
            for q_tok in range(act_s1):
                valid = q_row_valid_kv_blocks(q_tok, act_s2, block_y, act_s1, mask_mode)
                if not valid:
                    continue
                pick_n = min(len(valid), MAX_KV_BLOCKS_PER_Q)
                perm = torch.randperm(len(valid), generator=gen)[:pick_n].tolist()
                for i in perm:
                    j_to_qs.setdefault(valid[i], []).append(q_tok)
            for j_idx, qs in j_to_qs.items():
                qs_sorted = sorted(qs)
                cnt[b, n2_idx, j_idx] = len(qs_sorted)
                idx[b, n2_idx, j_idx, : len(qs_sorted)] = torch.tensor(qs_sorted, dtype=torch.int32)
    return idx, cnt


def iter_sparse_entries(rsvd_idx, rsvd_cnt, seqlens_kv, block_y):
    batch_size, n2_size, _, _ = rsvd_idx.shape
    for b_idx in range(batch_size):
        act_s2 = seqlens_kv[b_idx]
        for n2_idx in range(n2_size):
            for j_idx in range(math.ceil(act_s2 / block_y)):
                count = int(rsvd_cnt[b_idx, n2_idx, j_idx].item())
                if count <= 0:
                    continue
                q_toks = rsvd_idx[b_idx, n2_idx, j_idx, :count].tolist()
                s2_start = j_idx * block_y
                kv_len = min(block_y, act_s2 - s2_start)
                yield b_idx, n2_idx, q_toks, s2_start, kv_len


def build_block_attn_mask(q_toks, s2_start, kv_len, mask_mode, act_s1, act_s2):
    m = len(q_toks)
    mask = torch.ones(1, 1, m, kv_len, dtype=GTYPE)
    if mask_mode == 1:
        for i, q_tok in enumerate(q_toks):
            for k_local in range(kv_len):
                if not is_right_down_causal_valid(q_tok, s2_start + k_local, act_s1, act_s2):
                    mask[0, 0, i, k_local] = 0.0
    return mask


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    y = torch.exp(x - x_max)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y / x_sum
    x_lse = x_max + torch.log(torch.sum(torch.exp(x - x_max), dim=-1, keepdim=True))
    return ans, x_lse


def sparse_forward(q, k, v, attn_mask, softmax_scale):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(softmax_scale)
    if attn_mask is not None:
        qk = qk.masked_fill(attn_mask == 0, float("-inf"))
    p, x_lse = tsoftmax(qk)
    if attn_mask is not None:
        am = attn_mask.to(p.dtype)
        p = p * am
        row_valid = am.sum(dim=-1, keepdim=True) > 0
        p = torch.where(row_valid, p, torch.zeros_like(p))
        x_lse = torch.where(row_valid, x_lse, torch.full_like(x_lse, float("-inf")))
    y = torch.matmul(p, v)
    return y, p, x_lse


def sparse_backward(dx, q, k, v, lse, atten_in, attn_mask, softmax_scale):
    qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(softmax_scale)
    if attn_mask is not None:
        qk = qk.masked_fill(attn_mask == 0, float("-inf"))
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    lse_safe = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
    p = torch.exp(qk - lse_safe)
    p = torch.where(torch.isfinite(lse), p, torch.zeros_like(p))
    atten_safe = torch.where(torch.isfinite(atten_in), atten_in, torch.zeros_like(atten_in))
    softmax_grad = (dx * atten_safe).sum(dim=-1, keepdims=True)
    ds = p * (dp - softmax_grad)
    if attn_mask is not None:
        am = attn_mask.to(p.dtype)
        p = p * am
        ds = ds * am
    ds = torch.where(torch.isfinite(ds), ds, torch.zeros_like(ds))
    p = torch.where(torch.isfinite(p), p, torch.zeros_like(p))
    dv = torch.matmul(p.permute(0, 1, 3, 2), dx)
    dq = torch.matmul(ds, k).mul(softmax_scale)
    dk = torch.matmul(ds.permute(0, 1, 3, 2), q).mul(softmax_scale)
    return dq, dk, dv


def merge_online_softmax_stats(out_old, lse_old, out_new, lse_new):
    lse_new_safe = torch.where(
        torch.isfinite(lse_new), lse_new, torch.full_like(lse_new, float("-inf"))
    )
    lse_m = torch.logaddexp(lse_old, lse_new_safe)
    alpha = torch.exp(lse_old - lse_m)
    beta = torch.exp(lse_new_safe - lse_m)
    both_empty = ~torch.isfinite(lse_m)
    alpha = torch.where(both_empty, torch.zeros_like(alpha), alpha)
    beta = torch.where(both_empty, torch.zeros_like(beta), beta)
    out_old_safe = torch.where(torch.isfinite(out_old), out_old, torch.zeros_like(out_old))
    out_new_safe = torch.where(torch.isfinite(out_new), out_new, torch.zeros_like(out_new))
    out_new_safe = torch.where(
        torch.isfinite(lse_new).unsqueeze(-1), out_new_safe, torch.zeros_like(out_new_safe)
    )
    out_m = alpha.unsqueeze(-1) * out_old_safe + beta.unsqueeze(-1) * out_new_safe
    lse_m = torch.where(both_empty, torch.full_like(lse_m, float("-inf")), lse_m)
    return out_m, lse_m


def run_cpu_golden_bnsd(query, key, value, dout, rsvd_idx, rsvd_cnt, seqlens_q, seqlens_kv, scale, mask_mode, block_y):
    """query/key/value/dout in BNSD [B,N,S,D]."""
    n1, n2, head_dim = query.shape[1], key.shape[1], query.shape[3]
    group = n1 // n2
    out_g = torch.zeros_like(query, dtype=GTYPE)
    dq_g = torch.zeros_like(query, dtype=GTYPE)
    dk_g = torch.zeros_like(key, dtype=GTYPE)
    dv_g = torch.zeros_like(value, dtype=GTYPE)
    lse_g = torch.full((query.shape[0], n1, query.shape[2]), float("-inf"), dtype=GTYPE)

    def _blocks():
        for b_idx, n2_idx, q_toks, s2_start, kv_len in iter_sparse_entries(
            rsvd_idx, rsvd_cnt, seqlens_kv, block_y
        ):
            s1 = seqlens_q[b_idx]
            q_toks = [t for t in q_toks if 0 <= t < s1]
            if not q_toks:
                continue
            q_idx = torch.tensor(q_toks, dtype=torch.long)
            attn_mask = build_block_attn_mask(
                q_toks, s2_start, kv_len, mask_mode, seqlens_q[b_idx], seqlens_kv[b_idx]
            )
            yield b_idx, n2_idx, q_idx, s2_start, kv_len, attn_mask

    for b_idx, n2_idx, q_idx, s2_start, kv_len, attn_mask in _blocks():
        row_valid = attn_mask[0, 0].sum(dim=-1) > 0
        kv_s, kv_e = s2_start, s2_start + kv_len
        for gi in range(group):
            n1_cur = n2_idx * group + gi
            qi = query[b_idx, n1_cur, q_idx, :].to(GTYPE).view(1, 1, -1, head_dim)
            ki = key[b_idx, n2_idx, kv_s:kv_e, :].to(GTYPE).view(1, 1, kv_len, head_dim)
            vi = value[b_idx, n2_idx, kv_s:kv_e, :].to(GTYPE).view(1, 1, kv_len, head_dim)
            out_i, _, lse_i = sparse_forward(qi, ki, vi, attn_mask, scale)
            out_new = out_i[0, 0]
            lse_new = lse_i[0, 0, :, 0]
            lse_new = torch.where(row_valid, lse_new, torch.full_like(lse_new, float("-inf")))
            out_new = torch.where(row_valid.unsqueeze(-1), out_new, torch.zeros_like(out_new))
            out_old = out_g[b_idx, n1_cur, q_idx, :]
            lse_old = lse_g[b_idx, n1_cur, q_idx]
            out_m, lse_m = merge_online_softmax_stats(out_old, lse_old, out_new, lse_new)
            out_g[b_idx, n1_cur, q_idx, :] = out_m
            lse_g[b_idx, n1_cur, q_idx] = lse_m

    for b_idx, n2_idx, q_idx, s2_start, kv_len, attn_mask in _blocks():
        kv_s, kv_e = s2_start, s2_start + kv_len
        m = q_idx.numel()
        for gi in range(group):
            n1_cur = n2_idx * group + gi
            qi = query[b_idx, n1_cur, q_idx, :].to(GTYPE).view(1, 1, m, head_dim)
            ki = key[b_idx, n2_idx, kv_s:kv_e, :].to(GTYPE).view(1, 1, kv_len, head_dim)
            vi = value[b_idx, n2_idx, kv_s:kv_e, :].to(GTYPE).view(1, 1, kv_len, head_dim)
            dxi = dout[b_idx, n1_cur, q_idx, :].to(GTYPE).view(1, 1, m, head_dim)
            lse_b = lse_g[b_idx, n1_cur, q_idx].view(1, 1, m, 1)
            out_b = out_g[b_idx, n1_cur, q_idx, :].view(1, 1, m, head_dim)
            dqi, dki, dvi = sparse_backward(dxi, qi, ki, vi, lse_b, out_b, attn_mask, scale)
            for row, gidx in enumerate(q_idx.tolist()):
                dq_g[b_idx, n1_cur, gidx, :] += dqi[0, 0, row]
            dk_g[b_idx, n2_idx, kv_s:kv_e, :] += dki[0, 0]
            dv_g[b_idx, n2_idx, kv_s:kv_e, :] += dvi[0, 0]

    return (
        out_g.to(query.dtype),
        lse_g.unsqueeze(-1),
        dq_g.to(query.dtype),
        dk_g.to(key.dtype),
        dv_g.to(value.dtype),
    )


def run_cpu_golden_tnd(query, key, value, dout, rsvd_idx, rsvd_cnt, seqlens_q, seqlens_kv, scale, mask_mode, block_y):
    """TND [T,N,D] golden via per-batch BNSD."""
    n1 = query.shape[1]
    cu_q = build_cu_seqlens(seqlens_q)
    cu_kv = build_cu_seqlens(seqlens_kv)
    out_g = torch.zeros_like(query, dtype=GTYPE)
    dq_g = torch.zeros_like(query, dtype=GTYPE)
    dk_g = torch.zeros_like(key, dtype=GTYPE)
    dv_g = torch.zeros_like(value, dtype=GTYPE)
    lse_g = torch.full((query.shape[0], n1), float("-inf"), dtype=GTYPE)

    for b_idx, (s1, s2) in enumerate(zip(seqlens_q, seqlens_kv)):
        s1, s2 = int(s1), int(s2)
        if s1 == 0 or s2 == 0:
            continue
        q_off = int(cu_q[b_idx].item())
        kv_off = int(cu_kv[b_idx].item())
        # [S,N,D] -> [1,N,S,D]
        q_b = query[q_off : q_off + s1].permute(1, 0, 2).unsqueeze(0)
        k_b = key[kv_off : kv_off + s2].permute(1, 0, 2).unsqueeze(0)
        v_b = value[kv_off : kv_off + s2].permute(1, 0, 2).unsqueeze(0)
        do_b = dout[q_off : q_off + s1].permute(1, 0, 2).unsqueeze(0)
        out_b, lse_b, dq_b, dk_b, dv_b = run_cpu_golden_bnsd(
            q_b,
            k_b,
            v_b,
            do_b,
            rsvd_idx[b_idx : b_idx + 1],
            rsvd_cnt[b_idx : b_idx + 1],
            [s1],
            [s2],
            scale,
            mask_mode,
            block_y,
        )
        out_g[q_off : q_off + s1] = out_b[0].permute(1, 0, 2)
        dq_g[q_off : q_off + s1] = dq_b[0].permute(1, 0, 2)
        dk_g[kv_off : kv_off + s2] = dk_b[0].permute(1, 0, 2)
        dv_g[kv_off : kv_off + s2] = dv_b[0].permute(1, 0, 2)
        lse_g[q_off : q_off + s1] = lse_b[0, :, :, 0].permute(1, 0)

    return (
        out_g.to(query.dtype),
        lse_g.unsqueeze(-1),
        dq_g.to(query.dtype),
        dk_g.to(key.dtype),
        dv_g.to(value.dtype),
    )


def relative_diff(actual, expected, diff_thd):
    base = torch.abs(expected)
    min_base = diff_thd
    return torch.abs(actual - expected) / (torch.maximum(base, torch.full_like(base, min_base)) + 1e-9)


def compare_tensors(name, actual, expected, diff_thd=0.001, pct_thd=0.001, max_diff_thd=0.1):
    actual_flat = actual.float().contiguous().view(-1)
    expected_flat = expected.float().contiguous().view(-1)
    if actual_flat.numel() != expected_flat.numel():
        print(f"{name}: failed, shape mismatch a={tuple(actual.shape)} e={tuple(expected.shape)}")
        return {"name": name, "result": "failed", "pct": 0.0, "max_error": float("inf")}
    rel = relative_diff(actual_flat, expected_flat, diff_thd)
    errors = rel[rel > diff_thd]
    pct = (actual_flat.numel() - errors.numel()) / max(actual_flat.numel(), 1) * 100.0
    max_error = float(errors.max().item()) if errors.numel() else 0.0
    max_abs = float(torch.abs(actual_flat - expected_flat).max().item()) if actual_flat.numel() else 0.0
    result = "success" if pct >= (1 - pct_thd) * 100.0 and max_error < max_diff_thd else "failed"
    print(f"{name}: {result}, pct={pct:.6f}%, max_abs_diff={max_abs:.8f}, max_error={max_error}")
    return {"name": name, "result": result, "pct": pct, "max_error": max_error}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GsagCase:
    name: str
    layout: str = "TND"
    dtype: str = "fp16"
    q_lens: tuple = (16,)
    kv_lens: tuple = (16,)
    # dense BSND/BNSD
    batch_num: int = 1
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    n1: int = 8
    n2: int = 2
    head_dim: int = 128
    block_y: int = 128
    mask_mode: int = 1
    seed: int = 42

    @property
    def scale(self):
        return 1.0 / math.sqrt(self.head_dim)


DEFAULT_CASES = (
    GsagCase(name="tnd_tiny_by128", q_lens=(4,), kv_lens=(1,), block_y=128, dtype="fp16"),
    GsagCase(name="tnd_tiny_by192", q_lens=(4,), kv_lens=(1,), block_y=192, dtype="fp16"),
    GsagCase(name="tnd_small_by128", q_lens=(32,), kv_lens=(48,), block_y=128, dtype="bf16"),
    GsagCase(name="tnd_multi_by192", q_lens=(16, 24), kv_lens=(20, 32), block_y=192, dtype="fp16"),
    GsagCase(
        name="bnsd_mid_by128",
        layout="BNSD",
        dtype="fp16",
        batch_num=1,
        max_seqlen_q=64,
        max_seqlen_k=80,
        block_y=128,
    ),
    GsagCase(
        name="bsnd_mid_by192",
        layout="BSND",
        dtype="bf16",
        batch_num=1,
        max_seqlen_q=48,
        max_seqlen_k=64,
        block_y=192,
    ),
)


def list_cases(cases):
    print("available cases:")
    for i, case in enumerate(cases):
        print(
            f"  {i}: {case.name} layout={case.layout} dtype={case.dtype} "
            f"q={case.q_lens} kv={case.kv_lens} By={case.block_y}"
        )


def select_cases(case_selector, cases):
    if case_selector == "all":
        return list(cases)
    if case_selector.isdigit():
        idx = int(case_selector)
        if idx < 0 or idx >= len(cases):
            raise ValueError(f"case index out of range: {idx}")
        return [cases[idx]]
    matched = [c for c in cases if c.name == case_selector]
    if not matched:
        raise ValueError(f"unknown case '{case_selector}'")
    return matched


def run_one_case(case: GsagCase, device: torch.device):
    dtype = dtype_from_name(case.dtype)
    layout = case.layout.upper()
    scale = case.scale
    block_x, block_y = 1, case.block_y
    mask_mode = case.mask_mode

    if layout == "TND":
        seqlens_q = list(case.q_lens)
        seqlens_kv = list(case.kv_lens)
        batch_num = len(seqlens_q)
        max_s1, max_s2 = max(seqlens_q), max(seqlens_kv)
        q_tot, kv_tot = sum(seqlens_q), sum(seqlens_kv)
        q = (torch.rand(q_tot, case.n1, case.head_dim, generator=torch.Generator().manual_seed(case.seed), dtype=dtype) * 2 - 1)
        k = (torch.rand(kv_tot, case.n2, case.head_dim, generator=torch.Generator().manual_seed(case.seed + 1), dtype=dtype) * 2 - 1)
        v = (torch.rand(kv_tot, case.n2, case.head_dim, generator=torch.Generator().manual_seed(case.seed + 2), dtype=dtype) * 2 - 1)
        dout = (torch.rand(q_tot, case.n1, case.head_dim, generator=torch.Generator().manual_seed(case.seed + 3), dtype=dtype) * 2 - 1)
        cu_q = build_cu_seqlens(seqlens_q)
        cu_kv = build_cu_seqlens(seqlens_kv)
    else:
        batch_num = case.batch_num
        max_s1, max_s2 = case.max_seqlen_q, case.max_seqlen_k
        seqlens_q = [max_s1] * batch_num
        seqlens_kv = [max_s2] * batch_num
        # allocate BNSD then rearrange for BSND
        q = (torch.rand(batch_num, case.n1, max_s1, case.head_dim, generator=torch.Generator().manual_seed(case.seed), dtype=dtype) * 2 - 1)
        k = (torch.rand(batch_num, case.n2, max_s2, case.head_dim, generator=torch.Generator().manual_seed(case.seed + 1), dtype=dtype) * 2 - 1)
        v = (torch.rand(batch_num, case.n2, max_s2, case.head_dim, generator=torch.Generator().manual_seed(case.seed + 2), dtype=dtype) * 2 - 1)
        dout = (torch.rand(batch_num, case.n1, max_s1, case.head_dim, generator=torch.Generator().manual_seed(case.seed + 3), dtype=dtype) * 2 - 1)
        cu_q = cu_kv = None

    rsvd_idx, rsvd_cnt = build_sparse_table(
        batch_num, case.n2, max_s1, max_s2, block_y, seqlens_q, seqlens_kv, mask_mode, seed=case.seed + 1
    )

    print(
        f"\n==== {case.name} layout={layout} dtype={dtype} "
        f"N1={case.n1} N2={case.n2} By={block_y} ===="
    )
    if layout == "TND":
        out_cpu, lse_cpu, dq_cpu, dk_cpu, dv_cpu = run_cpu_golden_tnd(
            q, k, v, dout, rsvd_idx, rsvd_cnt, seqlens_q, seqlens_kv, scale, mask_mode, block_y
        )
    else:
        out_cpu, lse_cpu, dq_cpu, dk_cpu, dv_cpu = run_cpu_golden_bnsd(
            q, k, v, dout, rsvd_idx, rsvd_cnt, seqlens_q, seqlens_kv, scale, mask_mode, block_y
        )
        if layout == "BSND":
            q = q.permute(0, 2, 1, 3).contiguous()
            k = k.permute(0, 2, 1, 3).contiguous()
            v = v.permute(0, 2, 1, 3).contiguous()
            dout = dout.permute(0, 2, 1, 3).contiguous()
            out_cpu = out_cpu.permute(0, 2, 1, 3).contiguous()
            dq_cpu = dq_cpu.permute(0, 2, 1, 3).contiguous()
            dk_cpu = dk_cpu.permute(0, 2, 1, 3).contiguous()
            dv_cpu = dv_cpu.permute(0, 2, 1, 3).contiguous()
    print("-------- finish cpu golden --------")

    q_n = q.contiguous().to(device)
    k_n = k.contiguous().to(device)
    v_n = v.contiguous().to(device)
    dout_n = dout.contiguous().to(device)
    out_n = out_cpu.to(dtype).contiguous().to(device)
    lse_n = lse_cpu.float().contiguous().to(device)
    idx_n = rsvd_idx.to(device)
    cnt_n = rsvd_cnt.to(device)
    if layout == "TND":
        cu_q_n = cu_q.to(device)
        cu_kv_n = cu_kv.to(device)
        seqused_q = torch.tensor(seqlens_q, dtype=torch.int32, device=device)
        seqused_kv = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    else:
        cu_q_n = cu_kv_n = seqused_q = seqused_kv = None
    bs = [block_x, block_y]

    meta = torch.ops.cann_ops_transformer.generic_block_sparse_attention_grad_metadata(
        idx_n,
        cnt_n,
        cu_seqlens_q=cu_q_n,
        cu_seqlens_kv=cu_kv_n,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        max_seqlen_q=max_s1,
        max_seqlen_kv=max_s2,
        num_heads_q=case.n1,
        num_heads_kv=case.n2,
        head_dim=case.head_dim,
        block_shape=bs,
        is_packed_gqa=1,
        layout_q=layout,
        layout_kv=layout,
        mask_mode=mask_mode,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
    )
    torch.npu.synchronize()
    torch_npu.npu.set_op_timeout_ms(120 * 1000)

    dq_n, dk_n, dv_n = torch.ops.cann_ops_transformer.generic_block_sparse_attention_grad(
        q_n,
        k_n,
        v_n,
        dout_n,
        out_n,
        lse_n,
        idx_n,
        cnt_n,
        meta,
        attn_mask=None,
        cu_seqlens_q=cu_q_n,
        cu_seqlens_kv=cu_kv_n,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        block_shape=bs,
        is_packed_gqa=1,
        layout_q=layout,
        layout_kv=layout,
        softmax_scale=scale,
        mask_mode=mask_mode,
        softmax_precision=0,
        win_left=-1,
        win_right=-1,
    )
    torch.npu.synchronize()

    diff_thd, pct_thd = compare_thresholds(dtype)
    results = [
        compare_tensors("dq", dq_n.cpu(), dq_cpu, diff_thd, pct_thd),
        compare_tensors("dk", dk_n.cpu(), dk_cpu, diff_thd, pct_thd),
        compare_tensors("dv", dv_n.cpu(), dv_cpu, diff_thd, pct_thd),
    ]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=int(os.environ.get("GSAG_DEVICE_ID", "0")))
    parser.add_argument("--case", default="all", help="case name/index, or 'all'")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()

    if args.list_cases:
        list_cases(DEFAULT_CASES)
        return

    torch_npu.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")
    cases = select_cases(args.case, DEFAULT_CASES)

    all_results = []
    for case in cases:
        results = run_one_case(case, device)
        all_results.append((case, results))

    print("\n================ suite summary ================")
    failed = False
    for case, results in all_results:
        ok = all(r["result"] == "success" for r in results)
        failed = failed or not ok
        detail = ", ".join(f"{r['name']}={r['result']}" for r in results)
        print(f"{case.name}: {'success' if ok else 'failed'} ({detail})")
    if failed:
        raise RuntimeError("One or more cases failed.")


# ---------------------------------------------------------------------------
# pytest entry
# ---------------------------------------------------------------------------
def _pytest_device_id() -> int:
    return int(os.environ.get("GSAG_DEVICE_ID", os.environ.get("DEVICE_ID", "0")))


if pytest is not None:

    @pytest.mark.parametrize("case", DEFAULT_CASES, ids=lambda c: c.name)
    def test_generic_block_sparse_attention_grad(case):
        if not torch.npu.is_available():
            pytest.skip("NPU not available")
        device_id = _pytest_device_id()
        torch_npu.npu.set_device(device_id)
        results = run_one_case(case, torch.device(f"npu:{device_id}"))
        failed = [r for r in results if r["result"] != "success"]
        assert not failed, f"{case.name} failed: {failed}"


if __name__ == "__main__":
    main()
