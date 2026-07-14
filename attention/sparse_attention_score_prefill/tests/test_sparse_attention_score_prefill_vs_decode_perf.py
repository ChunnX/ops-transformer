# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""
Prefill vs Decode accuracy comparison for sparse attention (long-sequence focus).

Same Q/KV/sparse-pattern, run through:
  - prefill op (KV-centric, k2q CSR) -> torch.ops...npu_sparse_attention_score_prefill
  - decode op  (Q-centric,  select_idx) -> torch_npu.npu_sparse_attention_score
and assert the two outputs agree (loose rtol: KV-centric vs Q-centric reduction orders and
bf16 cast points differ). q is a long prefill (Q_SEQLEN, default 16K, NOT chunked); the sweep
variable is the KV cache length.

Perf is intentionally NOT measured here (use a dedicated profiler, e.g. msprof/torch profiler,
wrapping this test). 
accuracy is checked on the last output.

NOTE on sizing: prefill op workspace = total_q * kv_heads * top_k * group_size * D * 4 bytes
(~8GB for q=16K, group=16, top_k=16, D=128) + output 256MB + Q 256MB + KV (<=125MB). Needs a
large-HBM device; if 16K OOMs, lower Q_SEQLEN.
"""

import importlib.util
import math
import sys
from math import ceil
from pathlib import Path

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
_GOLDEN_TEST = Path(__file__).resolve().parent / "test_sparse_attention_score_prefill_golden.py"
sys.path.insert(0, str(_TORCH_EXT))
sys.path.insert(0, str(_GOLDEN_TEST.parent))

# load both custom ops (registers torch_npu.npu_sparse_attention_score as well)
import cann_ops_transformer.ops.sparse_attention_score_prefill  # noqa: F401
import cann_ops_transformer.ops.sparse_attention_score  # noqa: F401

_golden_spec = importlib.util.spec_from_file_location("prefill_golden", _GOLDEN_TEST)
prefill_golden = importlib.util.module_from_spec(_golden_spec)
_golden_spec.loader.exec_module(prefill_golden)
build_k2q_csr_torch_reference = prefill_golden.build_k2q_csr_torch_reference

# ---- config ----
KV_SEQLENS = [
    16*1024,
    #32 * 1024, 64 * 1024, 128 * 1024
]  # KV cache length sweep
Q_HEADS = 64
KV_HEADS = 4
TOP_K = 16
HEAD_DIM = 128
BLOCK_SIZE = 128
Q_SEQLEN = 16*1024        # long prefill q (not chunked); ~8GB prefill workspace
SEED = 42
PREC = 1e-2                  # prefill vs decode consistency rtol (different reduction orders)


def _fast_causal_select_idx(q_seqlen, kv_seqlen, kv_heads, block_size, top_k):
    """Vectorized causal block selection: each q token picks its last top_k visible blocks.
    O(q*top_k) only (no per-(q,kv,head) score pooling) -> fast for large kv. For batch=1 the
    result is batch-local, which is exactly what both the decode op (select_idx) and the k2q
    builder (q2k) expect."""
    total_q = q_seqlen
    total_blocks = ceil(kv_seqlen / block_size)
    q_tokens = torch.arange(total_q)
    history = kv_seqlen - q_seqlen
    causal_bound = history + q_tokens                       # last visible kv position per q
    q_block = causal_bound // block_size
    visible = torch.minimum(torch.full_like(q_block, total_blocks), q_block + 1)
    valid_k = torch.minimum(torch.full_like(visible, top_k), visible)
    k_idx = torch.arange(top_k)
    # selected[q, k] = q_block[q] - valid_k[q] + 1 + k  (the last valid_k blocks, causal)
    selected = q_block.unsqueeze(1) - valid_k.unsqueeze(1) + 1 + k_idx.unsqueeze(0)
    mask = k_idx.unsqueeze(0) < valid_k.unsqueeze(1)
    selected = torch.where(mask, selected, torch.full_like(selected, -1))
    select_idx = selected.unsqueeze(0).expand(kv_heads, total_q, top_k).contiguous().to(torch.int32)
    select_num = valid_k.unsqueeze(0).expand(kv_heads, total_q).contiguous().to(torch.int32)
    print("select_idx:", select_idx)
    return select_idx, select_num


def _gen_data(kv_seqlen, seed):
    batch = 1
    q_seqlen = Q_SEQLEN
    max_blocks = ceil(kv_seqlen / BLOCK_SIZE)
    total_physical = batch * max_blocks
    torch.manual_seed(seed)
    query = (torch.rand(q_seqlen, Q_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    key = (torch.rand(total_physical, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    value = (torch.rand(total_physical, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    block_table = torch.arange(total_physical, dtype=torch.int32).reshape(batch, max_blocks)

    select_idx, select_num = _fast_causal_select_idx(q_seqlen, kv_seqlen, KV_HEADS, BLOCK_SIZE, TOP_K)
    actual_seq_lengths = torch.tensor([q_seqlen], dtype=torch.int32)
    actual_seq_lengths_kv = torch.tensor([kv_seqlen], dtype=torch.int32)

    # batch=1: select_idx is batch-local == q2k; skip the slow global<->local conversion.
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = build_k2q_csr_torch_reference(
        select_idx, actual_seq_lengths, actual_seq_lengths_kv, BLOCK_SIZE, q_global_offset=True)

    scale_value = 1.0 / math.sqrt(HEAD_DIM)
    return dict(query=query, key=key, value=value, block_table=block_table,
                select_idx=select_idx.to(torch.int32), select_num=select_num.to(torch.int32),
                k2q_row_ptr=k2q_row_ptr.to(torch.int32),
                k2q_q_indices=k2q_q_indices.to(torch.int32),
                k2q_slot_indices=k2q_slot_indices.to(torch.int32),
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                scale_value=scale_value)


class TestSparseAttentionPrefillVsDecode(TestCase):
    """Prefill vs decode accuracy (long q=16K prefill), kv cache sweep. Perf via external profiler."""

    def _run_one(self, kv_seqlen):
        data = _gen_data(kv_seqlen, SEED)
        q = data["query"].npu()
        k = data["key"].npu()
        v = data["value"].npu()
        bt = data["block_table"].npu()
        asl = data["actual_seq_lengths"].npu()
        asl_kv = data["actual_seq_lengths_kv"].npu()
        sel_idx = data["select_idx"].npu()
        sel_num = data["select_num"].npu()
        rpw = data["k2q_row_ptr"].npu()
        qidx = data["k2q_q_indices"].npu()
        sidx = data["k2q_slot_indices"].npu()
        scale = data["scale_value"]

        def prefill_fn():
            return torch.ops.cann_ops_transformer.npu_sparse_attention_score_prefill(
                q, k, v, bt, rpw, qidx, sidx, asl, asl_kv,
                KV_HEADS, scale, BLOCK_SIZE, TOP_K, 4)

        def decode_fn():
            return torch_npu.npu_sparse_attention_score(
                q, k, v, sel_idx, bt,
                select_num_idx=sel_num,
                actual_seq_lengths=asl,
                actual_seq_lengths_kv=asl_kv,
                num_key_value_heads=KV_HEADS,
                scale_value=scale,
                block_size=BLOCK_SIZE, top_k=TOP_K, inner_precise=4)
        
        p_out = p_out = prefill_fn()
        d_out = d_out = decode_fn()
        
        torch.npu.synchronize()
        p_out = p_out.cpu()
        d_out = d_out.cpu()
        
        print("p_out:", p_out)
        print("d_out:", d_out)
        diff = (p_out.float() - d_out.float()).abs()
        print(f"[kv={kv_seqlen//1024}K q={Q_SEQLEN//1024}K kvh={KV_HEADS} tk={TOP_K}] "
              f"max_diff={diff.max().item():.6f} mean={diff.mean().item():.6f}")

        # accuracy: prefill vs decode must agree (loose rtol). assertRtolEqual also fails on NaN.
        self.assertRtolEqual(d_out.float().numpy(), p_out.float().numpy(), prec=PREC)

    def test_prefill_vs_decode_sweep(self):
        print(f"\n=== prefill vs decode accuracy (batch=1, q_heads={Q_HEADS}, kv_heads={KV_HEADS}, "
              f"top_k={TOP_K}, q_seqlen={Q_SEQLEN}) ===")
        print("kv(K)  max_diff    mean_diff")
        for s in KV_SEQLENS:
            self._run_one(s)


if __name__ == "__main__":
    run_tests()
