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
import os
import sys
from math import ceil
from pathlib import Path

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch_npu.npu.set_device(os.getenv("NPU_DEVICE", "npu:0"))

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
    int(value.strip())
    for value in os.getenv("KV_SEQLENS", str(128 * 1024)).split(",")
    if value.strip()
]  # Per-batch KV cache length sweep.
Q_HEADS = 64
KV_HEADS = 4
TOP_K = 16
HEAD_DIM = 128
BLOCK_SIZE = 128
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
Q_SEQLEN = int(os.getenv("Q_SEQLEN", str(16 * 1024)))  # Per-batch S; B*S is the generalization axis.
SEED = int(os.getenv("SEED", "42"))
PREC = float(os.getenv("PREC", "1e-2"))  # Different reduction orders need a loose rtol.

# Production k2q_row_ptr dump (captured on-device): each .pt holds {'k2q_row_ptr': [kv_heads,
# numKvBlocks+1] int32}. Production config matches this test (q=16K, top_k=16, kv_heads=4,
# block=128): total nnz = Q*top_k*kv_heads. These cases load REAL production row_ptr and
# reconstruct a causal select_idx whose per-block nnz matches it (so build_k2q_csr reproduces
# the production row_ptr incl. zero-nnz rows -> production-faithful per-core partition stress,
# incl. the heavy-tail + zero-row pattern the synthetic bias cases under-represent). The dump
# path/step list may be overridden via env (PROD_ROWPTR_DIR / PROD_ROWPTR_STEPS="0,18,56,...").
import os as _os
PRODUCTION_ROWPTR_DIR = _os.environ.get(
    "PROD_ROWPTR_DIR", "/home/k30071013/profiling/k2q_row_ptr_dump")
PRODUCTION_STEPS = [int(s) for s in _os.environ.get(
    #"PROD_ROWPTR_STEPS", "0,18,56,3510,4679").split(",") if s.strip()]
    "PROD_ROWPTR_STEPS", "18,").split(",") if s.strip()]


def _fast_causal_select_idx(q_seqlen, kv_seqlen, kv_heads, block_size, top_k, *,
                            random_select=False, bias=None, rng_seed=0):
    """Vectorized causal block selection: each q token picks (at most) top_k visible blocks.

    O(q*top_k) only (no per-(q,kv,head) score pooling) -> fast for large kv. For batch=1 the
    result is batch-local, which is exactly what both the decode op (select_idx) and the k2q
    builder (q2k) expect.

    random_select=False (default): the LAST valid_k contiguous causal blocks.
    random_select=True: a random subset of (at most) top_k blocks drawn from the causal-visible
    range, GUARANTEEING the local block (the block holding q's own KV position, i.e. the top of
    the visible range) is always selected — matching case 1's local-block guarantee. The local
    block is force-included; the remaining (valid_k - 1) blocks are drawn at random from the
    rest of the visible range. Order is irrelevant for the result: softmax + weighted sum is
    commutative over blocks, and the same select_idx feeds both the decode op and the k2q CSR
    builder, so prefill vs decode still compare the identical sparse pattern.

    bias: one of "heavy_tail" | "dual_focus" | "spread" — three production-mirroring SKEWED
    KV-selection patterns (see _build_bias_pop). Selected via Gumbel-top-k weighted sampling
    (score = Gumbel(0,1) + log(pop)); the local block is still force-included and no block is
    picked twice for a given q, so the pattern stays causal/valid for both ops. The three cases
    mirror the R=1024 production row_ptr distribution (total nnz = total_q*top_k*kv_heads) and
    stress per-core work-distribution in the prefill op on realistic heavy-tailed / dual-focus /
    spread-sparse access patterns:
      - heavy_tail: ~6 saturated hot blocks + a mid-hot tail; ~1.5% of blocks carry ~60-70% of
        nnz, ~40-50% of blocks zero-nnz (production 16384-65536 bucket: 15 rows, 64.7%).
      - dual_focus: hot blocks at BOTH the front (saturated) and a back-quarter cluster; middle
        quartiles cold (production step56: top-50 hottest Q1=7, Q4=38).
      - spread: ~16 hot blocks sampled uniformly at MEDIUM weight -> heavier mid buckets and a
        lower zero-nnz fraction (production step4679: top-50 Q1=11, Q2=12, Q4=23, 12.6% zero).
    """
    total_q = q_seqlen
    total_blocks = ceil(kv_seqlen / block_size)
    q_tokens = torch.arange(total_q)
    history = kv_seqlen - q_seqlen
    causal_bound = history + q_tokens                       # last visible kv position per q
    q_block = causal_bound // block_size
    visible = torch.minimum(torch.full_like(q_block, total_blocks), q_block + 1)
    valid_k = torch.minimum(torch.full_like(visible, top_k), visible)

    if not random_select and bias is None:
        k_idx = torch.arange(top_k)
        # selected[q, k] = q_block[q] - valid_k[q] + 1 + k  (the last valid_k blocks, causal)
        selected = q_block.unsqueeze(1) - valid_k.unsqueeze(1) + 1 + k_idx.unsqueeze(0)
        mask = k_idx.unsqueeze(0) < valid_k.unsqueeze(1)
        selected = torch.where(mask, selected, torch.full_like(selected, -1))
    elif random_select:
        # Random min(top_k, visible) blocks from the causal-visible range, GUARANTEEING the local
        # block (visible-1 = the block holding q's own KV position) is always selected. Force-
        # include the local block, then draw (valid_k - 1) blocks at random from {0, ..., visible-2}.
        gen = torch.Generator(device="cpu").manual_seed(rng_seed)
        rand_scores = torch.rand((total_q, total_blocks), generator=gen)
        block_ids = torch.arange(total_blocks).unsqueeze(0)        # [1, total_blocks]
        # remaining candidates: visible range EXCLUDING the local block (visible-1)
        rem_mask = block_ids < (visible - 1).unsqueeze(1)           # [total_q, total_blocks]
        rand_scores = torch.where(rem_mask, rand_scores,
                                  torch.full_like(rand_scores, -1.0))
        rem_vals, rem_ids = rand_scores.topk(top_k - 1, dim=1)     # [total_q, top_k-1]
        rem_chosen = rem_vals >= 0.0                                # exactly valid_k - 1 picks per q
        rem_selected = torch.where(rem_chosen, rem_ids,
                                   torch.full_like(rem_ids, -1))   # [total_q, top_k-1]
        # local block: always selected when valid_k >= 1 (valid_k==0 -> nothing visible, all -1)
        local = (visible - 1).unsqueeze(1)                          # [total_q, 1] local block id
        local_chosen = (valid_k >= 1).unsqueeze(1)                   # [total_q, 1]
        local_selected = torch.where(local_chosen, local,
                                     torch.full_like(local, -1))    # [total_q, 1]
        selected = torch.cat([local_selected, rem_selected], dim=1)  # [total_q, top_k]

    elif bias in ("heavy_tail", "dual_focus", "spread"):
        pop = _build_bias_pop(total_blocks, block_size, history, bias)
        selected, valid_k = _gumbel_topk_causal_select(
            total_q, total_blocks, block_size, history, top_k, pop, rng_seed)
        _report_block_bias(selected, total_blocks, bias)

    else:
        raise ValueError("at most one of random_select / bias may be requested")

    select_idx = selected.unsqueeze(0).expand(kv_heads, total_q, top_k).contiguous().to(torch.int32)
    select_num = valid_k.unsqueeze(0).expand(kv_heads, total_q).contiguous().to(torch.int32)
    print("select_idx:", select_idx)
    return select_idx, select_num


def _build_bias_pop(total_blocks, block_size, history, bias):
    """Popularity weight vector for the three production-mirroring skewed KV-selection cases.

    Mirrors the R=1024 production row_ptr analysis (total nnz = total_q*top_k*kv_heads). `sat`
    (0.5*total_blocks) gives ~99.5% selection — it dominates the base-weight-1 blocks so a hot
    block is picked by ~all q that see it. The local block (visible-1) is force-included by the
    sampler and excluded from candidates, so a hot id that coincides with some q's local block is
    still selected exactly once (no per-q duplicate).
      heavy_tail: ~6 fully-saturated front hot blocks (always visible to every q, so they can
        saturate) + a mid-hot tail; ~1.5% of blocks carry ~60-70% of nnz, ~40-50% zero-nnz.
        Mirrors the production 16384-65536 bucket (15 rows, 64.7% of nnz).
      dual_focus: hot blocks at BOTH the front (saturated, visible to all q) AND a back-quarter
        cluster (mid weight). Middle quartiles stay cold. Back-cluster blocks are only visible
        to late q (causal), so they saturate among late q but stay mid-nnz overall — mirrors
        production step56 (top-50 hottest: Q1=7, Q4=38, back rows ~18k-25k nnz).
      spread: ~16 hot blocks sampled UNIFORMLY across the block range at MEDIUM weight (not
        saturated) -> heavier mid buckets (256-4095, 4096-16383) and a lower zero-nnz fraction.
        Mirrors production step4679 (top-50: Q1=11, Q2=12, Q4=23, 12.6% zero rows).
    """
    pop = torch.ones(total_blocks, dtype=torch.float32)
    history_blocks = history // block_size          # blocks visible to q0 => visible to all q
    sat = 0.5 * total_blocks                         # ~99.5% selection (dominates base-1 blocks)

    if bias == "heavy_tail":
        n_sat, n_mid = 6, 6
        sat_ids = torch.arange(n_sat)                                 # 0..5 (front, visible to all)
        mid_ids = torch.arange(n_sat, n_sat + n_mid)                  # 6..11
        mid_w = torch.linspace(0.10 * total_blocks, 4.0, n_mid)        # high->low mid weight
        pop[sat_ids] = sat
        pop[mid_ids] = mid_w
    elif bias == "dual_focus":
        front_ids = torch.tensor([0, 1, 2], dtype=torch.int64)        # saturated front cluster
        back_start = int(0.90 * total_blocks)                          # back cluster (late-q only)
        back_ids = torch.arange(back_start, min(back_start + 3, total_blocks))
        pop[front_ids] = sat
        pop[back_ids] = sat                                            # saturated back cluster
    elif bias == "spread":
        gen = torch.Generator(device="cpu").manual_seed(0)
        hot_ids = torch.randint(0, total_blocks, (16,), generator=gen).unique()
        pop[hot_ids] = 0.05 * total_blocks                            # medium (not saturated)
    else:
        raise ValueError(f"unknown bias {bias!r}")
    return pop


def _gumbel_topk_causal_select(total_q, total_blocks, block_size, history, top_k, pop, rng_seed):
    """Weighted sampling without replacement via Gumbel-top-k: score = Gumbel(0,1) + log(pop).
    The local block (visible-1, the block holding q's own KV position) is force-included; the
    remaining (top_k-1) picks are the top-scoring candidates from the visible range EXCLUDING
    the local block. Causal (only visible blocks are candidates), no per-q duplicate. Returns
    selected[total_q, top_k] (int64, -1 pad) and valid_k[total_q].
    """
    q_tokens = torch.arange(total_q)
    causal_bound = history + q_tokens
    q_block = causal_bound // block_size
    visible = torch.minimum(torch.full_like(q_block, total_blocks), q_block + 1)
    valid_k = torch.minimum(torch.full_like(visible, top_k), visible)

    gen = torch.Generator(device="cpu").manual_seed(rng_seed)
    u = torch.rand((total_q, total_blocks), generator=gen)
    gumbel = -torch.log(-torch.log(u.clamp(min=1e-20)))
    score = gumbel + torch.log(pop).unsqueeze(0)
    block_ids = torch.arange(total_blocks).unsqueeze(0)
    cand_mask = block_ids < (visible - 1).unsqueeze(1)
    score = torch.where(cand_mask, score, torch.full_like(score, float("-inf")))
    rem_vals, rem_ids = score.topk(top_k - 1, dim=1)
    rem_chosen = torch.isfinite(rem_vals)
    rem_selected = torch.where(rem_chosen, rem_ids, torch.full_like(rem_ids, -1))
    local = (visible - 1).unsqueeze(1)
    local_chosen = (valid_k >= 1).unsqueeze(1)
    local_selected = torch.where(local_chosen, local, torch.full_like(local, -1))
    selected = torch.cat([local_selected, rem_selected], dim=1)
    return selected, valid_k


def _report_block_bias(selected, total_blocks, bias):
    """Log the realized per-block hit distribution so the bias shape is visible — mirrors the
    production row_ptr report (top-k concentration, zero fraction, max/mean)."""
    flat = selected.reshape(-1)
    hit = flat[flat >= 0]
    n_sel = int(hit.numel())
    if n_sel == 0:
        return
    counts = torch.bincount(hit, minlength=total_blocks)
    nz = counts[counts > 0]
    order = torch.argsort(counts, descending=True)
    top5 = int(counts[order[:min(5, total_blocks)]].sum())
    top10 = int(counts[order[:min(10, total_blocks)]].sum())
    zero_blocks = int((counts == 0).sum())
    nz_mean = float(nz.float().mean()) if nz.numel() > 0 else 0.0
    print(f"  [{bias}] block-hits: n_sel={n_sel} total_blocks={total_blocks} "
          f"zero-blocks={zero_blocks}({100.0*zero_blocks/total_blocks:.1f}%) "
          f"top5={100.0*top5/n_sel:.1f}% top10={100.0*top10/n_sel:.1f}% "
          f"max/mean={float(counts.max())/nz_mean:.1f}x")


def _reconstruct_select_idx_from_rowptr(row_ptr, q_seqlen, block_size, top_k, history, rng_seed):
    """Reconstruct a causal Q-centric select_idx whose per-block selection count matches the
    production k2q_row_ptr's per-head nnz, so build_k2q_csr_torch_reference reproduces the
    production row_ptr (incl. zero-nnz rows) for a production-faithful partition-stress test.

    The production dump only stores k2q_row_ptr (no q_indices/select_idx), so we reverse-derive
    a valid causal select_idx from the per-block nnz targets. row_ptr: [kv_heads, numKvBlocks+1]
    int32; all heads are equal in production (verified imbalance 1.00x), so head 0 drives the
    reconstruction and is replicated across heads.

    Greedy b-matching: blocks processed in DESCENDING target order (hot first, when Q-capacity
    is fullest); for each block r we pick `need` Q-tokens that causally see it (q_block >= r)
    with remaining capacity, preferring the SMALLEST q_block candidates — those can only serve
    low-r blocks, so conserving large-q_block Q's for the high-r blocks only they can see makes
    the matching feasible (this achieves 100% per-block match on the R>=384 production steps and
    ~96% on R=128). Causal, no per-q duplicate, -1-padded -> valid for both ops and build_k2q_csr.
    Returns select_idx [kv_heads, q_seqlen, top_k] (int32) and select_num [q_seqlen] (per-q count).
    """
    kv_heads, rp1 = row_ptr.shape
    num_blocks = rp1 - 1
    rp = row_ptr[0].to(torch.int64)
    target = rp[1:] - rp[:-1]                          # [num_blocks] per-block q-count (head 0)
    total_q = q_seqlen
    q_block = (history + torch.arange(total_q)) // block_size   # last visible block per q
    cap = torch.full((total_q,), top_k, dtype=torch.int64)
    slot = torch.zeros(total_q, dtype=torch.int64)
    select = torch.full((total_q, top_k), -1, dtype=torch.int64)
    for r in torch.argsort(target, descending=True).tolist():
        need = int(target[r].item())
        if need <= 0:
            continue
        cand = torch.nonzero((q_block >= r) & (cap > 0), as_tuple=False).squeeze(-1)
        m = min(need, int(cand.numel()))
        if m == 0:
            continue
        # smallest q_block first: conserve large-q_block Q's for high-r blocks only they can see
        sidx = torch.argsort(q_block[cand])
        chosen = cand[sidx[:m]]
        slots = slot[chosen]
        select[chosen, slots] = r
        slot[chosen] = slots + 1
        cap[chosen] = cap[chosen] - 1
    select_num = (top_k - cap).unsqueeze(0).expand(kv_heads, total_q).contiguous().to(torch.int32)
    select = select.unsqueeze(0).expand(kv_heads, total_q, top_k).contiguous().to(torch.int32)
    return select, select_num


def _report_rowptr_fidelity(rebuilt_rowptr, prod_rowptr):
    """Log how closely the rebuilt k2q_row_ptr (from the reconstructed select_idx) matches the
    loaded production row_ptr — per-block nnz match rate, total abs diff, and zero-nnz row count
    (the zero rows are what drive the contiguous-partition imbalance)."""
    r = rebuilt_rowptr.to(torch.int64)
    o = prod_rowptr.to(torch.int64)
    on = o[0, 1:] - o[0, :-1]
    rn = r[0, 1:] - r[0, :-1]
    diff = (rn - on).abs()
    n = int(on.numel())
    print(f"  [prod-rowptr] per-block nnz match {int((diff == 0).sum())}/{n} "
          f"({100.0 * int((diff == 0).sum()) / max(1, n):.1f}%) "
          f"sum_abs_diff={int(diff.sum())}/{int(on.sum())} "
          f"zero-nnz rows orig={int((on == 0).sum())} rebuilt={int((rn == 0).sum())}")


def _gen_data_from_production_rowptr(row_ptr, seed):
    """Like _gen_data, but the sparse pattern is reconstructed from a REAL production
    k2q_row_ptr (loaded from the on-device dump) instead of synthesized. Q/KV/block_table are
    still random (the sparse PATTERN, not the data, is what drives per-core partition stress)."""
    batch = 1
    q_seqlen = Q_SEQLEN
    prod_kv_heads, rp1 = row_ptr.shape
    num_blocks = rp1 - 1
    if prod_kv_heads != KV_HEADS:
        raise ValueError(f"production row_ptr kv_heads {prod_kv_heads} != test KV_HEADS {KV_HEADS}")
    kv_seqlen = num_blocks * BLOCK_SIZE
    if kv_seqlen < q_seqlen:
        raise ValueError(f"production num_blocks {num_blocks} -> kv_seqlen {kv_seqlen} < "
                         f"q_seqlen {q_seqlen}; need R >= {ceil(q_seqlen / BLOCK_SIZE)}")
    history = kv_seqlen - q_seqlen
    max_blocks = num_blocks
    total_physical = batch * max_blocks
    torch.manual_seed(seed)
    query = (torch.rand(q_seqlen, Q_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    key = (torch.rand(total_physical, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    value = (torch.rand(total_physical, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    block_table = torch.arange(total_physical, dtype=torch.int32).reshape(batch, max_blocks)

    select_idx, select_num = _reconstruct_select_idx_from_rowptr(
        row_ptr, q_seqlen, BLOCK_SIZE, TOP_K, history, seed)
    actual_seq_lengths = torch.tensor([q_seqlen], dtype=torch.int32)
    actual_seq_lengths_kv = torch.tensor([kv_seqlen], dtype=torch.int32)
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = build_k2q_csr_torch_reference(
        select_idx, actual_seq_lengths, actual_seq_lengths_kv, BLOCK_SIZE, q_global_offset=True)
    _report_rowptr_fidelity(k2q_row_ptr, row_ptr)
    scale_value = 1.0 / math.sqrt(HEAD_DIM)
    return dict(query=query, key=key, value=value, block_table=block_table,
                select_idx=select_idx.to(torch.int32), select_num=select_num.to(torch.int32),
                k2q_row_ptr=k2q_row_ptr.to(torch.int32),
                k2q_q_indices=k2q_q_indices.to(torch.int32),
                k2q_slot_indices=k2q_slot_indices.to(torch.int32),
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                scale_value=scale_value)


def _gen_data(kv_seqlen, seed, *, random_select=False, bias=None):
    batch = BATCH_SIZE
    q_seqlen = Q_SEQLEN
    total_q_tokens = batch * q_seqlen
    max_blocks = ceil(kv_seqlen / BLOCK_SIZE)
    total_physical = batch * max_blocks
    torch.manual_seed(seed)
    query = (torch.rand(total_q_tokens, Q_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    key = (torch.rand(total_physical, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    value = (torch.rand(total_physical, BLOCK_SIZE, KV_HEADS, HEAD_DIM, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    block_table = torch.arange(total_physical, dtype=torch.int32).reshape(batch, max_blocks)

    select_idx_parts = []
    select_num_parts = []
    for batch_idx in range(batch):
        select_idx_batch, select_num_batch = _fast_causal_select_idx(
            q_seqlen, kv_seqlen, KV_HEADS, BLOCK_SIZE, TOP_K,
            random_select=random_select, bias=bias, rng_seed=seed + batch_idx)
        select_idx_parts.append(select_idx_batch)
        select_num_parts.append(select_num_batch)
    select_idx = torch.cat(select_idx_parts, dim=1)
    select_num = torch.cat(select_num_parts, dim=1)
    actual_seq_lengths = torch.full((batch,), q_seqlen, dtype=torch.int32)
    actual_seq_lengths_kv = torch.full((batch,), kv_seqlen, dtype=torch.int32)

    # select_idx is batch-local, which is the format shared by decode and the CSR builder.
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

    def _run_data(self, data, kv_seqlen, mode):
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
                KV_HEADS, scale, BLOCK_SIZE, TOP_K, 1)

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

        diff = (p_out.float() - d_out.float()).abs()
        print(
            f"[kv={kv_seqlen//1024}K B={BATCH_SIZE} S={Q_SEQLEN} "
            f"BxS={BATCH_SIZE * Q_SEQLEN} kvh={KV_HEADS} tk={TOP_K} mode={mode}] "
            f"shape={tuple(p_out.shape)} nonfinite={int((~torch.isfinite(p_out.float())).sum())} "
            f"max_diff={diff.max().item():.6f} mean={diff.mean().item():.6f}")

        # accuracy: prefill vs decode must agree (loose rtol). assertRtolEqual also fails on NaN.
        self.assertRtolEqual(d_out.float().numpy(), p_out.float().numpy(), prec=PREC)

    def _run_one(self, kv_seqlen, *, random_select=False, bias=None):
        data = _gen_data(kv_seqlen, SEED, random_select=random_select,
                         bias=bias)
        mode = bias if bias is not None else ("random" if random_select else "last-blocks")
        self._run_data(data, kv_seqlen, mode)

    def _run_production(self, step):
        path = Path(PRODUCTION_ROWPTR_DIR) / f"k2q_row_ptr_step{step:06d}.pt"
        obj = torch.load(path, map_location="cpu")
        row_ptr = obj["k2q_row_ptr"]
        num_blocks = row_ptr.shape[1] - 1
        kv_seqlen = num_blocks * BLOCK_SIZE
        data = _gen_data_from_production_rowptr(row_ptr, SEED)
        self._run_data(data, kv_seqlen, f"prod-step{step}")

    def test_prefill_vs_decode_sweep(self):
        print(f"\n=== prefill vs decode accuracy, case 1: last contiguous causal blocks "
              f"(batch=1, q_heads={Q_HEADS}, kv_heads={KV_HEADS}, top_k={TOP_K}, "
              f"q_seqlen={Q_SEQLEN}) ===")
        print("kv(K)  max_diff    mean_diff")
        for s in KV_SEQLENS:
            self._run_one(s, random_select=False)

    def test_prefill_vs_decode_random(self):
        print(f"\n=== prefill vs decode accuracy, case 2: random causal block selection "
              f"(at most top_k blocks drawn from the causal-visible range) ===")
        print("kv(K)  max_diff    mean_diff")
        for s in KV_SEQLENS:
            self._run_one(s, random_select=True)

    def test_prefill_vs_decode_unbalanced_heavy_tail(self):
        print(f"\n=== prefill vs decode accuracy, case 3a: UNBALANCED heavy-tail KV selection "
              f"(~6 saturated hot blocks + mid-hot tail; ~1.5% of blocks carry ~60-70% of "
              f"nnz, ~40-50% zero-nnz) ===")
        print("kv(K)  max_diff    mean_diff")
        for s in KV_SEQLENS:
            self._run_one(s, bias="heavy_tail")

    def test_prefill_vs_decode_unbalanced_dual_focus(self):
        print(f"\n=== prefill vs decode accuracy, case 3b: UNBALANCED dual-focus KV selection "
              f"(saturated front cluster + back-quarter cluster; middle quartiles cold) ===")
        print("kv(K)  max_diff    mean_diff")
        for s in KV_SEQLENS:
            self._run_one(s, bias="dual_focus")

    def test_prefill_vs_decode_unbalanced_spread(self):
        print(f"\n=== prefill vs decode accuracy, case 3c: UNBALANCED spread-sparse KV selection "
              f"(~16 hot blocks sampled uniformly at medium weight; heavier mid buckets, lower "
              f"zero-nnz fraction) ===")
        print("kv(K)  max_diff    mean_diff")
        for s in KV_SEQLENS:
            self._run_one(s, bias="spread")

    def test_prefill_vs_decode_production_rowptr(self):
        print(f"\n=== prefill vs decode accuracy, case 4: REAL production k2q_row_ptr loaded "
              f"from {PRODUCTION_ROWPTR_DIR} (steps={PRODUCTION_STEPS}) — production-faithful "
              f"heavy-tail + zero-row distribution; stresses per-core partition like on-device ===")
        print("step  R      max_diff    mean_diff")
        for step in PRODUCTION_STEPS:
            self._run_production(step)


if __name__ == "__main__":
    run_tests()
