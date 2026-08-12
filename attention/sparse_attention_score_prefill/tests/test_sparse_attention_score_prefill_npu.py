# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""
NPU precision test for sparse_attention_score_prefill.

Reuses golden make_case + cpu_golden_prefill_bf16 (KV-centric two-phase path)
and compares against NPU kernel output.
"""

import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
torch_npu.npu.set_device("npu:2")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
_GOLDEN_TEST = Path(__file__).resolve().parent / "test_sparse_attention_score_prefill_golden.py"
sys.path.insert(0, str(_TORCH_EXT))
sys.path.insert(0, str(_GOLDEN_TEST.parent))

_golden_spec = importlib.util.spec_from_file_location("prefill_golden", _GOLDEN_TEST)
prefill_golden = importlib.util.module_from_spec(_golden_spec)
_golden_spec.loader.exec_module(prefill_golden)

make_case = prefill_golden.make_case
cpu_golden_prefill_bf16 = prefill_golden.cpu_golden_prefill_bf16


def _load_prefill_op():
  import cann_ops_transformer.ops.sparse_attention_score_prefill  # noqa: F401


def _npu_bf16(t):
  return t.to(device="npu", dtype=torch.bfloat16).contiguous()


def _npu_i32(t):
  return t.to(device="npu", dtype=torch.int32).contiguous()


_load_prefill_op()


class TestSparseAttentionScorePrefillNPU(TestCase):
  """NPU vs CPU KV-centric golden for sparse_attention_score_prefill."""

  def _max_abs_diff(self, a, b):
    return (a.float() - b.float()).abs().max().item()

  def _run_npu_test(self, batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
                    head_dim=128, block_size=128, top_k=4, seed=42, prec=7e-3):
    torch.npu.synchronize()
    torch.npu.empty_cache()

    data = make_case(
        batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
        head_dim, block_size, top_k, seed)

    kv_centric = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"],
        data["k2q_row_ptr"], data["k2q_q_indices"], data["k2q_slot_indices"],
        data["block_table"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], data["top_k"], data["scale_value"])

    npu_out = torch.ops.cann_ops_transformer.npu_sparse_attention_score_prefill(
        data["query"].npu(),
        data["key"].npu(),
        data["value"].npu(),
        data["block_table"].npu(),
        data["k2q_row_ptr"].npu(),
        data["k2q_q_indices"].npu(),
        data["k2q_slot_indices"].npu(),
        data["actual_seq_lengths"].npu(),
        data["actual_seq_lengths_kv"].npu(),
        kv_heads,
        data["scale_value"],
        block_size,
        top_k,
        1,
    )

    npu_out_cpu = npu_out.cpu()
    max_diff = self._max_abs_diff(kv_centric, npu_out_cpu)
    mean_diff = (kv_centric.float() - npu_out_cpu.float()).abs().mean().item()
    print(
        f"[npu] batch={batch} q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
        f"kv_centric_vs_npu max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    print("npu_out_cpu:", npu_out_cpu)
    print("kv_centric:", kv_centric)

    self.assertFalse(
        torch.any(torch.isnan(npu_out_cpu.float())),
        "NPU output contains NaN")
    self.assertFalse(
        torch.any(torch.isinf(npu_out_cpu.float())),
        "NPU output contains Inf")
    # Phase2 scale-weights zeroing out (invalid-slot masking bug) produces an
    # all-zero output that NaN/rtol checks can miss when the golden is also ~0.
    self.assertFalse(
        torch.all(npu_out_cpu.float() == 0),
        "NPU output is all zeros (suspect Phase2 combine / scale-weight mask)")
    self.assertRtolEqual(
        kv_centric.float().numpy(), npu_out_cpu.float().numpy(), prec=prec)

  def test_bf16_single_batch_small(self):
    print("[test_bf16_single_batch_small]")
    self._run_npu_test(
        batch=1, q_seqlens=[4], kv_seqlens=[512],
        #q_heads=8, kv_heads=2, top_k=1)
        q_heads=8, kv_heads=2, top_k=3)

  def test_bf16_single_batch_medium(self):
    print("[test_bf16_single_batch_medium]")
    self._run_npu_test(
        batch=1, q_seqlens=[32], kv_seqlens=[2048],
        q_heads=16, kv_heads=4, top_k=5, seed=123)

  def test_bf16_multi_batch(self):
    print("[test_bf16_multi_batch]")
    self._run_npu_test(
        batch=3, q_seqlens=[8, 16, 4], kv_seqlens=[1024, 2048, 512],
        q_heads=8, kv_heads=2, top_k=4, seed=456, prec=1e-2)

  def test_bf16_gqa_large_group(self):
    print("[test_bf16_gqa_large_group]")
    self._run_npu_test(
        batch=1, q_seqlens=[16], kv_seqlens=[1024],
        q_heads=32, kv_heads=4, top_k=6, seed=789)

  def test_bf16_partial_last_block(self):
    print("[test_bf16_partial_last_block]")
    self._run_npu_test(
        batch=1, q_seqlens=[8], kv_seqlens=[300],
        q_heads=8, kv_heads=2, top_k=3, seed=101)

  def test_bf16_top_k_1(self):
    print("[test_bf16_top_k_1]")
    self._run_npu_test(
        batch=1, q_seqlens=[16], kv_seqlens=[1024],
        q_heads=8, kv_heads=2, top_k=1, seed=202)

  def test_bf16_long_prefill(self):
    print("[test_bf16_long_prefill]")
    self._run_npu_test(
        batch=1, q_seqlens=[128], kv_seqlens=[4096],
        q_heads=8, kv_heads=2, top_k=8, seed=303)

  def test_bf16_minimal_debug(self):
    print("[test_bf16_minimal_debug]")
    self._run_npu_test(
        batch=1, q_seqlens=[1], kv_seqlens=[128],
        q_heads=1, kv_heads=1, top_k=1, seed=0, prec=1e-2)

  def test_bf16_minimal_debug_2(self):
    print("[test_bf16_minimal_debug_2]")
    self._run_npu_test(
        batch=1, q_seqlens=[2], kv_seqlens=[128],
        q_heads=2, kv_heads=2, top_k=1, seed=1, prec=1e-2)

  def test_bf16_minimal_debug_gqa(self):
    print("[test_bf16_minimal_debug_gqa]")
    self._run_npu_test(
        batch=1, q_seqlens=[1], kv_seqlens=[128],
        q_heads=2, kv_heads=1, top_k=1, seed=2, prec=1e-2)


  def test_bf16_debug(self):
    print("[test_bf16_q2_kv2_q64_kv4_tk16_debug]")
    self._run_npu_test(
        batch=1, q_seqlens=[16], kv_seqlens=[16],
        q_heads=64, kv_heads=4, top_k=16, seed=42, prec=1e-2)

  # ---------------------------------------------------------------------------
  # Production-config precision sweep. FIXED dims (do NOT vary):
  #   head_dim=128, block_size=128, top_k=16,
  #   q_heads=64, kv_heads=4, group_size=16 (batchGroupsMax=8).
  # Only batch / q_seqlens / kv_seqlens / seed vary. Production shows
  # INTERMITTENT (seed/shape-dependent) precision regressions, so the cases
  # target the accumulation / per-core split-row paths that only fire on
  # specific inputs:
  #   * seed sweeps       -> same shape, many seeds (catches seed-driven 偶现;
  #                          accumulation order / split-row boundaries depend
  #                          on the exact value distribution)
  #   * q-boundary sweep  -> q_seqlen straddling block_size multiples (causal
  #                          clipping / packed-row walker re-seed at boundaries)
  #   * kv-blockcount     -> N=16 full-topK threshold, N<16 partial slots,
  #                          N>16 deep Phase2 combine, partial-tail variants
  #   * edge shapes       -> pure prefill / decode-like / hot packed rows /
  #                          uneven multi-batch (split rows) / very long KV
  #   * uneven multi-batch-> the per-core split-row path (hot batch with many
  #                          q-tokens on few KV rows), the key precision lever
  # Each case uses subTest so a single flake is reported in isolation.
  # ---------------------------------------------------------------------------

  _HEAD_DIM = 128
  _BLOCK_SIZE = 128
  _TOP_K = 16
  _Q_HEADS = 64
  _KV_HEADS = 4
  _GROUP_SIZE = 16  # = _Q_HEADS // _KV_HEADS

  def _run_prod_config(self, batch, q_seqlens, kv_seqlens,
                       seed, prec=1e-2, tag=""):
    if tag:
      print(f"[prod:{tag}]")
    self._run_npu_test(
        batch, q_seqlens, kv_seqlens, self._Q_HEADS, self._KV_HEADS,
        head_dim=self._HEAD_DIM, block_size=self._BLOCK_SIZE,
        top_k=self._TOP_K, seed=seed, prec=prec)

  def test_prod_topk16_seed_sweep(self):
    """Same representative shape, many seeds -> reproduce 偶现 precision flakes.

    Multi-batch uneven (a hot batch with many q-tokens on few KV rows) drives
    the nnz-balanced per-core split-row path; mixed KV lengths exercise both
    full-topK and partial-slot (valid_k<16) Phase2 masking.
    """
    for seed in range(40):
      with self.subTest(seed=seed):
        self._run_prod_config(
            batch=3, q_seqlens=[1, 256, 8], kv_seqlens=[512, 2048, 1024],
            seed=1000 + seed, prec=1e-2, tag=f"seed_sweep/{seed}")

  def test_prod_topk16_edge_shapes(self):
    """Boundary shapes that stress causal clipping, partial slots, and splits."""
    # (batch, q_seqlens, kv_seqlens, seed, prec, tag)
    edge_configs = [
      # pure prefill: q==kv, no history -> every token's current block == q_block
      (1, [2048], [2048], 4101, 1.5e-2, "pure_prefill_q_eq_kv"),
      # decode-like single token inside prefill
      (1, [1], [2048], 4102, 1e-2, "q1_decode_like"),
      # single KV block -> valid_k==1 (15 empty slots, Phase2 must mask)
      (1, [4], [128], 4103, 1e-2, "kv1block"),
      # short KV -> valid_k<=4<16 (many empty partial slots)
      (1, [32], [512], 4104, 1e-2, "kv4block_partial_slots"),
      # partial last KV block (2176 = 17*128, last block valid_size=0)
      (1, [16], [2176], 4105, 1e-2, "partial_last_block"),
      # very long KV -> deep Phase2 combine over 16 slots
      (1, [128], [8192], 4106, 1.5e-2, "kv8192_long"),
      # many tiny batches, each 2 KV blocks -> valid_k<=2 everywhere
      (8, [1] * 8, [256] * 8, 4107, 1e-2, "8batch_tiny"),
      # uneven multi-batch: batch1 is hot (512 q on 16 KV rows) -> split rows
      (2, [1, 512], [256, 2048], 4108, 1.5e-2, "uneven_hot_batch"),
      # single KV block across two batches
      (2, [2, 2], [128, 128], 4109, 1e-2, "2batch_kv1block"),
      # mid prefill, deep KV
      (1, [256], [4096], 4110, 1e-2, "mid_prefill_kv4096"),
      # uneven 3-batch with very long middle KV (head-outer walk + splits)
      (3, [8, 64, 4], [1024, 8192, 512], 4111, 1.5e-2, "3batch_mid_long"),
      # extreme long prefill
      (1, [1024], [8192], 4112, 1.5e-2, "extreme_q1024_kv8192"),
      # kv just under a full topK (15 blocks -> valid_k<=15)
      (1, [64], [1920], 4113, 1e-2, "kv15block"),
      # batch with an empty-ish KV row (kv<block -> 1 block, 1 q-token)
      (1, [1], [64], 4114, 1e-2, "kv_subblock"),
      # long prefill, deep KV
      (1, [256], [4096], 4115, 1.5e-2, "long_prefill_kv4096"),
      # mixed: one batch pure-decode (q=1) + one long prefill in same call
      (2, [1, 512], [2048, 4096], 4116, 1.5e-2, "mixed_decode_prefill"),
    ]
    for cfg in edge_configs:
      batch, q_sl, kv_sl, seed, prec, tag = cfg
      with self.subTest(tag=tag):
        self._run_prod_config(
            batch=batch, q_seqlens=q_sl, kv_seqlens=kv_sl,
            seed=seed, prec=prec, tag=tag)

  def test_prod_topk16_q_boundary_sweep(self):
    """q_seqlen across block_size boundaries (causal q_block / tile edges).

    Values straddle 128/256/512/1024/2048 to catch off-by-one in causal clipping
    and packed-row walker re-seed at batch/head boundaries. KV held at 2048
    (full 16-block topK) so the q boundary is the only varying axis.
    """
    q_vals = [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 64, 127, 128, 129,
              255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048]
    for idx, q in enumerate(q_vals):
      with self.subTest(q=q):
        self._run_prod_config(
            batch=1, q_seqlens=[q], kv_seqlens=[2048],
            seed=8000 + idx, prec=(1.5e-2 if q >= 1024 else 1e-2),
            tag=f"q_boundary/q{q}")

  def test_prod_topk16_kv_blockcount_sweep(self):
    """KV seqlen = N*128 for N in {1,2,4,8,15,16,17,32,64}.

    N=16 is the full-topK threshold (>=16 blocks -> valid_k==16); N<16 leaves
    partial slots; N>16 stresses deep Phase2 combine. Plus partial-tail
    variants (N*128+1) for causal valid-size clipping.
    """
    block_counts = [1, 2, 4, 8, 15, 16, 17, 32, 64]
    for idx, n in enumerate(block_counts):
      kv = n * 128
      with self.subTest(n_blocks=n):
        self._run_prod_config(
            batch=1, q_seqlens=[64], kv_seqlens=[kv],
            seed=9000 + idx, prec=(1.5e-2 if n >= 32 else 1e-2),
            tag=f"kv_blocks/n{n}")
    # partial tail variants (last block valid_size = 1)
    for idx, n in enumerate([1, 4, 16, 17, 32]):
      kv = n * 128 + 1
      with self.subTest(n_blocks=n, tail=1):
        self._run_prod_config(
            batch=1, q_seqlens=[64], kv_seqlens=[kv],
            seed=9100 + idx, prec=1e-2, tag=f"kv_blocks/n{n}_tail1")

  def test_prod_topk16_batch_count_sweep(self):
    """batch count 1..8 with one hot batch, rest small (packed-row imbalance)."""
    for b in [1, 2, 3, 4, 6, 8]:
      q_sl = [4] * (b - 1) + [128]
      kv_sl = [2048] * (b - 1) + [4096]
      with self.subTest(batch=b):
        self._run_prod_config(
            batch=b, q_seqlens=q_sl, kv_seqlens=kv_sl,
            seed=9300 + b, prec=1.5e-2, tag=f"batch_count/b{b}")

  def test_prod_topk16_history_heavy(self):
    """kv >> q (long history) -> the production decode-in-prefill path.

    Small q over very long KV stresses deep Phase2 combine (16 full slots) and
    causal-bound clipping at large q_position.
    """
    for idx, (q, kv) in enumerate([(1, 8192), (2, 8192), (4, 4096),
                                   (8, 8192), (4, 16384), (16, 16384)]):
      with self.subTest(q=q, kv=kv):
        self._run_prod_config(
            batch=1, q_seqlens=[q], kv_seqlens=[kv],
            seed=9400 + idx, prec=1.5e-2, tag=f"history/q{q}_kv{kv}")

  def test_prod_topk16_pure_prefill_seed_sweep(self):
    """q==kv (no history) seed sweep — pure prefill accumulation path."""
    for seed in range(25):
      with self.subTest(seed=seed):
        self._run_prod_config(
            batch=1, q_seqlens=[2048], kv_seqlens=[2048],
            seed=9500 + seed, prec=1.5e-2, tag=f"pure_prefill_sweep/{seed}")

  def test_prod_topk16_decode_seed_sweep(self):
    """q=1 (decode-like) seed sweep — single-token, 16-slot Phase2 combine."""
    for seed in range(25):
      with self.subTest(seed=seed):
        self._run_prod_config(
            batch=1, q_seqlens=[1], kv_seqlens=[2048],
            seed=9600 + seed, prec=1e-2, tag=f"decode_sweep/{seed}")

  def test_prod_topk16_uneven_multibatch_matrix(self):
    """Uneven multi-batch layouts -> the per-core split-row precision path.

    A hot batch (many q-tokens on few KV rows) forces nnz-balanced splits; a
    cold batch (q=1, short KV) forces partial-slot masking in the same call.
    These layouts are the most prone to 偶现 partition-coupled drift.
    """
    # (batch, q_seqlens, kv_seqlens, seed, prec, tag)
    uneven_configs = [
      (2, [1, 1024], [256, 2048], 9801, 1.5e-2, "2b_hot1024"),
      (2, [1, 2048], [256, 4096], 9802, 1.5e-2, "2b_hot2048"),
      (2, [2, 512], [128, 4096], 9803, 1.5e-2, "2b_hot512_kv1short"),
      (3, [1, 512, 1], [256, 2048, 256], 9804, 1.5e-2, "3b_mid_hot"),
      (3, [4, 4, 1024], [512, 512, 2048], 9805, 1.5e-2, "3b_tail_hot"),
      (4, [1, 256, 1, 256], [256, 2048, 256, 2048], 9806, 1.5e-2, "4b_alt_hot"),
      (2, [8, 2048], [512, 8192], 9807, 1.5e-2, "2b_hot2048_kv8192"),
      (3, [1, 1, 2048], [128, 128, 4096], 9808, 1.5e-2, "3b_two_cold_one_hot"),
      (2, [128, 1], [2048, 256], 9809, 1.5e-2, "2b_hot_first_then_cold"),
      (5, [1, 1, 1, 1, 512], [256, 256, 256, 256, 4096], 9810, 1.5e-2, "5b_cold_then_hot"),
    ]
    for cfg in uneven_configs:
      batch, q_sl, kv_sl, seed, prec, tag = cfg
      with self.subTest(tag=tag):
        self._run_prod_config(
            batch=batch, q_seqlens=q_sl, kv_seqlens=kv_sl,
            seed=seed, prec=prec, tag=f"uneven/{tag}")

  def test_prod_topk16_uneven_seed_sweep(self):
    """Seed sweep on a split-row-heavy uneven layout — broaden the 偶现 net."""
    for seed in range(25):
      with self.subTest(seed=seed):
        self._run_prod_config(
            batch=2, q_seqlens=[1, 1024], kv_seqlens=[256, 4096],
            seed=9900 + seed, prec=1.5e-2, tag=f"uneven_sweep/{seed}")

  def test_prod_topk16_edge_multi_seed(self):
    """Prone-to-flake edge shapes × 3 seeds each — widen the 偶现 net."""
    base_edges = [
      (1, [16], [2176], "partial_last_block"),
      (1, [1], [128], "kv1block"),
      (1, [32], [512], "kv4block_partial_slots"),
      (2, [1, 512], [256, 2048], "uneven_hot_batch"),
      (1, [1024], [8192], "extreme_q1024_kv8192"),
      (1, [256], [4096], "long_prefill_kv4096"),
    ]
    for (batch, q_sl, kv_sl, tag0), s in (
        (cfg, s) for cfg in base_edges for s in range(3)):
      with self.subTest(tag=tag0, seed_off=s):
        self._run_prod_config(
            batch=batch, q_seqlens=q_sl, kv_seqlens=kv_sl,
            seed=9700 + s * 37, prec=1.5e-2, tag=f"edge_ms/{tag0}/s{s}")


if __name__ == '__main__':
  run_tests()
