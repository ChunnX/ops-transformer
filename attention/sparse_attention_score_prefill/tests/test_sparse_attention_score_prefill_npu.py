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
        4,
    )

    npu_out_cpu = npu_out.cpu()
    max_diff = self._max_abs_diff(kv_centric, npu_out_cpu)
    mean_diff = (kv_centric.float() - npu_out_cpu.float()).abs().mean().item()
    print(
        f"[npu] batch={batch} q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
        f"kv_centric_vs_npu max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    self.assertFalse(
        torch.any(torch.isnan(npu_out_cpu.float())),
        "NPU output contains NaN")
    self.assertRtolEqual(
        kv_centric.float().numpy(), npu_out_cpu.float().numpy(), prec=prec)

  def test_bf16_single_batch_small(self):
    print("[test_bf16_single_batch_small]")
    self._run_npu_test(
        batch=1, q_seqlens=[4], kv_seqlens=[512],
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


if __name__ == '__main__':
  run_tests()
