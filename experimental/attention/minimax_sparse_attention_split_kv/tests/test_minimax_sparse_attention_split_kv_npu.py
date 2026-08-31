# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""
NPU precision test for minimax_sparse_attention_split_kv.

Reuses golden make_case + cpu_golden_prefill_bf16 (KV-centric two-phase path)
and compares against NPU kernel output.
"""

import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
torch_npu.npu.set_device(os.getenv("NPU_DEVICE", "npu:0"))

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
_GOLDEN_TEST = Path(__file__).resolve().parent / "test_minimax_sparse_attention_split_kv_golden.py"
sys.path.insert(0, str(_TORCH_EXT))
sys.path.insert(0, str(_GOLDEN_TEST.parent))

_golden_spec = importlib.util.spec_from_file_location("prefill_golden", _GOLDEN_TEST)
prefill_golden = importlib.util.module_from_spec(_golden_spec)
_golden_spec.loader.exec_module(prefill_golden)

make_case = prefill_golden.make_case
cpu_golden_prefill_bf16 = prefill_golden.cpu_golden_prefill_bf16
cpu_golden_prefill_phase1 = prefill_golden.cpu_golden_prefill_phase1

_NPU_OP_TIMEOUT_SECONDS = 180.0


def _load_prefill_op():
  # Avoid importing unrelated custom-op builders (for example mega_moe) during
  # collection; their headers can be version-sensitive across CANN installs.
  os.environ.setdefault("CANN_OPS_TRANSFORMER_LAZY_IMPORT", "1")
  import cann_ops_transformer.ops.sparse_attention_score  # noqa: F401
  import cann_ops_transformer.ops.minimax_sparse_attention_split_kv  # noqa: F401


def _npu_bf16(t):
  return t.to(device="npu", dtype=torch.bfloat16).contiguous()


def _npu_i32(t):
  return t.to(device="npu", dtype=torch.int32).contiguous()


def _npu_axis0_strided(t, stride_factor):
  storage_shape = (t.shape[0] * stride_factor, *t.shape[1:])
  storage = torch.empty(storage_shape, device="npu", dtype=t.dtype)
  view = storage[::stride_factor]
  view.copy_(t.to(device="npu"))
  assert not view.is_contiguous()
  return view


_load_prefill_op()


class TestMinimaxSparseAttentionSplitKvNPU(TestCase):
  """NPU vs CPU KV-centric golden for minimax_sparse_attention_split_kv."""

  def _max_abs_diff(self, a, b):
    return (a.float() - b.float()).abs().max().item()

  def _make_arch22_133120_case(self, q_seqlen=2048):
    """Fast-build the msa_compare_arch22 shape without CPU QK top-k pooling."""
    batch = 1
    kv_seqlen = 133120
    q_heads = 16
    kv_heads = 1
    head_dim = 128
    block_size = 128
    top_k = 16
    seed = 2026
    layout_seed = 137
    max_blocks = kv_seqlen // block_size
    physical_blocks = 1563
    total_q_tokens = batch * q_seqlen

    torch.manual_seed(seed)
    query = (torch.rand(total_q_tokens, q_heads, head_dim, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    key = (torch.rand(physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)
    value = (torch.rand(physical_blocks, block_size, kv_heads, head_dim, dtype=torch.float32) * 2 - 1).to(torch.bfloat16)

    import random
    physical_ids = list(range(physical_blocks))
    random.Random(layout_seed).shuffle(physical_ids)
    block_table = torch.tensor(physical_ids[:max_blocks * batch], dtype=torch.int32).reshape(batch, max_blocks)

    history_len = kv_seqlen - q_seqlen
    q_tokens = torch.arange(q_seqlen, dtype=torch.int64)
    current_blocks = (history_len + q_tokens) // block_size
    slot_offsets = torch.arange(top_k, dtype=torch.int64)
    selected = current_blocks.unsqueeze(1) - top_k + 1 + slot_offsets.unsqueeze(0)
    selected = torch.clamp(selected, min=0)
    select_idx = selected.unsqueeze(0).expand(kv_heads, total_q_tokens, top_k).contiguous().to(torch.int32)
    select_num_idx = torch.full((kv_heads, total_q_tokens), top_k, dtype=torch.int32)

    actual_seq_lengths = torch.tensor([q_seqlen], dtype=torch.int32)
    actual_seq_lengths_kv = torch.tensor([kv_seqlen], dtype=torch.int32)
    k2q_row_ptr, k2q_q_indices, k2q_slot_indices = prefill_golden.build_k2q_csr_torch_reference(
        select_idx, actual_seq_lengths, actual_seq_lengths_kv, block_size, q_global_offset=True)

    return {
        "query": query,
        "key": key,
        "value": value,
        "block_table": block_table,
        "select_idx": select_idx,
        "select_num_idx": select_num_idx,
        "k2q_row_ptr": k2q_row_ptr.to(torch.int32),
        "k2q_q_indices": k2q_q_indices.to(torch.int32),
        "k2q_slot_indices": k2q_slot_indices.to(torch.int32),
        "actual_seq_lengths": actual_seq_lengths,
        "actual_seq_lengths_kv": actual_seq_lengths_kv,
        "kv_heads": kv_heads,
        "scale_value": 1.0 / (head_dim ** 0.5),
        "block_size": block_size,
        "top_k": top_k,
        "inner_precise": 4,
    }

  def _run_arch22_precision_once(self, data, sample_tokens, label):
    """Run one A2 prefill invocation and compare selected rows with CPU golden."""
    q = data["query"].npu()
    k = data["key"].npu()
    v = data["value"].npu()
    bt = data["block_table"].npu()
    row_ptr = data["k2q_row_ptr"].npu()
    q_indices = data["k2q_q_indices"].npu()
    slot_indices = data["k2q_slot_indices"].npu()
    actual_seq_lengths = data["actual_seq_lengths"].npu()
    actual_seq_lengths_kv = data["actual_seq_lengths_kv"].npu()

    def prefill_fn():
      return torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
          q, k, v, bt, row_ptr, q_indices, slot_indices,
          actual_seq_lengths, actual_seq_lengths_kv,
          data["kv_heads"], data["scale_value"], data["block_size"],
          data["top_k"], data["inner_precise"])

    prefill_out = prefill_fn()
    torch.npu.synchronize()
    prefill_cpu = prefill_out.cpu()
    reference_sample = self._arch22_cpu_reference_sample(data, sample_tokens)
    actual_sample = prefill_cpu[sample_tokens]
    diff = (actual_sample.float() - reference_sample.float()).abs()
    print(
        f"[{label}] query={tuple(data['query'].shape)} "
        f"max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f} "
        f"samples={sample_tokens}")
    self.assertFalse(torch.any(torch.isnan(prefill_cpu.float())), "Prefill output contains NaN")
    self.assertFalse(torch.any(torch.isinf(prefill_cpu.float())), "Prefill output contains Inf")
    self.assertFalse(torch.all(prefill_cpu.float() == 0), "Prefill output is all zeros")
    self.assertRtolEqual(
        reference_sample.float().numpy(), actual_sample.float().numpy(), prec=1.5e-2)
    return prefill_fn

  def _run_arch22_phase1_dump_once(self, data, label):
    """Return the raw debug-mode workspace view for the deterministic probe."""
    tensors = [
        data["query"].npu(), data["key"].npu(), data["value"].npu(),
        data["block_table"].npu(), data["k2q_row_ptr"].npu(),
        data["k2q_q_indices"].npu(), data["k2q_slot_indices"].npu(),
        data["actual_seq_lengths"].npu(), data["actual_seq_lengths_kv"].npu(),
    ]
    out = torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
        *tensors, data["kv_heads"], data["scale_value"], data["block_size"],
        data["top_k"], data["inner_precise"])
    torch.npu.synchronize()
    result = out.cpu()
    print(f"[{label}] range=({result.float().min().item():.6f}, {result.float().max().item():.6f})")
    return result

  def _time_npu_op(self, fn, warmup=2, repeat=5):
    for _ in range(warmup):
      fn()
    torch.npu.synchronize()

    timings_ms = []
    out = None
    for _ in range(repeat):
      start = time.perf_counter()
      out = fn()
      torch.npu.synchronize()
      timings_ms.append((time.perf_counter() - start) * 1000.0)
    return out, timings_ms

  def _arch22_cpu_reference_sample(self, data, q_tokens):
    """Compute a small independent two-phase reference for selected rows.

    The production case is intentionally too large for a full CPU QK/PV
    reference.  Reusing the same block math on the first, middle, and last
    query rows still checks the causal boundary, block-table indirection,
    Phase1 statistics, and Phase2 combine without invoking the decode op.
    """
    query = data["query"]
    key = data["key"]
    value = data["value"]
    block_table = data["block_table"]
    select_idx = data["select_idx"]
    q_seqlen = int(data["actual_seq_lengths"][0].item())
    kv_seqlen = int(data["actual_seq_lengths_kv"][0].item())
    block_size = data["block_size"]
    top_k = data["top_k"]
    group_size = query.shape[1] // data["kv_heads"]
    head_dim = query.shape[2]
    output = torch.zeros(len(q_tokens), query.shape[1], head_dim, dtype=torch.bfloat16)

    history_len = kv_seqlen - q_seqlen
    for out_row, q_token in enumerate(q_tokens):
      q_position = history_len + q_token
      for kv_head in range(data["kv_heads"]):
        q_start = kv_head * group_size
        partials = []
        maxima = []
        sums = []
        for slot in range(top_k):
          logical_block = int(select_idx[kv_head, q_token, slot].item())
          block_begin = logical_block * block_size
          valid_end = min(block_begin + block_size, kv_seqlen, q_position + 1)
          valid_len = max(valid_end - block_begin, 0)
          if valid_len == 0:
            partials.append(torch.zeros(group_size, head_dim, dtype=torch.float32))
            maxima.append(torch.full((group_size,), -float("inf")))
            sums.append(torch.zeros(group_size))
            continue
          physical_block = int(block_table[0, logical_block].item())
          partial, row_max, row_sum = prefill_golden._bf16_block_qk_sm_pv(
              query[q_token, q_start:q_start + group_size, :],
              key[physical_block, :valid_len, kv_head, :],
              value[physical_block, :valid_len, kv_head, :],
              data["scale_value"], q_token, kv_head, slot, data["inner_precise"])
          partials.append(partial)
          maxima.append(row_max)
          sums.append(row_sum)

        partials = torch.stack(partials, dim=0)
        maxima = torch.stack(maxima, dim=0)
        sums = torch.stack(sums, dim=0)
        valid = torch.isfinite(maxima) & (sums > 0)
        masked_maxima = torch.where(
            valid, maxima, torch.full_like(maxima, -float("inf")))
        global_max = masked_maxima.max(dim=0).values
        exp_weight = torch.where(
            valid, torch.exp(maxima - global_max.unsqueeze(0)), torch.zeros_like(sums))
        denominator = (sums * exp_weight).sum(dim=0)
        combined_scale = torch.where(
            denominator.unsqueeze(0) > 0,
            exp_weight / denominator.clamp_min(1e-30).unsqueeze(0),
            torch.zeros_like(exp_weight))
        combined = (combined_scale.unsqueeze(-1) * partials).sum(dim=0)
        output[out_row, q_start:q_start + group_size, :] = combined.to(torch.bfloat16)
    return output

  def _run_npu_test(self, batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
                    head_dim=128, block_size=128, top_k=4, seed=42, prec=7e-3,
                    inner_precise=4):
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
        data["block_size"], data["top_k"], data["scale_value"], inner_precise)

    def abort_hung_npu():
      print(
          f"[npu-timeout] exceeded {_NPU_OP_TIMEOUT_SECONDS:.0f}s; "
          "terminating the test process",
          flush=True)
      os._exit(124)

    timeout = threading.Timer(_NPU_OP_TIMEOUT_SECONDS, abort_hung_npu)
    timeout.daemon = True
    npu_start = time.perf_counter()
    print(
        f"[npu-launch] batch={batch} q={q_seqlens} kv={kv_seqlens} "
        f"head_dim={head_dim} top_k={top_k} inner_precise={inner_precise}",
        flush=True)
    timeout.start()
    try:
      npu_out = torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
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
          inner_precise,
      )
      npu_out_cpu = npu_out.cpu()
    finally:
      timeout.cancel()
    npu_elapsed_ms = (time.perf_counter() - npu_start) * 1000.0
    print(f"[npu-done] elapsed_ms={npu_elapsed_ms:.3f}", flush=True)
    max_diff = self._max_abs_diff(kv_centric, npu_out_cpu)
    mean_diff = (kv_centric.float() - npu_out_cpu.float()).abs().mean().item()
    print(
        f"[npu] batch={batch} q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
        f"kv_centric_vs_npu max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

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

  def _run_constant_boundary_npu_test(
      self, q_seqlens, kv_seqlens, label, inner_precise=4):
    """Validate large indexing boundaries with an exact constant result."""
    batch = len(q_seqlens)
    self.assertEqual(batch, len(kv_seqlens))
    block_size = 128
    head_dim = 128
    q_heads = 1
    kv_heads = 1
    top_k = 1
    total_q = sum(q_seqlens)
    rows_per_batch = [
        (kv_len + block_size - 1) // block_size for kv_len in kv_seqlens]
    max_blocks = max(rows_per_batch, default=0)
    physical_blocks = sum(rows_per_batch)

    actual_q = torch.tensor(q_seqlens, dtype=torch.int32)
    actual_kv = torch.tensor(kv_seqlens, dtype=torch.int32)
    block_table = torch.zeros((batch, max_blocks), dtype=torch.int32)
    select_idx = torch.empty((kv_heads, total_q, top_k), dtype=torch.int32)
    q_offset = 0
    physical_offset = 0
    for batch_idx, (q_len, kv_len, kv_rows) in enumerate(
        zip(q_seqlens, kv_seqlens, rows_per_batch)):
      self.assertGreaterEqual(kv_len, q_len)
      if kv_rows > 0:
        block_table[batch_idx, :kv_rows] = (
            torch.arange(kv_rows, dtype=torch.int32) + physical_offset)
      if q_len > 0:
        history_len = kv_len - q_len
        q_positions = torch.arange(q_len, dtype=torch.int64)
        select_idx[0, q_offset:q_offset + q_len, 0] = (
            (history_len + q_positions) // block_size).to(torch.int32)
      q_offset += q_len
      physical_offset += kv_rows

    row_ptr, q_indices, slot_indices = (
        prefill_golden.build_k2q_csr_torch_reference(
            select_idx, actual_q, actual_kv, block_size,
            q_global_offset=True))

    torch.npu.synchronize()
    torch.npu.empty_cache()
    query = torch.zeros(
        (total_q, q_heads, head_dim), dtype=torch.bfloat16, device="npu")
    key = torch.zeros(
        (physical_blocks, block_size, kv_heads, head_dim),
        dtype=torch.bfloat16, device="npu")
    value = torch.full_like(key, 0.25)
    metadata = [
        block_table.npu(), row_ptr.npu(), q_indices.npu(),
        slot_indices.npu(), actual_q.npu(), actual_kv.npu()]

    print(
        f"[{label}] launch total_q={total_q} kv={kv_seqlens} "
        f"inner_precise={inner_precise} "
        f"physical_blocks={physical_blocks}")
    torch.npu.synchronize()
    start = time.perf_counter()
    output = torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
        query, key, value, *metadata,
        kv_heads, 1.0 / (head_dim ** 0.5), block_size, top_k, inner_precise)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    expected = torch.tensor(0.25, dtype=torch.bfloat16, device="npu")
    exact = bool(torch.all(output == expected).item())
    sample_indices = [0, total_q // 2, total_q - 1]
    samples = output[sample_indices, 0, 0].cpu().tolist()
    print(
        f"[{label}] elapsed_ms={elapsed_ms:.3f} exact={exact} "
        f"samples={samples}")
    self.assertEqual(tuple(output.shape), (total_q, q_heads, head_dim))
    self.assertEqual(output.dtype, torch.bfloat16)
    self.assertTrue(exact, "constant boundary output is not exactly 0.25")

    del output, expected, query, key, value, metadata
    torch.npu.empty_cache()

  def _run_npu_lse_test(self, batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
                        top_k, seed, inner_precise=4, prec=7e-3):
    data = make_case(
        batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
        128, 128, top_k, seed)
    attention_golden = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"],
        data["k2q_row_ptr"], data["k2q_q_indices"], data["k2q_slot_indices"],
        data["block_table"], data["actual_seq_lengths"],
        data["actual_seq_lengths_kv"], data["block_size"], data["top_k"],
        data["scale_value"], inner_precise)
    _, ws_max, ws_sum = cpu_golden_prefill_phase1(
        data["query"], data["key"], data["value"],
        data["k2q_row_ptr"], data["k2q_q_indices"], data["k2q_slot_indices"],
        data["block_table"], data["actual_seq_lengths"],
        data["actual_seq_lengths_kv"], data["block_size"], data["top_k"],
        data["scale_value"], inner_precise)

    valid = torch.isfinite(ws_max) & (ws_sum > 0)
    safe_max = torch.where(valid, ws_max, torch.full_like(ws_max, -float("inf")))
    global_max = safe_max.max(dim=2).values
    exp_weight = torch.where(
        valid,
        torch.exp(ws_max - global_max.unsqueeze(2)),
        torch.zeros_like(ws_sum))
    global_sum = (ws_sum * exp_weight).sum(dim=2)
    lse_golden = torch.where(
        global_sum > 0,
        global_max + torch.log(global_sum),
        torch.full_like(global_sum, float("inf")))
    lse_golden = lse_golden.reshape(data["query"].shape[0], q_heads, 1)

    attention_out, softmax_lse = (
        torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv_with_lse(
            data["query"].npu(), data["key"].npu(), data["value"].npu(),
            data["block_table"].npu(), data["k2q_row_ptr"].npu(),
            data["k2q_q_indices"].npu(), data["k2q_slot_indices"].npu(),
            data["actual_seq_lengths"].npu(), data["actual_seq_lengths_kv"].npu(),
            kv_heads, data["scale_value"], data["block_size"], top_k,
            inner_precise))
    attention_cpu = attention_out.cpu()
    lse_cpu = softmax_lse.cpu()

    self.assertEqual(tuple(softmax_lse.shape), (sum(q_seqlens), q_heads, 1))
    self.assertEqual(softmax_lse.dtype, torch.float32)
    self.assertFalse(torch.any(torch.isnan(lse_cpu)))
    self.assertTrue(torch.equal(torch.isposinf(lse_golden), torch.isposinf(lse_cpu)))
    self.assertFalse(torch.any(torch.isneginf(lse_cpu)))
    self.assertRtolEqual(
        attention_golden.float().numpy(), attention_cpu.float().numpy(), prec=prec)
    finite = torch.isfinite(lse_golden)
    if torch.any(finite):
      self.assertRtolEqual(
          lse_golden[finite].numpy(), lse_cpu[finite].numpy(), prec=2e-3)
      lse_diff = (lse_golden[finite] - lse_cpu[finite]).abs()
      max_diff = lse_diff.max().item()
      mean_diff = lse_diff.mean().item()
    else:
      max_diff = 0.0
      mean_diff = 0.0
    print(
        f"[lse] q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
        f"inner_precise={inner_precise} max_diff={max_diff:.6f} "
        f"mean_diff={mean_diff:.6f}")

  def test_bf16_softmax_lse(self):
    """LSE output has fp32 [T,N,1] metadata and matches the stable golden."""
    cases = (
        (1, [4], [512], 8, 2, 3, 601),
        (1, [8], [300], 8, 2, 3, 602),
        (1, [32], [2048], 16, 4, 5, 603),
        (2, [4, 4], [0, 128], 8, 2, 1, 605),
    )
    for case in cases:
      with self.subTest(q=case[1], kv=case[2]):
        self._run_npu_lse_test(*case)
    with self.subTest(q=[4], kv=[512], inner_precise=1):
      self._run_npu_lse_test(
          1, [4], [512], 8, 2, 3, 604, inner_precise=1, prec=1.5e-2)
    with self.subTest(q=[4], kv=[512], inner_precise=0):
      self._run_npu_lse_test(
          1, [4], [512], 8, 2, 3, 606, inner_precise=0, prec=7e-3)

  def test_bf16_single_batch_small(self):
    print("[test_bf16_single_batch_small]")
    self._run_npu_test(
        batch=1, q_seqlens=[4], kv_seqlens=[512],
        #q_heads=8, kv_heads=2, top_k=1)
        q_heads=8, kv_heads=2, top_k=3)

  def test_bf16_high_precision_softmax(self):
    """innerPrecise=0 keeps QK score S in fp32 until softmax."""
    print("[test_bf16_high_precision_softmax]")
    self._run_npu_test(
        batch=1, q_seqlens=[8], kv_seqlens=[512],
        q_heads=8, kv_heads=2, top_k=3, seed=405, inner_precise=0)

  def test_bf16_inner_precise_1(self):
    """BF16 O_partial remains accurate for regular and tail blocks."""
    print("[test_bf16_inner_precise_1]")
    cases = (
        ([4], [512], 8, 2, 3, 501),
        ([32], [2048], 16, 4, 5, 502),
        ([8], [300], 8, 2, 3, 503),
    )
    for q_seqlens, kv_seqlens, q_heads, kv_heads, top_k, seed in cases:
      with self.subTest(q=q_seqlens, kv=kv_seqlens):
        self._run_npu_test(
            batch=1, q_seqlens=q_seqlens, kv_seqlens=kv_seqlens,
            q_heads=q_heads, kv_heads=kv_heads, top_k=top_k, seed=seed,
            prec=1.5e-2, inner_precise=1)

  def test_bf16_head_dim_256_precision_paths(self):
    """D=256 works with every published innerPrecise path and a short block."""
    for inner_precise, prec in ((0, 7e-3), (1, 1.5e-2), (4, 7e-3)):
      with self.subTest(inner_precise=inner_precise):
        self._run_npu_test(
            batch=1, q_seqlens=[4], kv_seqlens=[72],
            q_heads=8, kv_heads=2, head_dim=256, block_size=16,
            top_k=3, seed=1800 + inner_precise, prec=prec,
            inner_precise=inner_precise)

  def test_bf16_gqa16_head_dim_256_precision_paths(self):
    """D=256 and groupSize=16 work together for every precision path."""
    for inner_precise, prec in ((0, 7e-3), (1, 1.5e-2), (4, 7e-3)):
      with self.subTest(inner_precise=inner_precise):
        self._run_npu_test(
            batch=1, q_seqlens=[4], kv_seqlens=[512],
            q_heads=16, kv_heads=1, head_dim=256, block_size=128,
            top_k=4, seed=1850 + inner_precise, prec=prec,
            inner_precise=inner_precise)

  def test_bf16_gqa16_head_dim_256_topk16_precision_paths(self):
    """D=256 covers the non-uniform 7/7/2 Phase2 split grouping."""
    for inner_precise, prec in ((0, 7e-3), (1, 1.5e-2), (4, 7e-3)):
      with self.subTest(inner_precise=inner_precise):
        self._run_npu_test(
            batch=1, q_seqlens=[4], kv_seqlens=[2048],
            q_heads=16, kv_heads=1, head_dim=256, block_size=128,
            top_k=16, seed=1860 + inner_precise, prec=prec,
            inner_precise=inner_precise)

  def test_bf16_max_heads_d256_mid_block_precision_paths(self):
    """H=64, GQA16, D=256, and blockSize=64 work on every precision path."""
    for inner_precise, prec in ((0, 7e-3), (1, 1.5e-2), (4, 7e-3)):
      with self.subTest(inner_precise=inner_precise):
        self._run_npu_test(
            batch=1, q_seqlens=[4], kv_seqlens=[288],
            q_heads=64, kv_heads=4, head_dim=256, block_size=64,
            top_k=4, seed=1870 + inner_precise, prec=prec,
            inner_precise=inner_precise)

  def test_bf16_paged_block_size_sweep(self):
    """Paged KV accepts every positive multiple of 16 through 128."""
    for index, block_size in enumerate(range(16, 129, 16)):
      kv_len = block_size * 4 + block_size // 2
      for inner_precise, prec in ((0, 7e-3), (1, 1.5e-2), (4, 7e-3)):
        with self.subTest(
            block_size=block_size, kv_len=kv_len,
            inner_precise=inner_precise):
          self._run_npu_test(
              batch=1, q_seqlens=[4], kv_seqlens=[kv_len],
              q_heads=8, kv_heads=2, head_dim=128, block_size=block_size,
              top_k=3, seed=1700 + index * 10 + inner_precise,
              prec=prec, inner_precise=inner_precise)

  def test_bf16_kv_seqlen_256k_boundary(self):
    """The maximum advertised paged KV sequence length executes correctly."""
    for inner_precise in (0, 1, 4):
      with self.subTest(inner_precise=inner_precise):
        self._run_constant_boundary_npu_test(
            q_seqlens=[128], kv_seqlens=[262144],
            label=f"kv_256k_boundary_inner{inner_precise}",
            inner_precise=inner_precise)

  def test_bf16_kv256k_gqa16_sample_precision_paths(self):
    """Random 256K KV data matches sampled CPU golden for all precision paths."""
    data = make_case(
        batch=1, q_seqlens=[16], kv_seqlens=[262144],
        q_heads=16, kv_heads=1, head_dim=128, block_size=128,
        top_k=16, seed=1880)
    for inner_precise in (0, 1, 4):
      with self.subTest(inner_precise=inner_precise):
        data["inner_precise"] = inner_precise
        self._run_arch22_precision_once(
            data, [0, 8, 15], f"kv256k_gqa16_inner{inner_precise}")

  def test_bf16_total_q_1024k_boundary(self):
    """The maximum advertised total Q length works across packed batches."""
    q_seqlens = [262144] * 4
    for inner_precise in (0, 1, 4):
      with self.subTest(inner_precise=inner_precise):
        self._run_constant_boundary_npu_test(
            q_seqlens=q_seqlens, kv_seqlens=q_seqlens,
            label=f"total_q_1024k_boundary_inner{inner_precise}",
            inner_precise=inner_precise)

  def test_bf16_axis0_strided_qkv(self):
    """Q/K/V views with independent leading strides match contiguous input."""
    print("[test_bf16_axis0_strided_qkv]")
    data = make_case(
        batch=1, q_seqlens=[8], kv_seqlens=[300], q_heads=8, kv_heads=2,
        head_dim=128, block_size=128, top_k=3, seed=404)

    q = data["query"].npu()
    k = data["key"].npu()
    v = data["value"].npu()
    inputs = [
        data["block_table"].npu(), data["k2q_row_ptr"].npu(),
        data["k2q_q_indices"].npu(), data["k2q_slot_indices"].npu(),
        data["actual_seq_lengths"].npu(), data["actual_seq_lengths_kv"].npu(),
    ]

    def run(query, key, value):
      return torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
          query, key, value, *inputs, data["kv_heads"], data["scale_value"],
          data["block_size"], data["top_k"], 4)

    contiguous_out = run(q, k, v)
    strided_out = run(
        _npu_axis0_strided(q, 2), _npu_axis0_strided(k, 2), _npu_axis0_strided(v, 3))
    torch.npu.synchronize()

    contiguous_cpu = contiguous_out.cpu()
    strided_cpu = strided_out.cpu()
    golden = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"], data["k2q_row_ptr"],
        data["k2q_q_indices"], data["k2q_slot_indices"], data["block_table"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], data["top_k"], data["scale_value"], 4)
    self.assertTrue(torch.equal(contiguous_cpu, strided_cpu))
    self.assertRtolEqual(golden.float().numpy(), strided_cpu.float().numpy(), prec=7e-3)

  def test_bf16_continuous_kv_multi_batch(self):
    """blockSize=0 supports D=128/256, GQA16, and every precision path."""
    print("[test_bf16_continuous_kv_multi_batch]")
    configs = (
        (128, 8, 2, 406),
        (256, 16, 1, 407),
    )
    for head_dim, q_heads, kv_heads, seed in configs:
      data = make_case(
          batch=2, q_seqlens=[8, 6], kv_seqlens=[144, 70],
          q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim,
          block_size=128, top_k=2, seed=seed)

      dense_key = []
      dense_value = []
      for batch_idx, kv_len in enumerate(data["actual_seq_lengths_kv"].tolist()):
        num_blocks = (kv_len + data["block_size"] - 1) // data["block_size"]
        physical = data["block_table"][batch_idx, :num_blocks].to(torch.long)
        dense_key.append(
            data["key"][physical].reshape(-1, data["kv_heads"], head_dim)[:kv_len])
        dense_value.append(
            data["value"][physical].reshape(-1, data["kv_heads"], head_dim)[:kv_len])
      dense_key = torch.cat(dense_key, dim=0)
      dense_value = torch.cat(dense_value, dim=0)

      for inner_precise, prec in ((0, 1e-2), (1, 1.5e-2), (4, 1e-2)):
        with self.subTest(head_dim=head_dim, inner_precise=inner_precise):
          golden = cpu_golden_prefill_bf16(
              data["query"], data["key"], data["value"], data["k2q_row_ptr"],
              data["k2q_q_indices"], data["k2q_slot_indices"], data["block_table"],
              data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
              data["block_size"], data["top_k"], data["scale_value"], inner_precise)
          out = torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
              data["query"].npu(), _npu_axis0_strided(dense_key, 2),
              _npu_axis0_strided(dense_value, 3), None,
              data["k2q_row_ptr"].npu(), data["k2q_q_indices"].npu(),
              data["k2q_slot_indices"].npu(), data["actual_seq_lengths"].npu(),
              data["actual_seq_lengths_kv"].npu(), data["kv_heads"],
              data["scale_value"], 0, data["top_k"], inner_precise).cpu()

          self.assertFalse(torch.any(torch.isnan(out.float())))
          self.assertFalse(torch.any(torch.isinf(out.float())))
          self.assertRtolEqual(
              golden.float().numpy(), out.float().numpy(), prec=prec)

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

  def test_bf16_zero_q_len_mixed_batch(self):
    """Batches with q_len=0 contribute no query rows and are skipped."""
    print("[test_bf16_zero_q_len_mixed_batch]")
    self._run_npu_test(
        batch=3, q_seqlens=[0, 4, 0], kv_seqlens=[128, 128, 0],
        q_heads=8, kv_heads=2, top_k=1, seed=457, prec=1e-2)

  def test_bf16_zero_kv_len_mixed_batch(self):
    """A leading batch with kv_len=0 produces zero output and is skipped."""
    print("[test_bf16_zero_kv_len_mixed_batch]")
    data = make_case(
        batch=2, q_seqlens=[4, 4], kv_seqlens=[0, 128],
        q_heads=8, kv_heads=2, head_dim=128, block_size=128, top_k=1, seed=458)
    golden = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"], data["k2q_row_ptr"],
        data["k2q_q_indices"], data["k2q_slot_indices"], data["block_table"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], data["top_k"], data["scale_value"])
    out = torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
        data["query"].npu(), data["key"].npu(), data["value"].npu(),
        data["block_table"].npu(), data["k2q_row_ptr"].npu(),
        data["k2q_q_indices"].npu(), data["k2q_slot_indices"].npu(),
        data["actual_seq_lengths"].npu(), data["actual_seq_lengths_kv"].npu(),
        data["kv_heads"], data["scale_value"], data["block_size"], data["top_k"], 0).cpu()
    self.assertTrue(torch.all(out[:4] == 0))
    self.assertRtolEqual(golden.float().numpy(), out.float().numpy(), prec=1e-2)

  def test_bf16_all_zero_lengths(self):
    """An entirely empty TND input returns an empty output without launching work."""
    print("[test_bf16_all_zero_lengths]")
    data = make_case(
        batch=1, q_seqlens=[0], kv_seqlens=[0],
        q_heads=8, kv_heads=2, head_dim=128, block_size=128, top_k=1, seed=459)
    out = torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
        data["query"].npu(), data["key"].npu(), data["value"].npu(),
        data["block_table"].npu(), data["k2q_row_ptr"].npu(),
        data["k2q_q_indices"].npu(), data["k2q_slot_indices"].npu(),
        data["actual_seq_lengths"].npu(), data["actual_seq_lengths_kv"].npu(),
        data["kv_heads"], data["scale_value"], data["block_size"], data["top_k"], 0)
    self.assertEqual(tuple(out.shape), (0, 8, 128))
    self.assertEqual(out.numel(), 0)

  def test_bf16_gqa_large_group(self):
    print("[test_bf16_gqa_large_group]")
    self._run_npu_test(
        batch=1, q_seqlens=[16], kv_seqlens=[1024],
        q_heads=32, kv_heads=4, top_k=6, seed=789)

  def test_bf16_group_size_sweep(self):
    """Every GQA group size has a golden; G=1/8 also cover inner 0 and 1."""
    for group_size in range(1, 17):
      inner_modes = (0, 1, 4) if group_size in (1, 8) else (4,)
      for inner_precise in inner_modes:
        with self.subTest(group_size=group_size, inner_precise=inner_precise):
          self._run_npu_test(
              batch=1, q_seqlens=[2], kv_seqlens=[256],
              q_heads=group_size, kv_heads=1, head_dim=128,
              block_size=128, top_k=2,
              seed=1900 + group_size * 10 + inner_precise,
              prec=(1.5e-2 if inner_precise == 1 else 1e-2),
              inner_precise=inner_precise)

  def test_bf16_partial_last_block(self):
    print("[test_bf16_partial_last_block]")
    for inner_precise, prec in ((0, 7e-3), (1, 1.5e-2), (4, 7e-3)):
      with self.subTest(inner_precise=inner_precise):
        self._run_npu_test(
            batch=1, q_seqlens=[8], kv_seqlens=[300],
            q_heads=8, kv_heads=2, top_k=3,
            seed=101 + inner_precise, prec=prec,
            inner_precise=inner_precise)

  def test_bf16_gqa16_uneven_batch_precision_paths(self):
    """Uneven multi-batch offsets and GQA16 work on every precision path."""
    for inner_precise, prec in ((0, 1e-2), (1, 1.5e-2), (4, 1e-2)):
      with self.subTest(inner_precise=inner_precise):
        self._run_npu_test(
            batch=2, q_seqlens=[1, 8], kv_seqlens=[256, 512],
            q_heads=64, kv_heads=4, head_dim=128, block_size=128,
            top_k=4, seed=1950 + inner_precise, prec=prec,
            inner_precise=inner_precise)

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

  def test_arch22_bf16_q2048_kv133120_precision_and_perf(self):
    """msa_compare_arch22 shape: B=1, Q=2048, KV=133120, H=16, KVH=1, D=128, topK=16."""
    print("[test_arch22_bf16_q2048_kv133120_precision_and_perf]")
    torch.npu.synchronize()
    torch.npu.empty_cache()
    data = self._make_arch22_133120_case()
    self.assertEqual(tuple(data["query"].shape), (2048, 16, 128))
    self.assertEqual(tuple(data["key"].shape), (1563, 128, 1, 128))
    self.assertEqual(tuple(data["value"].shape), (1563, 128, 1, 128))
    self.assertEqual(tuple(data["block_table"].shape), (1, 1040))
    self.assertEqual(tuple(data["k2q_row_ptr"].shape), (1, 1041))

    q = data["query"].npu()
    k = data["key"].npu()
    v = data["value"].npu()
    bt = data["block_table"].npu()
    row_ptr = data["k2q_row_ptr"].npu()
    q_indices = data["k2q_q_indices"].npu()
    slot_indices = data["k2q_slot_indices"].npu()
    actual_seq_lengths = data["actual_seq_lengths"].npu()
    actual_seq_lengths_kv = data["actual_seq_lengths_kv"].npu()
    kv_heads = data["kv_heads"]
    scale_value = data["scale_value"]
    block_size = data["block_size"]
    top_k = data["top_k"]
    inner_precise = data["inner_precise"]

    def prefill_fn():
      return torch.ops.cann_ops_transformer.npu_minimax_sparse_attention_split_kv(
          q, k, v, bt, row_ptr, q_indices, slot_indices,
          actual_seq_lengths, actual_seq_lengths_kv,
          kv_heads, scale_value, block_size, top_k, inner_precise)

    prefill_out = prefill_fn()
    torch.npu.synchronize()

    prefill_cpu = prefill_out.cpu()
    sample_tokens = [0, data["query"].shape[0] // 2, data["query"].shape[0] - 1]
    reference_sample = self._arch22_cpu_reference_sample(data, sample_tokens)
    actual_sample = prefill_cpu[sample_tokens]
    diff = (actual_sample.float() - reference_sample.float()).abs()
    per_token = diff.amax(dim=(1, 2))
    per_head = diff.amax(dim=2)
    print(
        f"[arch22_shape] query={tuple(data['query'].shape)} key={tuple(data['key'].shape)} "
        f"block_table={tuple(data['block_table'].shape)} row_ptr={tuple(data['k2q_row_ptr'].shape)} "
        f"max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f}")
    print(f"[arch22_shape_diff] sample_tokens={sample_tokens} per_token={per_token.tolist()}")
    print(f"[arch22_shape_diff] per_head={per_head.tolist()}")

    self.assertFalse(torch.any(torch.isnan(prefill_cpu.float())), "Prefill output contains NaN")
    self.assertFalse(torch.any(torch.isinf(prefill_cpu.float())), "Prefill output contains Inf")
    self.assertFalse(torch.all(prefill_cpu.float() == 0), "Prefill output is all zeros")
    self.assertRtolEqual(
        reference_sample.float().numpy(), actual_sample.float().numpy(), prec=1.5e-2)

    _, timings_ms = self._time_npu_op(prefill_fn)
    mean_ms = sum(timings_ms) / len(timings_ms)
    print(
        f"[arch22_shape_perf] prefill warmup=2 repeat={len(timings_ms)} "
        f"min={min(timings_ms):.3f}ms max={max(timings_ms):.3f}ms mean={mean_ms:.3f}ms "
        "(informational; use run_minimax_sparse_attention_split_kv_perf.py/msprof for the gate)")

  def test_arch22_bf16_q128_hot_csr_precision(self):
    """Regression: one hot CSR row split into many four-group batches."""
    print("[test_arch22_bf16_q128_hot_csr_precision]")
    torch.npu.synchronize()
    torch.npu.empty_cache()
    data = self._make_arch22_133120_case(q_seqlen=128)
    self._run_arch22_precision_once(data, [0, 64, 127], "arch22_q128_hot_csr")

  def test_arch22_bf16_q16_hot_csr_precision(self):
    """Regression: groupSize=16 fills an M=64 tile in every precision path."""
    print("[test_arch22_bf16_q16_hot_csr_precision]")
    torch.npu.synchronize()
    torch.npu.empty_cache()
    data = self._make_arch22_133120_case(q_seqlen=16)
    for inner_precise in (0, 1, 4):
      with self.subTest(inner_precise=inner_precise):
        data["inner_precise"] = inner_precise
        self._run_arch22_precision_once(
            data, [0, 8, 15], f"arch22_q16_hot_csr_inner{inner_precise}")

  def _manual_arch22_bf16_q16_hot_csr_phase1_dump(self):
    """Manual DEBUG_MODE=1/2/3 probe; not a production precision test."""
    data = self._make_arch22_133120_case(q_seqlen=16)
    self._run_arch22_phase1_dump_once(data, "arch22_q16_hot_csr_phase1")

  # ---------------------------------------------------------------------------
  # Production-config precision sweep. FIXED dims (do NOT vary):
  #   head_dim=128, block_size=128, top_k=16,
  #   q_heads=64, kv_heads=4, group_size=16 (A2 batchGroupsMax=4;
  #   the kernel derives this from its L0/UB capacity formula).
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
                       seed, prec=1e-2, tag="", inner_precise=4):
    if tag:
      print(f"[prod:{tag}]")
    self._run_npu_test(
        batch, q_seqlens, kv_seqlens, self._Q_HEADS, self._KV_HEADS,
        head_dim=self._HEAD_DIM, block_size=self._BLOCK_SIZE,
        top_k=self._TOP_K, seed=seed, prec=prec,
        inner_precise=inner_precise)

  def test_prod_inner_precise_matrix(self):
    """All precision modes have a full-output golden on production GQA."""
    for inner_precise, prec in ((0, 1e-2), (1, 1.5e-2), (4, 1e-2)):
      with self.subTest(inner_precise=inner_precise):
        self._run_prod_config(
            batch=1, q_seqlens=[64], kv_seqlens=[4096],
            seed=9700 + inner_precise, prec=prec,
            tag=f"inner_precise/{inner_precise}",
            inner_precise=inner_precise)

  def test_prod_inner_precise_hot_csr_sample(self):
    """Q=64 exercises repeated M=64 batches with sampled CPU references."""
    sample_tokens = [0, 32, 63]
    for inner_precise in (0, 1, 4):
      with self.subTest(inner_precise=inner_precise):
        data = make_case(
            batch=1, q_seqlens=[64], kv_seqlens=[4096],
            q_heads=self._Q_HEADS, kv_heads=self._KV_HEADS,
            head_dim=self._HEAD_DIM, block_size=self._BLOCK_SIZE,
            top_k=self._TOP_K, seed=9750 + inner_precise)
        data["inner_precise"] = inner_precise
        self._run_arch22_precision_once(
            data, sample_tokens, f"prod_hot_csr_inner{inner_precise}")

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
