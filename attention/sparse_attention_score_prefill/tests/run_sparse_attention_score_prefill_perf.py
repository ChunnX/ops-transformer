# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""Standalone A2 performance runner for sparse_attention_score_prefill.

This script only constructs inputs and invokes the prefill operator. It does
not call the decode operator or run a CPU golden. Use msprof Task Duration as
the authoritative kernel latency; the NPU Event statistics printed here are a
quick local check.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch_npu


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "torch_extension"))
os.environ.setdefault("CANN_OPS_TRANSFORMER_LAZY_IMPORT", "1")

import cann_ops_transformer.ops.sparse_attention_score_prefill  # noqa: E402,F401


def parse_args():
  parser = argparse.ArgumentParser(
      description="Profile npu_sparse_attention_score_prefill without decode/golden")
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--q-seqlen", type=int, default=2048)
  parser.add_argument("--kv-seqlen", type=int, default=131072)
  parser.add_argument("--q-heads", type=int, default=16)
  parser.add_argument("--kv-heads", type=int, default=1)
  parser.add_argument("--head-dim", type=int, default=128)
  parser.add_argument("--block-size", type=int, default=128)
  parser.add_argument("--top-k", type=int, default=16)
  parser.add_argument("--inner-precise", type=int, default=4)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--repeat", type=int, default=20)
  parser.add_argument("--seed", type=int, default=2026)
  parser.add_argument("--device", default=os.getenv("NPU_DEVICE", "npu:0"))
  return parser.parse_args()


def validate_args(args):
  positive_names = (
      "batch", "q_seqlen", "kv_seqlen", "q_heads", "kv_heads",
      "head_dim", "block_size", "top_k", "repeat")
  for name in positive_names:
    if getattr(args, name) <= 0:
      raise ValueError(f"--{name.replace('_', '-')} must be positive")
  if args.warmup < 0:
    raise ValueError("--warmup must be non-negative")
  if args.q_heads % args.kv_heads != 0:
    raise ValueError("--q-heads must be divisible by --kv-heads")
  if args.q_seqlen > args.kv_seqlen:
    raise ValueError("--q-seqlen must not exceed --kv-seqlen")
  if args.kv_seqlen % args.block_size != 0:
    raise ValueError("--kv-seqlen must be an integer multiple of --block-size")
  if args.top_k > args.kv_seqlen // args.block_size:
    raise ValueError("--top-k must not exceed the number of KV blocks")


def build_q2k(args):
  """Build a causal, contiguous top-k block selection for every query token."""
  total_q = args.batch * args.q_seqlen
  select_idx = torch.full(
      (args.kv_heads, total_q, args.top_k), -1, dtype=torch.int32)
  history_len = args.kv_seqlen - args.q_seqlen
  slots = torch.arange(args.top_k, dtype=torch.int64)

  for batch_idx in range(args.batch):
    q_start = batch_idx * args.q_seqlen
    q_end = q_start + args.q_seqlen
    q_tokens = torch.arange(args.q_seqlen, dtype=torch.int64)
    current_blocks = (history_len + q_tokens) // args.block_size
    selected = current_blocks.unsqueeze(1) - args.top_k + 1 + slots.unsqueeze(0)
    selected.masked_fill_(selected < 0, -1)
    select_idx[:, q_start:q_end, :] = selected.to(torch.int32).unsqueeze(0)
  return select_idx


def build_k2q_csr(select_idx, args):
  """Convert q2k indices to packed-row k2q CSR without golden dependencies."""
  total_q = args.batch * args.q_seqlen
  blocks_per_batch = args.kv_seqlen // args.block_size
  total_rows = args.batch * blocks_per_batch
  batch_ids = torch.arange(args.batch, dtype=torch.int64).repeat_interleave(
      args.q_seqlen)
  row_offsets = (batch_ids * blocks_per_batch).view(1, total_q, 1)
  valid = select_idx >= 0
  rows = select_idx.to(torch.int64) + row_offsets
  q_ids = torch.arange(total_q, dtype=torch.int32).view(1, total_q, 1).expand(
      args.kv_heads, total_q, args.top_k)
  slot_ids = torch.arange(args.top_k, dtype=torch.int32).view(1, 1, args.top_k).expand(
      args.kv_heads, total_q, args.top_k)

  valid_entries = int(valid[0].sum().item())
  rows_flat = rows.masked_select(valid).view(args.kv_heads, valid_entries)
  q_flat = q_ids.masked_select(valid).view(args.kv_heads, valid_entries)
  slot_flat = slot_ids.masked_select(valid).view(args.kv_heads, valid_entries)
  sort_keys = rows_flat * max(total_q, 1) + q_flat.to(torch.int64)
  sort_order = torch.argsort(sort_keys, dim=1, stable=True)

  sorted_rows = rows_flat.gather(1, sort_order)
  q_indices = q_flat.gather(1, sort_order).contiguous()
  slot_indices = slot_flat.gather(1, sort_order).contiguous()
  counts = torch.zeros((args.kv_heads, total_rows), dtype=torch.int32)
  counts.scatter_add_(1, sorted_rows, torch.ones_like(sorted_rows, dtype=torch.int32))
  row_ptr = torch.zeros((args.kv_heads, total_rows + 1), dtype=torch.int32)
  row_ptr[:, 1:] = counts.cumsum(dim=1, dtype=torch.int32)
  return row_ptr, q_indices, slot_indices


def make_inputs(args):
  torch.manual_seed(args.seed)
  total_q = args.batch * args.q_seqlen
  blocks_per_batch = args.kv_seqlen // args.block_size
  physical_blocks = args.batch * blocks_per_batch

  query = (torch.rand(total_q, args.q_heads, args.head_dim) * 2 - 1).to(
      torch.bfloat16)
  key = (torch.rand(
      physical_blocks, args.block_size, args.kv_heads, args.head_dim) * 2 - 1).to(
          torch.bfloat16)
  value = (torch.rand(
      physical_blocks, args.block_size, args.kv_heads, args.head_dim) * 2 - 1).to(
          torch.bfloat16)
  block_table = torch.arange(physical_blocks, dtype=torch.int32).reshape(
      args.batch, blocks_per_batch)
  q2k = build_q2k(args)
  row_ptr, q_indices, slot_indices = build_k2q_csr(q2k, args)
  actual_q = torch.full((args.batch,), args.q_seqlen, dtype=torch.int32)
  actual_kv = torch.full((args.batch,), args.kv_seqlen, dtype=torch.int32)

  return tuple(t.to(args.device) for t in (
      query, key, value, block_table, row_ptr, q_indices, slot_indices,
      actual_q, actual_kv))


def percentile(sorted_values, fraction):
  index = round((len(sorted_values) - 1) * fraction)
  return sorted_values[index]


def main():
  args = parse_args()
  validate_args(args)
  torch_npu.npu.set_device(args.device)
  inputs = make_inputs(args)

  def run_prefill():
    return torch.ops.cann_ops_transformer.npu_sparse_attention_score_prefill(
        *inputs,
        args.kv_heads,
        1.0 / (args.head_dim ** 0.5),
        args.block_size,
        args.top_k,
        args.inner_precise)

  with torch.inference_mode():
    output = None
    for _ in range(args.warmup):
      output = run_prefill()
    torch_npu.npu.synchronize()

    timings_ms = []
    for _ in range(args.repeat):
      start = torch_npu.npu.Event(enable_timing=True)
      end = torch_npu.npu.Event(enable_timing=True)
      start.record()
      output = run_prefill()
      end.record()
      end.synchronize()
      timings_ms.append(start.elapsed_time(end))

  timings_ms.sort()
  mean_ms = sum(timings_ms) / len(timings_ms)
  print(
      "PREFILL_SHAPE "
      f"B={args.batch} Q={args.q_seqlen} H={args.q_heads} "
      f"KV={args.kv_seqlen} KVH={args.kv_heads} D={args.head_dim} "
      f"block={args.block_size} topK={args.top_k} innerPrecise={args.inner_precise}")
  print(
      "PREFILL_EVENT_MS "
      f"warmup={args.warmup} repeat={args.repeat} min={timings_ms[0]:.3f} "
      f"mean={mean_ms:.3f} p50={percentile(timings_ms, 0.50):.3f} "
      f"p90={percentile(timings_ms, 0.90):.3f} max={timings_ms[-1]:.3f}")
  print(f"PREFILL_OUTPUT shape={tuple(output.shape)} dtype={output.dtype}")


if __name__ == "__main__":
  main()
