# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""
Golden reference tests for sparse_attention_score_prefill (KV-Gather-Q).

Based on sparse_attention_score decode bf16 golden helpers, extended with:
1. Host dense packed-row CSR (k2q_row_ptr / k2q_q_indices / k2q_slot_indices)
2. Phase1 KV-block outer loop: QK + SM + PV only; workspace stores O_tmp(fp32, row-normalized) + LSE
3. Phase2 LSE logsumexp combine across topK partials (design doc §7.2, same as kernel)
"""

import math
import unittest
from math import ceil

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared helpers (aligned with test_sparse_attention_score_golden.py)
# ---------------------------------------------------------------------------

def generate_block_index_with_causal(query_fp32, key_fp32, q_seqlen, kv_seqlen,
                                     kv_heads, group_size, block_size=128, top_k=16):
  """Generate per-kv-head block indices with causal pooling (decode style)."""
  his_seq_len = kv_seqlen - q_seqlen
  total_blocks = ceil(kv_seqlen / block_size)
  head_dim = query_fp32.shape[-1]

  select_idx = torch.full((kv_heads, q_seqlen, top_k), -1, dtype=torch.int32)
  select_num_idx = torch.zeros((kv_heads, q_seqlen), dtype=torch.int32)

  for kv_head in range(kv_heads):
    representative_q_head = kv_head * group_size
    k_head = key_fp32[:, kv_head, :]

    for q_token in range(q_seqlen):
      q_vec = query_fp32[q_token, representative_q_head, :]
      causal_bound = his_seq_len + q_token

      scores = torch.matmul(q_vec, k_head[:kv_seqlen, :].transpose(0, 1))

      pooled = torch.full((total_blocks,), -float("inf"), dtype=torch.float32)
      q_block = causal_bound // block_size

      for block_idx in range(total_blocks):
        block_begin = block_idx * block_size
        block_end = min(block_begin + block_size, kv_seqlen)

        if block_idx > q_block:
          pooled[block_idx] = -float("inf")
        elif block_idx == q_block:
          pooled[block_idx] = float("inf")
        else:
          effective_end = min(block_end, causal_bound + 1)
          if effective_end > block_begin:
            pooled[block_idx] = torch.max(scores[block_begin:effective_end]).item()

      visible_blocks = min(total_blocks, q_block + 1)
      valid_k = min(top_k, visible_blocks)
      select_num_idx[kv_head, q_token] = valid_k
      if valid_k > 0:
        topk_indices = torch.topk(pooled, k=valid_k, largest=True).indices.to(torch.int32)
        select_idx[kv_head, q_token, :valid_k] = topk_indices

  return select_idx, select_num_idx


def generate_block_table(batch, max_blocks_per_batch, shuffle=True):
  """Logical -> physical block mapping (decode style)."""
  total_physical = batch * max_blocks_per_batch
  all_physical_ids = list(range(total_physical))
  if shuffle:
    import random
    rng = random.Random(137)
    rng.shuffle(all_physical_ids)
  block_table = torch.zeros(batch, max_blocks_per_batch, dtype=torch.int32)
  for b in range(batch):
    for i in range(max_blocks_per_batch):
      block_table[b, i] = all_physical_ids[b * max_blocks_per_batch + i]
  return block_table


def cpu_sparse_attention_score_bf16(
    query, key, value, select_idx, block_table,
    actual_seq_lengths, actual_seq_lengths_kv,
    num_key_value_heads, select_num_idx=None,
    block_size=128, scale_value=1.0,
):
  """Decode Q-centric BF16 golden (online softmax over topK KV blocks)."""
  select_idx_cpu = select_idx.to(torch.int64)
  block_table_cpu = block_table.to(torch.int64)

  total_q_tokens, q_heads, head_dim = query.shape
  kv_heads = num_key_value_heads
  group_size = q_heads // kv_heads
  top_k = select_idx.shape[2]
  batch = len(actual_seq_lengths)
  scale_value = float(scale_value)

  output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.bfloat16)
  q_offset = 0
  for batch_idx, q_seqlen in enumerate(actual_seq_lengths):
    kv_seqlen = int(actual_seq_lengths_kv[batch_idx])
    history_len = kv_seqlen - q_seqlen

    for q_token_in_batch in range(q_seqlen):
      global_q_token = q_offset + q_token_in_batch
      causal_bound = history_len + q_token_in_batch

      for kv_head in range(kv_heads):
        valid_top_k = top_k
        if select_num_idx is not None:
          valid_top_k = int(select_num_idx[kv_head, global_q_token].item())
          valid_top_k = min(valid_top_k, top_k)
        if valid_top_k == 0:
          continue

        q_start = kv_head * group_size
        q_group_bf16 = query[global_q_token, q_start:q_start + group_size, :]

        last_max_fp32 = torch.full((group_size,), -float("inf"), dtype=torch.float32)
        last_sum_fp32 = torch.zeros(group_size, dtype=torch.float32)
        o_acc_fp32 = torch.zeros(group_size, head_dim, dtype=torch.float32)
        is_first = True

        for topk_idx in range(valid_top_k):
          logical_id = int(select_idx_cpu[kv_head, global_q_token, topk_idx].item())
          if logical_id < 0:
            continue
          block_begin = logical_id * block_size
          block_end = min(block_begin + block_size, kv_seqlen)
          effective_end = min(block_end, causal_bound + 1)
          if effective_end <= block_begin:
            continue

          physical_id = int(block_table_cpu[batch_idx, logical_id].item())
          valid_len = effective_end - block_begin
          k_bf16 = key[physical_id, :valid_len, kv_head, :]
          v_bf16 = value[physical_id, :valid_len, kv_head, :]

          s_fp32 = torch.matmul(q_group_bf16.float(), k_bf16.float().t()) * scale_value
          now_max_fp32 = s_fp32.max(dim=1).values
          if not is_first:
            now_max_fp32 = torch.max(now_max_fp32, last_max_fp32)

          p_fp32 = torch.exp(s_fp32 - now_max_fp32.unsqueeze(1))
          now_sum_fp32 = p_fp32.sum(dim=1)
          p_bf16 = p_fp32.to(torch.bfloat16)

          if is_first:
            last_sum_fp32 = now_sum_fp32
            last_max_fp32 = now_max_fp32
          else:
            correction_fp32 = torch.exp(last_max_fp32 - now_max_fp32)
            last_sum_fp32 = correction_fp32 * last_sum_fp32 + now_sum_fp32
            last_max_fp32 = now_max_fp32

          pv_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())
          if is_first:
            o_acc_fp32 = pv_fp32
          else:
            o_acc_fp32 = o_acc_fp32 * correction_fp32.unsqueeze(1) + pv_fp32

          is_first = False

        if last_sum_fp32.max() > 0:
          result_fp32 = o_acc_fp32 / last_sum_fp32.unsqueeze(1)
          output[global_q_token, q_start:q_start + group_size, :] = result_fp32.to(torch.bfloat16)

    q_offset += q_seqlen

  return output


# ---------------------------------------------------------------------------
# BF16 block helpers (shared by phase1 and q-centric reference)
# ---------------------------------------------------------------------------

def _softmax_p_from_scores_fp32(s_fp32):
  """fp32 softmax; cast P to bf16 only for PV matmul dtype."""
  row_max_fp32 = s_fp32.max(dim=1).values
  print("row_max_fp32:", row_max_fp32)
  p_fp32 = torch.exp(s_fp32 - row_max_fp32.unsqueeze(1))
  row_sum_fp32 = p_fp32.sum(dim=1)
  p_bf16 = p_fp32.to(torch.bfloat16)
  return p_bf16, row_max_fp32, row_sum_fp32


def _bf16_block_qk_sm_pv(q_group_bf16, k_bf16, v_bf16, scale_value):
  """Single KV block: QK -> SM -> PV. Returns row-normalized O_tmp and LSE."""
  s_fp32 = torch.matmul(q_group_bf16.float(), k_bf16.float().t())
  #print("s_fp32:", s_fp32[0][:16])
  s_fp32 = s_fp32* scale_value
  p_bf16, row_max_fp32, row_sum_fp32 = _softmax_p_from_scores_fp32(s_fp32)
  print("p_bf16:", p_bf16[0][:16])
  #print("row_sum_fp32:", row_sum_fp32[:16])

  o_unnorm_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())
  o_norm_fp32 = o_unnorm_fp32 / row_sum_fp32.unsqueeze(1).clamp_min(1e-20)
  lse_fp32 = row_max_fp32 + torch.log(row_sum_fp32.clamp_min(1e-20))
  lse_fp32 = torch.where(row_sum_fp32 > 0, lse_fp32, torch.full_like(lse_fp32, float('-inf')))
  return o_norm_fp32, lse_fp32


def _combine_partials_lse_ws(o_norm_slots, lse_slots):
  """Combine topK workspace partials (O_tmp + LSE) via LSE logsumexp weighting.

  Matches kernel block_epilogue_rescale_o_prefill_arch35 and design doc §7.2.
  O_tmp is row-normalized partial output; LSE = rowMax + log(rowSum).
  """
  valid = torch.isfinite(lse_slots)
  if not torch.any(valid):
    return None

  lse_v = lse_slots[valid]
  o_norm_v = o_norm_slots[valid]

  max_lse = lse_v.max()
  weights = torch.exp(lse_v - max_lse)
  weight_sum = weights.sum()
  if weight_sum <= 0:
    return None

  lse_global = max_lse + torch.log(weight_sum)
  weights = torch.exp(lse_v - lse_global)
  return (weights.unsqueeze(-1) * o_norm_v).sum(dim=0)


# ---------------------------------------------------------------------------
# Prefill-specific: k2q CSR + KV outer loop golden
# ---------------------------------------------------------------------------

def _batch_offsets(q_seqlens, kv_seqlens, block_size):
  batch_size = len(q_seqlens)
  batch_q_offset = [0] * (batch_size + 1)
  batch_kv_block_offset = [0] * (batch_size + 1)
  for b in range(batch_size):
    batch_q_offset[b + 1] = batch_q_offset[b] + q_seqlens[b]
    batch_kv_block_offset[b + 1] = (
        batch_kv_block_offset[b] + ceil(kv_seqlens[b] / block_size))
  return batch_q_offset, batch_kv_block_offset


def _find_batch_for_token(token_idx, batch_offsets):
  for b in range(len(batch_offsets) - 1):
    if token_idx < batch_offsets[b + 1]:
      return b
  return len(batch_offsets) - 2


def _normalize_select_idx_3d(select_idx, select_num_idx=None, kv_heads=None):
  """Return (select_idx_3d, select_num_3d, kv_heads, total_q, top_k) as numpy arrays."""
  if isinstance(select_idx, torch.Tensor):
    select_idx_np = select_idx.detach().cpu().numpy()
  else:
    select_idx_np = np.asarray(select_idx)

  if select_num_idx is not None:
    if isinstance(select_num_idx, torch.Tensor):
      select_num_np = select_num_idx.detach().cpu().numpy()
    else:
      select_num_np = np.asarray(select_num_idx)
  else:
    select_num_np = None

  if select_idx_np.ndim == 2:
    if kv_heads is None:
      kv_heads = 1
    total_q_tokens, top_k = select_idx_np.shape
    select_idx_3d = np.broadcast_to(
        select_idx_np[np.newaxis, :, :], (kv_heads, total_q_tokens, top_k)).copy()
    select_num_3d = select_num_np if select_num_np is not None and select_num_np.ndim == 2 else None
  elif select_idx_np.ndim == 3:
    kv_heads, total_q_tokens, top_k = select_idx_np.shape
    select_idx_3d = select_idx_np
    select_num_3d = select_num_np
  else:
    raise ValueError(f"select_idx must be 2D or 3D, got shape {select_idx_np.shape}")

  return select_idx_3d, select_num_3d, kv_heads, total_q_tokens, top_k


def _rows_per_batch_from_actual_seq_lengths(actual_seq_lengths_kv, kv_block_size):
  seqlens_k = np.asarray(actual_seq_lengths_kv, dtype=np.int64)
  return (seqlens_k + kv_block_size - 1) // kv_block_size


def _q_batch_offsets_from_actual(actual_seq_lengths):
  actual = np.asarray(actual_seq_lengths, dtype=np.int32)
  offsets = np.zeros(int(actual.shape[0]) + 1, dtype=np.int32)
  for i, slen in enumerate(actual.tolist()):
    offsets[i + 1] = offsets[i] + int(slen)
  return offsets


def _build_packed_row_map(rows_per_batch):
  """Map (batch, local_kv_block) -> packed dense row id (CUDA-compatible)."""
  rows_per_batch = np.asarray(rows_per_batch, dtype=np.int64)
  batch = int(rows_per_batch.shape[0])
  max_rows = int(rows_per_batch.max()) if batch > 0 else 0
  row_map = np.full((batch, max_rows), -1, dtype=np.int32)
  row_linear = 0
  for kv_block_idx in range(max_rows):
    for batch_idx, row_count in enumerate(rows_per_batch.tolist()):
      if kv_block_idx < row_count:
        row_map[batch_idx, kv_block_idx] = row_linear
        row_linear += 1
  return row_map, row_linear


def _init_packed_row_coord(rows_per_batch):
  """MSA packing coord for packed_row == 0."""
  batch_size = len(rows_per_batch)
  kv_block_idx = 0
  batch_idx = 0
  while batch_idx < batch_size and kv_block_idx >= rows_per_batch[batch_idx]:
    batch_idx += 1
  return batch_idx, kv_block_idx


def _advance_packed_row_coord(batch_idx, kv_block_idx, batch_size, rows_per_batch):
  """Step to the next (batch_idx, kv_block_idx) in MSA packing order."""
  batch_idx += 1
  while batch_idx < batch_size and kv_block_idx >= rows_per_batch[batch_idx]:
    batch_idx += 1
  if batch_idx >= batch_size:
    kv_block_idx += 1
    batch_idx = 0
    while batch_idx < batch_size and kv_block_idx >= rows_per_batch[batch_idx]:
      batch_idx += 1
  return batch_idx, kv_block_idx


def decode_packed_row(packed_row, actual_seq_lengths_kv, block_size):
  """Decode CUDA packed CSR row id -> (batch_idx, local_kv_block_idx)."""
  kv_seqlens = _as_seq_list(actual_seq_lengths_kv)
  rows_per_batch = [ceil(int(kv) / block_size) for kv in kv_seqlens]
  max_kv_block_rows = max(rows_per_batch) if rows_per_batch else 0
  row_linear = 0
  for kv_block_idx in range(max_kv_block_rows):
    for batch_idx, rows_b in enumerate(rows_per_batch):
      if kv_block_idx >= rows_b:
        continue
      if row_linear == packed_row:
        return batch_idx, kv_block_idx
      row_linear += 1
  raise ValueError(f"packed_row {packed_row} out of range")


def _as_seq_list(seq):
  if isinstance(seq, torch.Tensor):
    return [int(x) for x in seq.detach().cpu().tolist()]
  return [int(x) for x in seq]


def _kv_block_valid_size(local_block_idx, kv_seqlen, block_size):
  tail_remain = int(kv_seqlen) - local_block_idx * block_size
  if tail_remain <= 0:
    return 0
  return min(block_size, tail_remain)


def global_select_idx_to_batch_local(select_idx_global, q_seqlens, kv_seqlens, block_size,
                                     select_num_idx=None, kv_heads=None):
  """
  Convert global logical KV block ids to batch-local ids (CUDA q2k layout).

  q2k_indices[head_kv, total_q, topK] uses per-token batch-local block indices.
  """
  q_seqlens = _as_seq_list(q_seqlens)
  kv_seqlens = _as_seq_list(kv_seqlens)
  select_idx_3d, select_num_3d, kv_heads, total_q, top_k = _normalize_select_idx_3d(
      select_idx_global, select_num_idx, kv_heads)
  batch_q_offset, batch_kv_block_offset = _batch_offsets(q_seqlens, kv_seqlens, block_size)

  q2k = np.full((kv_heads, total_q, top_k), -1, dtype=np.int32)
  for h in range(kv_heads):
    for q in range(total_q):
      batch_idx = _find_batch_for_token(q, batch_q_offset)
      valid_k = top_k
      if select_num_3d is not None:
        valid_k = min(int(select_num_3d[h, q]), top_k)
      for k in range(valid_k):
        global_block = int(select_idx_3d[h, q, k])
        if global_block < 0:
          continue
        if global_block < batch_kv_block_offset[batch_idx] or \
            global_block >= batch_kv_block_offset[batch_idx + 1]:
          continue
        q2k[h, q, k] = global_block - batch_kv_block_offset[batch_idx]
  return q2k, batch_q_offset, batch_kv_block_offset


def build_k2q_csr_torch_reference(
    q2k_indices,
    actual_seq_lengths,
    actual_seq_lengths_kv,
    kv_block_size,
    *,
    q_global_offset=True,
):
  """
  Torch reference for q2k -> k2q CSR (ported from fmha_sm100 sparse_index_utils.py).

  Dense packed rows: one CSR row per (head_kv, packed_kv_block) across the batch.

  Args:
    q2k_indices: [head_kv, total_q, topK] int32, batch-local KV block ids, -1 padded.
    actual_seq_lengths / actual_seq_lengths_kv: [batch] int32, per-batch Q/KV seqlen.
    kv_block_size: tokens per KV block.

  Returns:
    k2q_row_ptr: [head_kv, total_rows + 1] int32
    k2q_q_indices: [head_kv, total_q * topK] int32 (global q if q_global_offset)
    k2q_slot_indices: [head_kv, total_q * topK] int32
  """
  if kv_block_size <= 0:
    raise ValueError(f"kv_block_size must be > 0, got {kv_block_size}")
  if not isinstance(q2k_indices, torch.Tensor):
    q2k_indices = torch.from_numpy(np.asarray(q2k_indices))
  if q2k_indices.dtype != torch.int32:
    q2k_indices = q2k_indices.to(torch.int32)
  if q2k_indices.ndim != 3:
    raise ValueError(
        f"q2k_indices must be [head_kv, total_q, topK], got {tuple(q2k_indices.shape)}")

  actual_seq_lengths = torch.as_tensor(
      actual_seq_lengths, dtype=torch.int32, device=q2k_indices.device)
  actual_seq_lengths_kv = torch.as_tensor(
      actual_seq_lengths_kv, dtype=torch.int32, device=q2k_indices.device)
  if actual_seq_lengths.shape != actual_seq_lengths_kv.shape:
    raise ValueError("actual_seq_lengths and actual_seq_lengths_kv must have the same shape [B]")
  if actual_seq_lengths.ndim != 1:
    raise ValueError(
        f"actual_seq_lengths must be rank-1 [B], got shape {tuple(actual_seq_lengths.shape)}")

  head_kv, total_q, topk = q2k_indices.shape
  total_q_expected = int(actual_seq_lengths.sum().item())
  if total_q != total_q_expected:
    raise ValueError(
        f"q2k_indices.shape[1] ({total_q}) must equal sum(actual_seq_lengths) "
        f"({total_q_expected})")

  rows_per_batch = _rows_per_batch_from_actual_seq_lengths(
      actual_seq_lengths_kv.cpu().numpy(), kv_block_size)
  row_map, total_rows = _build_packed_row_map(rows_per_batch)
  row_map_t = torch.from_numpy(row_map).to(device=q2k_indices.device)
  nnz_upper_bound = total_q * topk

  k2q_row_ptr = torch.zeros((head_kv, total_rows + 1), dtype=torch.int32, device=q2k_indices.device)
  k2q_q_indices = torch.full(
      (head_kv, nnz_upper_bound), -1, dtype=torch.int32, device=q2k_indices.device)
  k2q_slot_indices = torch.full(
      (head_kv, nnz_upper_bound), -1, dtype=torch.int32, device=q2k_indices.device)
  if total_rows == 0 or total_q == 0 or topk == 0:
    return k2q_row_ptr, k2q_q_indices, k2q_slot_indices

  counts = torch.zeros((head_kv, total_rows), dtype=torch.int32, device=q2k_indices.device)
  total_entries = total_q * topk
  row_all = torch.empty((head_kv, total_entries), dtype=torch.int64, device=q2k_indices.device)
  q_all = torch.empty((head_kv, total_entries), dtype=torch.int32, device=q2k_indices.device)
  slot_all = torch.empty((head_kv, total_entries), dtype=torch.int32, device=q2k_indices.device)
  valid_all = torch.empty((head_kv, total_entries), dtype=torch.bool, device=q2k_indices.device)

  rows_per_batch_list = rows_per_batch.tolist()
  q_offsets = _q_batch_offsets_from_actual(actual_seq_lengths.cpu().numpy())
  entry_cursor = 0

  for batch_idx, kv_rows in enumerate(rows_per_batch_list):
    q_start = int(q_offsets[batch_idx])
    q_end = int(q_offsets[batch_idx + 1])
    q_len = q_end - q_start
    if q_len == 0:
      continue
    num_entries = q_len * topk
    q2k_batch = q2k_indices[:, q_start:q_end, :]
    valid_batch = q2k_batch >= 0
    if valid_batch.any():
      max_valid_kv = int(q2k_batch[valid_batch].max().item())
      if max_valid_kv >= kv_rows:
        raise ValueError(
            f"q2k_indices references kv_block {max_valid_kv} for batch {batch_idx}, "
            f"but that batch only has {kv_rows} logical kv blocks")

    kv_flat = q2k_batch.reshape(head_kv, num_entries).long()
    valid_flat = valid_batch.reshape(head_kv, num_entries)
    safe_kv_flat = torch.where(valid_flat, kv_flat, torch.zeros_like(kv_flat))
    row_flat = row_map_t[batch_idx][safe_kv_flat]
    q_flat = (
        torch.arange(q_len, device=q2k_indices.device, dtype=torch.int32)
        .view(1, q_len, 1)
        .expand(head_kv, q_len, topk)
        .reshape(head_kv, num_entries)
    )
    if q_global_offset:
      q_flat = q_flat + q_start
    slot_flat = (
        torch.arange(topk, device=q2k_indices.device, dtype=torch.int32)
        .view(1, 1, topk)
        .expand(head_kv, q_len, topk)
        .reshape(head_kv, num_entries)
    )
    row_all[:, entry_cursor:entry_cursor + num_entries] = row_flat
    q_all[:, entry_cursor:entry_cursor + num_entries] = q_flat
    slot_all[:, entry_cursor:entry_cursor + num_entries] = slot_flat
    valid_all[:, entry_cursor:entry_cursor + num_entries] = valid_flat
    counts.scatter_add_(1, row_flat.to(torch.int64), valid_flat.to(torch.int32))
    entry_cursor += num_entries

  k2q_row_ptr[:, 1:] = counts.cumsum(dim=1, dtype=torch.int32)

  sort_stride = max(total_q, 1)
  invalid_key = total_rows * sort_stride
  sort_keys = torch.full_like(row_all, invalid_key, dtype=torch.int64)
  sort_keys[valid_all] = row_all[valid_all] * sort_stride + q_all[valid_all].to(torch.int64)
  _, sort_idx = sort_keys.sort(dim=1, stable=True)
  sorted_q = q_all.gather(1, sort_idx)
  sorted_slot = slot_all.gather(1, sort_idx)

  valid_counts = valid_all.sum(dim=1)
  write_mask = (
      torch.arange(total_entries, device=q2k_indices.device)
      .unsqueeze(0)
      .expand(head_kv, -1)
      < valid_counts.unsqueeze(1)
  )
  k2q_q_indices[write_mask] = sorted_q[write_mask]
  k2q_slot_indices[write_mask] = sorted_slot[write_mask]
  return k2q_row_ptr, k2q_q_indices, k2q_slot_indices


def build_k2q_csr(select_idx, select_num_idx, actual_seq_lengths, actual_seq_lengths_kv,
                  block_size, kv_heads=None):
  """
  Build dense packed-row CSR reverse index (CUDA / MSA compatible).

  Row id is packed (batch, local_kv_block); empty rows have row_ptr[i]==row_ptr[i+1].

  Args:
    actual_seq_lengths / actual_seq_lengths_kv: [batch] per-batch Q/KV seqlen (non-cumsum).

  Returns:
    k2q_row_ptr: [kv_heads, total_rows + 1] int32
    k2q_q_indices: [kv_heads, total_q * top_k] int32, global q token ids
    k2q_slot_indices: [kv_heads, total_q * top_k] int32
  """
  q2k, _, _ = global_select_idx_to_batch_local(
      select_idx, actual_seq_lengths, actual_seq_lengths_kv, block_size,
      select_num_idx=select_num_idx, kv_heads=kv_heads)

  row_ptr, q_idx, slot_idx = build_k2q_csr_torch_reference(
      torch.from_numpy(q2k), actual_seq_lengths, actual_seq_lengths_kv,
      block_size, q_global_offset=True)
  return (
      row_ptr.cpu().numpy(),
      q_idx.cpu().numpy(),
      slot_idx.cpu().numpy(),
  )


def decode_select_idx_to_prefill_global(select_idx_decode, q_seqlens, kv_seqlens, block_size):
  """
  Convert decode select_idx [kv_heads, total_q, top_k] (local logical ids)
  to prefill global select_idx [kv_heads, total_q, top_k] (global kv block ids).
  """
  kv_heads, total_q, top_k = select_idx_decode.shape
  batch_q_offset, batch_kv_block_offset = _batch_offsets(q_seqlens, kv_seqlens, block_size)
  batch_size = len(q_seqlens)

  select_idx_global = torch.full((kv_heads, total_q, top_k), -1, dtype=torch.int32)
  for h in range(kv_heads):
    for b in range(batch_size):
      for qi in range(q_seqlens[b]):
        q_global = batch_q_offset[b] + qi
        for k in range(top_k):
          logical_id = int(select_idx_decode[h, q_global, k].item())
          if logical_id >= 0:
            select_idx_global[h, q_global, k] = batch_kv_block_offset[b] + logical_id
  return select_idx_global


def cpu_golden_prefill_phase1(
    query, key, value, k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
    block_table, actual_seq_lengths, actual_seq_lengths_kv,
    block_size, top_k, scale_value,
):
  """
  Phase1 golden: QK + SM + PV only; write workspace partials (O_tmp row-normalized, LSE).

  Mirrors kernel Phase1 workspace layout; LSE combine is deferred to Phase2.
  """
  total_q_tokens, num_q_heads, d = query.shape
  kv_heads = key.shape[2]
  group_size = num_q_heads // kv_heads
  scale_value = float(scale_value)

  q_seqlens = _as_seq_list(actual_seq_lengths)
  kv_seqlens = _as_seq_list(actual_seq_lengths_kv)
  batch_q_offset, _ = _batch_offsets(q_seqlens, kv_seqlens, block_size)

  if isinstance(k2q_row_ptr, torch.Tensor):
    k2q_row_ptr_np = k2q_row_ptr.detach().cpu().numpy()
    k2q_q_indices_np = k2q_q_indices.detach().cpu().numpy()
    k2q_slot_indices_np = k2q_slot_indices.detach().cpu().numpy()
  else:
    k2q_row_ptr_np = np.asarray(k2q_row_ptr)
    k2q_q_indices_np = np.asarray(k2q_q_indices)
    k2q_slot_indices_np = np.asarray(k2q_slot_indices)

  ws_o_tmp = torch.zeros(total_q_tokens, kv_heads, top_k, group_size, d, dtype=torch.float32)
  ws_lse = torch.full(
      (total_q_tokens, kv_heads, top_k, group_size), float('-inf'), dtype=torch.float32)
  # Workspace layout per partial (bytes): O_tmp[groupSize,D] fp32 + LSE[groupSize]

  total_rows = int(k2q_row_ptr_np.shape[1] - 1)
  total_task_num_p1 = total_rows * kv_heads
  rows_per_batch = [ceil(int(kv) / block_size) for kv in kv_seqlens]
  batch_size = len(kv_seqlens)

  cur_packed_row = -1
  batch_idx = 0
  kv_block_idx = 0

  for task_idx in range(total_task_num_p1):
    packed_row = task_idx // kv_heads
    kv_head_idx = task_idx % kv_heads
    if packed_row >= total_rows:
      continue

    if packed_row != cur_packed_row:
      if cur_packed_row < 0:
        batch_idx, kv_block_idx = _init_packed_row_coord(rows_per_batch)
        cur_packed_row = 0
      while cur_packed_row < packed_row:
        batch_idx, kv_block_idx = _advance_packed_row_coord(
            batch_idx, kv_block_idx, batch_size, rows_per_batch)
        cur_packed_row += 1

    local_block_idx = kv_block_idx

    physical_block_id = int(block_table[batch_idx, local_block_idx].item())

    valid_size = _kv_block_valid_size(local_block_idx, kv_seqlens[batch_idx], block_size)
    if valid_size == 0:
      continue

    csr_start = int(k2q_row_ptr_np[kv_head_idx, packed_row])
    csr_end = int(k2q_row_ptr_np[kv_head_idx, packed_row + 1])
    num_q_tokens = csr_end - csr_start
    if num_q_tokens == 0:
      continue

    kv_start_pos = local_block_idx * block_size
    k_block = key[physical_block_id, :valid_size, kv_head_idx, :]
    v_block = value[physical_block_id, :valid_size, kv_head_idx, :]
    q_head_start = kv_head_idx * group_size

    for qi in range(num_q_tokens):
      edge = csr_start + qi
      q_token = int(k2q_q_indices_np[kv_head_idx, edge])
      slot_k = int(k2q_slot_indices_np[kv_head_idx, edge])

      q_row_bf16 = query[q_token, q_head_start:q_head_start + group_size, :]

      q_batch = _find_batch_for_token(q_token, batch_q_offset)
      local_q_idx = q_token - batch_q_offset[q_batch]
      q_position = kv_seqlens[q_batch] - q_seqlens[q_batch] + local_q_idx
      causal_valid_len = min(valid_size, q_position - kv_start_pos + 1)
      causal_valid_len = max(causal_valid_len, 0)
      if causal_valid_len == 0:
        continue

      o_norm, lse = _bf16_block_qk_sm_pv(
          q_row_bf16, k_block[:causal_valid_len], v_block[:causal_valid_len], scale_value)

      ws_o_tmp[q_token, kv_head_idx, slot_k] = o_norm
      ws_lse[q_token, kv_head_idx, slot_k] = lse

  return ws_o_tmp, ws_lse


def cpu_golden_prefill_phase2(ws_o_tmp, ws_lse, top_k, kv_heads, group_size, d):
  """Phase2: LSE logsumexp combine over topK workspace partials per (qToken, kvHead)."""
  total_q_tokens = ws_o_tmp.shape[0]
  num_q_heads = kv_heads * group_size
  output = torch.zeros(total_q_tokens, num_q_heads, d, dtype=torch.bfloat16)
  total_task_num_p2 = total_q_tokens * kv_heads

  for task_idx in range(total_task_num_p2):
    q_token = task_idx // kv_heads
    kv_head_idx = task_idx % kv_heads
    q_head_start = kv_head_idx * group_size
    for gh in range(group_size):
      slot_o_tmp = ws_o_tmp[q_token, kv_head_idx, :, gh, :]
      slot_lse = ws_lse[q_token, kv_head_idx, :, gh]
      combined = _combine_partials_lse_ws(slot_o_tmp, slot_lse)
      if combined is not None:
        output[q_token, q_head_start + gh, :] = combined.to(torch.bfloat16)

  return output


def cpu_golden_prefill_qcentric_bf16(
    query, key, value, select_idx, block_table,
    q_seqlens, kv_seqlens, block_size, top_k, scale_value,
    select_num_idx=None,
):
  """
  Q-centric BF16 golden using per-kv-head select_idx [kv_heads, total_q, top_k].

  Same online-softmax math as decode bf16 with global logical KV block ids.
  """
  select_idx_cpu = select_idx.to(torch.int64)
  block_table_cpu = block_table.to(torch.int64)

  total_q_tokens, q_heads, head_dim = query.shape
  kv_heads = key.shape[2]
  group_size = q_heads // kv_heads
  scale_value = float(scale_value)

  batch_q_offset, batch_kv_block_offset = _batch_offsets(q_seqlens, kv_seqlens, block_size)

  output = torch.zeros(total_q_tokens, q_heads, head_dim, dtype=torch.bfloat16)

  for q_token in range(total_q_tokens):
    batch_idx = _find_batch_for_token(q_token, batch_q_offset)
    local_q_idx = q_token - batch_q_offset[batch_idx]
    kv_seqlen = int(kv_seqlens[batch_idx])
    q_seqlen = int(q_seqlens[batch_idx])
    causal_bound = kv_seqlen - q_seqlen + local_q_idx

    for kv_head in range(kv_heads):
      valid_top_k = top_k
      if select_num_idx is not None:
        valid_top_k = int(select_num_idx[kv_head, q_token].item())
        valid_top_k = min(valid_top_k, top_k)
      if valid_top_k == 0:
        continue

      q_start = kv_head * group_size
      q_group_bf16 = query[q_token, q_start:q_start + group_size, :]

      last_max_fp32 = torch.full((group_size,), -float("inf"), dtype=torch.float32)
      last_sum_fp32 = torch.zeros(group_size, dtype=torch.float32)
      o_acc_fp32 = torch.zeros(group_size, head_dim, dtype=torch.float32)
      is_first = True

      for topk_idx in range(valid_top_k):
        global_kv_block = int(select_idx_cpu[kv_head, q_token, topk_idx].item())
        if global_kv_block < 0:
          continue

        block_batch = _find_batch_for_token(global_kv_block, batch_kv_block_offset)
        if block_batch != batch_idx:
          continue

        local_block_idx = global_kv_block - batch_kv_block_offset[block_batch]
        block_begin = local_block_idx * block_size
        block_end = min(block_begin + block_size, kv_seqlen)
        effective_end = min(block_end, causal_bound + 1)
        if effective_end <= block_begin:
          continue

        physical_id = int(block_table_cpu[batch_idx, local_block_idx].item())
        valid_len = effective_end - block_begin
        k_bf16 = key[physical_id, :valid_len, kv_head, :]
        v_bf16 = value[physical_id, :valid_len, kv_head, :]

        s_fp32 = torch.matmul(q_group_bf16.float(), k_bf16.float().t()) * scale_value
        now_max_fp32 = s_fp32.max(dim=1).values
        if not is_first:
          now_max_fp32 = torch.max(now_max_fp32, last_max_fp32)

        p_fp32 = torch.exp(s_fp32 - now_max_fp32.unsqueeze(1))
        now_sum_fp32 = p_fp32.sum(dim=1)
        p_bf16 = p_fp32.to(torch.bfloat16)

        if is_first:
          last_sum_fp32 = now_sum_fp32
          last_max_fp32 = now_max_fp32
        else:
          correction_fp32 = torch.exp(last_max_fp32 - now_max_fp32)
          last_sum_fp32 = correction_fp32 * last_sum_fp32 + now_sum_fp32
          last_max_fp32 = now_max_fp32

        pv_fp32 = torch.matmul(p_bf16.float(), v_bf16.float())
        if is_first:
          o_acc_fp32 = pv_fp32
        else:
          o_acc_fp32 = o_acc_fp32 * correction_fp32.unsqueeze(1) + pv_fp32

        is_first = False

      if last_sum_fp32.max() > 0:
        result_fp32 = o_acc_fp32 / last_sum_fp32.unsqueeze(1)
        output[q_token, q_start:q_start + group_size, :] = result_fp32.to(torch.bfloat16)

  return output


def cpu_golden_prefill_bf16(
    query, key, value, k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
    block_table, actual_seq_lengths, actual_seq_lengths_kv,
    block_size, top_k, scale_value,
):
  """Full two-phase prefill golden: Phase1 QK/SM/PV -> workspace; Phase2 LSE combine."""
  kv_heads = key.shape[2]

  ws_o_tmp, ws_lse = cpu_golden_prefill_phase1(
      query, key, value,
      k2q_row_ptr, k2q_q_indices, k2q_slot_indices,
      block_table, actual_seq_lengths, actual_seq_lengths_kv,
      block_size, top_k, scale_value)

  group_size = query.shape[1] // kv_heads
  d = query.shape[2]
  return cpu_golden_prefill_phase2(
      ws_o_tmp, ws_lse, top_k, kv_heads, group_size, d)


def make_case(batch, q_seqlens, kv_seqlens, q_heads, kv_heads, head_dim=128,
              block_size=128, top_k=4, seed=42, shuffle_block_table=True):
  """Generate test tensors using decode-style causal block selection."""
  assert len(q_seqlens) == batch and len(kv_seqlens) == batch
  group_size = q_heads // kv_heads
  max_kv_seqlen = max(kv_seqlens)
  max_blocks_per_batch = ceil(max_kv_seqlen / block_size)
  total_q_tokens = sum(q_seqlens)

  torch.manual_seed(seed)
  query_fp32 = torch.rand(total_q_tokens, q_heads, head_dim, dtype=torch.float32) * 2 - 1
  key_fp32 = torch.rand(
      max_blocks_per_batch * batch, block_size, kv_heads, head_dim,
      dtype=torch.float32) * 2 - 1
  value_fp32 = torch.rand(
      max_blocks_per_batch * batch, block_size, kv_heads, head_dim,
      dtype=torch.float32) * 2 - 1

  block_table = generate_block_table(batch, max_blocks_per_batch, shuffle=shuffle_block_table)

  select_idx_decode = torch.full((kv_heads, total_q_tokens, top_k), -1, dtype=torch.int32)
  select_num_idx = torch.zeros((kv_heads, total_q_tokens), dtype=torch.int32)

  q_offset = 0
  for b in range(batch):
    q_seqlen_b = q_seqlens[b]
    kv_seqlen_b = kv_seqlens[b]
    total_blocks_b = ceil(kv_seqlen_b / block_size)

    key_logical_b = torch.zeros(
        total_blocks_b * block_size, kv_heads, head_dim, dtype=torch.float32)
    for logical_id in range(total_blocks_b):
      physical_id = int(block_table[b, logical_id].item())
      key_logical_b[logical_id * block_size:(logical_id + 1) * block_size] = key_fp32[physical_id]
    key_flat_b = key_logical_b[:kv_seqlen_b, :, :]

    q_for_batch = query_fp32[q_offset:q_offset + q_seqlen_b, :, :]
    batch_select_idx, batch_select_num = generate_block_index_with_causal(
        q_for_batch, key_flat_b, q_seqlen_b, kv_seqlen_b,
        kv_heads, group_size, block_size, top_k)

    select_idx_decode[:, q_offset:q_offset + q_seqlen_b, :] = batch_select_idx
    select_num_idx[:, q_offset:q_offset + q_seqlen_b] = batch_select_num
    q_offset += q_seqlen_b

  select_idx_global = decode_select_idx_to_prefill_global(
      select_idx_decode, q_seqlens, kv_seqlens, block_size)

  scale_value = 1.0 / math.sqrt(head_dim)
  actual_seq_lengths = torch.tensor(q_seqlens, dtype=torch.int32)
  actual_seq_lengths_kv = torch.tensor(kv_seqlens, dtype=torch.int32)

  k2q_row_ptr, k2q_q_indices, k2q_slot_indices = build_k2q_csr(
      select_idx_global, select_num_idx,
      actual_seq_lengths, actual_seq_lengths_kv,
      block_size, kv_heads=kv_heads)

  return {
      "query": query_fp32.to(torch.bfloat16),
      "key": key_fp32.to(torch.bfloat16),
      "value": value_fp32.to(torch.bfloat16),
      "select_idx_decode": select_idx_decode,
      "select_num_idx": select_num_idx,
      "select_idx": select_idx_global,
      "block_table": block_table,
      "k2q_row_ptr": torch.from_numpy(k2q_row_ptr).to(torch.int32),
      "k2q_q_indices": torch.from_numpy(k2q_q_indices).to(torch.int32),
      "k2q_slot_indices": torch.from_numpy(k2q_slot_indices).to(torch.int32),
      "q_seqlens": q_seqlens,
      "kv_seqlens": kv_seqlens,
      "actual_seq_lengths": actual_seq_lengths,
      "actual_seq_lengths_kv": actual_seq_lengths_kv,
      "kv_heads": kv_heads,
      "block_size": block_size,
      "top_k": top_k,
      "scale_value": scale_value,
  }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSparseAttentionScorePrefillGolden(unittest.TestCase):
  """Validate prefill two-phase bf16 golden (k2q CSR + KV outer loop)."""

  def _max_abs_diff(self, a, b):
    return (a.float() - b.float()).abs().max().item()

  def _run_case(self, batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
                head_dim=128, block_size=128, top_k=4, seed=42,
                check_decode=False, tol=5e-3):
    data = make_case(batch, q_seqlens, kv_seqlens, q_heads, kv_heads,
                     head_dim, block_size, top_k, seed)

    two_phase = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"],
        data["k2q_row_ptr"], data["k2q_q_indices"], data["k2q_slot_indices"],
        data["block_table"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], data["top_k"], data["scale_value"])

    qcentric = cpu_golden_prefill_qcentric_bf16(
        data["query"], data["key"], data["value"],
        data["select_idx"], data["block_table"],
        data["q_seqlens"], data["kv_seqlens"],
        block_size, top_k, data["scale_value"],
        data["select_num_idx"])

    self.assertFalse(torch.any(torch.isnan(two_phase.float())), "two-phase output contains NaN")
    self.assertFalse(torch.all(two_phase == 0), "two-phase output is all zeros")

    max_diff = self._max_abs_diff(two_phase, qcentric)
    mean_diff = (two_phase.float() - qcentric.float()).abs().mean().item()
    print(f"[golden] batch={batch} q={q_seqlens} kv={kv_seqlens} top_k={top_k} "
          f"two_phase_vs_qcentric max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")

    self.assertLess(max_diff, tol,
                    f"two-phase vs q-centric mismatch: max_diff={max_diff}")

    if check_decode:
      decode_ref = cpu_sparse_attention_score_bf16(
          data["query"], data["key"], data["value"],
          data["select_idx_decode"], data["block_table"],
          data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
          data["kv_heads"], data["select_num_idx"],
          block_size, data["scale_value"])
      decode_vs_qc = self._max_abs_diff(decode_ref, qcentric)
      print(f"  decode_vs_qcentric max_diff={decode_vs_qc:.6f}")
      self.assertLess(decode_vs_qc, tol,
                      f"decode vs q-centric mismatch: max_diff={decode_vs_qc}")

  def test_single_batch_small(self):
    self._run_case(batch=1, q_seqlens=[4], kv_seqlens=[512],
                   q_heads=8, kv_heads=2, top_k=3, seed=42)

  def test_single_batch_medium(self):
    self._run_case(batch=1, q_seqlens=[32], kv_seqlens=[2048],
                   q_heads=16, kv_heads=4, top_k=5, seed=123)

  def test_multi_batch(self):
    self._run_case(batch=3, q_seqlens=[8, 16, 4], kv_seqlens=[1024, 2048, 512],
                   q_heads=8, kv_heads=2, top_k=4, seed=456)

  def test_gqa_large_group(self):
    self._run_case(batch=1, q_seqlens=[16], kv_seqlens=[1024],
                   q_heads=32, kv_heads=4, top_k=6, seed=789, check_decode=True)

  def test_matches_decode_gqa(self):
    self._run_case(batch=1, q_seqlens=[16], kv_seqlens=[1024],
                   q_heads=8, kv_heads=2, top_k=4, seed=606, check_decode=True)

  def test_partial_last_block(self):
    self._run_case(batch=1, q_seqlens=[8], kv_seqlens=[300],
                   q_heads=8, kv_heads=2, top_k=3, seed=101)

  def test_top_k_1(self):
    self._run_case(batch=1, q_seqlens=[16], kv_seqlens=[1024],
                   q_heads=8, kv_heads=2, top_k=1, seed=202)

  def test_long_prefill(self):
    self._run_case(batch=1, q_seqlens=[128], kv_seqlens=[4096],
                   q_heads=8, kv_heads=2, top_k=8, seed=303)

  def test_k2q_csr_packed_row_q_list(self):
    """CSR packed row 0 lists non-contiguous Q tokens for a single batch."""
    select_idx = np.full((1, 6, 3), -1, dtype=np.int32)
    select_idx[0] = [
        [0, 2, 3],
        [1, 2, 3],
        [0, 2, 4],
        [1, 3, 4],
        [0, 4, 5],
        [2, 4, 5],
    ]
    select_num = np.full((1, 6), 3, dtype=np.int32)
    q_seqlens = [6]
    kv_seqlens = [768]
    block_size = 128
    row_ptr, q_idx, slot_idx = build_k2q_csr(
        select_idx, select_num, q_seqlens, kv_seqlens, block_size, kv_heads=1)
    start = int(row_ptr[0, 0])
    end = int(row_ptr[0, 1])
    self.assertEqual(q_idx[0, start:end].tolist(), [0, 2, 4])
    self.assertEqual(slot_idx[0, start:end].tolist(), [0, 0, 0])

  def test_k2q_csr_gqa_per_head(self):
    select_idx = np.full((2, 3, 2), -1, dtype=np.int32)
    select_idx[0, 0, 0] = 0
    select_idx[0, 2, 0] = 0
    select_idx[1, 1, 0] = 1
    select_idx[1, 2, 0] = 1
    select_num = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int32)
    q_seqlens = [3]
    kv_seqlens = [256]
    block_size = 128
    row_ptr, q_idx, _ = build_k2q_csr(
        select_idx, select_num, q_seqlens, kv_seqlens, block_size, kv_heads=2)
    s0, e0 = int(row_ptr[0, 0]), int(row_ptr[0, 1])
    s1, e1 = int(row_ptr[1, 1]), int(row_ptr[1, 2])
    self.assertEqual(q_idx[0, s0:e0].tolist(), [0, 2])
    self.assertEqual(q_idx[1, s1:e1].tolist(), [1, 2])

  def test_k2q_csr_matches_torch_reference(self):
    data = make_case(batch=2, q_seqlens=[4, 6], kv_seqlens=[512, 768],
                     q_heads=8, kv_heads=2, top_k=4, seed=909)
    k2q_row_ptr, k2q_q_idx, k2q_slot_idx = build_k2q_csr(
        data["select_idx"], data["select_num_idx"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], kv_heads=data["kv_heads"])

    q2k, _, _ = global_select_idx_to_batch_local(
        data["select_idx"], data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], data["select_num_idx"], kv_heads=data["kv_heads"])

    ref_row_ptr, ref_q_idx, ref_slot_idx = build_k2q_csr_torch_reference(
        torch.from_numpy(q2k),
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], q_global_offset=True)
    np.testing.assert_array_equal(k2q_row_ptr, ref_row_ptr.numpy())
    np.testing.assert_array_equal(k2q_q_idx, ref_q_idx.numpy())
    np.testing.assert_array_equal(k2q_slot_idx, ref_slot_idx.numpy())

  def test_decode_packed_row_roundtrip(self):
    q_seqlens = [4, 6]
    kv_seqlens = [300, 500]
    block_size = 128
    rows_per_batch = [ceil(k / block_size) for k in kv_seqlens]
    row_map, total_rows = _build_packed_row_map(np.array(rows_per_batch, dtype=np.int64))
    for batch_idx in range(len(q_seqlens)):
      for local_blk in range(rows_per_batch[batch_idx]):
        packed = int(row_map[batch_idx, local_blk])
        b2, l2 = decode_packed_row(packed, kv_seqlens, block_size)
        self.assertEqual((b2, l2), (batch_idx, local_blk))
    self.assertEqual(total_rows, sum(rows_per_batch))

  def test_combine_required_for_topk_partials(self):
    data = make_case(batch=1, q_seqlens=[16], kv_seqlens=[1024],
                     q_heads=8, kv_heads=2, top_k=4, seed=505)
    combined = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"],
        data["k2q_row_ptr"], data["k2q_q_indices"], data["k2q_slot_indices"],
        data["block_table"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], data["top_k"], data["scale_value"])

    one_block_idx = data["select_idx"].clone()
    one_block_idx[:, :, 1:] = -1
    one_select_num = data["select_num_idx"].clone()
    one_select_num[:] = 1
    one_k2q_row_ptr, one_k2q_q_indices, one_k2q_slot_indices = build_k2q_csr(
        one_block_idx, one_select_num,
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], kv_heads=data["kv_heads"])
    one_partial = cpu_golden_prefill_bf16(
        data["query"], data["key"], data["value"],
        one_k2q_row_ptr, one_k2q_q_indices, one_k2q_slot_indices,
        data["block_table"],
        data["actual_seq_lengths"], data["actual_seq_lengths_kv"],
        data["block_size"], 1, data["scale_value"])
    max_diff = self._max_abs_diff(combined, one_partial)
    self.assertGreater(max_diff, 1e-3)


if __name__ == "__main__":
  unittest.main()
