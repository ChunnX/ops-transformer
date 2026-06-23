# Copyright (c) 2026, Huawei Technologies. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause

"""
NPU precision test for sparse_attention_score_prefill.

Compares NPU kernel output against CPU golden (two-phase partial + combine).
Uses torch_npu custom op loading via OpBuilder.
"""

import math
import os
import sys
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TORCH_EXT = _REPO_ROOT / "torch_extension"
sys.path.insert(0, str(_TORCH_EXT))


def _load_prefill_op():
    """Load the sparse_attention_score_prefill custom op via OpBuilder."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sparse_attention_score_prefill",
        _TORCH_EXT / "cann_ops_transformer" / "ops" / "sparse_attention_score_prefill.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_prefill_mod = _load_prefill_op()


# ============================================================================
# Helper: build reverse index (select_idx -> kv_to_q tables)
# ============================================================================
def build_reverse_index(select_idx, num_kv_blocks):
    """
    Build reverse index from select_idx [total_q_tokens, topK].

    Returns:
        kv_to_q_select_idx: [num_kv_blocks, max_q_per_block] int32
        kv_to_q_select_num: [num_kv_blocks] int32
        kv_to_q_slot_idx: [num_kv_blocks, max_q_per_block] int32
        max_q_per_block: int
    """
    total_q_tokens, topK = select_idx.shape

    block_count = np.zeros(num_kv_blocks, dtype=np.int32)
    for q in range(total_q_tokens):
        for k in range(topK):
            kv_block = int(select_idx[q, k])
            if 0 <= kv_block < num_kv_blocks:
                block_count[kv_block] += 1

    max_q_per_block = int(block_count.max()) if num_kv_blocks > 0 else 1
    max_q_per_block = max(max_q_per_block, 1)

    kv_to_q_select_idx = np.zeros((num_kv_blocks, max_q_per_block), dtype=np.int32)
    kv_to_q_select_num = np.zeros(num_kv_blocks, dtype=np.int32)
    kv_to_q_slot_idx = np.zeros((num_kv_blocks, max_q_per_block), dtype=np.int32)

    for q in range(total_q_tokens):
        for k in range(topK):
            kv_block = int(select_idx[q, k])
            if 0 <= kv_block < num_kv_blocks:
                qi = kv_to_q_select_num[kv_block]
                kv_to_q_select_idx[kv_block, qi] = q
                kv_to_q_slot_idx[kv_block, qi] = k
                kv_to_q_select_num[kv_block] += 1

    return kv_to_q_select_idx, kv_to_q_select_num, kv_to_q_slot_idx, max_q_per_block


# ============================================================================
# Helper: compute kv_block_valid_size
# ============================================================================
def compute_kv_block_valid_size(kv_seqlens, block_size, batch_kv_block_offset, num_kv_blocks):
    """Compute valid token count for each logical KV block."""
    valid_size = np.zeros(num_kv_blocks, dtype=np.int32)
    batch_size = len(kv_seqlens)
    for b in range(batch_size):
        start = int(batch_kv_block_offset[b])
        end = int(batch_kv_block_offset[b + 1])
        kv_seq = int(kv_seqlens[b])
        num_blocks_b = end - start
        for i in range(num_blocks_b):
            if i == num_blocks_b - 1:
                valid_size[start + i] = kv_seq - i * block_size
            else:
                valid_size[start + i] = block_size
    return valid_size


# ============================================================================
# CPU golden: two-phase sparse attention (matches kernel logic)
# ============================================================================
def cpu_golden_prefill_bf16(query_bf16, key_bf16, value_bf16,
                            select_idx, block_table,
                            q_seqlens, kv_seqlens,
                            kv_heads, block_size, top_k, scale_value):
    """
    CPU golden with bf16 inputs (cast to fp32 for compute, matches NPU accumulator).

    Args:
        query_bf16: [total_q_tokens, num_q_heads, D] bf16
        key_bf16: [num_physical_blocks, block_size, kv_heads, D] bf16
        value_bf16: same shape as key
        select_idx: [total_q_tokens, topK] int32
        block_table: [batch, max_blocks_per_batch] int32
        q_seqlens: list[int]
        kv_seqlens: list[int]
    """
    query = query_bf16.float()
    key = key_bf16.float()
    value = value_bf16.float()

    total_q_tokens, num_q_heads, D = query.shape
    group_size = num_q_heads // kv_heads
    batch_size = len(q_seqlens)

    batch_q_offset = [0] * (batch_size + 1)
    batch_kv_block_offset = [0] * (batch_size + 1)
    for b in range(batch_size):
        batch_q_offset[b + 1] = batch_q_offset[b] + q_seqlens[b]
        batch_kv_block_offset[b + 1] = batch_kv_block_offset[b] + ceil(kv_seqlens[b] / block_size)

    num_kv_blocks = batch_kv_block_offset[batch_size]

    kv_to_q_select_idx, kv_to_q_select_num, kv_to_q_slot_idx, _ = \
        build_reverse_index(select_idx.numpy(), num_kv_blocks)

    # Phase 1: partial compute
    partial_O = torch.zeros(total_q_tokens, kv_heads, top_k, group_size, D)
    partial_max = torch.full((total_q_tokens, kv_heads, top_k, group_size), -torch.inf)
    partial_sum = torch.zeros(total_q_tokens, kv_heads, top_k, group_size)

    for kv_block_idx in range(num_kv_blocks):
        batch_idx = 0
        for b in range(batch_size):
            if kv_block_idx < batch_kv_block_offset[b + 1]:
                batch_idx = b
                break

        local_block_idx = kv_block_idx - batch_kv_block_offset[batch_idx]
        physical_block_id = int(block_table[batch_idx, local_block_idx].item())

        kv_seq = kv_seqlens[batch_idx]
        num_blocks_b = ceil(kv_seq / block_size)
        if local_block_idx == num_blocks_b - 1:
            valid_size = kv_seq - local_block_idx * block_size
        else:
            valid_size = block_size

        num_q_tokens = int(kv_to_q_select_num[kv_block_idx])

        for kv_head_idx in range(kv_heads):
            K_block = key[physical_block_id, :valid_size, kv_head_idx, :]
            V_block = value[physical_block_id, :valid_size, kv_head_idx, :]

            for qi in range(num_q_tokens):
                q_token = int(kv_to_q_select_idx[kv_block_idx, qi])
                slot_k = int(kv_to_q_slot_idx[kv_block_idx, qi])
                q_head_start = kv_head_idx * group_size

                Q_row = query[q_token, q_head_start:q_head_start + group_size, :]

                # QK: [group_size, valid_size]
                S = Q_row @ K_block.T * scale_value

                # Causal mask
                q_batch = batch_idx
                local_q_idx = q_token - batch_q_offset[q_batch]
                q_position = kv_seqlens[q_batch] - q_seqlens[q_batch] + local_q_idx
                kv_start_pos = local_block_idx * block_size
                causal_valid_len = min(valid_size, q_position - kv_start_pos + 1)
                causal_valid_len = max(causal_valid_len, 0)

                if causal_valid_len == 0:
                    continue

                if causal_valid_len < valid_size:
                    S[:, causal_valid_len:] = -torch.inf

                # Softmax (unnormalized partial)
                row_max, _ = S.max(dim=1)
                P = torch.exp(S - row_max.unsqueeze(1))
                row_sum = P.sum(dim=1)

                # PV
                O_tmp = P @ V_block

                partial_O[q_token, kv_head_idx, slot_k] = O_tmp
                partial_max[q_token, kv_head_idx, slot_k] = row_max
                partial_sum[q_token, kv_head_idx, slot_k] = row_sum

    # Phase 2: CombineScale (online softmax)
    output = torch.zeros(total_q_tokens, num_q_heads, D)

    for q_token in range(total_q_tokens):
        for kv_head_idx in range(kv_heads):
            q_head_start = kv_head_idx * group_size

            global_max = partial_max[q_token, kv_head_idx, 0].clone()
            global_sum = partial_sum[q_token, kv_head_idx, 0].clone()
            o_acc = partial_O[q_token, kv_head_idx, 0].clone()

            for k in range(1, top_k):
                p_max_k = partial_max[q_token, kv_head_idx, k]
                p_sum_k = partial_sum[q_token, kv_head_idx, k]
                p_O_k = partial_O[q_token, kv_head_idx, k]

                if torch.all(p_max_k == -torch.inf):
                    continue

                new_max = torch.maximum(global_max, p_max_k)
                corr_old = torch.exp(global_max - new_max)
                corr_new = torch.exp(p_max_k - new_max)

                o_acc = corr_old.unsqueeze(1) * o_acc + corr_new.unsqueeze(1) * p_O_k
                global_sum = corr_old * global_sum + corr_new * p_sum_k
                global_max = new_max

            safe_sum = global_sum.clone()
            safe_sum[safe_sum == 0] = 1.0
            o_final = o_acc / safe_sum.unsqueeze(1)
            output[q_token, q_head_start:q_head_start + group_size, :] = o_final

    return output.to(torch.bfloat16)


# ============================================================================
# Helper: generate select_idx with causal constraint
# ============================================================================
def generate_select_idx_causal(q_seqlens, kv_seqlens, block_size, top_k, seed=42):
    """Generate select_idx respecting causal constraint."""
    rng = np.random.RandomState(seed)
    batch_size = len(q_seqlens)
    total_q_tokens = sum(q_seqlens)

    batch_q_offset = [0] * (batch_size + 1)
    batch_kv_block_offset = [0] * (batch_size + 1)
    for b in range(batch_size):
        batch_q_offset[b + 1] = batch_q_offset[b] + q_seqlens[b]
        batch_kv_block_offset[b + 1] = batch_kv_block_offset[b] + ceil(kv_seqlens[b] / block_size)

    select_idx = np.zeros((total_q_tokens, top_k), dtype=np.int32)

    for b in range(batch_size):
        kv_block_start = batch_kv_block_offset[b]
        num_blocks_b = batch_kv_block_offset[b + 1] - kv_block_start

        for qi in range(q_seqlens[b]):
            q_global = batch_q_offset[b] + qi
            q_position = kv_seqlens[b] - q_seqlens[b] + qi
            max_valid_block = min(q_position // block_size + 1, num_blocks_b)

            if max_valid_block <= top_k:
                selected = list(range(max_valid_block))
                while len(selected) < top_k:
                    selected.append(selected[-1] if selected else 0)
            else:
                selected = sorted(rng.choice(max_valid_block, top_k, replace=False).tolist())

            select_idx[q_global, :] = [kv_block_start + s for s in selected]

    return torch.from_numpy(select_idx).to(torch.int32)


# ============================================================================
# Helper: generate block table (identity or shuffled)
# ============================================================================
def generate_block_table(batch, max_blocks_per_batch, shuffle=True):
    total_physical = batch * max_blocks_per_batch
    all_ids = list(range(total_physical))
    if shuffle:
        import random
        rng = random.Random(137)
        rng.shuffle(all_ids)
    block_table = torch.zeros(batch, max_blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = all_ids[b * max_blocks_per_batch + i]
    return block_table


# ============================================================================
# Test class
# ============================================================================
class TestSparseAttentionScorePrefillNPU(TestCase):
    """NPU precision test for sparse_attention_score_prefill kernel."""

    def make_case(self, batch_size, q_seqlens, kv_seqlens,
                  num_q_heads, kv_heads, D, block_size, top_k, seed=42):
        """Construct full test case data."""
        total_q_tokens = sum(q_seqlens)
        num_kv_blocks_list = [ceil(ks / block_size) for ks in kv_seqlens]
        total_kv_blocks = sum(num_kv_blocks_list)
        max_blocks_per_batch = max(num_kv_blocks_list)
        num_physical_blocks = max_blocks_per_batch * batch_size

        torch.manual_seed(seed)
        query_fp32 = torch.rand(total_q_tokens, num_q_heads, D) * 2 - 1
        key_fp32 = torch.rand(num_physical_blocks, block_size, kv_heads, D) * 2 - 1
        value_fp32 = torch.rand(num_physical_blocks, block_size, kv_heads, D) * 2 - 1

        block_table = generate_block_table(batch_size, max_blocks_per_batch)

        select_idx = generate_select_idx_causal(
            q_seqlens, kv_seqlens, block_size, top_k, seed)

        scale_value = 1.0 / math.sqrt(D)

        # Build reverse index
        batch_kv_block_offset = [0] * (batch_size + 1)
        for b in range(batch_size):
            batch_kv_block_offset[b + 1] = batch_kv_block_offset[b] + num_kv_blocks_list[b]

        kv_to_q_select_idx, kv_to_q_select_num, kv_to_q_slot_idx, max_q_per_block = \
            build_reverse_index(select_idx.numpy(), total_kv_blocks)

        kv_block_valid_size = compute_kv_block_valid_size(
            kv_seqlens, block_size, batch_kv_block_offset, total_kv_blocks)

        return {
            'query': query_fp32.to(torch.bfloat16),
            'key': key_fp32.to(torch.bfloat16),
            'value': value_fp32.to(torch.bfloat16),
            'select_idx': select_idx,
            'block_table': block_table,
            'kv_to_q_select_idx': torch.from_numpy(kv_to_q_select_idx).to(torch.int32),
            'kv_to_q_select_num': torch.from_numpy(kv_to_q_select_num).to(torch.int32),
            'kv_to_q_slot_idx': torch.from_numpy(kv_to_q_slot_idx).to(torch.int32),
            'kv_block_valid_size': torch.from_numpy(kv_block_valid_size).to(torch.int32),
            'q_seqlens': torch.tensor(q_seqlens, dtype=torch.int64),
            'kv_seqlens': torch.tensor(kv_seqlens, dtype=torch.int64),
            'batch_size': batch_size,
            'num_q_heads': num_q_heads,
            'kv_heads': kv_heads,
            'D': D,
            'block_size': block_size,
            'top_k': top_k,
            'scale_value': scale_value,
            'max_q_per_block': max_q_per_block,
        }

    def _run_npu_test(self, batch_size, q_seqlens, kv_seqlens,
                      num_q_heads, kv_heads, D, block_size, top_k,
                      seed=42, prec=7e-3):
        """Run NPU kernel and compare with CPU golden."""
        torch.npu.synchronize()
        torch.npu.empty_cache()

        data = self.make_case(batch_size, q_seqlens, kv_seqlens,
                              num_q_heads, kv_heads, D, block_size, top_k, seed)

        # CPU golden
        cpu_out = cpu_golden_prefill_bf16(
            data['query'], data['key'], data['value'],
            data['select_idx'], data['block_table'],
            q_seqlens, kv_seqlens,
            kv_heads, block_size, top_k, data['scale_value'])

        # NPU call via OpBuilder-loaded custom op
        npu_out = torch.ops.cann_ops_transformer.npu_sparse_attention_score_prefill(
            data['query'].npu(),
            data['key'].npu(),
            data['value'].npu(),
            data['select_idx'].npu(),
            data['block_table'].npu(),
            data['kv_to_q_select_idx'].npu(),
            data['kv_to_q_select_num'].npu(),
            data['kv_to_q_slot_idx'].npu(),
            data['kv_block_valid_size'].npu(),
            data['q_seqlens'].npu(),
            data['kv_seqlens'].npu(),
            kv_heads,
            data['scale_value'],
            block_size,
            top_k,
            4,  # inner_precise
        )

        npu_out_cpu = npu_out.cpu()
        diff = (npu_out_cpu.float() - cpu_out.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        self.assertRtolEqual(cpu_out.float().numpy(), npu_out_cpu.float().numpy(), prec=prec)

    # === Test Cases ===

    def test_bf16_single_batch_small(self):
        """Single batch, 4 Q tokens, 512 KV."""
        print("[test_bf16_single_batch_small]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[4], kv_seqlens=[512],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=3)

    def test_bf16_single_batch_medium(self):
        """Single batch, 32 Q tokens, 2048 KV."""
        print("[test_bf16_single_batch_medium]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[32], kv_seqlens=[2048],
            num_q_heads=16, kv_heads=4, D=128, block_size=128, top_k=5, seed=123)

    def test_bf16_multi_batch(self):
        """Multi-batch with different seqlens."""
        print("[test_bf16_multi_batch]")
        self._run_npu_test(
            batch_size=3, q_seqlens=[8, 16, 4], kv_seqlens=[1024, 2048, 512],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=4,
            seed=456, prec=1e-2)

    def test_bf16_gqa_large_group(self):
        """GQA group_size=8."""
        print("[test_bf16_gqa_large_group]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[16], kv_seqlens=[1024],
            num_q_heads=32, kv_heads=4, D=128, block_size=128, top_k=6, seed=789)

    def test_bf16_partial_last_block(self):
        """KV seqlen not aligned to block_size."""
        print("[test_bf16_partial_last_block]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[8], kv_seqlens=[300],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=3, seed=101)

    def test_bf16_top_k_1(self):
        """TopK = 1."""
        print("[test_bf16_top_k_1]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[16], kv_seqlens=[1024],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=1, seed=202)

    def test_bf16_long_prefill(self):
        """128 Q tokens, 4096 KV."""
        print("[test_bf16_long_prefill]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[128], kv_seqlens=[4096],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=8, seed=303)

    def test_bf16_minimal_debug(self):
        """Minimal case for debugging: 1 Q token, 1 KV block, 1 head, topK=1."""
        print("[test_bf16_minimal_debug]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[1], kv_seqlens=[128],
            num_q_heads=1, kv_heads=1, D=128, block_size=128, top_k=1, seed=0, prec=1e-2)

    def test_bf16_minimal_debug_2(self):
        """2 Q tokens, 1 KV block, 2 heads (groupSize=1), topK=1."""
        print("[test_bf16_minimal_debug_2]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[2], kv_seqlens=[128],
            num_q_heads=2, kv_heads=2, D=128, block_size=128, top_k=1, seed=1, prec=1e-2)

    def test_bf16_minimal_debug_gqa(self):
        """1 Q token, 1 KV block, groupSize=2, topK=1."""
        print("[test_bf16_minimal_debug_gqa]")
        self._run_npu_test(
            batch_size=1, q_seqlens=[1], kv_seqlens=[128],
            num_q_heads=2, kv_heads=1, D=128, block_size=128, top_k=1, seed=2, prec=1e-2)


if __name__ == '__main__':
    run_tests()
