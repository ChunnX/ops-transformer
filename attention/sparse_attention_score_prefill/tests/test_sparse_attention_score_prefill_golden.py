"""
Golden reference test for sparse_attention_score_prefill (KV-Gather-Q).

This script implements:
1. Host-side reverse index construction (select_idx -> kv_to_q tables)
2. Two-phase CPU golden: Phase1 partial compute + Phase2 CombineScale
3. Test cases with various configurations
"""

import os
import sys
import math
import unittest
import numpy as np


def build_reverse_index(select_idx, num_kv_blocks):
    """
    Build reverse index from select_idx.

    Args:
        select_idx: [total_q_tokens, topK] int32 - per Q token selected KV block indices
        num_kv_blocks: total number of KV blocks

    Returns:
        kv_to_q_select_idx: [num_kv_blocks, max_q_per_block] int32
        kv_to_q_select_num: [num_kv_blocks] int32
        kv_to_q_slot_idx: [num_kv_blocks, max_q_per_block] int32
    """
    total_q_tokens, topK = select_idx.shape

    # First pass: count max_q_per_block
    block_count = np.zeros(num_kv_blocks, dtype=np.int32)
    for q in range(total_q_tokens):
        for k in range(topK):
            kv_block = select_idx[q, k]
            if 0 <= kv_block < num_kv_blocks:
                block_count[kv_block] += 1

    max_q_per_block = int(block_count.max()) if num_kv_blocks > 0 else 1
    max_q_per_block = max(max_q_per_block, 1)

    # Second pass: fill reverse index
    kv_to_q_select_idx = np.zeros((num_kv_blocks, max_q_per_block), dtype=np.int32)
    kv_to_q_select_num = np.zeros(num_kv_blocks, dtype=np.int32)
    kv_to_q_slot_idx = np.zeros((num_kv_blocks, max_q_per_block), dtype=np.int32)

    for q in range(total_q_tokens):
        for k in range(topK):
            kv_block = select_idx[q, k]
            if 0 <= kv_block < num_kv_blocks:
                qi = kv_to_q_select_num[kv_block]
                kv_to_q_select_idx[kv_block, qi] = q
                kv_to_q_slot_idx[kv_block, qi] = k
                kv_to_q_select_num[kv_block] += 1

    return kv_to_q_select_idx, kv_to_q_select_num, kv_to_q_slot_idx, max_q_per_block


def compute_golden_prefill(query, key, value, select_idx, block_table,
                           q_seqlens, kv_seqlens, block_size, scale, causal=True):
    """
    Two-phase golden reference for KV-Gather-Q sparse attention.

    Args:
        query: [total_q_tokens, num_q_heads, D] float32
        key: [num_physical_blocks, block_size, kv_heads, D] float32
        value: [num_physical_blocks, block_size, kv_heads, D] float32
        select_idx: [total_q_tokens, topK] int32
        block_table: [batch_size, max_num_blocks] int32
        q_seqlens: [batch_size] int64
        kv_seqlens: [batch_size] int64
        block_size: int
        scale: float
        causal: bool

    Returns:
        output: [total_q_tokens, num_q_heads, D] float32
    """
    total_q_tokens, num_q_heads, D = query.shape
    _, _, kv_heads, _ = key.shape
    topK = select_idx.shape[1]
    group_size = num_q_heads // kv_heads
    batch_size = len(q_seqlens)

    # Compute batch offsets
    batch_q_offset = np.zeros(batch_size + 1, dtype=np.int64)
    batch_kv_block_offset = np.zeros(batch_size + 1, dtype=np.int64)
    for b in range(batch_size):
        batch_q_offset[b + 1] = batch_q_offset[b] + q_seqlens[b]
        num_blocks_b = (kv_seqlens[b] + block_size - 1) // block_size
        batch_kv_block_offset[b + 1] = batch_kv_block_offset[b] + num_blocks_b

    num_kv_blocks = int(batch_kv_block_offset[batch_size])

    # Build reverse index
    kv_to_q_select_idx, kv_to_q_select_num, kv_to_q_slot_idx, max_q_per_block = \
        build_reverse_index(select_idx, num_kv_blocks)

    # Phase 1: compute partials
    # workspace: [total_q_tokens, kv_heads, topK] -> (partial_O, partial_max, partial_sum)
    partial_O = np.zeros((total_q_tokens, kv_heads, topK, group_size, D), dtype=np.float32)
    partial_max = np.full((total_q_tokens, kv_heads, topK, group_size), -np.inf, dtype=np.float32)
    partial_sum = np.zeros((total_q_tokens, kv_heads, topK, group_size), dtype=np.float32)

    for kv_block_idx in range(num_kv_blocks):
        # Find batch for this KV block
        batch_idx = 0
        for b in range(batch_size):
            if kv_block_idx < batch_kv_block_offset[b + 1]:
                batch_idx = b
                break

        local_block_idx = kv_block_idx - int(batch_kv_block_offset[batch_idx])
        physical_block_id = block_table[batch_idx, local_block_idx]

        # Valid block size
        kv_seq = int(kv_seqlens[batch_idx])
        num_blocks_this_batch = (kv_seq + block_size - 1) // block_size
        if local_block_idx == num_blocks_this_batch - 1:
            valid_size = kv_seq - local_block_idx * block_size
        else:
            valid_size = block_size

        num_q_tokens = int(kv_to_q_select_num[kv_block_idx])

        for kv_head_idx in range(kv_heads):
            # K, V for this block and head: [valid_size, D]
            K_block = key[physical_block_id, :valid_size, kv_head_idx, :]  # [valid_size, D]
            V_block = value[physical_block_id, :valid_size, kv_head_idx, :]  # [valid_size, D]

            for qi in range(num_q_tokens):
                q_token = int(kv_to_q_select_idx[kv_block_idx, qi])
                slot_k = int(kv_to_q_slot_idx[kv_block_idx, qi])
                q_head_start = kv_head_idx * group_size

                # Q for this token: [group_size, D]
                Q_row = query[q_token, q_head_start:q_head_start + group_size, :]

                # QK: [group_size, valid_size]
                S = Q_row @ K_block.T * scale  # [group_size, valid_size]

                # Causal mask
                if causal:
                    q_batch = 0
                    for b in range(batch_size):
                        if q_token < batch_q_offset[b + 1]:
                            q_batch = b
                            break
                    local_q_idx = q_token - int(batch_q_offset[q_batch])
                    q_position = int(kv_seqlens[q_batch]) - int(q_seqlens[q_batch]) + local_q_idx
                    kv_start_pos = local_block_idx * block_size
                    causal_valid_len = min(valid_size, q_position - kv_start_pos + 1)
                    if causal_valid_len < valid_size:
                        S[:, causal_valid_len:] = -np.inf

                # Softmax (unnormalized)
                max_val = S.max(axis=1, keepdims=False)  # [group_size]
                P = np.exp(S - max_val[:, None])  # [group_size, valid_size]
                sum_val = P.sum(axis=1)  # [group_size]

                # PV: [group_size, D]
                O_tmp = P @ V_block  # [group_size, D]

                # Store partial
                partial_O[q_token, kv_head_idx, slot_k] = O_tmp
                partial_max[q_token, kv_head_idx, slot_k] = max_val
                partial_sum[q_token, kv_head_idx, slot_k] = sum_val

    # Phase 2: CombineScale
    output = np.zeros((total_q_tokens, num_q_heads, D), dtype=np.float32)

    for q_token in range(total_q_tokens):
        for kv_head_idx in range(kv_heads):
            q_head_start = kv_head_idx * group_size

            # Initialize with first partial
            o_acc = partial_O[q_token, kv_head_idx, 0].copy()  # [group_size, D]
            global_max = partial_max[q_token, kv_head_idx, 0].copy()  # [group_size]
            global_sum = partial_sum[q_token, kv_head_idx, 0].copy()  # [group_size]

            # Combine remaining partials
            for k in range(1, topK):
                p_max_k = partial_max[q_token, kv_head_idx, k]
                p_sum_k = partial_sum[q_token, kv_head_idx, k]
                p_O_k = partial_O[q_token, kv_head_idx, k]

                # Skip invalid partials (max = -inf means no valid data)
                if np.all(p_max_k == -np.inf):
                    continue

                new_max = np.maximum(global_max, p_max_k)
                corr_old = np.exp(global_max - new_max)
                corr_new = np.exp(p_max_k - new_max)

                o_acc = corr_old[:, None] * o_acc + corr_new[:, None] * p_O_k
                global_sum = corr_old * global_sum + corr_new * p_sum_k
                global_max = new_max

            # Normalize
            o_final = o_acc / global_sum[:, None]
            output[q_token, q_head_start:q_head_start + group_size, :] = o_final

    return output


def make_test_data(batch_size, q_seqlens, kv_seqlens, num_q_heads, kv_heads, D,
                   block_size, top_k, seed=42):
    """Generate test data for sparse attention prefill."""
    np.random.seed(seed)

    total_q_tokens = sum(q_seqlens)
    num_kv_blocks_list = [(ks + block_size - 1) // block_size for ks in kv_seqlens]
    total_kv_blocks = sum(num_kv_blocks_list)
    max_num_blocks = max(num_kv_blocks_list)

    # Use enough physical blocks (could be more than logical for paging)
    num_physical_blocks = total_kv_blocks + 10

    # Q, K, V with values in [-1, 1]
    query = (np.random.rand(total_q_tokens, num_q_heads, D).astype(np.float32) * 2 - 1)
    key = (np.random.rand(num_physical_blocks, block_size, kv_heads, D).astype(np.float32) * 2 - 1)
    value = (np.random.rand(num_physical_blocks, block_size, kv_heads, D).astype(np.float32) * 2 - 1)

    # block_table: identity mapping for simplicity
    block_table = np.full((batch_size, max_num_blocks), 0, dtype=np.int32)
    block_offset = 0
    for b in range(batch_size):
        for i in range(num_kv_blocks_list[b]):
            block_table[b, i] = block_offset + i
        block_offset += num_kv_blocks_list[b]

    # select_idx: random selection of top_k KV blocks per Q token (within batch's blocks)
    select_idx = np.zeros((total_q_tokens, top_k), dtype=np.int32)
    q_offset = 0
    kv_block_offset = 0
    for b in range(batch_size):
        num_blocks_b = num_kv_blocks_list[b]
        for qi in range(q_seqlens[b]):
            q_global = q_offset + qi
            # For causal: only select blocks that are causally valid
            q_position = kv_seqlens[b] - q_seqlens[b] + qi
            max_valid_block = q_position // block_size + 1
            max_valid_block = min(max_valid_block, num_blocks_b)

            if max_valid_block <= top_k:
                selected = list(range(max_valid_block))
                while len(selected) < top_k:
                    selected.append(selected[-1] if selected else 0)
            else:
                selected = sorted(np.random.choice(max_valid_block, top_k, replace=False).tolist())

            select_idx[q_global, :] = [kv_block_offset + s for s in selected]
        q_offset += q_seqlens[b]
        kv_block_offset += num_blocks_b

    scale = 1.0 / math.sqrt(D)

    return {
        'query': query,
        'key': key,
        'value': value,
        'select_idx': select_idx,
        'block_table': block_table,
        'q_seqlens': np.array(q_seqlens, dtype=np.int64),
        'kv_seqlens': np.array(kv_seqlens, dtype=np.int64),
        'block_size': block_size,
        'scale': scale,
        'top_k': top_k,
        'num_q_heads': num_q_heads,
        'kv_heads': kv_heads,
        'D': D,
    }


class TestSparseAttentionScorePrefillGolden(unittest.TestCase):
    """Test the golden reference implementation."""

    def _run_golden_test(self, batch_size, q_seqlens, kv_seqlens,
                         num_q_heads, kv_heads, D, block_size, top_k, seed=42):
        data = make_test_data(batch_size, q_seqlens, kv_seqlens,
                              num_q_heads, kv_heads, D, block_size, top_k, seed)

        output = compute_golden_prefill(
            data['query'], data['key'], data['value'],
            data['select_idx'], data['block_table'],
            data['q_seqlens'], data['kv_seqlens'],
            data['block_size'], data['scale'], causal=True)

        # Basic sanity: output should not be all zeros or NaN
        self.assertFalse(np.any(np.isnan(output)),
                         "Output contains NaN")
        self.assertFalse(np.all(output == 0),
                         "Output is all zeros")

        # Compare with standard attention (direct compute without two-phase split)
        output_ref = self._compute_reference_attention(data)
        max_diff = np.abs(output - output_ref).max()
        mean_diff = np.abs(output - output_ref).mean()
        print(f"[golden] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        # Two-phase and direct should be numerically identical (both fp32)
        self.assertLess(max_diff, 1e-5,
                        f"Two-phase vs direct mismatch: max_diff={max_diff}")

    def _compute_reference_attention(self, data):
        """Direct sparse attention computation (single pass) for reference."""
        query = data['query']
        key = data['key']
        value = data['value']
        select_idx = data['select_idx']
        block_table = data['block_table']
        q_seqlens = data['q_seqlens']
        kv_seqlens = data['kv_seqlens']
        block_size = data['block_size']
        scale = data['scale']
        num_q_heads = data['num_q_heads']
        kv_heads = data['kv_heads']
        D = data['D']
        top_k = data['top_k']
        group_size = num_q_heads // kv_heads
        batch_size = len(q_seqlens)

        total_q_tokens = query.shape[0]
        output = np.zeros_like(query)

        batch_q_offset = np.zeros(batch_size + 1, dtype=np.int64)
        batch_kv_block_offset = np.zeros(batch_size + 1, dtype=np.int64)
        for b in range(batch_size):
            batch_q_offset[b + 1] = batch_q_offset[b] + q_seqlens[b]
            batch_kv_block_offset[b + 1] = batch_kv_block_offset[b] + \
                (kv_seqlens[b] + block_size - 1) // block_size

        for q_token in range(total_q_tokens):
            # Find batch
            q_batch = 0
            for b in range(batch_size):
                if q_token < batch_q_offset[b + 1]:
                    q_batch = b
                    break

            local_q_idx = q_token - int(batch_q_offset[q_batch])
            q_position = int(kv_seqlens[q_batch]) - int(q_seqlens[q_batch]) + local_q_idx

            for kv_head_idx in range(kv_heads):
                q_head_start = kv_head_idx * group_size
                Q_row = query[q_token, q_head_start:q_head_start + group_size, :]

                # Gather all KV from topK blocks
                all_K = []
                all_V = []
                all_valid_mask = []

                for k in range(top_k):
                    kv_block_global = select_idx[q_token, k]
                    # Find batch for this kv block
                    kv_batch = 0
                    for b in range(batch_size):
                        if kv_block_global < batch_kv_block_offset[b + 1]:
                            kv_batch = b
                            break

                    local_block = kv_block_global - int(batch_kv_block_offset[kv_batch])
                    phys_block = block_table[kv_batch, local_block]

                    kv_seq = int(kv_seqlens[kv_batch])
                    num_blocks_b = (kv_seq + block_size - 1) // block_size
                    if local_block == num_blocks_b - 1:
                        valid_sz = kv_seq - local_block * block_size
                    else:
                        valid_sz = block_size

                    K_blk = key[phys_block, :valid_sz, kv_head_idx, :]
                    V_blk = value[phys_block, :valid_sz, kv_head_idx, :]

                    # Causal mask
                    kv_start_pos = local_block * block_size
                    causal_len = min(valid_sz, q_position - kv_start_pos + 1)
                    causal_len = max(causal_len, 0)

                    if causal_len > 0:
                        all_K.append(K_blk[:causal_len])
                        all_V.append(V_blk[:causal_len])

                if len(all_K) == 0:
                    continue

                K_concat = np.concatenate(all_K, axis=0)  # [total_valid, D]
                V_concat = np.concatenate(all_V, axis=0)

                S = Q_row @ K_concat.T * scale  # [group_size, total_valid]
                max_val = S.max(axis=1, keepdims=True)
                P = np.exp(S - max_val)
                sum_val = P.sum(axis=1, keepdims=True)
                P_norm = P / sum_val
                O = P_norm @ V_concat

                output[q_token, q_head_start:q_head_start + group_size, :] = O

        return output

    # === Test Cases ===

    def test_single_batch_small(self):
        """Single batch, small sequence."""
        self._run_golden_test(
            batch_size=1, q_seqlens=[4], kv_seqlens=[512],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=3, seed=42)

    def test_single_batch_medium(self):
        """Single batch, medium prefill."""
        self._run_golden_test(
            batch_size=1, q_seqlens=[32], kv_seqlens=[2048],
            num_q_heads=16, kv_heads=4, D=128, block_size=128, top_k=5, seed=123)

    def test_multi_batch(self):
        """Multi-batch with different seqlens."""
        self._run_golden_test(
            batch_size=3, q_seqlens=[8, 16, 4], kv_seqlens=[1024, 2048, 512],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=4, seed=456)

    def test_gqa_large_group(self):
        """GQA with large group size."""
        self._run_golden_test(
            batch_size=1, q_seqlens=[16], kv_seqlens=[1024],
            num_q_heads=32, kv_heads=4, D=128, block_size=128, top_k=6, seed=789)

    def test_partial_last_block(self):
        """KV seqlen not aligned to block_size."""
        self._run_golden_test(
            batch_size=1, q_seqlens=[8], kv_seqlens=[300],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=3, seed=101)

    def test_top_k_1(self):
        """TopK = 1 (each Q selects only 1 block)."""
        self._run_golden_test(
            batch_size=1, q_seqlens=[16], kv_seqlens=[1024],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=1, seed=202)

    def test_long_prefill(self):
        """Longer prefill with many Q tokens sharing blocks."""
        self._run_golden_test(
            batch_size=1, q_seqlens=[128], kv_seqlens=[4096],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=8, seed=303)

    def test_multi_batch_uneven(self):
        """Multi-batch with very uneven seqlens."""
        self._run_golden_test(
            batch_size=4, q_seqlens=[1, 64, 8, 32], kv_seqlens=[256, 8192, 1000, 4096],
            num_q_heads=16, kv_heads=4, D=128, block_size=128, top_k=5, seed=404)

    def test_reverse_index_contains_non_contiguous_q_tokens(self):
        """A KV block can gather Q tokens that are not contiguous in TND order."""
        select_idx = np.array([
            [0, 2, 3],
            [1, 2, 3],
            [0, 2, 4],
            [1, 3, 4],
            [0, 4, 5],
            [2, 4, 5],
        ], dtype=np.int32)
        kv_to_q_select_idx, kv_to_q_select_num, kv_to_q_slot_idx, _ = build_reverse_index(select_idx, 6)
        q_list_for_block0 = kv_to_q_select_idx[0, :kv_to_q_select_num[0]].tolist()
        slot_list_for_block0 = kv_to_q_slot_idx[0, :kv_to_q_select_num[0]].tolist()
        self.assertEqual(q_list_for_block0, [0, 2, 4])
        self.assertEqual(slot_list_for_block0, [0, 0, 0])
        self.assertTrue(any((q_list_for_block0[i + 1] - q_list_for_block0[i]) > 1
                            for i in range(len(q_list_for_block0) - 1)))

    def test_combine_is_required_for_topk_partials(self):
        """Using only one partial is not equivalent to CombineScale across topK."""
        data = make_test_data(
            batch_size=1, q_seqlens=[16], kv_seqlens=[1024],
            num_q_heads=8, kv_heads=2, D=128, block_size=128, top_k=4, seed=505)
        combined = compute_golden_prefill(
            data['query'], data['key'], data['value'], data['select_idx'], data['block_table'],
            data['q_seqlens'], data['kv_seqlens'], data['block_size'], data['scale'], causal=True)

        one_block_data = dict(data)
        one_block_data['select_idx'] = data['select_idx'][:, :1]
        one_partial = compute_golden_prefill(
            one_block_data['query'], one_block_data['key'], one_block_data['value'],
            one_block_data['select_idx'], one_block_data['block_table'],
            one_block_data['q_seqlens'], one_block_data['kv_seqlens'], one_block_data['block_size'],
            one_block_data['scale'], causal=True)
        max_diff = np.abs(combined - one_partial).max()
        self.assertGreater(max_diff, 1e-3)


if __name__ == '__main__':
    unittest.main()
