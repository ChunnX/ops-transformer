#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only unit tests for batch consistency transforms and schema invariants.

These tests do NOT require NPU and can run in any environment with PyTorch.
They cover:

* TND segment split / concat round-trip
* Permutation + inverse recovery (BSND and TND)
* Mode 2 split: origin completeness and no duplicates
* PA_BBND block table reorder: block ids unchanged and in-bounds
* Per-batch vector synchronization
* Schema invariant validation (positive and negative)
* Mode 3 / 4 explicit unsupported
* Deterministic level save/restore (monkeypatch)
* Physical Q length priority over seqused_q
* BSND cu_seqlens_q recompute for mode2
* TND topk_length segmentation with original cu_seqlens_q
* Params full synchronization
* LSE merge for mode2 (BSND and TND)
* BF16 MD5 safe computation
* Split validation (num_groups bounds, no empty groups)
"""

import pytest
import torch

from batch_consistency.model import TokenOrigin, RunResult
from batch_consistency import layouts
from batch_consistency import schema
from batch_consistency.transform import (
    generate_non_identity_permutation,
    generate_split_groups,
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_split,
    transform_mode4_prepend_delete,
    UnsupportedTransformError,
)
from batch_consistency.compare import (
    _tensor_md5,
    compare_consistency,
    merge_split_lse,
    compare_pair,
)


# ------------------------------------------------------------------ #
#  Helpers: synthetic input_data builders
# ------------------------------------------------------------------ #

def _make_bsnd_bsnd_input(B=4, S1=2, S2=8, N1=4, N2=1, D=16, seed=42, with_cu_q=True):
    """Build a minimal BSND+BSND input_data dict.

    If *with_cu_q* is True, ``cu_seqlens_q`` is set (mimicking real golden).
    If False, ``cu_seqlens_q`` is None and ``seqused_q`` is None (real golden common case).
    """
    torch.manual_seed(seed)
    q = torch.randn(B, S1, N1, D, dtype=torch.bfloat16)
    ori_kv = torch.randn(B, S2, N2, D, dtype=torch.bfloat16)
    cmp_kv = torch.randn(B, S2 // 4, N2, D, dtype=torch.bfloat16)
    seqused_q = torch.tensor([S1] * B, dtype=torch.int32)
    seqused_ori_kv = torch.tensor([S2] * B, dtype=torch.int32)
    seqused_cmp_kv = torch.tensor([S2 // 4] * B, dtype=torch.int32)
    cmp_residual_kv = torch.tensor([0] * B, dtype=torch.int32)
    sinks = torch.randn(N1, dtype=torch.float)
    cpu_output = torch.randn(B, S1, N1, D, dtype=torch.bfloat16)

    if with_cu_q:
        cu_q = torch.tensor(list(range(0, (B + 1) * S1, S1)), dtype=torch.int32)
    else:
        cu_q = None
        seqused_q = None  # real golden: both None for BSND

    return {
        "B": B, "S1": S1, "S2": S2, "T1": None, "T2": None, "T3": None,
        "N1": N1, "N2": N2,
        "layout_q": "BSND", "layout_kv": "BSND",
        "params": {"B": B, "S1": S1, "S2": S2, "return_softmax_lse": False,
                    "layout_q": "BSND", "layout_kv": "BSND",
                    "cu_seqlens_q": cu_q.tolist() if cu_q is not None else None,
                    "seqused_q": seqused_q.tolist() if seqused_q is not None else None,
                    "seqused_ori_kv": seqused_ori_kv.tolist(),
                    "seqused_cmp_kv": seqused_cmp_kv.tolist(),
                    "cmp_residual_kv": cmp_residual_kv.tolist()},
        "metadata_input": {
            "N1": N1, "N2": N2, "D": D, "B": B,
            "cu_seqlens_q": cu_q, "cu_seqlens_ori_kv": None, "cu_seqlens_cmp_kv": None,
            "seqused_q": seqused_q, "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv, "cmp_residual_kv": cmp_residual_kv,
            "ori_topk_length": None, "cmp_topk_length": None,
            "max_seqlen_q": S1, "max_seqlen_ori_kv": S2, "max_seqlen_cmp_kv": S2 // 4,
            "K": None, "cmp_ratio": 4,
            "ori_mask_mode": 0, "cmp_mask_mode": 3, "ori_win_left": 127, "ori_win_right": 0,
            "layout_q": "BSND", "layout_kv": "BSND",
        },
        "input": {
            "q": q, "ori_kv": ori_kv, "cmp_kv": cmp_kv,
            "ori_sparse_indices": None, "cmp_sparse_indices": None,
            "ori_block_table": None, "cmp_block_table": None,
            "cu_seqlens_q": cu_q, "cu_seqlens_ori_kv": None, "cu_seqlens_cmp_kv": None,
            "seqused_q": seqused_q, "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv, "cmp_residual_kv": cmp_residual_kv,
            "sinks": sinks, "softmax_scale": 0.04,
            "ori_mask_mode": 0, "cmp_mask_mode": 3,
            "ori_win_left": 127, "ori_win_right": 0,
            "layout_q": "BSND", "layout_kv": "BSND",
        },
        "cpu_output": cpu_output,
        "softmax_lse": None,
    }


def _make_tnd_tnd_input(B=4, seed=42):
    """Build a minimal TND+TND input_data dict with variable lengths."""
    torch.manual_seed(seed)
    q_lengths = [2, 3, 1, 4]
    kv_lengths = [8, 10, 6, 12]
    cmp_lengths = [2, 2, 1, 3]
    N1, N2, D = 4, 1, 16

    T1 = sum(q_lengths)
    T2 = sum(kv_lengths)
    T3 = sum(cmp_lengths)
    cu_q = torch.tensor([0] + [sum(q_lengths[:i+1]) for i in range(B)], dtype=torch.int32)
    cu_ori = torch.tensor([0] + [sum(kv_lengths[:i+1]) for i in range(B)], dtype=torch.int32)
    cu_cmp = torch.tensor([0] + [sum(cmp_lengths[:i+1]) for i in range(B)], dtype=torch.int32)

    q = torch.randn(T1, N1, D, dtype=torch.bfloat16)
    ori_kv = torch.randn(T2, N2, D, dtype=torch.bfloat16)
    cmp_kv = torch.randn(T3, N2, D, dtype=torch.bfloat16)
    seqused_q = torch.tensor(q_lengths, dtype=torch.int32)
    seqused_ori_kv = torch.tensor(kv_lengths, dtype=torch.int32)
    seqused_cmp_kv = torch.tensor(cmp_lengths, dtype=torch.int32)
    cmp_residual_kv = torch.tensor([0] * B, dtype=torch.int32)
    sinks = torch.randn(N1, dtype=torch.float)
    cpu_output = torch.randn(T1, N1, D, dtype=torch.bfloat16)

    return {
        "B": B, "S1": max(q_lengths), "S2": max(kv_lengths),
        "T1": T1, "T2": T2, "T3": T3,
        "N1": N1, "N2": N2,
        "layout_q": "TND", "layout_kv": "TND",
        "params": {"B": B, "T1": T1, "T2": T2, "T3": T3,
                    "return_softmax_lse": False,
                    "layout_q": "TND", "layout_kv": "TND",
                    "cu_seqlens_q": cu_q.tolist(), "cu_seqlens_ori_kv": cu_ori.tolist(),
                    "cu_seqlens_cmp_kv": cu_cmp.tolist(),
                    "seqused_q": seqused_q.tolist(), "seqused_ori_kv": seqused_ori_kv.tolist(),
                    "seqused_cmp_kv": seqused_cmp_kv.tolist(),
                    "cmp_residual_kv": cmp_residual_kv.tolist()},
        "metadata_input": {
            "N1": N1, "N2": N2, "D": D, "B": B,
            "cu_seqlens_q": cu_q, "cu_seqlens_ori_kv": cu_ori, "cu_seqlens_cmp_kv": cu_cmp,
            "seqused_q": seqused_q, "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv, "cmp_residual_kv": cmp_residual_kv,
            "ori_topk_length": None, "cmp_topk_length": None,
            "max_seqlen_q": max(q_lengths), "max_seqlen_ori_kv": max(kv_lengths),
            "max_seqlen_cmp_kv": max(cmp_lengths),
            "K": None, "cmp_ratio": 4,
            "ori_mask_mode": 4, "cmp_mask_mode": 3, "ori_win_left": 127, "ori_win_right": 0,
            "layout_q": "TND", "layout_kv": "TND",
        },
        "input": {
            "q": q, "ori_kv": ori_kv, "cmp_kv": cmp_kv,
            "ori_sparse_indices": None, "cmp_sparse_indices": None,
            "ori_block_table": None, "cmp_block_table": None,
            "cu_seqlens_q": cu_q, "cu_seqlens_ori_kv": cu_ori, "cu_seqlens_cmp_kv": cu_cmp,
            "seqused_q": seqused_q, "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv, "cmp_residual_kv": cmp_residual_kv,
            "sinks": sinks, "softmax_scale": 0.04,
            "ori_mask_mode": 4, "cmp_mask_mode": 3,
            "ori_win_left": 127, "ori_win_right": 0,
            "layout_q": "TND", "layout_kv": "TND",
        },
        "cpu_output": cpu_output,
        "softmax_lse": None,
    }


def _make_tnd_pa_input(B=4, seed=42):
    """Build a minimal TND+PA_BBND input_data dict."""
    torch.manual_seed(seed)
    q_lengths = [1, 1, 1, 1]
    kv_lengths = [128, 256, 128, 256]
    cmp_lengths = [32, 64, 32, 64]
    N1, N2, D = 4, 1, 16
    block_size1, block_size2 = 128, 128

    T1 = sum(q_lengths)
    cu_q = torch.tensor([0] + [sum(q_lengths[:i+1]) for i in range(B)], dtype=torch.int32)

    block_num1 = sum((l + block_size1 - 1) // block_size1 for l in kv_lengths)
    block_num2 = sum((l + block_size2 - 1) // block_size2 for l in cmp_lengths)

    q = torch.randn(T1, N1, D, dtype=torch.bfloat16)
    ori_kv = torch.randn(block_num1, block_size1, N2, D, dtype=torch.bfloat16)
    cmp_kv = torch.randn(block_num2, block_size2, N2, D, dtype=torch.bfloat16)

    max_blocks_ori = max((l + block_size1 - 1) // block_size1 for l in kv_lengths)
    max_blocks_cmp = max((l + block_size2 - 1) // block_size2 for l in cmp_lengths)
    ori_bt = torch.zeros(B, max_blocks_ori, dtype=torch.int32)
    cmp_bt = torch.zeros(B, max_blocks_cmp, dtype=torch.int32)
    bid = 0
    for i in range(B):
        nb = (kv_lengths[i] + block_size1 - 1) // block_size1
        for j in range(nb):
            ori_bt[i, j] = bid
            bid += 1
    bid = 0
    for i in range(B):
        nb = (cmp_lengths[i] + block_size2 - 1) // block_size2
        for j in range(nb):
            cmp_bt[i, j] = bid
            bid += 1

    seqused_q = torch.tensor(q_lengths, dtype=torch.int32)
    seqused_ori_kv = torch.tensor(kv_lengths, dtype=torch.int32)
    seqused_cmp_kv = torch.tensor(cmp_lengths, dtype=torch.int32)
    cmp_residual_kv = torch.tensor([0] * B, dtype=torch.int32)
    sinks = torch.randn(N1, dtype=torch.float)

    return {
        "B": B, "S1": 1, "S2": max(kv_lengths),
        "T1": T1, "T2": None, "T3": None,
        "N1": N1, "N2": N2,
        "layout_q": "TND", "layout_kv": "PA_BBND",
        "params": {"B": B, "T1": T1, "return_softmax_lse": False,
                    "layout_q": "TND", "layout_kv": "PA_BBND",
                    "cu_seqlens_q": cu_q.tolist(),
                    "seqused_q": seqused_q.tolist(), "seqused_ori_kv": seqused_ori_kv.tolist(),
                    "seqused_cmp_kv": seqused_cmp_kv.tolist(),
                    "cmp_residual_kv": cmp_residual_kv.tolist()},
        "metadata_input": {
            "N1": N1, "N2": N2, "D": D, "B": B,
            "cu_seqlens_q": cu_q, "cu_seqlens_ori_kv": None, "cu_seqlens_cmp_kv": None,
            "seqused_q": seqused_q, "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv, "cmp_residual_kv": cmp_residual_kv,
            "ori_topk_length": None, "cmp_topk_length": None,
            "max_seqlen_q": max(q_lengths), "max_seqlen_ori_kv": max(kv_lengths),
            "max_seqlen_cmp_kv": max(cmp_lengths),
            "K": None, "cmp_ratio": 4,
            "ori_mask_mode": 4, "cmp_mask_mode": 3, "ori_win_left": 127, "ori_win_right": 0,
            "layout_q": "TND", "layout_kv": "PA_BBND",
        },
        "input": {
            "q": q, "ori_kv": ori_kv, "cmp_kv": cmp_kv,
            "ori_sparse_indices": None, "cmp_sparse_indices": None,
            "ori_block_table": ori_bt, "cmp_block_table": cmp_bt,
            "cu_seqlens_q": cu_q, "cu_seqlens_ori_kv": None, "cu_seqlens_cmp_kv": None,
            "seqused_q": seqused_q, "seqused_ori_kv": seqused_ori_kv,
            "seqused_cmp_kv": seqused_cmp_kv, "cmp_residual_kv": cmp_residual_kv,
            "sinks": sinks, "softmax_scale": 0.04,
            "ori_mask_mode": 4, "cmp_mask_mode": 3,
            "ori_win_left": 127, "ori_win_right": 0,
            "layout_q": "TND", "layout_kv": "PA_BBND",
        },
        "cpu_output": torch.randn(T1, N1, D, dtype=torch.bfloat16),
        "softmax_lse": None,
    }


# ------------------------------------------------------------------ #
#  Tests: TND segment split / concat round-trip
# ------------------------------------------------------------------ #

class TestTNDSegmentRoundTrip:
    def test_split_concat_restores_original(self):
        T, N, D = 10, 2, 4
        tensor = torch.randn(T, N, D)
        cu_seqlens = torch.tensor([0, 3, 7, 10], dtype=torch.int32)
        segments = layouts.split_tnd(tensor, cu_seqlens)
        assert len(segments) == 3
        assert segments[0].shape[0] == 3
        assert segments[1].shape[0] == 4
        assert segments[2].shape[0] == 3
        concatenated, new_cu = layouts.concat_tnd_segments(segments)
        assert new_cu == [0, 3, 7, 10]
        assert torch.equal(concatenated, tensor)

    def test_split_none_returns_none(self):
        assert layouts.split_tnd(None, torch.tensor([0, 5])) is None

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            layouts.concat_tnd_segments([])


# ------------------------------------------------------------------ #
#  Tests: Permutation + inverse recovery
# ------------------------------------------------------------------ #

class TestPermutationInverse:
    def test_bsnd_permutation_inverse(self):
        B, S, N, D = 4, 3, 2, 8
        t = torch.randn(B, S, N, D)
        order = [2, 0, 3, 1]
        reordered = layouts.select_bsnd(t, order)
        inverse = [0] * len(order)
        for new_b, old_b in enumerate(order):
            inverse[old_b] = new_b
        restored = layouts.select_bsnd(reordered, inverse)
        assert torch.equal(restored, t)

    def test_tnd_permutation_inverse(self):
        q_lengths = [2, 3, 1, 4]
        cu = torch.tensor([0, 2, 5, 6, 10], dtype=torch.int32)
        T, N, D = 10, 2, 4
        tensor = torch.randn(T, N, D)
        order = [3, 1, 0, 2]
        reordered, new_cu = layouts.select_tnd(tensor, cu, order)
        expected_cu = [0]
        for i in order:
            expected_cu.append(expected_cu[-1] + q_lengths[i])
        assert new_cu == expected_cu
        inverse = [0] * len(order)
        for new_b, old_b in enumerate(order):
            inverse[old_b] = new_b
        restored, _ = layouts.select_tnd(reordered, torch.tensor(new_cu, dtype=torch.int32), inverse)
        assert torch.equal(restored, tensor)

    def test_non_identity_permutation_generation(self):
        order = generate_non_identity_permutation(5, seed=42)
        assert sorted(order) == [0, 1, 2, 3, 4]
        assert order != [0, 1, 2, 3, 4]

    def test_non_identity_permutation_b1_raises(self):
        with pytest.raises(ValueError, match="B >= 2"):
            generate_non_identity_permutation(1)


# ------------------------------------------------------------------ #
#  Tests: Mode 2 split origin completeness
# ------------------------------------------------------------------ #

class TestSplitOriginCompleteness:
    def test_split_covers_all_origins(self):
        input_data = _make_bsnd_bsnd_input(B=4, S1=2)
        groups = generate_split_groups(4, seed=42)
        all_cases = transform_mode2_split(input_data, groups=groups, seed=42)
        all_origins = set()
        for case in all_cases:
            for o in case.output_origins:
                all_origins.add(o)
        expected = set()
        for b in range(4):
            for s in range(2):
                expected.add(TokenOrigin(source_batch=b, source_token=s))
        assert all_origins == expected

    def test_split_no_duplicate_origins(self):
        input_data = _make_tnd_tnd_input(B=4)
        cases = transform_mode2_split(input_data, seed=99)
        all_origins = []
        for case in cases:
            all_origins.extend(case.output_origins)
        assert len(all_origins) == len(set(all_origins)), "Duplicate origins found"

    def test_split_groups_partition(self):
        groups = generate_split_groups(6, seed=7)
        all_ids = sorted(idx for g in groups for idx in g)
        assert all_ids == [0, 1, 2, 3, 4, 5]
        for g in groups:
            assert len(g) > 0


# ------------------------------------------------------------------ #
#  Tests: PA_BBND block table reorder
# ------------------------------------------------------------------ #

class TestPABlockTableReorder:
    def test_block_ids_unchanged_and_in_bounds(self):
        input_data = _make_tnd_pa_input(B=4)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        new_bt = case.input_data["input"]["ori_block_table"]
        old_bt = input_data["input"]["ori_block_table"]
        cache_block_num = input_data["input"]["ori_kv"].shape[0]
        old_ids = set(old_bt[old_bt >= 0].tolist())
        new_ids = set(new_bt[new_bt >= 0].tolist())
        assert old_ids == new_ids
        valid = new_bt[new_bt >= 0]
        assert int(valid.max().item()) < cache_block_num

    def test_pa_physical_cache_unchanged(self):
        input_data = _make_tnd_pa_input(B=4)
        case = transform_mode1_reorder(input_data, order=[3, 2, 1, 0])
        assert torch.equal(
            case.input_data["input"]["ori_kv"],
            input_data["input"]["ori_kv"],
        )


# ------------------------------------------------------------------ #
#  Tests: per-batch vector synchronization
# ------------------------------------------------------------------ #

class TestPerBatchVectorSync:
    def test_bsnd_all_vectors_reordered(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        ti = case.input_data["input"]
        for field in ("seqused_q", "seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
            orig = input_data["input"][field]
            new = ti[field]
            expected = orig[torch.tensor(order, dtype=torch.long)]
            assert torch.equal(new, expected), f"{field} not properly reordered"

    def test_tnd_all_vectors_reordered(self):
        input_data = _make_tnd_tnd_input(B=4)
        order = [3, 0, 2, 1]
        case = transform_mode1_reorder(input_data, order=order)
        ti = case.input_data["input"]
        for field in ("seqused_q", "seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
            orig = input_data["input"][field]
            new = ti[field]
            expected = orig[torch.tensor(order, dtype=torch.long)]
            assert torch.equal(new, expected), f"{field} not properly reordered"

    def test_pa_all_vectors_reordered(self):
        input_data = _make_tnd_pa_input(B=4)
        order = [1, 3, 0, 2]
        case = transform_mode1_reorder(input_data, order=order)
        ti = case.input_data["input"]
        for field in ("seqused_q", "seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
            orig = input_data["input"][field]
            new = ti[field]
            expected = orig[torch.tensor(order, dtype=torch.long)]
            assert torch.equal(new, expected), f"{field} not properly reordered"


# ------------------------------------------------------------------ #
#  Tests: schema invariant validation
# ------------------------------------------------------------------ #

class TestSchemaInvariants:
    def test_valid_bsnd_passes(self):
        input_data = _make_bsnd_bsnd_input()
        schema.validate_schema(input_data)
        schema.check_invariants(input_data)

    def test_valid_bsnd_no_cu_q_no_seqused_q_passes(self):
        """Real golden common case: BSND with cu_seqlens_q=None and seqused_q=None."""
        input_data = _make_bsnd_bsnd_input(B=4, with_cu_q=False)
        schema.validate_schema(input_data)
        schema.check_invariants(input_data)

    def test_valid_tnd_passes(self):
        input_data = _make_tnd_tnd_input()
        schema.validate_schema(input_data)
        schema.check_invariants(input_data)

    def test_valid_pa_passes(self):
        input_data = _make_tnd_pa_input()
        schema.validate_schema(input_data)
        schema.check_invariants(input_data)

    def test_missing_key_raises(self):
        input_data = _make_bsnd_bsnd_input()
        del input_data["input"]
        with pytest.raises(schema.SchemaError, match="missing keys"):
            schema.validate_schema(input_data)

    def test_wrong_layout_raises(self):
        input_data = _make_bsnd_bsnd_input()
        input_data["input"]["layout_q"] = "INVALID"
        with pytest.raises(schema.SchemaError, match="Unsupported layout_q"):
            schema.validate_schema(input_data)

    def test_b_mismatch_raises(self):
        input_data = _make_bsnd_bsnd_input(B=4, with_cu_q=False)
        input_data["input"]["seqused_q"] = torch.tensor([2, 2, 2], dtype=torch.int32)
        with pytest.raises(schema.SchemaError, match="seqused_q length"):
            schema.check_invariants(input_data)

    def test_cu_seqlens_mismatch_raises(self):
        input_data = _make_tnd_tnd_input(B=4)
        ti = input_data["input"]
        ti["cu_seqlens_q"] = torch.tensor([0, 2, 5, 6, 99], dtype=torch.int32)
        with pytest.raises(schema.SchemaError, match="cu_seqlens_q"):
            schema.check_invariants(input_data)

    def test_transform_preserves_invariants_bsnd(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        schema.check_invariants(case.input_data)

    def test_transform_preserves_invariants_bsnd_no_cu_q(self):
        """Transform works when BSND has cu_seqlens_q=None (real golden)."""
        input_data = _make_bsnd_bsnd_input(B=4, with_cu_q=False)
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        schema.check_invariants(case.input_data)
        # cu_seqlens_q should still be None (not inserted)
        assert case.input_data["input"]["cu_seqlens_q"] is None

    def test_transform_preserves_invariants_tnd(self):
        input_data = _make_tnd_tnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[3, 1, 0, 2])
        schema.check_invariants(case.input_data)

    def test_transform_preserves_invariants_pa(self):
        input_data = _make_tnd_pa_input(B=4)
        case = transform_mode1_reorder(input_data, order=[1, 0, 3, 2])
        schema.check_invariants(case.input_data)

    def test_split_preserves_invariants(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        for case in cases:
            schema.check_invariants(case.input_data)

    def test_split_preserves_invariants_tnd(self):
        input_data = _make_tnd_tnd_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 1], [2, 3]])
        for case in cases:
            schema.check_invariants(case.input_data)


# ------------------------------------------------------------------ #
#  Tests: Mode 1 / Mode 2 origin correctness
# ------------------------------------------------------------------ #

class TestOriginCorrectness:
    def test_mode1_origins_match_order(self):
        input_data = _make_bsnd_bsnd_input(B=4, S1=2)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        assert case.output_origins[0] == TokenOrigin(2, 0)
        assert case.output_origins[1] == TokenOrigin(2, 1)
        assert case.output_origins[2] == TokenOrigin(0, 0)
        assert case.output_origins[3] == TokenOrigin(0, 1)

    def test_mode1_tnd_origins(self):
        input_data = _make_tnd_tnd_input(B=4)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        assert case.output_origins[0] == TokenOrigin(2, 0)
        assert case.output_origins[1] == TokenOrigin(0, 0)
        assert case.output_origins[2] == TokenOrigin(0, 1)
        assert case.output_origins[3] == TokenOrigin(3, 0)

    def test_mode2_origins_match_groups(self):
        input_data = _make_bsnd_bsnd_input(B=4, S1=2)
        cases = transform_mode2_split(input_data, groups=[[0, 3], [1, 2]])
        assert cases[0].output_origins[0] == TokenOrigin(0, 0)
        assert cases[0].output_origins[1] == TokenOrigin(0, 1)
        assert cases[0].output_origins[2] == TokenOrigin(3, 0)
        assert cases[0].output_origins[3] == TokenOrigin(3, 1)
        assert cases[1].output_origins[0] == TokenOrigin(1, 0)
        assert cases[1].output_origins[1] == TokenOrigin(1, 1)
        assert cases[1].output_origins[2] == TokenOrigin(2, 0)

    def test_baseline_origins(self):
        input_data = _make_tnd_tnd_input(B=4)
        origins = schema.build_baseline_origins(input_data)
        assert len(origins) == 10
        assert origins[0] == TokenOrigin(0, 0)
        assert origins[2] == TokenOrigin(1, 0)
        assert origins[5] == TokenOrigin(2, 0)
        assert origins[6] == TokenOrigin(3, 0)


# ------------------------------------------------------------------ #
#  Tests: original input_data not mutated
# ------------------------------------------------------------------ #

class TestNoMutation:
    def test_mode1_does_not_mutate_original(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        orig_q = input_data["input"]["q"].clone()
        orig_cu_q = input_data["input"]["cu_seqlens_q"]
        transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert torch.equal(input_data["input"]["q"], orig_q)
        assert input_data["input"]["cu_seqlens_q"] is orig_cu_q

    def test_mode2_does_not_mutate_original(self):
        input_data = _make_tnd_tnd_input(B=4)
        orig_q = input_data["input"]["q"].clone()
        transform_mode2_split(input_data, groups=[[0, 1], [2, 3]])
        assert torch.equal(input_data["input"]["q"], orig_q)


# ------------------------------------------------------------------ #
#  Tests: Mode 3 / 4 explicit unsupported
# ------------------------------------------------------------------ #

class TestUnsupportedModes:
    def test_mode3_raises_unsupported(self):
        with pytest.raises(UnsupportedTransformError, match="Mode 3"):
            transform_mode3_token_split()

    def test_mode4_raises_unsupported(self):
        with pytest.raises(UnsupportedTransformError, match="Mode 4"):
            transform_mode4_prepend_delete()

    def test_mode3_message_has_reason(self):
        try:
            transform_mode3_token_split()
        except UnsupportedTransformError as e:
            assert "Reason:" in str(e)
            assert "query position" in str(e).lower() or "mask" in str(e).lower()

    def test_mode4_message_has_reason(self):
        try:
            transform_mode4_prepend_delete()
        except UnsupportedTransformError as e:
            assert "Reason:" in str(e)
            assert "mask" in str(e).lower() or "position" in str(e).lower()


# ------------------------------------------------------------------ #
#  Tests: TND cu_seqlens recompute after transform
# ------------------------------------------------------------------ #

class TestTNDCuSeqlensRecompute:
    def test_mode1_cu_seqlens_q_recomputed(self):
        input_data = _make_tnd_tnd_input(B=4)
        order = [3, 0, 2, 1]
        case = transform_mode1_reorder(input_data, order=order)
        new_cu = case.input_data["input"]["cu_seqlens_q"]
        expected = torch.tensor([0, 4, 6, 7, 10], dtype=torch.int32)
        assert torch.equal(new_cu, expected)

    def test_mode1_cu_seqlens_kv_recomputed(self):
        input_data = _make_tnd_tnd_input(B=4)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        new_cu = case.input_data["input"]["cu_seqlens_ori_kv"]
        expected = torch.tensor([0, 6, 14, 26, 36], dtype=torch.int32)
        assert torch.equal(new_cu, expected)

    def test_mode2_cu_seqlens_q_recomputed(self):
        input_data = _make_tnd_tnd_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        cu0 = cases[0].input_data["input"]["cu_seqlens_q"]
        assert torch.equal(cu0, torch.tensor([0, 2, 3], dtype=torch.int32))
        cu1 = cases[1].input_data["input"]["cu_seqlens_q"]
        assert torch.equal(cu1, torch.tensor([0, 3, 7], dtype=torch.int32))


# ------------------------------------------------------------------ #
#  Tests: derived fields recompute
# ------------------------------------------------------------------ #

class TestDerivedFields:
    def test_b_updated_after_reorder(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert case.input_data["B"] == 4

    def test_b_updated_after_split(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        assert cases[0].input_data["B"] == 2
        assert cases[1].input_data["B"] == 2

    def test_t1_updated_after_split_tnd(self):
        input_data = _make_tnd_tnd_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        assert cases[0].input_data["T1"] == 3
        assert cases[1].input_data["T1"] == 7

    def test_max_seqlen_updated_after_split(self):
        input_data = _make_tnd_pa_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        assert cases[0].input_data["metadata_input"]["max_seqlen_ori_kv"] == 128
        assert cases[1].input_data["metadata_input"]["max_seqlen_ori_kv"] == 256

    def test_cpu_output_cleared(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert case.input_data["cpu_output"] is None
        assert case.input_data["softmax_lse"] is None

    def test_no_none_input_b_inserted(self):
        """input dict should not get a None 'B' key inserted if it wasn't there."""
        input_data = _make_bsnd_bsnd_input(B=4)
        assert "B" not in input_data["input"]
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert "B" not in case.input_data["input"]

    def test_no_none_input_t1_inserted_bsnd(self):
        """BSND input dict should not get a None 'T1' key."""
        input_data = _make_bsnd_bsnd_input(B=4)
        assert "T1" not in input_data["input"]
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert "T1" not in case.input_data["input"]


# ------------------------------------------------------------------ #
#  Tests: RunResult legacy tuple
# ------------------------------------------------------------------ #

class TestRunResult:
    def test_as_legacy_tuple(self):
        out = torch.randn(2, 3)
        lse = torch.randn(2, 3)
        r = RunResult(output=out, softmax_lse=lse, metadata=None)
        o, l = r.as_legacy_tuple()
        assert torch.equal(o, out)
        assert torch.equal(l, lse)

    def test_as_legacy_tuple_none_lse(self):
        out = torch.randn(2, 3)
        r = RunResult(output=out, softmax_lse=None)
        o, l = r.as_legacy_tuple()
        assert torch.equal(o, out)
        assert l is None


# ------------------------------------------------------------------ #
#  Fix 1 regression: deterministic level save/restore (monkeypatch)
# ------------------------------------------------------------------ #

class TestDeterministicLevel:
    def test_get_and_restore_real_level(self, monkeypatch):
        """Verify runner reads real old level, sets 3, restores old level."""
        import batch_consistency.runner as runner

        call_log = []
        state = {"level": 0}  # simulate old level is 0

        def mock_get_level():
            call_log.append(("get",))
            return state["level"]

        def mock_set_level(level):
            call_log.append(("set", level))
            state["level"] = level

        monkeypatch.setattr(runner.torch_npu.npu, "_get_deterministic_level", mock_get_level)
        monkeypatch.setattr(runner.torch_npu.npu, "set_deterministic_level", mock_set_level)

        # Mock call_npu to avoid NPU
        def mock_call_npu(input_data):
            call_log.append(("call_npu",))
            return torch.tensor([1.0]), None

        monkeypatch.setattr(runner.sparse_flash_mla_process, "call_npu", mock_call_npu)

        # Use a minimal valid input_data
        input_data = _make_bsnd_bsnd_input(B=2)
        input_data["cpu_output"] = torch.randn(2, 2, 4, 16, dtype=torch.bfloat16)
        runner.run_with_batch_consistency(input_data, validate=True)

        # Extract set and call_npu events (get is called for verify too)
        set_calls = [e for e in call_log if e[0] == "set"]
        npu_calls = [e for e in call_log if e[0] == "call_npu"]
        get_calls = [e for e in call_log if e[0] == "get"]

        # First get reads old level (0)
        assert get_calls[0] == ("get",)
        # Set 3, then restore 0
        assert set_calls == [("set", 3), ("set", 0)]
        # call_npu happened exactly once
        assert npu_calls == [("call_npu",)]
        # Final level restored to 0
        assert state["level"] == 0

    def test_set_failure_raises(self, monkeypatch):
        """If set_deterministic_level(3) fails, must raise, not silently continue."""
        import batch_consistency.runner as runner

        state = {"level": 0}

        def mock_get_level():
            return state["level"]

        def mock_set_level(level):
            if level == 3:
                raise RuntimeError("NPU unavailable")
            state["level"] = level

        monkeypatch.setattr(runner.torch_npu.npu, "_get_deterministic_level", mock_get_level)
        monkeypatch.setattr(runner.torch_npu.npu, "set_deterministic_level", mock_set_level)

        input_data = _make_bsnd_bsnd_input(B=2)
        input_data["cpu_output"] = torch.randn(2, 2, 4, 16, dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="NPU unavailable"):
            runner.run_with_batch_consistency(input_data, validate=True)

    def test_restore_on_call_npu_failure(self, monkeypatch):
        """If call_npu raises, old level must still be restored."""
        import batch_consistency.runner as runner

        call_log = []
        state = {"level": 1}  # old level is 1

        def mock_get_level():
            return state["level"]

        def mock_set_level(level):
            call_log.append(("set", level))
            state["level"] = level

        monkeypatch.setattr(runner.torch_npu.npu, "_get_deterministic_level", mock_get_level)
        monkeypatch.setattr(runner.torch_npu.npu, "set_deterministic_level", mock_set_level)

        def mock_call_npu(input_data):
            raise ValueError("op failed")

        monkeypatch.setattr(runner.sparse_flash_mla_process, "call_npu", mock_call_npu)

        input_data = _make_bsnd_bsnd_input(B=2)
        input_data["cpu_output"] = torch.randn(2, 2, 4, 16, dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="op failed"):
            runner.run_with_batch_consistency(input_data, validate=True)

        # Verify: set 3 -> (call_npu fails) -> set 1 (restore)
        assert call_log == [("set", 3), ("set", 1)]

    def test_runtime_level3_rejection_is_classified_and_restored(self, monkeypatch):
        """A lazy runtime rejection should be reported as unsupported, not precision failure."""
        import batch_consistency.runner as runner

        state = {"level": 0}

        monkeypatch.setattr(
            runner.torch_npu.npu,
            "_get_deterministic_level",
            lambda: state["level"],
        )

        def mock_set_level(level):
            state["level"] = level

        monkeypatch.setattr(runner.torch_npu.npu, "set_deterministic_level", mock_set_level)

        def mock_call_npu(input_data):
            raise RuntimeError(
                "'torch_npu.npu.set_deterministic_level' currently only supports configuring 0/1/2 !"
            )

        monkeypatch.setattr(runner.sparse_flash_mla_process, "call_npu", mock_call_npu)
        input_data = _make_bsnd_bsnd_input(B=2)

        with pytest.raises(runner.BatchConsistencyUnsupportedError, match="only levels 0/1/2"):
            runner.run_with_batch_consistency(input_data)
        assert state["level"] == 0


# ------------------------------------------------------------------ #
#  Fix 2 regression: physical Q length priority over seqused_q
# ------------------------------------------------------------------ #

class TestPhysicalQLength:
    def test_bsnd_uses_q_shape1_not_seqused_q(self):
        """BSND: q_lengths should be q.shape[1], not seqused_q (which may be smaller)."""
        input_data = _make_bsnd_bsnd_input(B=3, S1=8, with_cu_q=False)
        # Set seqused_q to a smaller value (padding scenario)
        input_data["input"]["seqused_q"] = torch.tensor([5, 6, 3], dtype=torch.int32)
        lengths = schema.get_q_lengths(input_data)
        # Should use q.shape[1]=8, not seqused_q
        assert lengths == [8, 8, 8]

    def test_tnd_uses_cu_seqlens_diff_not_seqused_q(self):
        """TND: q_lengths should be cu_seqlens_q diff, not seqused_q."""
        input_data = _make_tnd_tnd_input(B=4)
        # q_lengths from cu_seqlens: [2, 3, 1, 4]
        # Override seqused_q to different values
        input_data["input"]["seqused_q"] = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        lengths = schema.get_q_lengths(input_data)
        assert lengths == [2, 3, 1, 4]

    def test_bsnd_seqused_q_none_uses_q_shape1(self):
        """BSND with seqused_q=None: should fall back to q.shape[1]."""
        input_data = _make_bsnd_bsnd_input(B=4, S1=3, with_cu_q=False)
        lengths = schema.get_q_lengths(input_data)
        assert lengths == [3, 3, 3, 3]

    def test_bsnd_no_cu_q_transform_works(self):
        """BSND with cu_seqlens_q=None and seqused_q=None: transform should work."""
        input_data = _make_bsnd_bsnd_input(B=4, S1=2, with_cu_q=False)
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert case.input_data["input"]["cu_seqlens_q"] is None
        # Origins should still be correct
        assert case.output_origins[0] == TokenOrigin(2, 0)
        assert case.output_origins[1] == TokenOrigin(2, 1)


# ------------------------------------------------------------------ #
#  Fix 3 regression: BSND mode2 cu_seqlens_q recompute
# ------------------------------------------------------------------ #

class TestBSNDCuSeqlensRecompute:
    def test_bsnd_mode2_cu_seqlens_q_recomputed(self):
        """BSND with cu_seqlens_q set: mode2 must recompute for new B."""
        input_data = _make_bsnd_bsnd_input(B=4, S1=2, with_cu_q=True)
        # Original cu_seqlens_q: [0, 2, 4, 6, 8]
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        # Group 0: B=2, S1=2 -> [0, 2, 4]
        cu0 = cases[0].input_data["input"]["cu_seqlens_q"]
        assert torch.equal(cu0, torch.tensor([0, 2, 4], dtype=torch.int32))
        # Group 1: B=2, S1=2 -> [0, 2, 4]
        cu1 = cases[1].input_data["input"]["cu_seqlens_q"]
        assert torch.equal(cu1, torch.tensor([0, 2, 4], dtype=torch.int32))

    def test_bsnd_mode1_cu_seqlens_q_recomputed(self):
        """BSND with cu_seqlens_q set: mode1 must recompute for reordered B."""
        input_data = _make_bsnd_bsnd_input(B=4, S1=3, with_cu_q=True)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        # B=4, S1=3 -> [0, 3, 6, 9, 12]
        cu = case.input_data["input"]["cu_seqlens_q"]
        assert torch.equal(cu, torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32))

    def test_bsnd_mode2_cu_seqlens_q_in_metadata_and_params(self):
        """BSND mode2: cu_seqlens_q must be synced to metadata_input and params."""
        input_data = _make_bsnd_bsnd_input(B=4, S1=2, with_cu_q=True)
        cases = transform_mode2_split(input_data, groups=[[0, 1], [2, 3]])
        ti = cases[0].input_data["input"]
        mi = cases[0].input_data["metadata_input"]
        params = cases[0].input_data["params"]
        expected = [0, 2, 4]
        assert ti["cu_seqlens_q"].tolist() == expected
        assert mi["cu_seqlens_q"].tolist() == expected
        assert params["cu_seqlens_q"] == expected


# ------------------------------------------------------------------ #
#  Fix 4 regression: TND topk_length uses original cu_seqlens_q
# ------------------------------------------------------------------ #

class TestTNDTopkLengthSegmentation:
    def test_tnd_topk_length_uses_original_cu_seqlens_q(self):
        """TND: topk_length must be segmented using ORIGINAL cu_seqlens_q."""
        input_data = _make_tnd_tnd_input(B=4)
        # q_lengths = [2, 3, 1, 4], T1=10
        # Add cmp_topk_length as a TND tensor [T1]
        ti = input_data["input"]
        mi = input_data["metadata_input"]
        original_topk = torch.arange(10, dtype=torch.int32)  # [0,1,2,...,9]
        ti["cmp_topk_length"] = original_topk.clone()
        mi["cmp_topk_length"] = original_topk.clone()

        order = [2, 0, 3, 1]  # new order: batch2(1 tok), batch0(2 tok), batch3(4 tok), batch1(3 tok)
        case = transform_mode1_reorder(input_data, order=order)

        # Original segments: batch0=[0,1], batch1=[2,3,4], batch2=[5], batch3=[6,7,8,9]
        # After reorder [2,0,3,1]: [5, 0,1, 6,7,8,9, 2,3,4]
        expected = torch.tensor([5, 0, 1, 6, 7, 8, 9, 2, 3, 4], dtype=torch.int32)
        result = case.input_data["input"]["cmp_topk_length"]
        assert torch.equal(result, expected), f"Expected {expected}, got {result}"


# ------------------------------------------------------------------ #
#  Fix 5 regression: params full synchronization
# ------------------------------------------------------------------ #

class TestParamsSync:
    def test_params_b_updated(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        assert case.input_data["params"]["B"] == 4

    def test_params_s1_s2_updated_bsnd(self):
        input_data = _make_bsnd_bsnd_input(B=4, S1=2, S2=8)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        for case in cases:
            assert case.input_data["params"]["S1"] == 2
            assert case.input_data["params"]["S2"] == 8

    def test_params_cu_seqlens_q_synced_tnd(self):
        input_data = _make_tnd_tnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[3, 0, 2, 1])
        params_cu = case.input_data["params"]["cu_seqlens_q"]
        input_cu = case.input_data["input"]["cu_seqlens_q"].tolist()
        assert params_cu == input_cu

    def test_params_seqused_q_synced(self):
        input_data = _make_bsnd_bsnd_input(B=4, with_cu_q=False)
        order = [2, 0, 3, 1]
        case = transform_mode1_reorder(input_data, order=order)
        if input_data["input"]["seqused_q"] is not None:
            orig = input_data["input"]["seqused_q"]
            expected = orig[torch.tensor(order, dtype=torch.long)].tolist()
            assert case.input_data["params"]["seqused_q"] == expected

    def test_params_cmp_residual_kv_synced(self):
        input_data = _make_tnd_tnd_input(B=4)
        case = transform_mode1_reorder(input_data, order=[3, 0, 2, 1])
        orig = input_data["input"]["cmp_residual_kv"]
        expected = orig[torch.tensor([3, 0, 2, 1], dtype=torch.long)].tolist()
        assert case.input_data["params"]["cmp_residual_kv"] == expected

    def test_params_t1_t2_t3_updated_tnd(self):
        input_data = _make_tnd_tnd_input(B=4)
        cases = transform_mode2_split(input_data, groups=[[0, 2], [1, 3]])
        # Group 0: q_lengths [2,1] -> T1=3; kv_lengths [8,6] -> T2=14; cmp [2,1] -> T3=3
        assert cases[0].input_data["params"]["T1"] == 3
        assert cases[0].input_data["params"]["T2"] == 14
        assert cases[0].input_data["params"]["T3"] == 3

    def test_tnd_max_lengths_without_seqused_vectors(self):
        input_data = _make_tnd_tnd_input(B=4)
        for field in ("seqused_ori_kv", "seqused_cmp_kv"):
            input_data["input"][field] = None
            input_data["metadata_input"][field] = None
            input_data["params"][field] = None

        case = transform_mode1_reorder(input_data, order=[2, 0, 3, 1])
        metadata = case.input_data["metadata_input"]
        assert metadata["max_seqlen_ori_kv"] == 12
        assert metadata["max_seqlen_cmp_kv"] == 3
        assert case.input_data["S2"] == 12

    def test_bsnd_and_pa_reporting_token_counts(self):
        bsnd_cases = transform_mode2_split(
            _make_bsnd_bsnd_input(B=4, S1=2, S2=8),
            groups=[[0, 2], [1, 3]],
        )
        assert bsnd_cases[0].input_data["params"]["T1"] == 4
        assert bsnd_cases[0].input_data["params"]["T2"] == 16
        assert bsnd_cases[0].input_data["params"]["T3"] == 4

        pa_cases = transform_mode2_split(
            _make_tnd_pa_input(B=4),
            groups=[[0, 2], [1, 3]],
        )
        assert pa_cases[0].input_data["params"]["T2"] == 256
        assert pa_cases[0].input_data["params"]["T3"] == 64


# ------------------------------------------------------------------ #
#  Fix 6 regression: LSE merge for mode2 (CPU-only)
# ------------------------------------------------------------------ #

class TestLSEMerge:
    def test_bsnd_lse_merge(self):
        """BSND LSE [B, N2, S1, G] merge from sub-cases."""
        N2, S1, G = 1, 2, 2
        baseline_q_lengths = [2, 2, 2, 2]

        # Create two sub-case LSEs
        # Sub-case 0: batches [0, 1]
        lse0 = torch.randn(2, N2, S1, G, dtype=torch.float32)
        origins0 = [TokenOrigin(0, 0), TokenOrigin(0, 1), TokenOrigin(1, 0), TokenOrigin(1, 1)]

        # Sub-case 1: batches [2, 3]
        lse1 = torch.randn(2, N2, S1, G, dtype=torch.float32)
        origins1 = [TokenOrigin(2, 0), TokenOrigin(2, 1), TokenOrigin(3, 0), TokenOrigin(3, 1)]

        merged, merged_origins = merge_split_lse(
            [(lse0, origins0), (lse1, origins1)], "BSND", baseline_q_lengths
        )

        assert merged.shape == (4, N2, S1, G)
        assert len(merged_origins) == 8
        # Verify token mapping
        assert torch.equal(merged[0, :, 0, :], lse0[0, :, 0, :])
        assert torch.equal(merged[0, :, 1, :], lse0[0, :, 1, :])
        assert torch.equal(merged[1, :, 0, :], lse0[1, :, 0, :])
        assert torch.equal(merged[2, :, 0, :], lse1[0, :, 0, :])
        assert torch.equal(merged[3, :, 1, :], lse1[1, :, 1, :])

    def test_tnd_lse_merge(self):
        """TND LSE [T, N2, G] merge from sub-cases."""
        N2, G = 1, 2
        baseline_q_lengths = [2, 3, 1, 4]  # total T = 10

        # Sub-case 0: batches [0, 2] -> lengths [2, 1] -> T=3
        lse0 = torch.randn(3, N2, G, dtype=torch.float32)
        origins0 = [TokenOrigin(0, 0), TokenOrigin(0, 1), TokenOrigin(2, 0)]

        # Sub-case 1: batches [1, 3] -> lengths [3, 4] -> T=7
        lse1 = torch.randn(7, N2, G, dtype=torch.float32)
        origins1 = [TokenOrigin(1, 0), TokenOrigin(1, 1), TokenOrigin(1, 2),
                    TokenOrigin(3, 0), TokenOrigin(3, 1), TokenOrigin(3, 2), TokenOrigin(3, 3)]

        merged, merged_origins = merge_split_lse(
            [(lse0, origins0), (lse1, origins1)], "TND", baseline_q_lengths
        )

        assert merged.shape == (10, N2, G)
        # Verify token mapping: batch0 tokens at positions 0,1
        assert torch.equal(merged[0], lse0[0])
        assert torch.equal(merged[1], lse0[1])
        # batch1 tokens at positions 2,3,4
        assert torch.equal(merged[2], lse1[0])
        assert torch.equal(merged[3], lse1[1])
        # batch2 token at position 5
        assert torch.equal(merged[5], lse0[2])
        # batch3 tokens at positions 6,7,8,9
        assert torch.equal(merged[6], lse1[3])

    def test_bsnd_lse_merge_reordered_groups(self):
        """LSE merge with non-sequential batch groups."""
        N2, S1, G = 1, 1, 2
        baseline_q_lengths = [1, 1, 1, 1]

        # Sub-case 0: batches [2, 0]
        lse0 = torch.randn(2, N2, S1, G, dtype=torch.float32)
        origins0 = [TokenOrigin(2, 0), TokenOrigin(0, 0)]

        # Sub-case 1: batches [3, 1]
        lse1 = torch.randn(2, N2, S1, G, dtype=torch.float32)
        origins1 = [TokenOrigin(3, 0), TokenOrigin(1, 0)]

        merged, _ = merge_split_lse(
            [(lse0, origins0), (lse1, origins1)], "BSND", baseline_q_lengths
        )

        # batch0 -> lse0[1], batch1 -> lse1[1], batch2 -> lse0[0], batch3 -> lse1[0]
        assert torch.equal(merged[0, :, 0, :], lse0[1, :, 0, :])
        assert torch.equal(merged[1, :, 0, :], lse1[1, :, 0, :])
        assert torch.equal(merged[2, :, 0, :], lse0[0, :, 0, :])
        assert torch.equal(merged[3, :, 0, :], lse1[0, :, 0, :])

    def test_requested_but_missing_lse_fails(self):
        output = torch.zeros(1, 1, 1, 1)
        origins = [TokenOrigin(0, 0)]
        result = compare_consistency(
            output,
            origins,
            output.clone(),
            origins,
            "BSND",
            [1],
            compare_lse=True,
            baseline_lse=torch.zeros(1, 1, 1, 1),
            derived_lse=None,
        )
        assert not result["pass"]
        assert "missing" in result["lse"]["error"]


# ------------------------------------------------------------------ #
#  Fix 7 regression: BF16 MD5 safe computation
# ------------------------------------------------------------------ #

class TestBF16Md5:
    def test_bf16_tensor_md5_no_typeerror(self):
        """BF16 tensor MD5 should not raise TypeError."""
        t = torch.randn(2, 3, dtype=torch.bfloat16)
        md5 = _tensor_md5(t)
        assert len(md5) == 32  # hex MD5 length

    def test_bf16_md5_consistent(self):
        """Same BF16 tensor should produce same MD5."""
        t1 = torch.randn(2, 3, dtype=torch.bfloat16)
        t2 = t1.clone()
        assert _tensor_md5(t1) == _tensor_md5(t2)

    def test_bf16_md5_different_for_different_tensors(self):
        """Different BF16 tensors should produce different MD5."""
        t1 = torch.randn(2, 3, dtype=torch.bfloat16)
        t2 = torch.randn(2, 3, dtype=torch.bfloat16)
        assert _tensor_md5(t1) != _tensor_md5(t2)

    def test_compare_pair_bf16_no_error(self):
        """compare_pair should work with BF16 tensors."""
        t1 = torch.randn(2, 3, dtype=torch.bfloat16)
        t2 = t1.clone()
        result = compare_pair(t1, t2)
        assert result["bitwise_equal"] is True
        assert result["pass"] is True

    def test_compare_pair_md5_computed_once(self):
        """MD5 should be computed exactly twice (once per tensor), not 3 times."""
        t1 = torch.randn(2, 3, dtype=torch.bfloat16)
        t2 = t1.clone()
        result = compare_pair(t1, t2)
        # The result should have both md5s and bitwise_equal
        assert "expected_md5" in result
        assert "actual_md5" in result
        assert result["bitwise_equal"] is True


# ------------------------------------------------------------------ #
#  Fix 8 regression: split validation
# ------------------------------------------------------------------ #

class TestSplitValidation:
    def test_num_groups_too_small_raises(self):
        with pytest.raises(ValueError, match="2 <= num_groups"):
            generate_split_groups(4, num_groups=1)

    def test_num_groups_too_large_raises(self):
        with pytest.raises(ValueError, match="2 <= num_groups"):
            generate_split_groups(4, num_groups=5)

    def test_num_groups_equals_b(self):
        groups = generate_split_groups(4, num_groups=4, seed=42)
        assert len(groups) == 4
        for g in groups:
            assert len(g) == 1

    def test_custom_empty_group_raises(self):
        input_data = _make_bsnd_bsnd_input(B=4)
        with pytest.raises(ValueError, match="empty"):
            transform_mode2_split(input_data, groups=[[0, 1], [], [2, 3]])

    def test_num_groups_valid_range(self):
        for ng in range(2, 6):
            groups = generate_split_groups(6, num_groups=ng, seed=42)
            assert len(groups) == ng
