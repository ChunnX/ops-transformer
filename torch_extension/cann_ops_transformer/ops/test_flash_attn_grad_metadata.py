# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Test script for FAG (Flash Attn Grad) metadata verification.

This script:
1. Calls flash_attn_metadata via torch interface to generate the metadata tensor.
2. Extracts FAG region from the metadata tensor (using head[4] offset).
3. Independently computes expected FAG split results in Python.
4. Compares AICPU output vs Python reference for each field.

Usage:
    python -m pytest test_flash_attn_grad_metadata.py -v
    # or standalone:
    python test_flash_attn_grad_metadata.py
"""

import math
import pytest
import torch
import torch_npu
import cann_ops_transformer

# ============================================================
# FAG metadata field indices (must match flash_attn_metadata.h)
# ============================================================
HEAD_FAG_START_OFFSET_INDEX = 4
FAG_SPLIT_AXIS_INDEX = 0
FAG_BLOCK_OUTER_INDEX = 1
FAG_BLOCK_FACTOR_INDEX = 2
FAG_S1_OUTER_INDEX = 3
FAG_S2_OUTER_INDEX = 4
FAG_LAYOUT_TYPE_INDEX = 5
FAG_IS_SPARSE_INDEX = 6
FAG_SPARSE_MODE_INDEX = 7
FAG_BLOCK_STARTS_OFFSET = 8
FAG_BLOCK_ENDS_OFFSET = 44
FAG_TND_START_BIDX_OFFSET = 80
FAG_CORE_LIST_NUM = 36

# FAG internal constants
S1CV_RATIO = 2
S2CV_RATIO = 1
ALIGN64 = 64

# Layout type values
LAYOUT_BSND = 1
LAYOUT_BNSD = 3
LAYOUT_TND = 4

# Sparse mode values
SPARSE_NO_MASK = 0
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4


# ============================================================
# Python reference implementation of FAG split logic
# ============================================================

def _align_to(value, alignment):
    if alignment <= 0:
        return value
    return (value + alignment - 1) // alignment * alignment


def _ref_dense_split(b, n2, g, s1_outer, s2_outer, aic_num):
    """Reference: DoFagDenseSplit (normal_regbase.cpp:701-723)"""
    fused_outer = b * n2 * g * s1_outer * s2_outer
    block_factor = (fused_outer + aic_num - 1) // aic_num
    block_outer = (fused_outer + block_factor - 1) // block_factor
    starts = [0] * FAG_CORE_LIST_NUM
    ends = [0] * FAG_CORE_LIST_NUM
    for i in range(block_outer):
        starts[i] = block_factor * i
        ends[i] = min(block_factor * (i + 1), fused_outer)
    return starts, ends, block_outer, block_factor


def _ref_get_parse(s1, s2, s1_cv_inner, cv_s2_inner, s2_outer, s1_token, s2_token):
    """Reference: GetFagParseS1S2OuterInfo (normal_regbase.cpp:1509-1551)"""
    parse_info = []
    for i in range(s2_outer):
        left_int = max(0, cv_s2_inner * i - s2_token)
        if left_int > s1:
            begin = (s1 + s1_cv_inner - 1) // s1_cv_inner
        else:
            begin = left_int // s1_cv_inner
        cv_block_tail = (s2 % cv_s2_inner == 0 if cv_s2_inner else 1) * cv_s2_inner if i == s2_outer - 1 else cv_s2_inner
        if i == s2_outer - 1:
            cv_block_tail = cv_s2_inner if (s2 % cv_s2_inner == 0) else (s2 % cv_s2_inner)
        right_point = min(max(0, cv_s2_inner * i + cv_block_tail + s1_token), s1)
        end = (right_point + s1_cv_inner - 1) // s1_cv_inner
        tmp_size = max(0, end - begin) if end > begin else 0
        parse_info.append((begin, end, tmp_size))
    # Compute cumulative length
    cum = 0
    for i in range(len(parse_info)):
        cum += parse_info[i][2]
        parse_info[i] = (parse_info[i][0], parse_info[i][1], cum)
    return parse_info


def _ref_sparse_block_info(b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
                           s1_outer, s2_outer, s1_token, s2_token, aic_num):
    """Reference: DoFagSparseBlockInfo (normal_regbase.cpp:1553-1622)"""
    parse_info = _ref_get_parse(s1, s2, s1_cv_inner, cv_s2_inner, s2_outer, s1_token, s2_token)
    s1s2o_count = parse_info[s2_outer - 1][2] if s2_outer > 0 else 0

    fused_outer = b * n2 * g * s1s2o_count
    if fused_outer == 0:
        return [0]*FAG_CORE_LIST_NUM, [0]*FAG_CORE_LIST_NUM, 0, 0
    block_factor = (fused_outer + aic_num - 1) // aic_num
    block_outer = (fused_outer + block_factor - 1) // block_factor

    n2gs = n2 * g * s1s2o_count
    gs = g * s1s2o_count

    starts = [0] * FAG_CORE_LIST_NUM
    ends = [0] * FAG_CORE_LIST_NUM
    starts[0] = 0
    ends[block_outer - 1] = b * n2 * g * s1_outer * s2_outer

    for c in range(1, block_outer):
        current_idx = min(c * block_factor, fused_outer)
        b_idx = current_idx // n2gs
        b_tail = current_idx % n2gs
        n2_idx = b_tail // gs
        n2_tail = b_tail % gs
        g_idx = n2_tail // s1s2o_count
        g_tail = n2_tail % s1s2o_count

        # Reverse map
        s1o_idx = 0
        s2o_idx = 0
        pre_size = 0
        for i in range(s2_outer):
            next_size = parse_info[i][2]
            if g_tail >= pre_size and g_tail < next_size:
                s2o_idx = i
                s1o_idx = parse_info[i][0] + g_tail - pre_size - 1
                break
            pre_size = next_size

        starts[c] = (((b_idx * n2 + n2_idx) * g + g_idx) * s2_outer + s2o_idx) * s1_outer + s1o_idx + 1
        ends[c - 1] = starts[c]

    return starts, ends, block_outer, block_factor


def _compute_fag_reference(batch_size, num_heads_q, num_heads_kv, head_dim,
                           max_seqlen_q, max_seqlen_kv,
                           mask_mode, win_left, win_right,
                           layout_q, aic_core_num=36):
    """
    Compute expected FAG metadata values in Python.
    Returns dict with all FAG fields.
    """
    s1 = max_seqlen_q
    s2 = max_seqlen_kv
    n1 = num_heads_q
    n2 = num_heads_kv
    g = n1 // n2
    b = batch_size

    # s1Inner=64, s2Inner=128 (fixed for FP16/BF16)
    s1_inner = 64
    s2_inner = 128
    s1_cv_inner = min(s1_inner * S1CV_RATIO, s1)
    cv_s2_inner = min(s2_inner * S2CV_RATIO, s2)
    s1_outer = (s1 + s1_cv_inner - 1) // s1_cv_inner
    s2_outer = (s2 + cv_s2_inner - 1) // cv_s2_inner

    # Layout type
    if layout_q == "TND":
        layout_type = LAYOUT_TND
    elif layout_q == "BNSD":
        layout_type = LAYOUT_BNSD
    else:
        layout_type = LAYOUT_BSND

    # Sparse mode and token mapping
    sparse_mode = SPARSE_NO_MASK
    is_sparse = False
    s1_token = win_left
    s2_token = win_right

    if mask_mode == 0:
        is_sparse = False
    elif mask_mode == 3:
        sparse_mode = SPARSE_RIGHT_DOWN_CAUSAL
        is_sparse = True
        s1_token = 2147483647  # INT32_MAX
        s2_token = 0
        if layout_type != LAYOUT_TND:
            s1_token = s1_token + s1 - s2
            s2_token = s2_token - s1 + s2
    elif mask_mode == 4:
        sparse_mode = SPARSE_BAND
        if s1 > s1_token or s2 > s2_token:
            is_sparse = True
        if layout_type != LAYOUT_TND:
            s1_token = s1_token + s1 - s2
            s2_token = s2_token - s1 + s2

    # Dispatch to appropriate split function
    if layout_type == LAYOUT_TND:
        # TND not fully implemented in reference; return basic checks
        starts, ends, block_outer, block_factor = [0]*FAG_CORE_LIST_NUM, [0]*FAG_CORE_LIST_NUM, 0, 0
    elif is_sparse:
        starts, ends, block_outer, block_factor = _ref_sparse_block_info(
            b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
            s1_outer, s2_outer, s1_token, s2_token, aic_core_num
        )
    else:
        starts, ends, block_outer, block_factor = _ref_dense_split(
            b, n2, g, s1_outer, s2_outer, aic_core_num
        )

    return {
        'split_axis': 0,  # BN2GS1S2
        'block_outer': block_outer,
        'block_factor': block_factor,
        's1_outer': s1_outer,
        's2_outer': s2_outer,
        'layout_type': layout_type,
        'is_sparse': 1 if is_sparse else 0,
        'sparse_mode': sparse_mode,
        'block_starts': starts,
        'block_ends': ends,
        'tnd_start_bidx': [0] * FAG_CORE_LIST_NUM,
    }


def _extract_fag_from_metadata(metadata_tensor):
    """
    Extract FAG region from metadata tensor.
    Returns dict with all FAG fields read from the tensor.
    """
    meta = metadata_tensor.cpu().tolist()

    # Read FAG start offset from head[4]
    fag_offset = meta[HEAD_FAG_START_OFFSET_INDEX]

    if fag_offset == 0:
        return None  # FAG not populated

    fag = meta[fag_offset:]

    result = {
        'split_axis': fag[FAG_SPLIT_AXIS_INDEX],
        'block_outer': fag[FAG_BLOCK_OUTER_INDEX],
        'block_factor': fag[FAG_BLOCK_FACTOR_INDEX],
        's1_outer': fag[FAG_S1_OUTER_INDEX],
        's2_outer': fag[FAG_S2_OUTER_INDEX],
        'layout_type': fag[FAG_LAYOUT_TYPE_INDEX],
        'is_sparse': fag[FAG_IS_SPARSE_INDEX],
        'sparse_mode': fag[FAG_SPARSE_MODE_INDEX],
        'block_starts': fag[FAG_BLOCK_STARTS_OFFSET:FAG_BLOCK_STARTS_OFFSET + FAG_CORE_LIST_NUM],
        'block_ends': fag[FAG_BLOCK_ENDS_OFFSET:FAG_BLOCK_ENDS_OFFSET + FAG_CORE_LIST_NUM],
        'tnd_start_bidx': fag[FAG_TND_START_BIDX_OFFSET:FAG_TND_START_BIDX_OFFSET + FAG_CORE_LIST_NUM],
        'raw_fag_offset': fag_offset,
    }
    return result


def _compare_fag(actual, expected, test_name):
    """Compare actual FAG metadata vs expected. Returns list of (field, passed, detail)."""
    results = []
    int_fields = ['split_axis', 'block_outer', 'block_factor', 's1_outer', 's2_outer',
                  'layout_type', 'is_sparse', 'sparse_mode']

    for field in int_fields:
        a = actual[field]
        e = expected[field]
        ok = (a == e)
        results.append((field, ok, f"got={a} expected={e}"))

    for field in ['block_starts', 'block_ends', 'tnd_start_bidx']:
        a_list = actual[field]
        e_list = expected[field]
        for i in range(FAG_CORE_LIST_NUM):
            a = a_list[i]
            e = e_list[i]
            ok = (a == e)
            if not ok:
                results.append((f"{field}[{i}]", False, f"got={a} expected={e}"))
        # If all matched, add a summary pass
        all_ok = all(a_list[i] == e_list[i] for i in range(FAG_CORE_LIST_NUM))
        results.append((f"{field}_all", all_ok, f"{sum(1 for i in range(FAG_CORE_LIST_NUM) if a_list[i]==e_list[i])}/{FAG_CORE_LIST_NUM} matched"))

    return results


# ============================================================
# Test cases
# ============================================================

class TestFagMetadata:
    """FAG metadata verification tests."""

    def _run_and_check(self, batch_size, num_heads_q, num_heads_kv, head_dim,
                       max_seqlen_q, max_seqlen_kv, mask_mode=0,
                       win_left=-1, win_right=-1,
                       layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                       aic_core_num=36):
        """Helper: call flash_attn_metadata and verify FAG region."""
        # Call flash_attn_metadata
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_out=layout_out,
        )
        assert metadata is not None, "metadata tensor is None"
        assert metadata.dtype == torch.int32, f"metadata dtype={metadata.dtype}, expected int32"

        # Extract FAG region
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None, "FAG region not populated (head[4]==0)"

        # Compute expected
        expected = _compute_fag_reference(
            batch_size, num_heads_q, num_heads_kv, head_dim,
            max_seqlen_q, max_seqlen_kv, mask_mode, win_left, win_right,
            layout_q, aic_core_num
        )

        # Compare
        results = _compare_fag(actual, expected, "test")
        failures = [(f, d) for f, ok, d in results if not ok]

        if failures:
            msg = "\n".join(f"  FAIL {f}: {d}" for f, d in failures)
            pytest.fail(f"FAG metadata mismatch:\n{msg}")

        return actual, expected

    def test_dense_bsnd_small(self):
        """Dense BSND: s1=128, s2=128, b=4, n2=4, mask_mode=0."""
        self._run_and_check(
            batch_size=4, num_heads_q=8, num_heads_kv=4, head_dim=128,
            max_seqlen_q=128, max_seqlen_kv=128, mask_mode=0,
        )

    def test_dense_bsnd_s1_neq_s2(self):
        """Dense BSND: s1=128, s2=256 (asymmetric)."""
        self._run_and_check(
            batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
            max_seqlen_q=128, max_seqlen_kv=256, mask_mode=0,
        )

    def test_dense_bsnd_large(self):
        """Dense BSND large: s1=1024, s2=1024 (full 36 cores)."""
        self._run_and_check(
            batch_size=4, num_heads_q=8, num_heads_kv=4, head_dim=128,
            max_seqlen_q=1024, max_seqlen_kv=1024, mask_mode=0,
        )

    def test_causal_bsnd(self):
        """Causal BSND: mask_mode=3, s1=256, s2=256."""
        self._run_and_check(
            batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
            max_seqlen_q=256, max_seqlen_kv=256, mask_mode=3,
        )

    def test_causal_bsnd_s1_neq_s2(self):
        """Causal BSND: mask_mode=3, s1=128, s2=256 (pad token correction)."""
        self._run_and_check(
            batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
            max_seqlen_q=128, max_seqlen_kv=256, mask_mode=3,
        )

    def test_band_bsnd(self):
        """Band BSND: mask_mode=4, win_left=128, win_right=128, s1=512, s2=512."""
        self._run_and_check(
            batch_size=1, num_heads_q=2, num_heads_kv=1, head_dim=128,
            max_seqlen_q=512, max_seqlen_kv=512, mask_mode=4,
            win_left=128, win_right=128,
        )

    def test_band_bsnd_s1_neq_s2(self):
        """Band BSND: mask_mode=4, s1=256, s2=512 (pad token correction)."""
        self._run_and_check(
            batch_size=2, num_heads_q=2, num_heads_kv=1, head_dim=128,
            max_seqlen_q=256, max_seqlen_kv=512, mask_mode=4,
            win_left=64, win_right=64,
        )

    def test_bnsd_layout(self):
        """Dense BNSD: layout_q=BNSD."""
        self._run_and_check(
            batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
            max_seqlen_q=128, max_seqlen_kv=128, mask_mode=0,
            layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD",
        )

    def test_fag_offset_nonzero(self):
        """Verify head[4] (FAG offset) is non-zero and points past FA data."""
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        meta = metadata.cpu().tolist()
        fag_offset = meta[HEAD_FAG_START_OFFSET_INDEX]
        assert fag_offset > 0, f"FAG offset is 0, FAG not populated"

        # FA data ends at 16 + 1728 * sectionNum
        # sectionNum is in head[0]
        section_num = meta[0]
        expected_min_offset = 16 + 1728 * section_num
        assert fag_offset >= expected_min_offset, \
            f"FAG offset {fag_offset} < expected min {expected_min_offset} (16 + 1728*{section_num})"

    def test_fag_block_starts_zero_init(self):
        """Verify blockStarts[0] == 0 (first core always starts at 0)."""
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None
        assert actual['block_starts'][0] == 0, \
            f"blockStarts[0]={actual['block_starts'][0]}, expected 0"

    def test_fag_block_ends_nonzero(self):
        """Verify blockEnds[blockOuter-1] > 0 (last core has work)."""
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None
        block_outer = actual['block_outer']
        assert block_outer > 0, f"blockOuter={block_outer}, expected > 0"
        last_end = actual['block_ends'][block_outer - 1]
        assert last_end > 0, f"blockEnds[{block_outer-1}]={last_end}, expected > 0"

    def test_fag_unused_cores_zero(self):
        """Verify blockStarts[blockOuter..35] == 0 (unused cores zeroed)."""
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None
        block_outer = actual['block_outer']
        for i in range(block_outer, FAG_CORE_LIST_NUM):
            assert actual['block_starts'][i] == 0, \
                f"blockStarts[{i}]={actual['block_starts'][i]}, expected 0 (unused core)"
            assert actual['block_ends'][i] == 0, \
                f"blockEnds[{i}]={actual['block_ends'][i]}, expected 0 (unused core)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
