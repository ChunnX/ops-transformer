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
5. Prints verbose blockStarts/blockEnds values after successful comparison.

Usage:
    python -m pytest test_flash_attn_grad_metadata.py -v -s
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

S1CV_RATIO = 2
S2CV_RATIO = 1

LAYOUT_BSND = 1
LAYOUT_BNSD = 3
LAYOUT_TND = 4

SPARSE_NO_MASK = 0
SPARSE_RIGHT_DOWN_CAUSAL = 3
SPARSE_BAND = 4


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
        lip = max(0, cv_s2_inner * i - s2_token)
        if lip > s1:
            begin = (s1 + s1_cv_inner - 1) // s1_cv_inner
        else:
            begin = lip // s1_cv_inner
        if i == s2_outer - 1:
            cv_block_tail = cv_s2_inner if (s2 % cv_s2_inner == 0) else (s2 % cv_s2_inner)
        else:
            cv_block_tail = cv_s2_inner
        rp = min(max(0, cv_s2_inner * i + cv_block_tail + s1_token), s1)
        end = (rp + s1_cv_inner - 1) // s1_cv_inner
        tmp_size = max(0, end - begin) if end > begin else 0
        parse_info.append((begin, end, tmp_size))
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


def _ref_bn2s2_dense_split(b, n2, g, s1_outer, s2_outer, aic_num):
    """Reference: DoFagBn2s2DenseSplit (normal_regbase.cpp:519-544)"""
    fused_outer = b * n2 * g * s2_outer
    bns2_factor = (fused_outer + aic_num - 1) // aic_num
    block_outer = (fused_outer + bns2_factor - 1) // bns2_factor
    total_block = fused_outer * s1_outer
    block_factor = bns2_factor * s1_outer
    starts = [0] * FAG_CORE_LIST_NUM
    ends = [0] * FAG_CORE_LIST_NUM
    for i in range(block_outer):
        starts[i] = block_factor * i
        ends[i] = min(block_factor * (i + 1), total_block)
    return starts, ends, block_outer, block_factor


def _ref_bn2_dense_split(b, n2, g, s1_outer, s2_outer, aic_num):
    """Reference: DoFagBn2DenseSplit (normal_regbase.cpp:629-655)"""
    fused_outer = b * n2 * g
    bn_factor = (fused_outer + aic_num - 1) // aic_num
    block_outer = (fused_outer + bn_factor - 1) // bn_factor
    block_factor = bn_factor * (s1_outer * s2_outer)
    fused_outer *= (s1_outer * s2_outer)
    starts = [0] * FAG_CORE_LIST_NUM
    ends = [0] * FAG_CORE_LIST_NUM
    for i in range(block_outer):
        starts[i] = block_factor * i
        ends[i] = min(block_factor * (i + 1), fused_outer)
    return starts, ends, block_outer, block_factor


def _ref_bn2_sparse_block_info(b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
                                s1_outer, s2_outer, s1_token, s2_token, aic_num):
    """Reference: DoFagBn2SparseBlockInfo (normal_regbase.cpp:549-619 GetSparseBlockInfoBn2)"""
    parse_info = _ref_get_parse(s1, s2, s1_cv_inner, cv_s2_inner, s2_outer, s1_token, s2_token)
    s1s2o_count = parse_info[s2_outer - 1][2] if s2_outer > 0 else 0
    fused_outer = b * n2 * g
    bn_factor = (fused_outer + aic_num - 1) // aic_num
    block_outer = (fused_outer + bn_factor - 1) // bn_factor
    fused_outer *= s1s2o_count
    block_factor = bn_factor * s1s2o_count
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
    s1 = max_seqlen_q
    s2 = max_seqlen_kv
    n1 = num_heads_q
    n2 = num_heads_kv
    g = n1 // n2
    b = batch_size
    s1_inner = 64
    s2_inner = 128
    s1_cv_inner = min(s1_inner * S1CV_RATIO, s1)
    cv_s2_inner = min(s2_inner * S2CV_RATIO, s2)
    s1_outer = (s1 + s1_cv_inner - 1) // s1_cv_inner
    s2_outer = (s2 + cv_s2_inner - 1) // cv_s2_inner

    if layout_q == "TND":
        layout_type = LAYOUT_TND
    elif layout_q == "BNSD":
        layout_type = LAYOUT_BNSD
    else:
        layout_type = LAYOUT_BSND

    sparse_mode = SPARSE_NO_MASK
    is_sparse = False
    s1_token = win_left
    s2_token = win_right

    if mask_mode == 0:
        is_sparse = False
    elif mask_mode == 3:
        sparse_mode = SPARSE_RIGHT_DOWN_CAUSAL
        is_sparse = True
        s1_token = 2147483647
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

    # SetFagSplitAxis simplified
    is_tnd = (layout_type == LAYOUT_TND)
    is_all_same = True
    is_bn2 = (s1 <= 128 and s2 <= 128) and (n1 == n2) and (head_dim <= 512)
    bn_limit = ((b * n1) >= 256) or ((b * n1) >= 128 and (s1 % 128 == 0) and (s2 % 128 == 0))
    bn_sparse_limit = bn_limit and not is_tnd
    is_bn2_multiblk = bn_sparse_limit and (s1 > 128 or s2 > 128) and (s1 <= 640 and s2 <= 640) and (n1 == n2) and (head_dim <= 512)
    is_bn2 = is_bn2_multiblk if is_bn2_multiblk else is_bn2
    if is_bn2 and not is_bn2_multiblk:
        if is_tnd and head_dim > 128:
            is_bn2 = False

    bn2s2_not_tnd_limit = (s1 < s2) and (s2 <= 1024) and (s2 - s1 >= 128) and (head_dim <= 128) and not is_sparse
    bn2s2_route_limit = (head_dim <= 512) and (is_tnd or is_all_same or bn2s2_not_tnd_limit) and (n1 == n2)

    if not is_bn2 and bn2s2_route_limit:
        split_axis = 5
    elif is_bn2:
        split_axis = 1
    else:
        split_axis = 0

    # Dispatch
    if split_axis == 5:
        if is_sparse or is_tnd:
            starts, ends, block_outer, block_factor = [0]*FAG_CORE_LIST_NUM, [0]*FAG_CORE_LIST_NUM, 0, 0
        else:
            starts, ends, block_outer, block_factor = _ref_bn2s2_dense_split(b, n2, g, s1_outer, s2_outer, aic_core_num)
        if block_outer < aic_core_num:
            split_axis = 0
            if is_bn2 and is_bn2_multiblk:
                if is_sparse:
                    starts, ends, block_outer, block_factor = _ref_bn2_sparse_block_info(
                        b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
                        s1_outer, s2_outer, s1_token, s2_token, aic_core_num)
                    split_axis = 1
                else:
                    starts, ends, block_outer, block_factor = _ref_bn2_dense_split(b, n2, g, s1_outer, s2_outer, aic_core_num)
                    split_axis = 1
            if split_axis == 0:
                if is_tnd:
                    starts, ends, block_outer, block_factor = [0]*FAG_CORE_LIST_NUM, [0]*FAG_CORE_LIST_NUM, 0, 0
                elif is_sparse:
                    starts, ends, block_outer, block_factor = _ref_sparse_block_info(
                        b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
                        s1_outer, s2_outer, s1_token, s2_token, aic_core_num)
                else:
                    starts, ends, block_outer, block_factor = _ref_dense_split(b, n2, g, s1_outer, s2_outer, aic_core_num)
    elif split_axis == 1 and is_bn2_multiblk:
        if is_sparse:
            starts, ends, block_outer, block_factor = _ref_bn2_sparse_block_info(
                b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
                s1_outer, s2_outer, s1_token, s2_token, aic_core_num)
        else:
            starts, ends, block_outer, block_factor = _ref_bn2_dense_split(b, n2, g, s1_outer, s2_outer, aic_core_num)
    else:
        if is_tnd:
            starts, ends, block_outer, block_factor = [0]*FAG_CORE_LIST_NUM, [0]*FAG_CORE_LIST_NUM, 0, 0
        elif is_sparse:
            starts, ends, block_outer, block_factor = _ref_sparse_block_info(
                b, n2, g, s1, s2, s1_inner, s2_inner, s1_cv_inner, cv_s2_inner,
                s1_outer, s2_outer, s1_token, s2_token, aic_core_num)
        else:
            starts, ends, block_outer, block_factor = _ref_dense_split(b, n2, g, s1_outer, s2_outer, aic_core_num)

    return {
        'split_axis': split_axis,
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
    meta = metadata_tensor.cpu().tolist()
    fag_offset = meta[HEAD_FAG_START_OFFSET_INDEX]
    if fag_offset == 0:
        return None
    fag = meta[fag_offset:]
    return {
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


def _print_fag_verbose(label, actual, expected):
    """Print FAG metadata values for debugging."""
    print(f"\n  --- {label} verbose ---")
    print(f"  splitAxis:    actual={actual['split_axis']}  expected={expected['split_axis']}")
    print(f"  blockOuter:   actual={actual['block_outer']}  expected={expected['block_outer']}")
    print(f"  blockFactor:  actual={actual['block_factor']}  expected={expected['block_factor']}")
    print(f"  s1Outer:      actual={actual['s1_outer']}  expected={expected['s1_outer']}")
    print(f"  s2Outer:      actual={actual['s2_outer']}  expected={expected['s2_outer']}")
    print(f"  layoutType:   actual={actual['layout_type']}  expected={expected['layout_type']}")
    print(f"  isSparse:     actual={actual['is_sparse']}  expected={expected['is_sparse']}")
    print(f"  sparseMode:   actual={actual['sparse_mode']}  expected={expected['sparse_mode']}")
    bo = actual['block_outer']
    print(f"  blockStarts (first {bo} active cores):")
    for i in range(min(bo, FAG_CORE_LIST_NUM)):
        a_val = actual['block_starts'][i]
        e_val = expected['block_starts'][i]
        match = "OK" if a_val == e_val else "MISMATCH"
        print(f"    blockStarts[{i:2d}] = {a_val:12d}  (expected={e_val:12d})  {match}")
    print(f"  blockEnds (first {bo} active cores):")
    for i in range(min(bo, FAG_CORE_LIST_NUM)):
        a_val = actual['block_ends'][i]
        e_val = expected['block_ends'][i]
        match = "OK" if a_val == e_val else "MISMATCH"
        print(f"    blockEnds[{i:2d}]   = {a_val:12d}  (expected={e_val:12d})  {match}")
    if bo < FAG_CORE_LIST_NUM:
        all_zero_start = all(actual['block_starts'][i] == 0 for i in range(bo, FAG_CORE_LIST_NUM))
        all_zero_end = all(actual['block_ends'][i] == 0 for i in range(bo, FAG_CORE_LIST_NUM))
        print(f"  unused cores [{bo}..{FAG_CORE_LIST_NUM-1}]: starts_all_zero={all_zero_start} ends_all_zero={all_zero_end}")
    print(f"  --- end verbose ---\n")


class TestFagMetadata:

    def _run_and_check(self, batch_size, num_heads_q, num_heads_kv, head_dim,
                       max_seqlen_q, max_seqlen_kv, mask_mode=0,
                       win_left=-1, win_right=-1,
                       layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                       aic_core_num=36):
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

        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None, "FAG region not populated (head[4]==0)"

        expected = _compute_fag_reference(
            batch_size, num_heads_q, num_heads_kv, head_dim,
            max_seqlen_q, max_seqlen_kv, mask_mode, win_left, win_right,
            layout_q, aic_core_num
        )

        int_fields = ['split_axis', 'block_outer', 'block_factor', 's1_outer', 's2_outer',
                      'layout_type', 'is_sparse', 'sparse_mode']
        failures = []
        for field in int_fields:
            a = actual[field]
            e = expected[field]
            if a != e:
                failures.append((field, f"got={a} expected={e}"))

        for field in ['block_starts', 'block_ends', 'tnd_start_bidx']:
            a_list = actual[field]
            e_list = expected[field]
            for i in range(FAG_CORE_LIST_NUM):
                if a_list[i] != e_list[i]:
                    failures.append((f"{field}[{i}]", f"got={a_list[i]} expected={e_list[i]}"))

        if failures:
            msg = "\n".join(f"  FAIL {f}: {d}" for f, d in failures)
            _print_fag_verbose("ACTUAL (mismatch)", actual, expected)
            pytest.fail(f"FAG metadata mismatch:\n{msg}")

        _print_fag_verbose("MATCHED", actual, expected)
        return actual, expected

    def test_dense_bsnd_small(self):
        self._run_and_check(batch_size=4, num_heads_q=8, num_heads_kv=4, head_dim=128,
                            max_seqlen_q=128, max_seqlen_kv=128, mask_mode=0)

    def test_dense_bsnd_s1_neq_s2(self):
        self._run_and_check(batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
                            max_seqlen_q=128, max_seqlen_kv=256, mask_mode=0)

    def test_dense_bsnd_large(self):
        self._run_and_check(batch_size=4, num_heads_q=8, num_heads_kv=4, head_dim=128,
                            max_seqlen_q=1024, max_seqlen_kv=1024, mask_mode=0)

    def test_causal_bsnd(self):
        self._run_and_check(batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
                            max_seqlen_q=256, max_seqlen_kv=256, mask_mode=3)

    def test_causal_bsnd_s1_neq_s2(self):
        self._run_and_check(batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
                            max_seqlen_q=128, max_seqlen_kv=256, mask_mode=3)

    def test_band_bsnd(self):
        self._run_and_check(batch_size=1, num_heads_q=2, num_heads_kv=1, head_dim=128,
                            max_seqlen_q=512, max_seqlen_kv=512, mask_mode=4,
                            win_left=128, win_right=128)

    def test_band_bsnd_s1_neq_s2(self):
        self._run_and_check(batch_size=2, num_heads_q=2, num_heads_kv=1, head_dim=128,
                            max_seqlen_q=256, max_seqlen_kv=512, mask_mode=4,
                            win_left=64, win_right=64)

    def test_bnsd_layout(self):
        self._run_and_check(batch_size=2, num_heads_q=4, num_heads_kv=2, head_dim=128,
                            max_seqlen_q=128, max_seqlen_kv=128, mask_mode=0,
                            layout_q="BNSD", layout_kv="BNSD", layout_out="BNSD")

    def test_fag_offset_nonzero(self):
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        meta = metadata.cpu().tolist()
        fag_offset = meta[HEAD_FAG_START_OFFSET_INDEX]
        assert fag_offset > 0, f"FAG offset is 0"
        section_num = meta[0]
        expected_min_offset = 16 + 1728 * section_num
        assert fag_offset >= expected_min_offset, \
            f"FAG offset {fag_offset} < expected min {expected_min_offset}"

    def test_fag_block_starts_zero_init(self):
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None
        assert actual['block_starts'][0] == 0, f"blockStarts[0]={actual['block_starts'][0]}"

    def test_fag_block_ends_nonzero(self):
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None
        block_outer = actual['block_outer']
        assert block_outer > 0
        assert actual['block_ends'][block_outer - 1] > 0

    def test_fag_unused_cores_zero(self):
        metadata = cann_ops_transformer.flash_attn_metadata(
            num_heads_q=8, num_heads_kv=4, head_dim=128,
            batch_size=4, max_seqlen_q=128, max_seqlen_kv=128,
            mask_mode=0, layout_q="BSND",
        )
        actual = _extract_fag_from_metadata(metadata)
        assert actual is not None
        block_outer = actual['block_outer']
        for i in range(block_outer, FAG_CORE_LIST_NUM):
            assert actual['block_starts'][i] == 0, f"blockStarts[{i}]={actual['block_starts'][i]}"
            assert actual['block_ends'][i] == 0, f"blockEnds[{i}]={actual['block_ends'][i]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
