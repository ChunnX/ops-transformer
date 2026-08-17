#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.

"""CPU-only contract tests for the TTK NumPy golden."""

from pathlib import Path
import sys

import numpy as np


ASSETS = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(ASSETS))

from impl.numpy_golden import (  # noqa: E402
    BF16_DTYPE,
    FP8_DTYPE,
    qkv_rms_norm_rope_cache_with_k_scale_numpy,
    saturating_fp8_cast,
)


def _inputs(layout_qkv="TND", layout_q_out="TND", mrope=False, zero=False):
    rng = np.random.default_rng(2026)
    token_num, q_heads, k_heads, v_heads, head_dim = 5, 8, 1, 1, 128
    qkv_tnd = rng.normal(0.0, 0.4, size=(token_num, 10, head_dim)).astype(np.float32)
    if zero:
        qkv_tnd[0] = 0
    qkv = qkv_tnd if layout_qkv == "TND" else np.transpose(qkv_tnd, (1, 0, 2))
    qkv = qkv.astype(BF16_DTYPE)
    angles = rng.normal(size=(64, head_dim // 2)).astype(np.float32)
    common = {
        "qkv": qkv,
        "q_gamma": rng.uniform(0.8, 1.2, size=(head_dim,)).astype(np.float32),
        "k_gamma": rng.uniform(0.8, 1.2, size=(head_dim,)).astype(np.float32),
        "cos_sin": np.concatenate((np.cos(angles), np.sin(angles)), axis=-1),
        "slot_mapping": np.array([3, 10, 1, 12, 7], dtype=np.int32),
        "k_cache": (
            rng.integers(-8, 9, size=(2, k_heads, 8, head_dim), dtype=np.int8)
            if mrope else
            saturating_fp8_cast(rng.uniform(-2, 2, size=(2, k_heads, 8, head_dim)))
        ),
        "v_cache": saturating_fp8_cast(
            rng.uniform(-2, 2, size=(2, v_heads, 8, head_dim))
        ),
        "k_scale_cache": rng.uniform(0.01, 0.5, size=(2, k_heads, 8, 1)).astype(np.float32),
        "rotation": (np.eye(head_dim) + rng.normal(0, 0.01, size=(head_dim, head_dim))).astype(BF16_DTYPE),
        "head_nums": [q_heads, k_heads, v_heads],
        "layout_qkv": layout_qkv,
        "layout_q_out": layout_q_out,
    }
    if mrope:
        common.update({
            "query_start_loc": None,
            "seq_lens": None,
            "v_scale": np.ones((v_heads, head_dim), dtype=np.float32),
            "mrope_position": np.array(
                [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
                dtype=np.int32,
            ),
            "mrope_section": [22, 21, 21],
            "q_quant_mode": "NoQuant",
            "q_out_dtype": 27,
        })
    else:
        common.update({
            "query_start_loc": np.array([0, 2, 5], dtype=np.int32),
            "seq_lens": np.array([5, 9], dtype=np.int32),
            "v_scale": np.ones((v_heads,), dtype=np.float32),
            "mrope_position": None,
            "mrope_section": [],
            "q_quant_mode": "PerTokenPerHead",
            "q_out_dtype": 36,
        })
    return common


def _assert_untouched(initial, updated, slots):
    block_size = initial.shape[2]
    touched = {(int(slot) // block_size, int(slot) % block_size) for slot in slots}
    for block in range(initial.shape[0]):
        for offset in range(block_size):
            if (block, offset) not in touched:
                assert np.array_equal(initial[block, :, offset], updated[block, :, offset])


def test_rope_layouts_and_cache_copy_semantics():
    for layout_qkv, layout_q_out, expected_q_shape in (
        ("NTD", "NTD", (8, 5, 128)),
        ("TND", "TND", (5, 8, 128)),
        ("TND", "NTD", (8, 5, 128)),
    ):
        data = _inputs(layout_qkv, layout_q_out)
        initial = (data["k_cache"].copy(), data["v_cache"].copy(), data["k_scale_cache"].copy())
        result = qkv_rms_norm_rope_cache_with_k_scale_numpy(**data)
        assert result.q_out.shape == expected_q_shape
        assert result.q_out.dtype == FP8_DTYPE
        assert result.q_scale.shape == expected_q_shape[:-1]
        for before, after in zip(initial, result[2:]):
            assert not np.shares_memory(before, after)
            _assert_untouched(before, after, data["slot_mapping"])


def test_mrope_bf16_q_int8_k_and_optional_scale():
    data = _inputs(mrope=True)
    result = qkv_rms_norm_rope_cache_with_k_scale_numpy(**data)
    assert result.q_out.dtype == BF16_DTYPE
    assert result.q_out.shape == (5, 8, 128)
    assert result.q_scale is None
    assert result.k_cache.dtype == np.int8
    assert result.v_cache.dtype == FP8_DTYPE


def test_v_scale_zero_writes_zero_v_rows():
    data = _inputs()
    data["v_scale"][:] = 0
    result = qkv_rms_norm_rope_cache_with_k_scale_numpy(**data)
    block_size = result.v_cache.shape[2]
    for slot in data["slot_mapping"]:
        assert np.all(result.v_cache[int(slot) // block_size, :, int(slot) % block_size] == 0)


def test_zero_quant_row_preserves_unguarded_divide_policy():
    data = _inputs(zero=True)
    result = qkv_rms_norm_rope_cache_with_k_scale_numpy(**data)
    assert np.all(result.q_scale[0] == 0)
    assert np.all(np.isnan(result.q_out[0].astype(np.float32)))
