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
"""Profile cases that must be covered to subsume LI / LIv2 / QLI / QLIv2 / PKI / Qwen CLI."""

import sys
from pathlib import Path

import numpy as np
import pytest

ASSET_IMPL = Path(__file__).resolve().parents[1] / "assets" / "impl"
sys.path.insert(0, str(ASSET_IMPL))
from golden import generic_lightning_indexer_golden  # noqa: E402

PROFILES = [
    {
        "id": "P-LI",
        "cmp_ratio": 1,
        "act_type": 0,
        "has_weighted_sum": True,
        "sparse_mode": 3,
        "expand_indices": False,
        "append_tail": False,
        "quant_mode": -1,
        "score_scale": 1.0,
        "topk": 8,
    },
    {
        "id": "P-LIv2",
        "cmp_ratio": 4,
        "act_type": 0,
        "has_weighted_sum": True,
        "sparse_mode": 3,
        "expand_indices": False,
        "append_tail": False,
        "quant_mode": -1,
        "score_scale": 1.0,
        "topk": 8,
    },
    {
        "id": "P-QLI",
        "cmp_ratio": 1,
        "act_type": 0,
        "has_weighted_sum": True,
        "sparse_mode": 3,
        "expand_indices": False,
        "append_tail": False,
        "quant_mode": 0,
        "score_scale": 1.0,
        "topk": 8,
    },
    {
        "id": "P-QLIv2",
        "cmp_ratio": 4,
        "act_type": 0,
        "has_weighted_sum": True,
        "sparse_mode": 3,
        "expand_indices": False,
        "append_tail": False,
        "quant_mode": 3,
        "score_scale": 1.0,
        "topk": 8,
    },
    {
        "id": "P-PKI",
        "cmp_ratio": 16,
        "act_type": 0,
        "has_weighted_sum": True,
        "sparse_mode": 3,
        "expand_indices": True,
        "append_tail": True,
        "quant_mode": -1,
        "score_scale": 1.0 / (128**0.5),
        "topk": 4,
    },
    {
        "id": "P-QWEN",
        "cmp_ratio": 4,
        "act_type": 0,
        "has_weighted_sum": False,
        "sparse_mode": 4,
        "expand_indices": True,
        "append_tail": True,
        "quant_mode": -1,
        "score_scale": 1.0,
        "topk": 4,
    },
    {
        "id": "P-MSA",
        "cmp_ratio": 8,
        "act_type": 1,
        "has_weighted_sum": False,
        "sparse_mode": 3,
        "expand_indices": False,
        "append_tail": False,
        "quant_mode": -1,
        "score_scale": 1.0 / (128**0.5),
        "topk": 16,
        "key_reduce_mode": 1,
        "select_mode": 1,
    },
]


def _expected_idx_last(cfg):
    topk = cfg["topk"]
    r = cfg["cmp_ratio"]
    if not cfg["expand_indices"]:
        return topk
    expanded = topk * r
    return expanded + r - 1 if cfg["append_tail"] else expanded


@pytest.mark.ci
@pytest.mark.parametrize("cfg", PROFILES, ids=lambda c: c["id"])
def test_golden_profile_shape_and_invalid_slot(cfg):
    b, s1, n1, d = 1, 8, 8, 128
    r = cfg["cmp_ratio"]
    s2 = 16 if r == 1 else 8
    rng = np.random.default_rng(0)
    query = rng.standard_normal((b, s1, n1, d)).astype(np.float32)
    key = rng.standard_normal((b, s2, 1, d)).astype(np.float32)
    weights = (
        np.ones((b, s1, n1), dtype=np.float32) if cfg["has_weighted_sum"] else None
    )
    residual = (
        np.array([1], dtype=np.int32)
        if r > 1 and cfg.get("select_mode", 0) == 0
        else None
    )
    idx, val = generic_lightning_indexer_golden(
        query,
        key,
        weights=weights,
        cmp_residual_k=residual,
        topk=cfg["topk"],
        cmp_ratio=r,
        layout_q="BSND",
        sparse_mode=cfg["sparse_mode"],
        act_type=cfg["act_type"],
        has_weighted_sum=cfg["has_weighted_sum"],
        expand_indices=cfg["expand_indices"],
        append_tail=cfg["append_tail"],
        return_value=True,
        score_scale=cfg["score_scale"],
        key_reduce_mode=cfg.get("key_reduce_mode", 0),
        select_mode=cfg.get("select_mode", 0),
    )
    idx_np = idx.numpy()
    val_np = val.numpy()
    if cfg.get("select_mode", 0) == 1:
        n_blocks = (s2 + r - 1) // r
        n_out = ((max(n_blocks, 1) + 15) // 16) * 16
        assert idx_np.shape == (0,)
        assert val_np.shape == (b, s1, n1, n_out)
        return
    last = _expected_idx_last(cfg)
    assert idx_np.shape == (b, s1, last)
    assert val_np.shape == (b, s1, cfg["topk"])
    assert np.all(idx_np >= -1)
    if cfg["expand_indices"]:
        assert last == cfg["topk"] * r + (r - 1 if cfg["append_tail"] else 0)
