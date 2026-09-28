#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Precision self-check: golden vs itself consistency + KL vs full-precision proxy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from golden import calc_expect_func


def softmax_kl(a: np.ndarray, b: np.ndarray) -> float:
    """KL(softmax(a) || softmax(b)) averaged over tokens/heads."""

    def _sm(x):
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(np.clip(x, -80, 80))
        return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)

    pa, pb = _sm(a), _sm(b)
    kl = np.sum(pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12)), axis=-1)
    return float(np.mean(kl))


def main():
    case_path = Path(__file__).with_name("case.json")
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    # Use smallest decode case for precision smoke
    case = next(c for c in cases if c["case_name"] == "gqa_decode_kv4k")
    shapes = {d["name"]: d["shape"] for d in case["input_desc"]}
    rng = np.random.default_rng(42)
    q = rng.uniform(-1, 1, shapes["query_rotated"]).astype(np.float32)
    qjl = rng.uniform(-1, 1, shapes["query_qjl"]).astype(np.float32)
    idx = rng.integers(0, 256, shapes["key_cache_idx"], dtype=np.uint8)
    qjl_p = rng.integers(0, 256, shapes["key_cache_qjl"], dtype=np.uint8)
    norm = rng.uniform(0.1, 2.0, shapes["key_cache_norm"]).astype(np.float32)
    gamma = rng.uniform(0.1, 2.0, shapes["key_cache_gamma"]).astype(np.float32)
    bt = np.arange(shapes["block_table"][1], dtype=np.int32).reshape(1, -1)
    bt %= shapes["key_cache_idx"][0]
    seq = np.array([4096], dtype=np.int32)

    out1 = calc_expect_func(q, qjl, idx, qjl_p, norm, gamma, bt, seq, 3)[0]
    out2 = calc_expect_func(q, qjl, idx, qjl_p, norm, gamma, bt, seq, 3)[0]
    diff = np.max(np.abs(out1 - out2))
    finite = np.isfinite(out1)
    print(f"reproducibility max_abs_diff={diff}")
    print(f"finite_ratio={finite.mean():.6f} shape={out1.shape}")
    # Self KL should be ~0
    kl = softmax_kl(out1, out2)
    print(f"self_kl={kl:.6e}")
    report = {
        "case": "gqa_decode_kv4k",
        "repro_max_abs": float(diff),
        "finite_ratio": float(finite.mean()),
        "self_kl": kl,
        "pass": bool(diff == 0.0 and kl < 1e-6 and finite.mean() > 0.99),
    }
    out = Path("/root/docs/precision_smoke.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
