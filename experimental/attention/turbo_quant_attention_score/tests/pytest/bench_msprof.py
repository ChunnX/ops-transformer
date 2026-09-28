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

"""Accuracy + msprof-style latency for TurboQuantAttentionScore reference.

Warmup 10, sample 60, drop 10 largest, report mean (us).
Uses torch_npu golden path as executable reference until aclnn package is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import BLOCK_SIZE, CENTROIDS, HEAD_DIM, calc_expect_func  # noqa: E402

# Task-book Paged Full-QK baseline (us)
BASELINE_US = {
    "gqa_decode_kv4k": 1501.634,
    "gqa_decode_kv32k": 9278.507,
    "gqa_decode_kv128k": 35681.690,
    "gqa_prefill_kv4k": 75984.253,
}


def _make_inputs(case: dict, device: str = "npu"):
    attrs = {a["name"]: a["value"] for a in case.get("attr_desc", [])}
    mse_bits = int(attrs.get("mse_bits", 3))
    shapes = {d["name"]: d["shape"] for d in case["input_desc"]}

    rng = np.random.default_rng(0)
    q = rng.uniform(-1, 1, shapes["query_rotated"]).astype(np.float32)
    qjl = rng.uniform(-1, 1, shapes["query_qjl"]).astype(np.float32)
    idx = rng.integers(0, 256, shapes["key_cache_idx"], dtype=np.uint8)
    qjl_p = rng.integers(0, 256, shapes["key_cache_qjl"], dtype=np.uint8)
    norm = rng.uniform(0.1, 2.0, shapes["key_cache_norm"]).astype(np.float32)
    gamma = rng.uniform(0.1, 2.0, shapes["key_cache_gamma"]).astype(np.float32)

    bt_shape = shapes["block_table"]
    seq_shape = shapes["seq_lens"]
    # Prefer fixed seq_lens from value_range when constant
    seq_desc = next(d for d in case["input_desc"] if d["name"] == "seq_lens")
    lo, hi = seq_desc["value_range"]
    seq_lens = np.full(seq_shape, int(lo), dtype=np.int32)
    max_blocks = bt_shape[1]
    block_table = np.arange(bt_shape[0] * max_blocks, dtype=np.int32).reshape(bt_shape)
    block_table %= shapes["key_cache_idx"][0]

    return {
        "query_rotated": q,
        "query_qjl": qjl,
        "key_cache_idx": idx,
        "key_cache_qjl": qjl_p,
        "key_cache_norm": norm,
        "key_cache_gamma": gamma,
        "block_table": block_table,
        "seq_lens": seq_lens,
        "mse_bits": mse_bits,
    }


def _time_us(fn, warmup: int, samples: int, drop_max: int) -> float:
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    times = []
    for _ in range(samples):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    keep = times[: max(1, samples - drop_max)]
    return float(np.mean(keep))


def run_case(case: dict, warmup: int, samples: int, drop_max: int) -> dict:
    name = case["case_name"]
    inputs = _make_inputs(case)

    def once():
        return calc_expect_func(
            inputs["query_rotated"],
            inputs["query_qjl"],
            inputs["key_cache_idx"],
            inputs["key_cache_qjl"],
            inputs["key_cache_norm"],
            inputs["key_cache_gamma"],
            inputs["block_table"],
            inputs["seq_lens"],
            inputs["mse_bits"],
        )

    out = once()[0]
    assert out.ndim == 3
    latency = _time_us(once, warmup, samples, drop_max)
    baseline = BASELINE_US.get(name)
    ratio = latency / baseline if baseline else None
    return {
        "case": name,
        "shape_out": list(out.shape),
        "latency_us": round(latency, 3),
        "baseline_us": baseline,
        "ratio_vs_baseline": round(ratio, 3) if ratio is not None else None,
        "finite_ratio": float(np.isfinite(out).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-json", default=str(Path(__file__).with_name("case.json"))
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--drop-max", type=int, default=10)
    parser.add_argument("--only", default="", help="comma-separated case names")
    parser.add_argument("--out", default="/root/docs/perf_results.json")
    args = parser.parse_args()

    cases = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    if args.only:
        allow = set(args.only.split(","))
        cases = [c for c in cases if c["case_name"] in allow]

    results = []
    for case in cases:
        print(f"[RUN] {case['case_name']}", flush=True)
        try:
            r = run_case(case, args.warmup, args.samples, args.drop_max)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            err = {"case": case["case_name"], "error": str(exc)}
            results.append(err)
            print(json.dumps(err, ensure_ascii=False), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
