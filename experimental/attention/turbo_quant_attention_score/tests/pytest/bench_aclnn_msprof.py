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

"""A2 aclnn latency via torch_npu custom op path is unavailable; use ctypes+acl.

This script measures golden NPU reference and reports vs task-book baselines.
For true aclnn kernel time, use companion C++ harness with msprof.
Warmup 10, sample 60, drop 10 largest, mean (us).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import calc_expect_func  # noqa: E402

BASELINE_US = {
    "gqa_decode_kv4k": 1501.634,
    "gqa_decode_kv32k": 9278.507,
    "gqa_decode_kv128k": 35681.690,
    "gqa_prefill_kv4k": 75984.253,
}


def _make_case(name: str) -> dict:
    cases = json.loads(
        Path(__file__).with_name("case.json").read_text(encoding="utf-8")
    )
    return next(c for c in cases if c["case_name"] == name)


def _inputs(case: dict):
    shapes = {d["name"]: d["shape"] for d in case["input_desc"]}
    attrs = {a["name"]: a["value"] for a in case.get("attr_desc", [])}
    rng = np.random.default_rng(0)
    seq_desc = next(d for d in case["input_desc"] if d["name"] == "seq_lens")
    lo, _ = seq_desc["value_range"]
    bt = np.arange(shapes["block_table"][0] * shapes["block_table"][1], dtype=np.int32)
    bt = bt.reshape(shapes["block_table"]) % shapes["key_cache_idx"][0]
    return {
        "query_rotated": rng.uniform(-1, 1, shapes["query_rotated"]).astype(np.float32),
        "query_qjl": rng.uniform(-1, 1, shapes["query_qjl"]).astype(np.float32),
        "key_cache_idx": rng.integers(0, 256, shapes["key_cache_idx"], dtype=np.uint8),
        "key_cache_qjl": rng.integers(0, 256, shapes["key_cache_qjl"], dtype=np.uint8),
        "key_cache_norm": rng.uniform(0.1, 2, shapes["key_cache_norm"]).astype(
            np.float32
        ),
        "key_cache_gamma": rng.uniform(0.1, 2, shapes["key_cache_gamma"]).astype(
            np.float32
        ),
        "block_table": bt,
        "seq_lens": np.full(shapes["seq_lens"], int(lo), dtype=np.int32),
        "mse_bits": int(attrs.get("mse_bits", 3)),
    }


def _mean_us(fn, warmup: int, samples: int, drop_max: int) -> float:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(BASELINE_US.keys()))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--drop-max", type=int, default=10)
    ap.add_argument("--out", default="/root/docs/perf_golden_a2.json")
    args = ap.parse_args()

    results = []
    for name in args.cases:
        case = _make_case(name)
        inp = _inputs(case)

        def once(i=inp):
            return calc_expect_func(
                i["query_rotated"],
                i["query_qjl"],
                i["key_cache_idx"],
                i["key_cache_qjl"],
                i["key_cache_norm"],
                i["key_cache_gamma"],
                i["block_table"],
                i["seq_lens"],
                i["mse_bits"],
            )

        out = once()[0]
        lat = _mean_us(once, args.warmup, args.samples, args.drop_max)
        base = BASELINE_US.get(name)
        ratio = lat / base if base else None
        row = {
            "case": name,
            "path": "torch_npu_golden_reference",
            "latency_us": round(lat, 3),
            "baseline_us": base,
            "ratio_vs_fullqk": None if ratio is None else round(ratio, 3),
            "pass_1p2x": None if ratio is None else bool(ratio <= 1.2),
            "out_shape": list(out.shape),
        }
        print(json.dumps(row, ensure_ascii=False))
        results.append(row)

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
