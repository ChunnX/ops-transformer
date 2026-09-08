# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark Sage/FIA sequentially with torch.npu.Event."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu
from cases import ALL_CASES
from sageattention2 import sageattn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=[case[0] for case in ALL_CASES], required=True
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq", type=int, help="Override context length")
    parser.add_argument("--smooth-v", action="store_true")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    _, seq, hk, batch, dtype_name, causal = next(
        c for c in ALL_CASES if c[0] == args.case
    )
    heads = 16 if args.case == "X-02" else 32
    dim = 64 if args.case == "X-04" else 128
    seq = args.seq if args.seq is not None else seq
    if seq < 1:
        parser.error("invalid context length")
    torch.manual_seed(71)
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    q = torch.randn(batch, heads, seq, dim, device="npu", dtype=dtype)
    k = torch.randn(batch, hk, seq, dim, device="npu", dtype=dtype)
    v = torch.randn_like(k)
    mask = (
        torch.ones(2048, 2048, dtype=torch.bool, device="npu").triu(1)
        if causal
        else None
    )

    def run_sage():
        return sageattn(q, k, v, is_causal=causal, smooth_v=args.smooth_v)

    def run_fia():
        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=heads,
            num_key_value_heads=hk,
            input_layout="BNSD",
            scale=dim**-0.5,
            atten_mask=mask,
            sparse_mode=2 if causal else 0,
        )[0]

    torch.npu.synchronize()
    outputs = {}
    event_samples = {}
    # Finish every Sage call before starting FIA. Each operator gets its own
    # warmup, excluded from the Event samples.
    for operator, fn in (("sage", run_sage), ("fia", run_fia)):
        samples = []
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        for iteration in range(args.warmup + args.repeats):
            measured = iteration >= args.warmup
            if measured:
                start.record()
            outputs[operator] = fn()
            if measured:
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000)
            else:
                torch.npu.synchronize()
        if samples:
            ordered = sorted(samples)
            event_samples[operator] = {
                "samples_us": samples,
                "median_us": statistics.median(samples),
                "mean_us": statistics.mean(samples),
                "min_us": min(samples),
                "max_us": max(samples),
                "p10_us": ordered[int(0.1 * (len(ordered) - 1))],
                "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
            }

    assert all(torch.isfinite(out).all().item() for out in outputs.values())
    error = (outputs["sage"].float() - outputs["fia"].float()).abs()
    report = {
        "case": args.case,
        "path": "public PyTorch -> aclnn",
        "sage_api": "sageattn -> aclnnSageAttention2",
        "fia_api": "torch_npu.npu_fused_infer_attention_score",
        "operator_stages": ["Fused"],
        "timing": "torch.npu.Event per call; warmup excluded",
        "order": "sage-first",
        "v_max": 448.0,
        "pv_accumulation": "fp32",
        "smooth_v": args.smooth_v,
        "shape": list(q.shape),
        "kv_heads": hk,
        "dtype": dtype_name,
        "causal": causal,
        "fia_sparse_mode": 2 if causal else 0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": torch.npu.get_device_name(),
        "max_abs_vs_fia": error.max().item(),
        "mean_abs_vs_fia": error.mean().item(),
        "rmse_vs_fia": error.square().mean().sqrt().item(),
    }
    if event_samples:
        report["event"] = event_samples
        report["event_speedup"] = (
            event_samples["fia"]["median_us"] / event_samples["sage"]["median_us"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
