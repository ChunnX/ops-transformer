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
import os
import time
from pathlib import Path

import torch
import torch_npu
from cases import ALL_CASES
from sageattention2 import sageattn


# Timing functions follow the task-provided test_perf.py verbatim.
def _sync(device: torch.device):
    if str(device).startswith("npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time_ms(fn, warmup: int = 5, iters: int = 20, device: torch.device = None) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)
    if str(device).startswith("npu"):
        starter = torch.npu.Event(enable_timing=True)
        ender = torch.npu.Event(enable_timing=True)
        starter.record()
        for _ in range(iters):
            fn()
        ender.record()
        _sync(device)
        return starter.elapsed_time(ender) / iters

    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=[case.name for case in ALL_CASES], required=True
    )
    parser.add_argument(
        "--warmup", type=int, default=int(os.environ.get("SAGEATTN_PERF_WARMUP", "50"))
    )
    parser.add_argument(
        "--repeats", type=int, default=int(os.environ.get("SAGEATTN_PERF_ITERS", "100"))
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq", type=int, help="Override context length")
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    case = next(c for c in ALL_CASES if c.name == args.case)
    batch, heads, hk, dim = case.batch, case.heads, case.hk, case.dim
    dtype_name, causal = case.dtype, case.causal
    seq = args.seq if args.seq is not None else case.seq
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
        return sageattn(q, k, v, is_causal=causal)

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

    event_samples = {}
    # Match the task script: FIA first; one Event pair around the entire batch.
    for operator, fn in (("fia", run_fia), ("sage", run_sage)):
        mean_ms = _time_ms(fn, warmup=args.warmup, iters=args.repeats, device=q.device)
        print(f"MEASURED {args.case} {operator}: {mean_ms:.2f} ms; cooldown 10s", flush=True)
        time.sleep(10)
        event_samples[operator] = {
            "mean_us": mean_ms * 1000,
            "batch_total_us": mean_ms * args.repeats * 1000,
            "calls": args.repeats,
            "warmup_calls": args.warmup,
        }
    # Accuracy outputs are obtained outside the timed batches.
    outputs = {"sage": run_sage(), "fia": run_fia()}
    torch.npu.synchronize()

    assert all(torch.isfinite(out).all().item() for out in outputs.values())
    error = (outputs["sage"].float() - outputs["fia"].float()).abs()
    report = {
        "case": args.case,
        "path": "public PyTorch -> dispatcher -> C++/torch_npu -> aclnn",
        "sage_api": "sageattn -> torch.ops.sageattention2.forward -> aclnnSageAttention2",
        "fia_api": "torch_npu.npu_fused_infer_attention_score",
        "operator_stages": ["Fused"],
        "timing": "torch.npu.Event batch total / repeats; synchronize after warmup and timed batch",
        "order": "fia-first",
        "pv_quantization": "mxfp8_e4m3_e8m0",
        "pv_accumulation": "fp32+fp32",
        "shape": list(q.shape),
        "kv_heads": hk,
        "dtype": dtype_name,
        "causal": causal,
        "fia_sparse_mode": 2 if causal else 0,
        "warmup_calls": args.warmup,
        "repeats": args.repeats,
        "cooldown_seconds_after_measurement": 10,
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
            event_samples["fia"]["mean_us"] / event_samples["sage"]["mean_us"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
