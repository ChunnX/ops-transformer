#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Profile block_attention_residuals forward: ??? vs ?????? + ?? kernel_details??"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[3]
TORCH_EXTENSION = REPO_ROOT / "torch_extension"
if str(TORCH_EXTENSION) not in sys.path:
    sys.path.insert(0, str(TORCH_EXTENSION))


def _forward_call(
    block_attention_residuals,
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num,
    norm_eps,
):
    """与 forward 精度脚本相同调用。"""
    return block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num,
        norm_eps,
    )


def _bench_wall_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _bench_profiler_us(fn, tag: str, warmup: int, iters: int, activities) -> dict:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            with record_function(tag):
                fn()
        torch.npu.synchronize()
    rows = [e for e in prof.key_averages() if e.key == tag]
    if not rows:
        print(f"  [{tag}] no record_function row; top events:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8))
        return {"cpu_us": 0, "npu_us": 0}
    ev = rows[0]
    npu_us = getattr(ev, "self_npu_time_total", 0) or getattr(
        ev, "self_device_time_total", 0
    )
    return {"cpu_us": ev.self_cpu_time_total, "npu_us": npu_us}


def _wait_kernel_details(trace_dir: Path, timeout_sec: float = 300.0) -> Path:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        files = sorted(trace_dir.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
        if files:
            return files[-1]
        for root, _, names in os.walk(trace_dir):
            if "analyse.done" in names:
                files = sorted(
                    trace_dir.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv")
                )
                if files:
                    return files[-1]
        time.sleep(1.0)
    raise TimeoutError(
        f"kernel_details.csv not found under {trace_dir} within {timeout_sec}s"
    )


def _summarize_kernel_details(csv_path: Path, repeat: int) -> tuple[list[dict], float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = row["Name"].strip()
            duration = float(row["Duration(us)"].strip())
            if name and duration > 0.0:
                grouped[name].append(duration)
    aggregates = [
        {
            "name": name,
            "count": len(values),
            "total_us": sum(values),
            "avg_us": sum(values) / len(values),
        }
        for name, values in grouped.items()
    ]
    aggregates.sort(key=lambda x: x["total_us"], reverse=True)
    bar_rows = [
        a
        for a in aggregates
        if "BlockAttention" in a["name"] or "block_attention" in a["name"].lower()
    ]
    if not bar_rows:
        bar_rows = aggregates[:5]
    total_us = sum(a["total_us"] for a in aggregates)
    return bar_rows, total_us


def _profile_npu_kernel_details(
    fn,
    label: str,
    trace_dir: Path,
    warmup: int,
    iters: int,
) -> Path:
    """torch_npu.profiler + prof.start()/stop()??? kernel_details.csv?"""
    import torch_npu.profiler as npu_profiler

    trace_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    handler = npu_profiler.tensorboard_trace_handler(str(trace_dir))
    prof = npu_profiler.profile(
        activities=[
            npu_profiler.ProfilerActivity.CPU,
            npu_profiler.ProfilerActivity.NPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        on_trace_ready=handler,
    )
    try:
        prof.start()
        with record_function(label):
            for _ in range(iters):
                fn()
        torch.npu.synchronize()
        prof.stop()
    finally:
        del prof

    csv_path = _wait_kernel_details(trace_dir)
    bar_rows, total_us = _summarize_kernel_details(csv_path, iters)
    print(f"\n  [kernel_details] {label}")
    print(f"    path: {csv_path}")
    print(f"    all-kernel total (sum Duration): {total_us:.1f} us over capture")
    for row in bar_rows[:8]:
        print(
            f"    {row['name']}: count={row['count']} "
            f"avg={row['avg_us']:.2f} us total={row['total_us']:.2f} us"
        )
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--storage-blocks", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--valid-block-num", type=int, default=-1)
    parser.add_argument("--norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100, help="?? benchmark ????")
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=8,
        help="torch.profiler ? kernel_details ?????????trace ???",
    )
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=None,
        help="????? prof.start/stop ?? kernel_details ????? after/?before/ ???",
    )
    parser.add_argument(
        "--no-npu-profiler", action="store_true", help="? CPU activity??? NPU trace ??"
    )
    args = parser.parse_args()

    storage_blocks = (
        args.storage_blocks if args.storage_blocks is not None else args.blocks * 2
    )
    if storage_blocks < args.blocks:
        raise ValueError("storage-blocks (k) must be >= blocks (N)")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    torch.manual_seed(args.seed)

    import torch_npu  # noqa: F401
    from cann_ops_transformer_custom.ops.mhc.block_attention_residuals import (
        block_attention_residuals,
    )

    device = torch.device(f"npu:{args.device_id}")
    partial_block = torch.randn(args.tokens, args.hidden_size, dtype=dtype).to(device)
    block_res_storage = torch.randn(
        args.tokens, storage_blocks, args.hidden_size, dtype=dtype
    )
    block_res_storage_npu = block_res_storage.to(device)
    block_res_sliced = block_res_storage_npu[:, : args.blocks, :]
    proj_weight = torch.randn(args.hidden_size, dtype=dtype).to(device)
    norm_weight = torch.randn(args.hidden_size, dtype=dtype).to(device)

    if block_res_sliced.is_contiguous() and storage_blocks > args.blocks:
        raise RuntimeError("expected sliced block_res to be non-contiguous on NPU")

    use_npu_prof = not args.no_npu_profiler
    profiler_activities = [ProfilerActivity.CPU]
    if use_npu_prof:
        try:
            import torch_npu.profiler as npu_profiler

            profiler_activities.append(npu_profiler.ProfilerActivity.NPU)
        except (ImportError, AttributeError):
            use_npu_prof = False

    print(
        f"T={args.tokens} N={args.blocks} k={storage_blocks} H={args.hidden_size} dtype={args.dtype}\n"
        f"slice stride={tuple(block_res_sliced.stride())} contiguous={block_res_sliced.is_contiguous()}\n"
        f"warmup={args.warmup} bench_iters={args.iters} profile_iters={args.profile_iters} "
        f"npu_profiler={use_npu_prof} trace_root={args.trace_root}"
    )

    def after_fn():
        with torch.no_grad():
            _forward_call(
                block_attention_residuals,
                partial_block,
                block_res_sliced,
                proj_weight,
                norm_weight,
                args.valid_block_num,
                args.norm_eps,
            )

    def before_fn():
        with torch.no_grad():
            br = block_res_sliced.contiguous()
            _forward_call(
                block_attention_residuals,
                partial_block,
                br,
                proj_weight,
                norm_weight,
                args.valid_block_num,
                args.norm_eps,
            )

    tag_after = "bar_fwd::after_stride_view"
    tag_before = "bar_fwd::before_legacy_contiguous"

    ms_after = _bench_wall_ms(after_fn, args.warmup, args.iters)
    ms_before = _bench_wall_ms(before_fn, args.warmup, args.iters)

    prof_after = _bench_profiler_us(
        after_fn, tag_after, args.warmup, args.profile_iters, profiler_activities
    )
    prof_before = _bench_profiler_us(
        before_fn, tag_before, args.warmup, args.profile_iters, profiler_activities
    )

    def _fmt_prof(name, prof, piters):
        cpu_ms = prof["cpu_us"] / piters / 1000.0
        npu_ms = prof["npu_us"] / piters / 1000.0 if prof["npu_us"] else float("nan")
        print(
            f"  torch.profiler [{name}] self_cpu={cpu_ms:.4f} ms/iter self_npu={npu_ms:.4f} ms/iter"
        )

    print("\n=== ????NPU slice ??? view?? test 78-81?===")
    print(f"  wall (sync) {ms_after:.4f} ms/iter")
    _fmt_prof(tag_after, prof_after, args.profile_iters)

    print("\n=== ????profile ??? contiguous + ???===")
    print(f"  wall (sync) {ms_before:.4f} ms/iter")
    _fmt_prof(tag_before, prof_before, args.profile_iters)

    if ms_after > 0:
        print(
            f"\n?? before/after = {ms_before / ms_after:.3f}x, "
            f"? {ms_before - ms_after:.4f} ms/iter"
        )
    if prof_after["npu_us"] and prof_before["npu_us"]:
        print(
            f"profiler self_npu before/after = {prof_before['npu_us'] / prof_after['npu_us']:.3f}x"
        )

    if args.trace_root is not None:
        print("\n=== NPU trace?prof.start/stop ? kernel_details.csv?===")
        trace_after = args.trace_root / "after_stride_view"
        trace_before = args.trace_root / "before_contiguous"
        csv_after = _profile_npu_kernel_details(
            after_fn, tag_after, trace_after, args.warmup, args.profile_iters
        )
        csv_before = _profile_npu_kernel_details(
            before_fn, tag_before, trace_before, args.warmup, args.profile_iters
        )
        print(f"\nkernel_details after:  {csv_after}")
        print(f"kernel_details before: {csv_before}")


if __name__ == "__main__":
    main()
