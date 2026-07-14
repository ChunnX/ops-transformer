#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Performance benchmark for stem_oam_prep_varlen_q on NPU (msprof) and GPU (CUDA Event).

Usage:
    # NPU only
    python benchmark_stem_oam_prep_varlen_q.py --device npu --device-id 0

    # GPU only
    python benchmark_stem_oam_prep_varlen_q.py --device gpu --device-id 0

    # NPU with msprof op-level profiling
    python benchmark_stem_oam_prep_varlen_q.py --device npu --device-id 0 --msprof

    # Custom parameters
    python benchmark_stem_oam_prep_varlen_q.py --device npu --warmup 10 --iters 100 \\
        --seq-lens 128 256 512 1024 4096 --heads 1 2 8
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def find_repo_root():
    for path in Path(__file__).resolve().parents:
        if (path / "MindSpeed").exists() or (path / "build.sh").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()
TORCH_EXTENSION_PATH = REPO_ROOT / "torch_extension"
if TORCH_EXTENSION_PATH.exists():
    sys.path.insert(0, str(TORCH_EXTENSION_PATH))


DIM_QK = 128
BENCH_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
BENCH_HEADS = [1, 2, 8]
DEFAULT_STEM_BLOCK_SIZE = 128
DEFAULT_STEM_STRIDE = 16


def generate_inputs(total_tokens, num_heads, device):
    q_fp32 = torch.randn(total_tokens, num_heads, DIM_QK, dtype=torch.float32, device=device) * 0.1
    q_fp8 = q_fp32.to(torch.float8_e4m3fn)
    q_scale = (torch.randn(total_tokens, num_heads, dtype=torch.float32, device=device).abs() + 0.5)
    return q_fp8, q_scale


def benchmark_npu_event(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_gpu_event(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_npu_benchmark(seq_lens, heads, warmup, iters, device_id, stem_block_size, stem_stride):
    import torch_npu
    import cann_ops_transformer  # noqa: F401

    torch_npu.npu.set_device(device_id)
    device = f'npu:{device_id}'
    op_fn = torch.ops.cann_ops_transformer.stem_oam_prep_varlen_q

    results = []
    for num_heads in heads:
        for seq_len in seq_lens:
            total_tokens = seq_len
            q_fp8, q_scale = generate_inputs(total_tokens, num_heads, device)
            q_seq_lens = [seq_len]
            cu_seqlens_q = [0, seq_len]

            def fn():
                return op_fn(q_fp8, q_seq_lens, cu_seqlens_q,
                             qScale=q_scale,
                             stemBlockSize=stem_block_size, stemStride=stem_stride)

            latency_ms = benchmark_npu_event(fn, warmup, iters)
            max_qb = (seq_len + stem_block_size - 1) // stem_block_size
            qflat_dim = stem_stride * DIM_QK
            output_size = 1 * num_heads * max_qb * qflat_dim * 2
            input_size = total_tokens * num_heads * DIM_QK * 1 + total_tokens * num_heads * 4
            total_bytes = input_size + output_size
            bandwidth_tbs = total_bytes / (latency_ms * 1e-3) / 1e12

            results.append({
                'device': 'NPU',
                'seq_len': seq_len,
                'heads': num_heads,
                'total_tokens': total_tokens,
                'latency_ms': latency_ms,
                'bandwidth_TBs': bandwidth_tbs,
                'max_Qb': max_qb,
            })
            print(f"  NPU  seq={seq_len:>6}  H={num_heads:>2}  "
                  f"latency={latency_ms:.4f} ms  bw={bandwidth_tbs:.2f} TB/s")

    return results


def run_gpu_benchmark(seq_lens, heads, warmup, iters, device_id, stem_block_size, stem_stride):
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device_id)

    R = stem_block_size // stem_stride

    def gpu_prep_q(q_fp8, q_scale, seq_len, num_heads, stem_bs, stem_s):
        num_qb = (seq_len + stem_bs - 1) // stem_bs
        q_padded = num_qb * stem_bs
        q_dense = torch.zeros(q_padded, num_heads, DIM_QK, dtype=torch.float32, device=device)
        q_dense[:seq_len] = q_fp8[:seq_len].float()
        q_blocks = q_dense.reshape(num_qb, R, stem_s, num_heads, DIM_QK)
        scales = q_scale[:q_padded].reshape(num_qb, R, stem_s, num_heads)
        q_group_sum = (q_blocks * scales.unsqueeze(-1)).sum(dim=1)
        return q_group_sum.reshape(num_qb, stem_s * DIM_QK, num_heads).permute(0, 2, 1).unsqueeze(0).to(torch.bfloat16)

    results = []
    for num_heads in heads:
        for seq_len in seq_lens:
            total_tokens = seq_len
            q_fp8, q_scale = generate_inputs(total_tokens, num_heads, device)

            def fn():
                return gpu_prep_q(q_fp8, q_scale, seq_len, num_heads, stem_block_size, stem_stride)

            latency_ms = benchmark_gpu_event(fn, warmup, iters)
            max_qb = (seq_len + stem_block_size - 1) // stem_block_size
            qflat_dim = stem_stride * DIM_QK
            output_size = 1 * num_heads * max_qb * qflat_dim * 2
            input_size = total_tokens * num_heads * DIM_QK * 1 + total_tokens * num_heads * 4
            total_bytes = input_size + output_size
            bandwidth_tbs = total_bytes / (latency_ms * 1e-3) / 1e12

            results.append({
                'device': 'GPU',
                'seq_len': seq_len,
                'heads': num_heads,
                'total_tokens': total_tokens,
                'latency_ms': latency_ms,
                'bandwidth_TBs': bandwidth_tbs,
                'max_Qb': max_qb,
            })
            print(f"  GPU  seq={seq_len:>6}  H={num_heads:>2}  "
                  f"latency={latency_ms:.4f} ms  bw={bandwidth_tbs:.2f} TB/s")

    return results


def build_msprof_script(seq_lens, heads, warmup, iters, device_id, stem_block_size, stem_stride):
    lines = [
        "import sys, torch, torch_npu",
        f"sys.path.insert(0, {str(TORCH_EXTENSION_PATH)!r})",
        "import cann_ops_transformer",
        f"torch_npu.npu.set_device({device_id})",
        f"device = 'npu:{device_id}'",
        "op_fn = torch.ops.cann_ops_transformer.stem_oam_prep_varlen_q",
        f"DIM_QK = {DIM_QK}",
        f"stem_bs = {stem_block_size}",
        f"stem_s = {stem_stride}",
    ]
    for num_heads in heads:
        for seq_len in seq_lens:
            tag = f"h{num_heads}_s{seq_len}"
            lines.extend([
                f"q_{tag} = (torch.randn({seq_len}, {num_heads}, DIM_QK, dtype=torch.float32, device=device) * 0.1).to(torch.float8_e4m3fn)",
                f"qs_{tag} = torch.randn({seq_len}, {num_heads}, dtype=torch.float32, device=device).abs() + 0.5",
                f"qsl_{tag} = [{seq_len}]",
                f"cusl_{tag} = [0, {seq_len}]",
                f"def fn_{tag}():",
                f"    return op_fn(q_{tag}, qsl_{tag}, cusl_{tag}, qScale=qs_{tag}, stemBlockSize=stem_bs, stemStride=stem_s)",
            ])
    lines.append("torch.npu.synchronize()")
    for num_heads in heads:
        for seq_len in seq_lens:
            tag = f"h{num_heads}_s{seq_len}"
            lines.append(f"for _ in range({warmup}): fn_{tag}()")
    lines.append("torch.npu.synchronize()")
    lines.append(f"for _ in range({iters}):")
    for num_heads in heads:
        for seq_len in seq_lens:
            tag = f"h{num_heads}_s{seq_len}"
            lines.append(f"    fn_{tag}()")
    lines.append("torch.npu.synchronize()")
    return "\n".join(lines)


def run_msprof(seq_lens, heads, warmup, iters, device_id, stem_block_size, stem_stride, output_dir):
    script = build_msprof_script(seq_lens, heads, warmup, iters, device_id, stem_block_size, stem_stride)
    script_path = output_dir / "_perf_bench.py"
    script_path.write_text(script)

    msprof_dir = output_dir / "msprof_output"
    if msprof_dir.exists():
        shutil.rmtree(msprof_dir)

    cmd = [
        "msprof",
        f"--output={msprof_dir}",
        "--aic-mode=task-based",
        sys.executable,
        str(script_path),
    ]
    print(f"\n[msprof] Running: {' '.join(cmd[:4])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[msprof] FAILED:\n{result.stderr}")
        return []

    prof_dirs = sorted(msprof_dir.glob("PROF_*"), reverse=True)
    if not prof_dirs:
        print("[msprof] No PROF_* directory found")
        return []

    csv_files = sorted(prof_dirs[0].glob("mindstudio_profiler_output/op_summary_*.csv"))
    if not csv_files:
        print("[msprof] No op_summary CSV found")
        return []

    results = []
    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_type = row.get("OP Type", "")
            if "StemOamPrepVarlenQ" in op_type or "stem_oam_prep_varlen_q" in op_type:
                duration_us = float(row.get("Task Duration(us)", 0))
                results.append({
                    'device': 'NPU(msprof)',
                    'op_type': op_type,
                    'duration_us': duration_us,
                })

    print(f"[msprof] Found {len(results)} StemOamPrepVarlenQ op records")
    for r in results[:20]:
        print(f"  op={r['op_type']}  duration={r['duration_us']:.2f} us")

    return results


def main():
    parser = argparse.ArgumentParser(description="Performance benchmark for stem_oam_prep_varlen_q")
    parser.add_argument('--device', choices=['npu', 'gpu'], default='npu',
                        help='Device to benchmark (default: npu)')
    parser.add_argument('--device-id', type=int, default=0, help='Device id')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=50, help='Timed iterations')
    parser.add_argument('--seq-lens', type=int, nargs='+', default=BENCH_SEQ_LENS,
                        help='Sequence lengths to benchmark')
    parser.add_argument('--heads', type=int, nargs='+', default=BENCH_HEADS,
                        help='Number of Q heads to benchmark')
    parser.add_argument('--stem-block-size', type=int, default=DEFAULT_STEM_BLOCK_SIZE)
    parser.add_argument('--stem-stride', type=int, default=DEFAULT_STEM_STRIDE)
    parser.add_argument('--msprof', action='store_true', help='Use msprof for NPU op-level profiling')
    parser.add_argument('--output', type=str, default='./perf_output',
                        help='Output directory for results and msprof data')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    npu_results = []
    gpu_results = []

    if args.device == 'npu':
        print(f"\n{'=' * 90}")
        print(f"  NPU Benchmark: stem_oam_prep_varlen_q")
        print(f"  warmup={args.warmup}  iters={args.iters}  B={args.stem_block_size}  S={args.stem_stride}")
        print(f"{'=' * 90}")
        npu_results = run_npu_benchmark(
            args.seq_lens, args.heads, args.warmup, args.iters,
            args.device_id, args.stem_block_size, args.stem_stride)

    if args.device == 'gpu':
        print(f"\n{'=' * 90}")
        print(f"  GPU Benchmark: PyTorch reference (weighted group sum)")
        print(f"  warmup={args.warmup}  iters={args.iters}  B={args.stem_block_size}  S={args.stem_stride}")
        print(f"{'=' * 90}")
        gpu_results = run_gpu_benchmark(
            args.seq_lens, args.heads, args.warmup, args.iters,
            args.device_id, args.stem_block_size, args.stem_stride)

    if args.msprof and args.device == 'npu':
        print(f"\n{'=' * 90}")
        print(f"  msprof Op-Level Profiling")
        print(f"{'=' * 90}")
        msprof_results = run_msprof(
            args.seq_lens, args.heads, max(args.warmup, 5), max(args.iters, 10),
            args.device_id, args.stem_block_size, args.stem_stride, output_dir)
        if msprof_results:
            durations = [r['duration_us'] for r in msprof_results]
            avg_us = sum(durations) / len(durations)
            print(f"\n  msprof avg op duration: {avg_us:.2f} us ({avg_us/1000:.4f} ms)")

    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['device', 'seq_len', 'heads', 'total_tokens',
                                                'latency_ms', 'bandwidth_TBs', 'max_Qb'])
        writer.writeheader()
        for r in npu_results + gpu_results:
            writer.writerow(r)
    print(f"\n  Results saved to {csv_path}")

    if args.msprof:
        print(f"  msprof data saved to {output_dir / 'msprof_output'}")


if __name__ == "__main__":
    main()
