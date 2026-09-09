#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Compare fused ClaGateMxQuantWithDualAxis against the equivalent small-op chain.

This script does NOT need profiling directories. You only maintain:

  1. SHAPES: all shapes you want to benchmark
  2. MODES: generated parameter combinations
  3. run_fused / run_small_op: the real fused/small-op call code

The small-op chain mirrors:
    sigmoid(global_gate) * global_out
    + sigmoid(local_gate) * local_out
    -> row/col quant (according to quant_mode)

For each mode and shape it runs warmup + repeat iterations and reports the mean
of the last `tail` iterations.

Examples:
    # use mode 1 and benchmark all SHAPES
    python3 perf_compare_fused_vs_small.py --mode 1 --repeat 20

    # benchmark a single shape by temporarily editing SHAPES, or add --shape-index
    python3 perf_compare_fused_vs_small.py --mode 2 --shape-index 0 --repeat 20
"""

import argparse
import itertools
import statistics
from pathlib import Path

import torch
import torch_npu
import cann_ops_transformer

# ============================================================================
# User editable configuration
# ============================================================================

# Shape list: each entry is (T, N, D). global_out/local_out = [T,N,D], gate = [T,N].
SHAPES = [
    (1, 24, 128),
    (1, 24, 256),
    (1, 64, 128),
    (1, 64, 256),
    (1, 96, 128),
    (1, 96, 256),
    (1, 128, 128),
    (1, 128, 256),

    (16, 24, 128),
    (16, 24, 256),
    (16, 64, 128),
    (16, 64, 256),
    (16, 96, 128),
    (16, 96, 256),
    (16, 128, 128),
    (16, 128, 256),

    (32, 24, 128),
    (32, 24, 256),
    (32, 64, 128),
    (32, 64, 256),
    (32, 96, 128),
    (32, 96, 256),
    (32, 128, 128),
    (32, 128, 256),

    (45, 24, 128),
    (45, 24, 256),
    (45, 64, 128),
    (45, 64, 256),
    (45, 96, 128),
    (45, 96, 256),
    (45, 128, 128),
    (45, 128, 256),

    (64, 24, 128),
    (64, 24, 256),
    (64, 64, 128),
    (64, 64, 256),
    (64, 96, 128),
    (64, 96, 256),
    (64, 128, 128),
    (64, 128, 256),

    (96, 24, 128),
    (96, 24, 256),
    (96, 64, 128),
    (96, 64, 256),
    (96, 96, 128),
    (96, 96, 256),
    (96, 128, 128),
    (96, 128, 256),

    (128, 24, 128),
    (128, 24, 256),
    (128, 64, 128),
    (128, 64, 256),
    (128, 96, 128),
    (128, 96, 256),
    (128, 128, 128),
    (128, 128, 256),

    (256, 24, 128),
    (256, 24, 256),
    (256, 64, 128),
    (256, 64, 256),
    (256, 96, 128),
    (256, 96, 256),
    (256, 128, 128),
    (256, 128, 256),

    (512, 24, 128),
    (512, 24, 256),
    (512, 64, 128),
    (512, 64, 256),
    (512, 96, 128),
    (512, 96, 256),
    (512, 128, 128),
    (512, 128, 256),

    (1024, 24, 128),
    (1024, 24, 256),
    (1024, 64, 128),
    (1024, 64, 256),
    (1024, 96, 128),
    (1024, 96, 256),
    (1024, 128, 128),
    (1024, 128, 256),

    (2048, 24, 128),
    (2048, 24, 256),
    (2048, 64, 128),
    (2048, 64, 256),
    (2048, 96, 128),
    (2048, 96, 256),
    (2048, 128, 128),
    (2048, 128, 256),

    (4096, 24, 128),
    (4096, 24, 256),
    (4096, 64, 128),
    (4096, 64, 256),
    (4096, 96, 128),
    (4096, 96, 256),
    (4096, 128, 128),
    (4096, 128, 256),

    (8192, 24, 128),
    (8192, 24, 256),
    # (8192, 64, 128),
    # (8192, 64, 256),
    # (8192, 96, 128),
    # (8192, 96, 256),
    # (8192, 128, 128),
    # (8192, 128, 256),

    # (16384, 24, 128),
    # (16384, 24, 256),
    # (16384, 64, 128),
    # (16384, 64, 256),
    # (16384, 96, 128),
    # (16384, 96, 256),
    # (16384, 128, 128),
    # (16384, 128, 256),

    # (32768, 24, 128),
    # (32768, 24, 256),
    # (32768, 64, 128),
    # (32768, 64, 256),
    # (32768, 96, 128),
    # (32768, 96, 256),
    # (32768, 128, 128),
    # (32768, 128, 256),
]
# Input dtype: "bf16" or "fp16". You can extend SHAPES with dtype if needed.
INPUT_DTYPE = "bf16"

# dst_type values and their friendly names.
DST_TYPES = [
    # (dst_type, label)
    (35, "e5m2"),
    (36, "e4m3fn"),
]

SCALE_ALGS = [0, 1]
QUANT_MODES = [0, 1]
ROUND_MODES = ["rint"]


def _dst_torch_dtype(dst_type):
    if dst_type == 35:
        return getattr(torch, "float8_e5m2", None) or getattr(torch_npu, "float8_e5m2", None)
    if dst_type == 36:
        return getattr(torch, "float8_e4m3fn", None) or getattr(torch_npu, "float8_e4m3fn", None)
    if dst_type == 40:
        return 296  # torch_npu FP4 packed enum
    if dst_type == 41:
        return 297
    raise ValueError(f"unsupported dst_type: {dst_type}")


def _input_dtype():
    if INPUT_DTYPE == "bf16":
        return torch.bfloat16
    if INPUT_DTYPE == "fp16":
        return torch.float16
    raise ValueError(f"unsupported INPUT_DTYPE: {INPUT_DTYPE}")


def _make_mode_list():
    """Auto generate all supported parameter combinations."""
    modes = []
    idx = 1
    for quant_mode in QUANT_MODES:
        for dst_type, dst_name in DST_TYPES:
            for scale_alg in SCALE_ALGS:
                if dst_type in (35, 36):
                    round_modes = ["rint"]
                else:
                    round_modes = ROUND_MODES
                for round_mode in round_modes:
                    modes.append({
                        "mode": idx,
                        "name": f"mode{idx}_q{quant_mode}_dst{dst_name}_alg{scale_alg}_{round_mode}",
                        "dst_type": dst_type,
                        "dst_label": dst_name,
                        "scale_alg": scale_alg,
                        "round_mode": round_mode,
                        "quant_mode": quant_mode,
                    })
                    idx += 1
    return modes


MODES = _make_mode_list()


# ============================================================================
# Operator call code. Modify these two functions for your actual op names.
# ============================================================================

def run_fused(global_out, local_out, global_gate, local_gate, mode):
    return cann_ops_transformer.cla_gate_mx_quant_with_dual_axis(
        global_out,
        local_out,
        global_gate,
        local_gate,
        round_mode=mode["round_mode"],
        scale_alg=mode["scale_alg"],
        dst_type=mode["dst_type"],
        quant_mode=mode["quant_mode"],
    )


def run_small_chain(global_out, local_out, global_gate, local_gate, mode):
    """Run the equivalent small-op chain and return each op's result.

    The returned dict can be used to inspect intermediate results if needed.
    """
    gg = global_gate
    lg = local_gate

    # Small ops in execution order.
    global_sigmoid = torch.sigmoid(gg)
    local_sigmoid = torch.sigmoid(lg)

    global_gated = global_out * global_sigmoid.unsqueeze(-1)
    local_gated = local_out * local_sigmoid.unsqueeze(-1)

    merged = global_gated + local_gated
    merged = merged.reshape(merged.shape[0], -1)

    dst = _dst_torch_dtype(mode["dst_type"])
    if mode["quant_mode"] == 0:
        row_data, row_scale = torch_npu.npu_dynamic_mx_quant(
            merged,
            round_mode=mode["round_mode"],
            dst_type=dst,
            scale_alg=mode["scale_alg"],
        )
        col_data = torch.empty((0,), dtype=row_data.dtype, device=row_data.device)
        col_scale = torch.empty((0,), dtype=row_scale.dtype, device=row_scale.device)
        return row_data, row_scale, col_data, col_scale

    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged,
        round_mode=mode["round_mode"],
        dst_type=dst,
        scale_alg=mode["scale_alg"],
    )


# ============================================================================
# Performance measurement helpers
# ============================================================================

def _new_event():
    return torch.npu.Event(enable_timing=True)


def _us(ms):
    return ms * 1000.0


def time_fused(inputs, mode, repeat, warmup):
    for _ in range(warmup):
        run_fused(*inputs, mode)
    torch.npu.synchronize()

    durations = []
    for _ in range(repeat):
        start = _new_event()
        end = _new_event()
        start.record()
        run_fused(*inputs, mode)
        end.record()
        torch.npu.synchronize()
        durations.append(_us(start.elapsed_time(end)))
    return durations


def time_small_ops(inputs, mode, repeat, warmup):
    for _ in range(warmup):
        run_small_chain(*inputs, mode)
    torch.npu.synchronize()

    op_names = [
        "sigmoid(global_gate)",
        "sigmoid(local_gate)",
        "mul(global_out*sig)",
        "mul(local_out*sig)",
        "add(global+local)",
        "quant",
    ]
    per_op = {name: [] for name in op_names}

    # More robust method: time each op one by one while keeping dependencies.
    # Since all ops are on the same stream, events around each op measure the
    # op's GPU time without syncing between ops.
    for _ in range(repeat):
        gg = inputs[2]
        lg = inputs[3]
        go = inputs[0]
        lo = inputs[1]

        e0 = _new_event()
        e1 = _new_event()
        e2 = _new_event()
        e3 = _new_event()
        e4 = _new_event()
        e5 = _new_event()
        e6 = _new_event()

        e0.record()
        gs = torch.sigmoid(gg)
        e1.record()

        ls = torch.sigmoid(lg)
        e2.record()

        gg_out = go * gs.unsqueeze(-1)
        e3.record()

        lg_out = lo * ls.unsqueeze(-1)
        e4.record()

        merged = gg_out + lg_out
        e5.record()

        merged2d = merged.reshape(merged.shape[0], -1)
        dst = _dst_torch_dtype(mode["dst_type"])
        if mode["quant_mode"] == 0:
            torch_npu.npu_dynamic_mx_quant(
                merged2d,
                round_mode=mode["round_mode"],
                dst_type=dst,
                scale_alg=mode["scale_alg"],
            )
        else:
            torch_npu.npu_dynamic_mx_quant_with_dual_axis(
                merged2d,
                round_mode=mode["round_mode"],
                dst_type=dst,
                scale_alg=mode["scale_alg"],
            )
        e6.record()
        torch.npu.synchronize()

        times = [
            e0.elapsed_time(e1),
            e1.elapsed_time(e2),
            e2.elapsed_time(e3),
            e3.elapsed_time(e4),
            e4.elapsed_time(e5),
            e5.elapsed_time(e6),
        ]
        for name, t in zip(op_names, times):
            per_op[name].append(_us(t))

    return per_op


def mean_last(values, tail):
    if not values:
        return 0.0, []
    last = values[-tail:] if tail and len(values) > tail else values
    return statistics.mean(last), last


# ============================================================================
# Report generation
# ============================================================================

def run_mode(mode, shapes, repeat, tail, device):
    dtype = _input_dtype()
    report = []
    rows = []

    report.append(f"# Mode {mode['mode']}: {mode['name']}")
    report.append("")
    report.append(f"- input dtype: {INPUT_DTYPE}")
    report.append(f"- dst_type={mode['dst_type']} ({mode['dst_label']}), "
                  f"scale_alg={mode['scale_alg']}, round_mode={mode['round_mode']}, "
                  f"quant_mode={mode['quant_mode']}")
    report.append(f"- repeat={repeat}, report uses last {tail} iterations")
    report.append("")

    for shape in shapes:
        t, n, d = shape
        torch.manual_seed(12345 + t + n + d)
        gen = torch.Generator(device="cpu").manual_seed(12345 + t + n + d)
        global_out = (torch.rand((t, n, d), generator=gen, dtype=torch.float32) * 2 - 1).to(dtype).to(device)
        local_out = (torch.rand((t, n, d), generator=gen, dtype=torch.float32) * 2 - 1).to(dtype).to(device)
        global_gate = (torch.rand((t, n), generator=gen, dtype=torch.float32) * 6 - 3).to(dtype).to(device)
        local_gate = (torch.rand((t, n), generator=gen, dtype=torch.float32) * 6 - 3).to(dtype).to(device)
        inputs = (global_out, local_out, global_gate, local_gate)

        warmup = max(2, repeat // 4)
        fused_durations = time_fused(inputs, mode, repeat, warmup)
        small_per_op = time_small_ops(inputs, mode, repeat, warmup)

        fused_mean, fused_last = mean_last(fused_durations, tail)
        small_means = {}
        for op, values in small_per_op.items():
            small_means[op], last = mean_last(values, tail)
        small_total = sum(small_means.values())

        shape_name = f"T{t}_N{n}_D{d}"
        report.append(f"## {shape_name}")
        report.append("")
        report.append("| Item | Mean(us) | Min(us) | Max(us) |")
        report.append("|---|---:|---:|---:|")
        for op, values in small_per_op.items():
            _, last = mean_last(values, tail)
            report.append(f"| Small {op} | {small_means[op]:.3f} | {min(last):.3f} | {max(last):.3f} |")
        report.append(f"| **Small total** | **{small_total:.3f}** | | |")
        report.append(f"| **Fused** | **{fused_mean:.3f}** | {min(fused_last):.3f} | {max(fused_last):.3f} |")
        report.append("")
        if small_total:
            report.append(f"- Fused/SmallTotal: {fused_mean / small_total:.4f}")
            report.append(f"- SmallTotal/Fused speedup: {small_total / fused_mean:.4f}")
        else:
            report.append("- SmallTotal is 0")
        report.append("")
        rows.append({
            "shape": shape_name,
            "fused_us": fused_mean,
            "small_total_us": small_total,
            "ratio": fused_mean / small_total if small_total else float("nan"),
        })

    return "\n".join(report), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, default=1,
                        help=f"mode index to benchmark; available modes: {[m['mode'] for m in MODES]}")
    parser.add_argument("--shape-index", type=int, default=-1,
                        help="if >=0, benchmark only SHAPES[shape-index]; otherwise all shapes")
    parser.add_argument("--repeat", type=int, default=20, help="total measured iterations")
    parser.add_argument("--tail", type=int, default=10, help="last N iterations used for mean")
    parser.add_argument("--device", type=int, default=6, help="NPU device id")
    parser.add_argument("--report", default="perf_fused_vs_small_report.md")
    args = parser.parse_args()

    mode = next((m for m in MODES if m["mode"] == args.mode), None)
    if mode is None:
        parser.error(f"--mode {args.mode} is invalid. Available: {[m['mode'] for m in MODES]}")

    shapes = SHAPES if args.shape_index < 0 else [SHAPES[args.shape_index]]
    if not shapes:
        parser.error("SHAPES is empty")

    torch.npu.set_device(args.device)
    print(f"Running mode {mode['mode']}: {mode['name']} on npu:{args.device}")
    report_text, rows = run_mode(mode, shapes, args.repeat, args.tail, args.device)

    Path(args.report).write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[Report saved] {args.report}")


if __name__ == "__main__":
    main()
