#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Compare fused operator vs small-op chain using msprof Task Duration.

Workflow:
  1. User only selects a mode and shape list.
  2. This script internally launches `msprof` for every shape.
  3. msprof runs this script in `--inner` mode, which executes the fused op and
     the small-op chain repeat times.
  4. After msprof finishes, this script parses op_summary_*.csv and generates a
     markdown report using Task Duration(us), taking the last `tail` iterations.

Usage:
    python3 perf_compare_msprof.py --mode 7 --repeat 20
    python3 perf_compare_msprof.py --mode 7 --shape-index 0 --repeat 20

Available modes are auto-generated from MODES config.
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Lazy NPU runtime imports. The outer driver process only needs to launch
# msprof, so avoid importing torch_npu/cann_ops_transformer there. They are
# imported when the --inner process actually runs the operators.
torch = None
torch_npu = None
cann_ops_transformer = None


def _ensure_runtime():
    global torch, torch_npu, cann_ops_transformer
    if torch is None:
        import torch
        import torch_npu
        import cann_ops_transformer
        torch = torch
        torch_npu = torch_npu
        cann_ops_transformer = cann_ops_transformer


# ============================================================================
# User editable configuration
# ============================================================================
SHAPES = [
    (1, 24, 128),
    (1, 24, 256),
    (1, 64, 128),
    (1, 64, 256),
    (1, 96, 128),
    (1, 96, 256),
    (1, 128, 128),
    (1, 128, 256),

    (8, 24, 128),
    (8, 24, 256),
    (8, 64, 128),
    (8, 64, 256),
    (8, 96, 128),
    (8, 96, 256),
    (8, 128, 128),
    (8, 128, 256),

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
]

INPUT_DTYPE = "bf16"

DST_TYPES = [
    (35, "e5m2"),
    (36, "e4m3fn"),
]

SCALE_ALGS = [0, 1]
QUANT_MODES = [0, 1]
ROUND_MODES = ["rint"]

# Fused op OP Type as it appears in op_summary.
FUSED_OP_TYPES = ["ClaGateQuant"]

# Small-op OP Types in execution order. No Cast is expected because the small
# path keeps the input dtype and does not insert FP32 conversion overhead.
SMALL_OP_TYPES = ["Sigmoid", "Mul", "Add", "DynamicMxQuant", "DynamicMxQuantWithDualAxis"]
SMALL_OP_ANCHOR = ["DynamicMxQuant", "DynamicMxQuantWithDualAxis"]


def _dst_torch_dtype(dst_type):
    _ensure_runtime()
    if dst_type == 35:
        return getattr(torch, "float8_e5m2", None) or getattr(torch_npu, "float8_e5m2", None)
    if dst_type == 36:
        return getattr(torch, "float8_e4m3fn", None) or getattr(torch_npu, "float8_e4m3fn", None)
    if dst_type == 40:
        return 296
    if dst_type == 41:
        return 297
    raise ValueError(f"unsupported dst_type: {dst_type}")


def _input_dtype():
    _ensure_runtime()
    if INPUT_DTYPE == "bf16":
        return torch.bfloat16
    if INPUT_DTYPE == "fp16":
        return torch.float16
    raise ValueError(f"unsupported INPUT_DTYPE: {INPUT_DTYPE}")


def _make_mode_list():
    modes = []
    idx = 1
    for quant_mode in QUANT_MODES:
        for dst_type, dst_name in DST_TYPES:
            for scale_alg in SCALE_ALGS:
                round_modes = ["rint"] if dst_type in (35, 36) else ROUND_MODES
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


def run_fused(global_out, local_out, global_gate, local_gate, mode):
    _ensure_runtime()
    return cann_ops_transformer.cla_gate_quant(
        global_out,
        local_out,
        global_gate,
        local_gate,
        round_mode=mode["round_mode"],
        scale_alg=mode["scale_alg"],
        dst_type=mode["dst_type"],
        input_attn_layout="TND",
        dual_axis_flag=bool(mode["quant_mode"]),
    )


def run_small_chain(global_out, local_out, global_gate, local_gate, mode):
    _ensure_runtime()
    """Small-op path; modify this to match the actual unfused implementation.

    Keep the same dtype as the original inputs; do NOT insert extra FP32 casts
    only for the benchmark, because that would add extra performance overhead.
    """
    global_scale = torch.sigmoid(global_gate).unsqueeze(-1)
    local_scale = torch.sigmoid(local_gate).unsqueeze(-1)

    merged = global_out * global_scale + local_out * local_scale
    merged = merged.reshape(merged.shape[0], -1)

    dst = _dst_torch_dtype(mode["dst_type"])
    if mode["quant_mode"] == 0:
        return torch_npu.npu_dynamic_mx_quant(
            merged,
            round_mode=mode["round_mode"],
            dst_type=dst,
            scale_alg=mode["scale_alg"],
        )
    return torch_npu.npu_dynamic_mx_quant_with_dual_axis(
        merged,
        round_mode=mode["round_mode"],
        dst_type=dst,
        scale_alg=mode["scale_alg"],
    )


def _make_inputs(shape, device):
    _ensure_runtime()
    t, n, d = shape
    torch.manual_seed(12345 + t + n + d)
    gen = torch.Generator(device="cpu").manual_seed(12345 + t + n + d)
    dtype = _input_dtype()
    global_out = (torch.rand((t, n, d), generator=gen, dtype=torch.float32) * 2 - 1).to(dtype).to(device)
    local_out = (torch.rand((t, n, d), generator=gen, dtype=torch.float32) * 2 - 1).to(dtype).to(device)
    global_gate = (torch.rand((t, n), generator=gen, dtype=torch.float32) * 6 - 3).to(dtype).to(device)
    local_gate = (torch.rand((t, n), generator=gen, dtype=torch.float32) * 6 - 3).to(dtype).to(device)
    return global_out, local_out, global_gate, local_gate


# ============================================================================
# Inner mode: this code is executed under msprof.
# ============================================================================

def inner_main(mode, shape_index, repeat, device):
    _ensure_runtime()
    torch.npu.set_device(device)
    shape = SHAPES[shape_index]
    inputs = _make_inputs(shape, device)
    warmup = max(2, repeat // 4)

    # Warmup is also profiled by msprof, but the report only uses the last tail.
    for _ in range(warmup):
        run_fused(*inputs, mode)
    for _ in range(repeat):
        run_fused(*inputs, mode)

    for _ in range(warmup):
        run_small_chain(*inputs, mode)
    for _ in range(repeat):
        run_small_chain(*inputs, mode)

    torch.npu.synchronize()


# ============================================================================
# Parse msprof output
# ============================================================================

def find_op_summary_candidates(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob("mindstudio_profiler_output/op_summary_*.csv")) + \
           sorted(root.rglob("op_summary_*.csv"))


def read_op_summary(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            op_type = raw.get("OP Type", "").strip()
            if not op_type:
                continue
            try:
                dur = float(str(raw.get("Task Duration(us)", "")).strip())
            except ValueError:
                continue
            rows.append({"op_type": op_type, "duration_us": dur})
    return rows


def partition_rows(rows, anchor_ops):
    anchors = set(anchor_ops)
    groups = []
    start = 0
    for i, row in enumerate(rows):
        if row["op_type"] in anchors:
            groups.append(rows[start:i + 1])
            start = i + 1
    return groups


def analyze_dir(prof_dir, mode):
    """Analyze one msprof product for one shape."""
    csv_files = find_op_summary_candidates(prof_dir)
    if not csv_files:
        raise FileNotFoundError(f"No op_summary csv under {prof_dir}")
    all_rows = []
    for f in csv_files:
        all_rows.extend(read_op_summary(f))

    fused_rows = [r for r in all_rows if r["op_type"] in FUSED_OP_TYPES]
    small_rows = [r for r in all_rows if r["op_type"] in SMALL_OP_TYPES]

    fused_groups = partition_rows(fused_rows, FUSED_OP_TYPES)
    small_groups = partition_rows(small_rows, SMALL_OP_ANCHOR)

    if not fused_groups or not small_groups:
        raise RuntimeError(f"No fused/small groups found in {prof_dir}")

    tail = 10
    fused_last = fused_groups[-tail:]
    small_last = small_groups[-tail:]

    fused_mean = statistics.mean(
        sum(r["duration_us"] for r in g) for g in fused_last
    )

    present_small_types = [op for op in SMALL_OP_TYPES if any(r["op_type"] == op for r in small_rows)]
    small_means = {op: [] for op in present_small_types}
    small_total = 0.0
    for group in small_last:
        iter_total = 0.0
        for op in small_means:
            d = sum(r["duration_us"] for r in group if r["op_type"] == op)
            small_means[op].append(d)
            iter_total += d
        small_total += iter_total
    small_total /= len(small_last)
    small_means = {op: statistics.mean(v) for op, v in small_means.items()}

    return {
        "fused_mean_us": fused_mean,
        "small_means_us": small_means,
        "small_total_us": small_total,
        "fused_iters": len(fused_groups),
        "small_iters": len(small_groups),
    }


# ============================================================================
# Report generation
# ============================================================================

def generate_report(mode, shape_results, tail):
    lines = []
    lines.append(f"# Mode {mode['mode']}: {mode['name']}")
    lines.append("")
    lines.append(f"- dst_type={mode['dst_type']} ({mode['dst_label']}), "
                 f"scale_alg={mode['scale_alg']}, round_mode={mode['round_mode']}, "
                 f"quant_mode={mode['quant_mode']}")
    lines.append(f"- report uses last {tail} msprof iterations")
    lines.append("")
    for shape_name, res in shape_results:
        lines.append(f"## {shape_name}")
        lines.append("")
        lines.append("| Item | Mean(us) |")
        lines.append("|---|---:|")
        for op, val in res["small_means_us"].items():
            lines.append(f"| Small {op} | {val:.3f} |")
        lines.append(f"| **Small total** | **{res['small_total_us']:.3f}** |")
        lines.append(f"| **Fused** | **{res['fused_mean_us']:.3f}** |")
        lines.append("")
        ratio = res["fused_mean_us"] / res["small_total_us"] if res["small_total_us"] else float("nan")
        speedup = res["small_total_us"] / res["fused_mean_us"] if res["fused_mean_us"] else float("nan")
        lines.append(f"- Fused/SmallTotal: {ratio:.4f}")
        lines.append(f"- SmallTotal/Fused speedup: {speedup:.4f}")
        lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--shape-index", type=int, default=-1)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--device", type=int, default=6)
    parser.add_argument("--report", default="perf_msprof_fused_vs_small_report.md")
    parser.add_argument("--work-dir", default=None,
                        help="where to store msprof PROF_* products; default is a temp dir")
    parser.add_argument("--inner", action="store_true",
                        help="internal flag: run under msprof")
    parser.add_argument("--collect-only", action="store_true",
                        help="do not run msprof; collect/analyze existing PROF results in --work-dir")
    args = parser.parse_args()

    mode = next((m for m in MODES if m["mode"] == args.mode), None)
    if mode is None:
        parser.error(f"invalid --mode {args.mode}")

    if args.inner:
        if args.shape_index < 0:
            parser.error("--inner requires --shape-index")
        inner_main(mode, args.shape_index, args.repeat, args.device)
        return

    shapes = SHAPES if args.shape_index < 0 else [SHAPES[args.shape_index]]
    if not shapes:
        parser.error("SHAPES is empty")

    script = Path(__file__).resolve()
    work_root = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="msprof_cla_"))
    work_root.mkdir(parents=True, exist_ok=True)

    shape_results = []
    failed = []
    total_shapes = len(shapes)

    if args.collect_only:
        for idx, shape in enumerate(shapes):
            shape_name = f"T{shape[0]}_N{shape[1]}_D{shape[2]}"
            shape_dir = work_root / f"shape_{idx}"
            result_file = shape_dir / "result.json"
            done_file = shape_dir / "DONE"

            if done_file.exists() and result_file.exists():
                try:
                    res = json.loads(result_file.read_text(encoding="utf-8"))
                    shape_results.append((shape_name, res))
                    print(f"[skip] {shape_name} already has result", flush=True)
                    continue
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] invalid {result_file}: {exc}", flush=True)

            prof_dirs = []
            if shape_dir.exists():
                prof_dirs = sorted(shape_dir.glob("PROF_*"))
                for attempt_dir in sorted(shape_dir.glob("attempt_*")):
                    prof_dirs.extend(sorted(attempt_dir.glob("PROF_*")))

            if not prof_dirs:
                failed.append((shape_name, "no PROF directory found"))
                print(f"[fail] {shape_name}: no PROF directory found", flush=True)
                continue

            found = prof_dirs[0]
            try:
                res = analyze_dir(found, mode)
            except Exception as exc:  # noqa: BLE001
                failed.append((shape_name, str(exc)))
                print(f"[fail] {shape_name}: analyze error: {exc}", flush=True)
                continue

            shape_dir.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(res), encoding="utf-8")
            done_file.write_text("ok\n", encoding="utf-8")
            shape_results.append((shape_name, res))
            print(f"[done] {shape_name} collected from {found}", flush=True)

        if shape_results:
            report = generate_report(mode, shape_results, args.tail)
            Path(args.report).write_text(report, encoding="utf-8")
            print(report)
            print(f"\n[Report saved] {Path(args.report)}")
        else:
            print("[error] no shape result available; report not generated", flush=True)
        print(f"[summary] completed={len(shape_results)}/{total_shapes}, failed={len(failed)}")
        if failed:
            print("[failed shapes]")
            for name, reason in failed:
                print(f"  - {name}: {reason}")
        return

    for idx, shape in enumerate(shapes):
        shape_name = f"T{shape[0]}_N{shape[1]}_D{shape[2]}"
        shape_dir = work_root / f"shape_{idx}"
        result_file = shape_dir / "result.json"
        done_file = shape_dir / "DONE"

        # Resume: if this shape was already successfully analyzed, load it.
        if done_file.exists() and result_file.exists():
            try:
                res = json.loads(result_file.read_text(encoding="utf-8"))
                shape_results.append((shape_name, res))
                print(f"[skip] {shape_name} already done", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] failed to load {result_file}: {exc}; will rerun", flush=True)

        shape_dir.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            attempt += 1
            prof_output = shape_dir / f"attempt_{attempt}"
            if prof_output.exists():
                # Previous failed attempt exists; continue with a new attempt dir.
                continue
            break
        prof_output.mkdir(parents=True, exist_ok=True)
        print(f"[{idx+1}/{total_shapes}] profiling shape {shape_name} ...", flush=True)

        app_cmd = (
            f"{sys.executable} {script} --inner --mode {args.mode} "
            f"--shape-index {idx} --repeat {args.repeat} --device {args.device}"
        )
        cmd = [
            "msprof",
            f"--output={prof_output}",
            f"--application={app_cmd}",
        ]
        print(" ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[fail] {shape_name} msprof failed: {exc}", flush=True)
            failed.append((shape_name, "msprof subprocess failed"))
            continue

        # Locate actual PROF_* directory created by msprof.
        found = None
        for p in prof_output.rglob("PROF_*"):
            if p.is_dir():
                found = p
                break
        if found is None:
            found = prof_output
        try:
            res = analyze_dir(found, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {shape_name} analyze failed: {exc}", flush=True)
            failed.append((shape_name, str(exc)))
            continue

        # Save successful result for breakpoint-resume.
        result_file.write_text(json.dumps(res), encoding="utf-8")
        done_file.write_text("ok\n", encoding="utf-8")
        shape_results.append((shape_name, res))
        print(f"[done] {shape_name}", flush=True)

    if shape_results:
        report = generate_report(mode, shape_results, args.tail)
        report_path = Path(args.report)
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"\n[Report saved] {report_path}")
    else:
        print("[error] no shape result available; report not generated", flush=True)

    print(f"[msprof products] {work_root}")
    print(f"[summary] completed={len(shape_results)}/{total_shapes}, failed={len(failed)}")
    if failed:
        print("[failed shapes]")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        print("\nRerun the same command to retry only failed shapes; successful shapes are skipped.")


if __name__ == "__main__":
    main()
