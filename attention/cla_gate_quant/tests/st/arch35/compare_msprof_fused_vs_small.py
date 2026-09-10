#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Compare fused operator performance with the sum of small operators from msprof
profiling products.

Usage examples:
  python3 compare_msprof_fused_vs_small.py \
      --fused-dir /path/to/PROF_fused \
      --small-dir /path/to/PROF_small \
      --shape T65_N24_D256 \
      --report perf_report.md

  python3 compare_msprof_fused_vs_small.py \
      --fused-dir /path/to/PROF_fused \
      --small-dir /path/to/PROF_small \
      --fused-op ClaGateQuant \
      --small-op Sigmoid \
      --small-op Mul \
      --small-op Mul \
      --small-op Add \
      --small-op DynamicMxQuantWithDualAxis \
      --tail 10
"""

import argparse
import csv
import glob
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# If you want to maintain multiple shape comparisons in one script, edit this
# list. Each entry can point to separate msprof products for fused/small ops.
# ---------------------------------------------------------------------------
SHAPE_CASES = [
    # {
    #     "name": "T65_N24_D256_e4m3fn_alg1",
    #     "fused_dir": "/path/to/PROF_fused_T65",
    #     "small_dir": "/path/to/PROF_small_T65",
    # },
]

DEFAULT_FUSED_OPS = ["ClaGateQuant"]
DEFAULT_SMALL_OPS = ["Sigmoid", "Mul", "Mul", "Add", "DynamicMxQuantWithDualAxis"]


def find_op_summary(prof_dir):
    """Return the op_summary csv under mindstudio_profiler_output."""
    root = Path(prof_dir)
    if root.is_dir():
        matches = sorted(root.glob("mindstudio_profiler_output/op_summary_*.csv"))
        if not matches:
            matches = sorted(root.glob("**/op_summary_*.csv"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No op_summary_*.csv found under {prof_dir}")


def read_op_summary(prof_dir):
    csv_file = find_op_summary(prof_dir)
    rows = []
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            op_type = raw.get("OP Type", "").strip()
            if not op_type:
                continue
            try:
                duration = float(str(raw.get("Task Duration(us)", "")).strip())
            except ValueError:
                continue
            rows.append({
                "op_type": op_type,
                "op_name": raw.get("Op Name", "").strip(),
                "duration_us": duration,
                "input_shapes": raw.get("Input Shapes", "").strip(),
            })
    if not rows:
        raise RuntimeError(f"No valid op_summary rows found in {csv_file}")
    return rows


def partition_rows(rows, anchor_ops):
    """Split rows into iterations.

    For small-op runs, each iteration ends with the last small op, e.g.
    DynamicMxQuantWithDualAxis. For fused runs, every fused kernel row can be
    treated as one iteration.
    """
    anchors = set(anchor_ops)
    groups = []
    start = 0
    for i, row in enumerate(rows):
        if row["op_type"] in anchors:
            # If there are non-anchor rows before this anchor, include them in
            # the same iteration. start is updated after the previous anchor.
            groups.append(rows[start:i + 1])
            start = i + 1
    # Usually the profile starts with iteration 0 and ends at an anchor. If any
    # trailing rows remain, they are ignored unless the user wants them.
    return groups


def mean_tail(values, tail):
    if len(values) <= tail:
        return statistics.mean(values), values
    last = values[-tail:]
    return statistics.mean(last), last


def summarize_rows(groups, op_types, tail):
    """For each op type, compute the mean of per-iteration sum from last N iters."""
    per_type_values = {op: [] for op in op_types}
    total_values = []
    used_groups = groups[-tail:] if tail > 0 and len(groups) > tail else groups

    for group in used_groups:
        total = 0.0
        for op in op_types:
            dur = sum(row["duration_us"] for row in group if row["op_type"] == op)
            per_type_values[op].append(dur)
            total += dur
        total_values.append(total)

    summary = {}
    for op, values in per_type_values.items():
        if not values:
            continue
        mean, last = mean_tail(values, tail)
        summary[op] = {
            "mean_us": mean,
            "min_us": min(last),
            "max_us": max(last),
            "count": len(values),
        }
    if total_values:
        mean, last = mean_tail(total_values, tail)
        summary["__sum__"] = {
            "mean_us": mean,
            "min_us": min(last),
            "max_us": max(last),
            "count": len(total_values),
        }
    return summary, len(groups)


def find_csv_columns(prof_dir, cols):
    """Read columns from the first op_summary for optional shape display."""
    try:
        rows = read_op_summary(prof_dir)
        out = {}
        for col in cols:
            if rows:
                out[col] = rows[0].get(col)
        return out
    except Exception:
        return {}


def run_one_case(name, fused_dir, small_dir, fused_ops, small_ops, tail):
    print(f"\n=== {name} ===")

    fused_rows = read_op_summary(fused_dir)
    small_rows = read_op_summary(small_dir)

    fused_groups = partition_rows(fused_rows, fused_ops)
    # Keep execution order but aggregate duplicate op types into one row for report.
    unique_small_ops = list(dict.fromkeys(small_ops))
    small_groups = partition_rows(small_rows, [unique_small_ops[-1]])

    fused_summary, fused_iter_count = summarize_rows(fused_groups, fused_ops, tail)
    small_summary, small_iter_count = summarize_rows(small_groups, unique_small_ops, tail)

    # The fused profile normally has one fused kernel per iteration.
    if fused_ops:
        fused_mean = fused_summary.get(fused_ops[0], {}).get("mean_us")
    else:
        fused_mean = fused_summary.get("__sum__", {}).get("mean_us")

    small_total = small_summary.get("__sum__", {}).get("mean_us")
    if fused_mean is None or small_total is None:
        raise RuntimeError(f"Cannot get valid fused/small duration for {name}")

    small_means = {op: small_summary[op]["mean_us"] for op in unique_small_ops if op in small_summary}

    # markdown text
    lines = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"- Fused profiling iterations: {fused_iter_count}, use last {min(tail, fused_iter_count)}")
    lines.append(f"- Small profiling iterations: {small_iter_count}, use last {min(tail, small_iter_count)}")
    lines.append("")
    lines.append("| Item | Mean Duration (us) | Min (us) | Max (us) | Iterations |")
    lines.append("|---|---:|---:|---:|---:|")
    for op in unique_small_ops:
        if op in small_summary:
            s = small_summary[op]
            lines.append(f"| Small {op} | {s['mean_us']:.3f} | {s['min_us']:.3f} | {s['max_us']:.3f} | {s['count']} |")
    lines.append(f"| **Small total** | **{small_total:.3f}** | "
                 f"{small_summary['__sum__']['min_us']:.3f} | "
                 f"{small_summary['__sum__']['max_us']:.3f} | "
                 f"{small_summary['__sum__']['count']} |")
    lines.append(f"| **Fused {'+'.join(fused_ops)}** | **{fused_mean:.3f}** | "
                 f"{fused_summary[fused_ops[0]]['min_us']:.3f} | "
                 f"{fused_summary[fused_ops[0]]['max_us']:.3f} | "
                 f"{fused_summary[fused_ops[0]]['count']} |")
    lines.append("")
    ratio = fused_mean / small_total if small_total != 0 else float("nan")
    speedup = small_total / fused_mean if fused_mean != 0 else float("nan")
    lines.append(f"- Fused / SmallTotal: **{ratio:.4f}**")
    lines.append(f"- SmallTotal / Fused speedup: **{speedup:.4f}**")
    lines.append("")
    if small_means:
        lines.append("Small op detail (mean us):")
        lines.append("")
        for op, val in small_means.items():
            lines.append(f"  - {op}: {val:.3f}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text, {
        "name": name,
        "fused_mean_us": fused_mean,
        "small_total_us": small_total,
        "fused_small_ratio": ratio,
        "speedup": speedup,
        "small_ops": small_means,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fused-dir", help="msprof product dir of the fused op")
    parser.add_argument("--small-dir", help="msprof product dir of the small-op path")
    parser.add_argument("--shape", default="default_shape", help="shape/comment label in the report")
    parser.add_argument("--fused-op", action="append", default=None,
                        help="fused op OP Type; repeat if needed (default: ClaGateQuant)")
    parser.add_argument("--small-op", action="append", default=None,
                        help="small op OP Type; repeat in execution order (default: Sigmoid,Mul,Mul,Add,DynamicMxQuantWithDualAxis)")
    parser.add_argument("--tail", type=int, default=10,
                        help="use last N iterations for mean (default: 10)")
    parser.add_argument("--report", default="msprof_perf_compare_report.md",
                        help="output markdown report path")
    args = parser.parse_args()

    fused_ops = args.fused_op or DEFAULT_FUSED_OPS
    small_ops = args.small_op or DEFAULT_SMALL_OPS

    if SHAPE_CASES and not (args.fused_dir and args.small_dir):
        reports = []
        results = []
        for case in SHAPE_CASES:
            text, data = run_one_case(
                case["name"],
                case["fused_dir"],
                case["small_dir"],
                fused_ops,
                small_ops,
                args.tail,
            )
            reports.append(text)
            results.append(data)
    elif args.fused_dir and args.small_dir:
        text, data = run_one_case(args.shape, args.fused_dir, args.small_dir,
                                  fused_ops, small_ops, args.tail)
        reports = [text]
        results = [data]
    else:
        parser.error("Provide --fused-dir/--small-dir, or fill SHAPE_CASES in the script")

    report_path = Path(args.report)
    report_path.write_text("\n---\n".join(reports), encoding="utf-8")
    print(f"\n[Report saved] {report_path}")

    if results:
        print("\nSummary:")
        for r in results:
            print(f"- {r['name']}: fused={r['fused_mean_us']:.3f}us, "
                  f"small_total={r['small_total_us']:.3f}us, "
                  f"fused/small={r['fused_small_ratio']:.4f}, "
                  f"small/fused={r['speedup']:.4f}")


if __name__ == "__main__":
    main()
