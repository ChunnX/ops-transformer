#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Extract one CSV row from a case's perf archive directory.

Reads:
  <case_dir>/bound.md            → bound_type, scalar_subtype, SCALAR%, SCALARLDST%
  <case_dir>/OpBasicInfo.csv     → task_duration_us, block_dim
  <case_dir>/PipeUtilization.csv → MTE2/MTE3/CUBE/SCALAR/vec ratios (max core)
plus the matching cases.json entry to flatten parameters.

Usage:
    python3 collect_row.py --case-dir <case_dir> --case-id N --cases cases.json [--header]

When --header is passed alone (without --case-dir), prints just the header line.
"""

import argparse
import csv
import json
import os
import re
import sys


COLUMNS = [
    "case_id", "tag", "scenario", "status",
    "in_layout", "dtype", "out_dtype", "batch_size",
    "q_head_num", "kv_head_num", "q_seq", "kv_seq", "head_dim",
    "block_size", "cache_layout",
    "antiquant_mode", "mask_subtype", "sparse_mode",
    "has_pse", "has_block_table",
    "task_duration_us", "block_dim", "core_prefix",
    "bound_type", "scalar_subtype",
    "scalar_ratio_pct",  # msprof 原生 <prefix>_scalar_ratio，SCALAR BOUND 判定输入
    "scalar_pct", "scalarldst_pct",
    "mte2_pct", "mte3_pct", "cube_pct", "vec_ratio_pct",
    "error_msg",
]


def safe_float(val, default=0.0):
    if val is None:
        return default
    s = str(val).strip()
    if s in ("", "N/A", "NA", "nan"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_bound_md(path):
    out = {"bound_type": "", "scalar_subtype": "", "scalar_pct": "",
           "scalarldst_pct": "", "scalar_ratio_pct": "", "core_prefix": ""}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    patterns = {
        "bound_type": r"\*\*bound_type\*\*\s*:\s*([^\n]+)",
        "scalar_subtype": r"\*\*scalar_subtype\*\*\s*:\s*([^\n]+)",
        "scalar_pct": r"\*\*SCALAR%\*\*\s*:\s*([0-9.]+)",
        "scalarldst_pct": r"\*\*SCALARLDST%\*\*\s*:\s*([0-9.]+)",
        "scalar_ratio_pct": r"\*\*scalar_ratio \(msprof native[^*]*\)\*\*\s*:\s*([0-9.]+)",
        "core_prefix": r"\*\*core_prefix\*\*\s*:\s*([^\n]+)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[k] = m.group(1).strip()
    return out


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def parse_op_basic(case_dir):
    rows = read_csv_rows(os.path.join(case_dir, "OpBasicInfo.csv"))
    if not rows:
        return {"task_duration_us": "", "block_dim": ""}
    r = rows[0]
    return {
        "task_duration_us": str(safe_float(r.get("Task Duration(us)"))),
        "block_dim": str(int(safe_float(r.get("Block Dim", "1")))),
    }


def parse_pipe_max(case_dir, prefix_hint):
    """与 msprof_pipeline_analyze.py:per_core_busy 同口径：用 `_time(us) / <prefix>_time(us)`。"""
    rows = read_csv_rows(os.path.join(case_dir, "PipeUtilization.csv"))
    if not rows:
        return {"mte2_pct": "", "mte3_pct": "", "cube_pct": "", "vec_ratio_pct": ""}

    prefix = prefix_hint or "aiv"
    if not prefix_hint:
        aiv_t = safe_float(rows[0].get("aiv_time(us)"))
        aic_t = safe_float(rows[0].get("aic_time(us)"))
        prefix = "aic" if aic_t > aiv_t else "aiv"

    max_row = max(rows, key=lambda r: safe_float(r.get(f"{prefix}_time(us)")))
    total = safe_float(max_row.get(f"{prefix}_time(us)"))

    def time_pct(time_field):
        if total <= 0:
            return ""
        return f"{safe_float(max_row.get(time_field)) / total * 100:.1f}"

    out = {
        "mte2_pct": time_pct(f"{prefix}_mte2_time(us)"),
        "mte3_pct": time_pct(f"{prefix}_mte3_time(us)"),
    }
    if prefix == "aic":
        out["cube_pct"] = time_pct("aic_cube_time(us)")
        out["vec_ratio_pct"] = ""  # aic 路径无 vec 流水
    else:
        out["cube_pct"] = ""
        out["vec_ratio_pct"] = time_pct("aiv_vec_time(us)")
    return out


def emit_header():
    return ",".join(COLUMNS)


def emit_row(case, case_dir, status, error_msg):
    bound = parse_bound_md(os.path.join(case_dir, "bound.md"))
    basic = parse_op_basic(case_dir)
    pipe = parse_pipe_max(case_dir, bound.get("core_prefix") or None)

    row = {c: "" for c in COLUMNS}
    row["case_id"] = case.get("case_id", "")
    row["tag"] = case.get("tag", "")
    row["scenario"] = case.get("scenario", "no_quant")
    row["status"] = status
    for k in ("in_layout", "dtype", "out_dtype", "batch_size", "q_head_num", "kv_head_num",
              "q_seq", "kv_seq", "head_dim", "block_size", "cache_layout",
              "antiquant_mode", "mask_subtype", "sparse_mode"):
        row[k] = case.get(k, "")
    # Derived flags inferred from case rather than from bound.md.
    sub = case.get("mask_subtype", "")
    has_pse = "1" if sub == "mask_pse_alibi" else ""
    has_bt = "" if case.get("scenario") == "contiguous" else "1"
    row["has_pse"] = has_pse
    row["has_block_table"] = has_bt
    row.update(basic)
    row.update(pipe)
    row["bound_type"] = bound.get("bound_type", "")
    row["scalar_subtype"] = bound.get("scalar_subtype", "")
    row["scalar_ratio_pct"] = bound.get("scalar_ratio_pct", "")
    row["scalar_pct"] = bound.get("scalar_pct", "")
    row["scalarldst_pct"] = bound.get("scalarldst_pct", "")
    row["core_prefix"] = bound.get("core_prefix", "")
    row["error_msg"] = error_msg

    # CSV quoting — keep simple by replacing commas/newlines in error_msg.
    if row["error_msg"]:
        row["error_msg"] = row["error_msg"].replace(",", ";").replace("\n", " ")

    return ",".join(str(row[c]) for c in COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Emit one CSV row per FIA perf case.")
    parser.add_argument("--header", action="store_true", help="Print header only and exit")
    parser.add_argument("--case-dir", help="Path to docs/perf/case_<id>/")
    parser.add_argument("--case-id", type=int, help="Case id to look up in cases.json")
    parser.add_argument("--cases", help="Path to cases.json")
    parser.add_argument("--status", default="OK", help="OK | ERROR")
    parser.add_argument("--error-msg", default="", help="Error message if status=ERROR")
    args = parser.parse_args()

    if args.header and not args.case_dir:
        print(emit_header())
        return

    if not (args.case_dir and args.cases and args.case_id is not None):
        print("Error: need --case-dir, --case-id, --cases", file=sys.stderr)
        sys.exit(2)

    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)
    case = next((c for c in cases if c["case_id"] == args.case_id), None)
    if case is None:
        print(f"Error: case_id {args.case_id} not in {args.cases}", file=sys.stderr)
        sys.exit(2)

    print(emit_row(case, args.case_dir, args.status, args.error_msg))


if __name__ == "__main__":
    main()
