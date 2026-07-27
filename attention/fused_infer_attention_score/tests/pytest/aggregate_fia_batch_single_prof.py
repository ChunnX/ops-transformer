#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from pathlib import Path


DEFAULT_OUTPUT_NAME = "fused_infer_attention_score_summary.csv"
OPERATOR_NAME = "FusedInferAttentionScore"


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_manifest(output_dir):
    manifest = output_dir / "summary_manifest.csv"
    if manifest.exists():
        rows = _read_csv_rows(manifest)
        return [
            {
                "case_name": row.get("case_name", ""),
                "op_summary": row.get("op_summary", ""),
                "log": row.get("log", ""),
            }
            for row in rows
            if row.get("op_summary")
        ]

    entries = []
    for op_summary in sorted(output_dir.glob("*.csv")):
        if op_summary.name in ("summary_manifest.csv", DEFAULT_OUTPUT_NAME):
            continue
        log = op_summary.with_suffix(".log")
        entries.append({
            "case_name": op_summary.stem,
            "op_summary": str(op_summary),
            "log": str(log) if log.exists() else "",
        })
    return entries


def _extract_perf_from_log(log_path):
    if not log_path or not Path(log_path).exists():
        return "", ""

    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    flops = ""
    mfu_time = ""

    flops_match = re.search(r"^FLOPS=([0-9]+(?:\.[0-9]+)?)\s*$", text, re.MULTILINE)
    if not flops_match:
        flops_match = re.search(r"FLOPS计算公式:.*=\s*([0-9]+(?:\.[0-9]+)?)", text)
    if flops_match:
        flops = flops_match.group(1)

    mfu_match = re.search(r"^MFU\*时间\(us\)=([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    if not mfu_match:
        mfu_match = re.search(r"MFU\*时间计算公式:.*=\s*([0-9.eE+-]+)", text)
    if mfu_match:
        mfu_time = mfu_match.group(1)
    return flops, mfu_time


def _op_name_columns(fieldnames):
    preferred = []
    for field in fieldnames:
        normalized = re.sub(r"[\s_]+", "", field).lower()
        if normalized in ("opname", "optype"):
            preferred.append(field)
    return preferred


def _find_fia_row(op_summary_path):
    rows = _read_csv_rows(op_summary_path)
    if not rows:
        return None, []

    fieldnames = list(rows[0].keys())
    op_cols = _op_name_columns(fieldnames)
    if not op_cols:
        return None, fieldnames

    for row in rows:
        for col in op_cols:
            if (row.get(col) or "").strip() == OPERATOR_NAME:
                return row, fieldnames

    for row in rows:
        for col in op_cols:
            value = row.get(col) or ""
            if OPERATOR_NAME in value:
                return row, fieldnames
    return None, fieldnames


def _append_field(fieldnames, field):
    if field not in fieldnames:
        fieldnames.append(field)


def aggregate(output_dir, output_path):
    output_dir = Path(output_dir).resolve()
    output_path = Path(output_path).resolve()
    entries = _load_manifest(output_dir)

    rows = []
    fieldnames = []
    for entry in entries:
        op_summary = entry["op_summary"]
        if not op_summary or not Path(op_summary).exists():
            continue
        fia_row, op_fields = _find_fia_row(op_summary)
        if fia_row is None:
            continue

        for field in op_fields:
            _append_field(fieldnames, field)
        row = dict(fia_row)
        row["FLOPS"], row["MFU*times(us)"] = _extract_perf_from_log(entry["log"])
        row["case_name"] = entry["case_name"] or Path(op_summary).stem
        rows.append(row)

    _append_field(fieldnames, "FLOPS")
    _append_field(fieldnames, "MFU*times(us)")
    _append_field(fieldnames, "case_name")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"归纳完成: {output_path}")
    print(f"归纳行数: {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description="归纳FIA batch_single采集到的FusedInferAttentionScore op_summary数据")
    parser.add_argument("output_dir", help="run_batch_single_prof.sh生成的归档目录")
    parser.add_argument("-o", "--output", default=None, help=f"总表输出路径，默认{DEFAULT_OUTPUT_NAME}")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else output_dir / DEFAULT_OUTPUT_NAME
    aggregate(output_dir, output_path)


if __name__ == "__main__":
    main()
