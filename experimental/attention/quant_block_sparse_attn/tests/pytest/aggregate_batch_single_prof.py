#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from pathlib import Path


DEFAULT_OUTPUT_NAME = "quant_block_sparse_attn_summary.csv"


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

    cases = []
    for op_summary in sorted(output_dir.glob("*.csv")):
        if op_summary.name in ("summary_manifest.csv", DEFAULT_OUTPUT_NAME):
            continue
        log = op_summary.with_suffix(".log")
        cases.append(
            {
                "case_name": op_summary.stem,
                "op_summary": str(op_summary),
                "log": str(log) if log.exists() else "",
            }
        )
    return cases


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


def _op_name_column(fieldnames):
    for field in fieldnames:
        if field.strip() == "Op Name":
            return field
    for field in fieldnames:
        normalized = re.sub(r"[\s_]+", "", field).lower()
        if normalized == "opname":
            return field
    return None


def _find_qbsa_row(op_summary_path):
    rows = _read_csv_rows(op_summary_path)
    if not rows:
        return None, []

    fieldnames = list(rows[0].keys())
    op_name_col = _op_name_column(fieldnames)
    if op_name_col is None:
        return None, fieldnames

    exact_matches = [
        row
        for row in rows
        if (row.get(op_name_col) or "").strip() == "QuantBlockSparseAttn"
    ]
    if exact_matches:
        return exact_matches[0], fieldnames

    contains_matches = [
        row
        for row in rows
        if "QuantBlockSparseAttn" in (row.get(op_name_col) or "")
        and "Metadata" not in (row.get(op_name_col) or "")
    ]
    if contains_matches:
        return contains_matches[0], fieldnames
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

        qbsa_row, op_fields = _find_qbsa_row(op_summary)
        if qbsa_row is None:
            continue

        for field in op_fields:
            _append_field(fieldnames, field)

        row = dict(qbsa_row)
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
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="归纳batch_single采集到的QuantBlockSparseAttn op_summary数据"
    )
    parser.add_argument("output_dir", help="run_batch_single_prof.sh生成的归档目录")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=f"总表输出路径，默认写入归档目录下的{DEFAULT_OUTPUT_NAME}",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = Path(args.output) if args.output else output_dir / DEFAULT_OUTPUT_NAME
    aggregate(output_dir, output_path)


if __name__ == "__main__":
    main()
