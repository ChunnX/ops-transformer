#!/usr/bin/env python3
"""
Summarize ChunkGDR / ChunkGDRNeumann msprof outputs into CSV tables.
"""
import argparse
import csv
from pathlib import Path

from compare_chunk_gdr_performance_paramset import PERFORMANCE_CASES, case_id

OP_TYPES = {
    "normal": "ChunkGatedDeltaRule",
    "neumann": "ChunkGatedDeltaRuleNeumann",
}

METRIC_COLUMNS = [
    ("aic_mac_time_us", "aic_mac_time(us)", True),
    ("aic_mac_ratio", "aic_mac_ratio", True),
    ("aic_scalar_time_us", "aic_scalar_time(us)", False),
    ("aic_mte1_time_us", "aic_mte1_time(us)", False),
    ("aic_mte2_time_us", "aic_mte2_time(us)", False),
    ("aic_fixpipe_time_us", "aic_fixpipe_time(us)", False),
    ("aic_icache_miss_rate", "aic_icache_miss_rate", False),
    ("aiv_time_us", "aiv_time(us)", False),
    ("aiv_total_cycles", "aiv_total_cycles", False),
    ("aiv_vec_time_us", "aiv_vec_time(us)", False),
    ("aiv_vec_ratio", "aiv_vec_ratio", True),
    ("aiv_scalar_time_us", "aiv_scalar_time(us)", False),
    ("aiv_mte2_time_us", "aiv_mte2_time(us)", False),
    ("aiv_mte3_time_us", "aiv_mte3_time(us)", False),
    ("aiv_icache_miss_rate", "aiv_icache_miss_rate", False),
]

COMPARE_VALUE_COLUMNS = [
    "task_duration_us",
    "task_wait_time_us",
    "aicore_time_us",
    *[metric_name for metric_name, _, _ in METRIC_COLUMNS],
]


def to_float(value):
    value = str(value).strip()
    return float(value) if value else 0.0


def to_int(value):
    return int(str(value).strip())


def latest_prof_dir(op_dir):
    prof_dirs = sorted(op_dir.glob("PROF_*"), key=lambda path: path.stat().st_mtime)
    return prof_dirs[-1] if prof_dirs else None


def find_op_summary(prof_dir):
    files = sorted((prof_dir / "mindstudio_profiler_output").glob("op_summary_*.csv"))
    return files[-1] if files else None


def as_root_list(root):
    if isinstance(root, (str, Path)):
        return [Path(root)]
    return [Path(item) for item in root]


def find_prof_dir(roots, case_name, ops):
    for root in roots:
        prof_dir = latest_prof_dir(root / case_name / ops)
        if prof_dir is not None:
            return prof_dir
    return None


def parse_target_rows(prof_dir, op_type):
    summary_path = find_op_summary(prof_dir)
    if summary_path is None:
        return []
    rows = []
    with summary_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("OP Type") != op_type:
                continue
            parsed = {
                "prof_dir": str(prof_dir),
                "summary_path": str(summary_path),
                "task_id": to_int(row["Task ID"]),
                "op_name": row["Op Name"],
                "op_type": row["OP Type"],
                "task_duration_us": to_float(row["Task Duration(us)"]),
                "task_wait_time_us": to_float(row["Task Wait Time(us)"]),
                "aicore_time_us": to_float(row["aicore_time(us)"]),
            }
            for metric_name, msprof_name, _ in METRIC_COLUMNS:
                parsed[metric_name] = to_float(row.get(msprof_name, "0"))
            rows.append(parsed)
    rows.sort(key=lambda row: row["task_id"])
    return rows


def ratio_value(numerator, denominator):
    return numerator / denominator if denominator else ""


def make_compare_op_row(case_name, case, ops, measurement, faster):
    row = {
        "case_name": case_name,
        "B": case["batch_size"],
        "S": case["seqlen"],
        "D": case["dk"],
        "Hq": case["nv"],
        "Hkv": case["nk"],
        "row_type": "measurement",
        "ops": ops,
        "faster": faster,
    }
    for column in COMPARE_VALUE_COLUMNS:
        row[column] = measurement[column]
    return row


def make_compare_ratio_row(case_name, case, normal, neumann, faster):
    row = {
        "case_name": case_name,
        "B": case["batch_size"],
        "S": case["seqlen"],
        "D": case["dk"],
        "Hq": case["nv"],
        "Hkv": case["nk"],
        "row_type": "ratio_neumann_over_normal",
        "ops": "neumann/normal",
        "faster": faster,
    }
    for column in COMPARE_VALUE_COLUMNS:
        row[column] = ratio_value(neumann[column], normal[column])
    return row


def summarize(root, extra_roots=None, output_root=None):
    roots = as_root_list(root)
    if extra_roots:
        roots.extend(as_root_list(extra_roots))
    output_root = Path(output_root) if output_root else roots[0]
    cases_by_name = {case_id(case): case for case in PERFORMANCE_CASES}
    raw_rows = []
    compare_rows = []

    for case in PERFORMANCE_CASES:
        case_name = case_id(case)
        measurements = {}
        for ops, op_type in OP_TYPES.items():
            prof_dir = find_prof_dir(roots, case_name, ops)
            if prof_dir is None:
                continue
            target_rows = parse_target_rows(prof_dir, op_type)
            if not target_rows:
                continue
            labeled_rows = [("warmup", target_rows[0]), ("measurement", target_rows[-1])]
            for occurrence, row in labeled_rows:
                raw_rows.append(
                    {
                        "case_name": case_name,
                        "ops": ops,
                        "occurrence": occurrence,
                        "B": case["batch_size"],
                        "S": case["seqlen"],
                        "D": case["dk"],
                        "Hq": case["nv"],
                        "Hkv": case["nk"],
                        **row,
                    }
                )
            measurements[ops] = target_rows[-1]

        if "normal" in measurements and "neumann" in measurements:
            normal_us = measurements["normal"]["task_duration_us"]
            neumann_us = measurements["neumann"]["task_duration_us"]
            faster = "neumann" if neumann_us < normal_us else "normal"
            compare_rows.append(make_compare_op_row(case_name, case, "normal", measurements["normal"], faster))
            compare_rows.append(make_compare_op_row(case_name, case, "neumann", measurements["neumann"], faster))
            compare_rows.append(
                make_compare_ratio_row(case_name, case, measurements["normal"], measurements["neumann"], faster)
            )

    raw_path = output_root / "performance_raw.csv"
    compare_path = output_root / "performance_compare.csv"
    write_csv(raw_path, raw_rows)
    write_csv(compare_path, compare_rows)
    return raw_path, compare_path, raw_rows, compare_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize ChunkGDR msprof outputs.")
    parser.add_argument("--root", required=True, help="msprof output root created by run_chunk_gdr_performance_msprof.py")
    parser.add_argument(
        "--extra-root",
        action="append",
        default=[],
        help="Additional msprof output root used to fill cases missing from --root. Can be repeated.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for generated performance_raw.csv and performance_compare.csv. Defaults to --root.",
    )
    args = parser.parse_args()
    raw_path, compare_path, raw_rows, compare_rows = summarize(args.root, args.extra_root, args.output_root)
    print(f"[SUMMARY] raw rows={len(raw_rows)} -> {raw_path}")
    print(f"[SUMMARY] compare rows={len(compare_rows)} -> {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
