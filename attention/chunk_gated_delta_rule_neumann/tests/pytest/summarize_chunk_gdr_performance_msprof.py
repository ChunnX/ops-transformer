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


def summarize(root):
    root = Path(root)
    cases_by_name = {case_id(case): case for case in PERFORMANCE_CASES}
    raw_rows = []
    compare_rows = []

    for case_name in sorted(cases_by_name):
        case = cases_by_name[case_name]
        measurements = {}
        for ops, op_type in OP_TYPES.items():
            prof_dir = latest_prof_dir(root / case_name / ops)
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
            compare_row = {
                "case_name": case_name,
                "B": case["batch_size"],
                "S": case["seqlen"],
                "D": case["dk"],
                "Hq": case["nv"],
                "Hkv": case["nk"],
                "normal_us": normal_us,
                "neumann_us": neumann_us,
                "neumann_speedup_vs_normal": normal_us / neumann_us if neumann_us > 0 else "",
                "neumann_over_normal": neumann_us / normal_us if normal_us > 0 else "",
                "faster": "neumann" if neumann_us < normal_us else "normal",
            }
            for ops in OP_TYPES:
                measurement = measurements[ops]
                compare_row[f"{ops}_task_wait_time_us"] = measurement["task_wait_time_us"]
                compare_row[f"{ops}_aicore_time_us"] = measurement["aicore_time_us"]
                for metric_name, _, include_ratio in METRIC_COLUMNS:
                    if metric_name.endswith("_ratio") and not include_ratio:
                        continue
                    compare_row[f"{ops}_{metric_name}"] = measurement[metric_name]
            compare_rows.append(compare_row)

    raw_path = root / "performance_raw.csv"
    compare_path = root / "performance_compare.csv"
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
    args = parser.parse_args()
    raw_path, compare_path, raw_rows, compare_rows = summarize(args.root)
    print(f"[SUMMARY] raw rows={len(raw_rows)} -> {raw_path}")
    print(f"[SUMMARY] compare rows={len(compare_rows)} -> {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
