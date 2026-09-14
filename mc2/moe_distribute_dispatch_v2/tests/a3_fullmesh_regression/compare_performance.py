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
"""Compare identical EP16 cases using only the requested device kernel rows."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def normalize(value):
    return value.lower().replace("_", "")


def collect(root, case_name, op, keep):
    if keep <= 0:
        raise ValueError("--keep must be positive")
    candidates = list(root.rglob(case_name))
    candidates = [
        p for p in candidates if p.is_dir() and list(p.rglob("op_summary*.csv"))
    ]
    if not candidates:
        raise ValueError(f"No profiler data for {root}/{case_name}")
    runs = [_collect_run(path, case_name, op, keep) for path in sorted(candidates)]
    return {
        "mean_us": statistics.mean(run["mean_us"] for run in runs),
        "rank_means_us": {
            rank: statistics.mean(run["rank_means_us"][rank] for run in runs)
            for rank in range(16)
        },
        "run_count": len(runs),
        "runs": runs,
        "retained_per_rank_per_run": keep,
        "raw_files": [path for run in runs for path in run["raw_files"]],
    }


def _collect_run(case_root, case_name, op, keep):
    summary_path = case_root.parent / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    reports = [c for c in summary["cases"] if c["case_name"] == case_name]
    if len(reports) != 1 or not reports[0]["report"]["passed"]:
        raise ValueError(f"Missing or failed accuracy report: {case_name}")
    ranks = {}
    paths = []
    for path in sorted(case_root.rglob("op_summary*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                identity = " ".join(
                    row.get(key, "") for key in ("Op Name", "Op Type", "OP Type")
                )
                if normalize(op) not in normalize(identity):
                    continue
                rank = int(row["Device_id"])
                duration = float(row["Task Duration(us)"])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid duration in {path}: {duration}")
                ranks.setdefault(rank, []).append(duration)
        paths.append(str(path))
    if set(ranks) != set(range(16)):
        raise ValueError(f"Expected ranks 0..15 for {case_name}; got {sorted(ranks)}")
    if any(len(samples) < keep for samples in ranks.values()):
        raise ValueError(f"Fewer than {keep} samples: {case_name}")
    means = {rank: statistics.mean(samples[-keep:]) for rank, samples in ranks.items()}
    return {
        "run_id": case_root.parent.parent.name,
        "mean_us": statistics.mean(means.values()),
        "rank_means_us": means,
        "samples_per_rank": {rank: len(samples) for rank, samples in ranks.items()},
        "retained_per_rank": keep,
        "raw_files": paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep", type=int, default=50)
    args = parser.parse_args()
    suites = json.loads(args.cases.read_text())
    rows, details = [], []
    for operation in ("dispatch", "combine"):
        op = "MoeDistribute" + operation.title() + "V2"
        for case in suites[operation + "performance"]:
            baseline = collect(args.baseline, case["case_name"], op, args.keep)
            optimized = collect(args.optimized, case["case_name"], op, args.keep)
            if baseline["run_count"] != optimized["run_count"]:
                raise ValueError(f"Unmatched repeat counts: {case['case_name']}")
            row = {
                "operation": operation,
                "case_name": case["case_name"],
                "tokens_per_rank": case["bs"],
                "hidden": case["h"],
                "topk": case["k"],
                "quant_mode": case["quant_mode"]
                if operation == "dispatch"
                else case["comm_quant_mode"],
                "runs_per_variant": baseline["run_count"],
                "baseline_us": baseline["mean_us"],
                "optimized_us": optimized["mean_us"],
                "reduction_pct": 100 * (1 - optimized["mean_us"] / baseline["mean_us"]),
            }
            rows.append(row)
            details.append({**row, "baseline": baseline, "optimized": optimized})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "performance_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "performance_details.json").write_text(
        json.dumps(details, indent=2) + "\n"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
