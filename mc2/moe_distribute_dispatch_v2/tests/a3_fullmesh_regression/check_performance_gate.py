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
"""Enforce the EP16 latency limit for every expected case and paired repeat."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def load_details(path):
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if data.get("format") != "partitioned-records-v1":
        raise ValueError("Unsupported performance details format")
    rows = []
    for part in data["parts"]:
        target = (path.parent / part["path"]).resolve()
        if target.parent != path.parent.resolve():
            raise ValueError("Performance part must be in the index directory")
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != part["sha256"]:
            raise ValueError(f"Performance part hash mismatch: {target.name}")
        records = json.loads(raw)
        if not isinstance(records, list) or len(records) != part["records"]:
            raise ValueError("Performance part record count mismatch")
        rows.extend(records)
    if len(rows) != data["records"]:
        raise ValueError("Performance index record count mismatch")
    return rows


def run_means(variant):
    runs = variant["runs"]
    if not runs or len(runs) != variant["run_count"]:
        raise ValueError("Missing or inconsistent repeat count")
    means = []
    for run in runs:
        ranks = {int(k): v for k, v in run["rank_means_us"].items()}
        samples = {int(k): v for k, v in run["samples_per_rank"].items()}
        if set(ranks) != set(range(16)) or set(samples) != set(range(16)):
            raise ValueError("Expected complete ranks 0..15")
        if any(n != 100 for n in samples.values()) or run["retained_per_rank"] != 50:
            raise ValueError("Expected 100 calls and the last 50 retained per rank")
        if any(not math.isfinite(t) or t <= 0 for t in ranks.values()):
            raise ValueError("Nonpositive or nonfinite rank duration")
        mean = statistics.mean(ranks.values())
        if not math.isclose(mean, run["mean_us"], rel_tol=1e-12):
            raise ValueError("Run mean disagrees with rank evidence")
        means.append(mean)
    if not math.isclose(statistics.mean(means), variant["mean_us"], rel_tol=1e-12):
        raise ValueError("Variant mean disagrees with repeat evidence")
    return means


def evaluate(details, expected, limit=1.0):
    if not math.isfinite(limit) or limit < 0 or limit > 1.0:
        raise ValueError("Regression limit must be between 0 and 1 percent")
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected inventory must be nonempty and unique")
    names = [row["case_name"] for row in details]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError("Missing, extra or duplicate cases versus expected inventory")
    rows = []
    for row in details:
        baseline = run_means(row["baseline"])
        candidate = run_means(row["optimized"])
        if len(baseline) != len(candidate):
            raise ValueError(f"Unequal paired repeats: {row['case_name']}")
        aggregate = statistics.mean(candidate) / statistics.mean(baseline)
        ratios = [b / a for a, b in zip(baseline, candidate)]
        # Compare unrounded ratios. All recorded pairs must meet the limit;
        # averaging another shape or a faster repeat cannot hide a regression.
        passed = max(aggregate, *ratios) <= 1 + limit / 100
        rows.append(
            {
                "case_name": row["case_name"],
                "operation": row["operation"],
                "paired_repeats": len(ratios),
                "aggregate_regression_pct": 100 * (aggregate - 1),
                "paired_regression_pct": [100 * (x - 1) for x in ratios],
                "passed": passed,
            }
        )
    return {
        "status": "passed" if all(row["passed"] for row in rows) else "failed",
        "max_allowed_regression_pct": limit,
        "rule": "Every case aggregate and every chronological paired repeat must meet the limit",
        "case_count": len(rows),
        "failed_cases": [row["case_name"] for row in rows if not row["passed"]],
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--suites", nargs="+", default=["dispatch_performance", "combine_performance"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-regression-pct", type=float, default=1.0)
    args = parser.parse_args()
    try:
        inventory = json.loads(args.cases.read_text())
        expected = [c["case_name"] for suite in args.suites for c in inventory[suite]]
        report = evaluate(load_details(args.details), expected, args.max_regression_pct)
        code = 0 if report["status"] == "passed" else 1
    except (ValueError, KeyError, TypeError, OSError) as error:
        report = {"status": "invalid_evidence", "error": str(error)}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
