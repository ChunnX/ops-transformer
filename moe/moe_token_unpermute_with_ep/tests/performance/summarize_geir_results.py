#!/usr/bin/env python3
"""Audit GEIR result counts and summarize end-to-end timings."""

import argparse
import csv
import json
import statistics
from pathlib import Path


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(values):
    return {
        "count": len(values),
        "min_us": min(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p95_us": percentile(values, 0.95),
        "max_us": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-result", required=True)
    parser.add_argument("--dynamic-result", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    static_rows = read_rows(args.static_result)
    dynamic_rows = read_rows(args.dynamic_result)
    static_values = [float(row["end_to_end_us"]) for row in static_rows]
    dynamic_values = [float(row["end_to_end_us"]) for row in dynamic_rows]

    first_compile = []
    steady_state = []
    seen_scenarios = set()
    for row, value in zip(dynamic_rows, dynamic_values):
        scenario = row["scenario"]
        if scenario in seen_scenarios:
            steady_state.append(value)
        else:
            seen_scenarios.add(scenario)
            first_compile.append(value)

    all_pass = all(
        row.get("status") == "PASS" for row in static_rows + dynamic_rows
    )
    summary = {
        "inputs": {
            "static": args.static_result,
            "dynamic": args.dynamic_result,
        },
        "static": summarize(static_values) if static_values else None,
        "dynamic_all": summarize(dynamic_values) if dynamic_values else None,
        "dynamic_first_compile_per_scenario_us": first_compile,
        "dynamic_steady_state": summarize(steady_state) if steady_state else None,
        "expected_counts": {"static": 14, "dynamic": 6},
        "passed": len(static_rows) == 14 and len(dynamic_rows) == 6 and all_pass,
        "note": (
            "Static cases and the first case of each dynamic scenario include GE graph "
            "compile/load time; dynamic_steady_state reports subsequent RunGraph calls."
        ),
    }
    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
