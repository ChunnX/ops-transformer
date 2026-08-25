#!/usr/bin/env python3
"""Audit Kernel result coverage and summarize the executed TTK timing column."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    if len(args.input) != len(args.result):
        parser.error("--input and --result must be supplied in matching pairs")

    expected = [row for path in args.input for row in read_csv(path)]
    results = [row for path in args.result for row in read_csv(path)]
    expected_names = [row["testcase_name"] for row in expected]
    result_by_name = {row["testcase_name"]: row for row in results}
    missing = sorted(set(expected_names) - set(result_by_name))
    unexpected = sorted(set(result_by_name) - set(expected_names))
    parsed = []
    unmeasured = []
    for name in expected_names:
        row = result_by_name.get(name)
        if row is None:
            continue
        selected = None
        for column in ("bin_perf_us", "dyn_perf_us", "cst_perf_us"):
            text = str(row.get(column, "")).strip()
            match = NUMBER.search(text)
            if match:
                selected = (column, float(match.group(0)))
                break
        if selected is None:
            unmeasured.append(name)
            continue
        parsed.append(
            {
                "testcase_name": name,
                "perf_column": selected[0],
                "device_time_us": selected[1],
                "precision_status": row.get("precision_status"),
                "perf_status": row.get("perf_status"),
            }
        )

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "testcase_name",
                "perf_column",
                "device_time_us",
                "precision_status",
                "perf_status",
            ),
        )
        writer.writeheader()
        writer.writerows(parsed)

    values = [row["device_time_us"] for row in parsed]
    summary = {
        "inputs": args.input,
        "results": args.result,
        "expected_count": len(expected_names),
        "result_count": len(results),
        "measured_count": len(parsed),
        "missing": missing,
        "unexpected": unexpected,
        "unmeasured": unmeasured,
        "device_time_us": {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }
        if values
        else None,
        "passed": len(expected_names) == len(results) == len(parsed)
        and not missing
        and not unexpected
        and not unmeasured,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
