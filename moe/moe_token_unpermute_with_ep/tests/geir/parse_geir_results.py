#!/usr/bin/env python3
"""Convert the GEIR example's machine-readable lines into auditable results."""

import argparse
import csv
import json
import re
from pathlib import Path


PERF_RE = re.compile(r"^GEIR_PERF case=(\S+) end_to_end_us=([0-9.eE+-]+)$")
SCENARIO_RE = re.compile(r"^Scenario (\S+), declared shape (.+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("static", "dynamic"), required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--process-return-code", type=int, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    log_path = Path(args.log)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    scenario = "static"
    declared_shape = "concrete"
    rows = []
    for line in lines:
        scenario_match = SCENARIO_RE.match(line.strip())
        if scenario_match:
            scenario, declared_shape = scenario_match.groups()
            continue
        perf_match = PERF_RE.match(line.strip())
        if perf_match:
            testcase_name, duration = perf_match.groups()
            rows.append(
                {
                    "testcase_name": testcase_name,
                    "route": args.route,
                    "scenario": scenario,
                    "declared_shape": declared_shape,
                    "end_to_end_us": duration,
                    "status": "PASS",
                }
            )

    error_lines = [line for line in lines if "ERROR" in line or "FAILED" in line]
    passed = (
        args.process_return_code == 0
        and len(rows) == args.expected_count
        and not error_lines
    )
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "testcase_name",
                "route",
                "scenario",
                "declared_shape",
                "end_to_end_us",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "route": args.route,
        "raw_log": str(log_path),
        "process_return_code": args.process_return_code,
        "expected_count": args.expected_count,
        "executed_count": len(rows),
        "passed_count": len(rows),
        "failed_count": 0 if passed else max(args.expected_count - len(rows), 1),
        "skipped_count": 0,
        "error_lines": error_lines,
        "passed": passed,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
