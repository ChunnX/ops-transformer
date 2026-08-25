#!/usr/bin/env python3
"""Audit generated whitebox cases against regular/oversized TTK executions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--result", action="append", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if len(args.input) != len(args.result):
        parser.error("--input and --result must be supplied in matching pairs")

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    source_names = [case["id"] for case in source["cases"]]
    input_rows = [row for path in args.input for row in read_csv(path)]
    result_rows = [row for path in args.result for row in read_csv(path)]
    input_names = [row["testcase_name"] for row in input_rows]
    result_names = [row["testcase_name"] for row in result_rows]

    failed = [
        row["testcase_name"]
        for row in result_rows
        if row.get("precision_status") != "PASS"
    ]
    skipped = [
        row["testcase_name"]
        for row in result_rows
        if str(row.get("is_enabled", "True")).strip().lower() in {"false", "0"}
    ]
    duplicate_source = sorted(name for name, count in Counter(source_names).items() if count != 1)
    duplicate_input = sorted(name for name, count in Counter(input_names).items() if count != 1)
    duplicate_result = sorted(name for name, count in Counter(result_names).items() if count != 1)
    missing_input = sorted(set(source_names) - set(input_names))
    unexpected_input = sorted(set(input_names) - set(source_names))
    missing_result = sorted(set(input_names) - set(result_names))
    unexpected_result = sorted(set(result_names) - set(input_names))
    problems = (
        duplicate_source
        + duplicate_input
        + duplicate_result
        + missing_input
        + unexpected_input
        + missing_result
        + unexpected_result
        + failed
        + skipped
    )
    report = {
        "source": str(Path(args.source)),
        "inputs": args.input,
        "results": args.result,
        "generated_count": len(source_names),
        "manifest_count": len(input_names),
        "executed_count": len(result_names),
        "passed_count": len(result_names) - len(failed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "duplicate_source": duplicate_source,
        "duplicate_input": duplicate_input,
        "duplicate_result": duplicate_result,
        "missing_input": missing_input,
        "unexpected_input": unexpected_input,
        "missing_result": missing_result,
        "unexpected_result": unexpected_result,
        "failed": failed,
        "skipped": skipped,
        "passed": not problems
        and len(source_names) == len(input_names) == len(result_names),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
