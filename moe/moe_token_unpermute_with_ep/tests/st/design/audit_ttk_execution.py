#!/usr/bin/env python3
"""Audit that a positive TTK run executed every listed case exactly once."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--layers", default="L0,L1")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    expected_rows = read_rows(args.input)
    result_rows = read_rows(args.result)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    layers = [layer.strip() for layer in args.layers.split(",") if layer.strip()]
    manifest_names = [
        case["testcase_name"] for layer in layers for case in manifest["layers"][layer]
    ]
    expected_names = [row["testcase_name"] for row in expected_rows]
    result_names = [row["testcase_name"] for row in result_rows]
    manifest_counts = Counter(manifest_names)
    expected_counts = Counter(expected_names)
    result_counts = Counter(result_names)
    duplicate_manifest = sorted(name for name, count in manifest_counts.items() if count != 1)
    duplicate_expected = sorted(name for name, count in expected_counts.items() if count != 1)
    duplicate_results = sorted(name for name, count in result_counts.items() if count != 1)
    missing = sorted(set(expected_names) - set(result_names))
    unexpected = sorted(set(result_names) - set(expected_names))
    missing_from_manifest = sorted(set(expected_names) - set(manifest_names))
    missing_from_input = sorted(set(manifest_names) - set(expected_names))
    failed = sorted(
        row["testcase_name"] for row in result_rows if row.get("precision_status") != "PASS"
    )
    skipped = sorted(
        row["testcase_name"]
        for row in result_rows
        if str(row.get("is_enabled", "")).strip().lower() in {"false", "0"}
    )
    report = {
        "input": str(Path(args.input)),
        "result": str(Path(args.result)),
        "generated_count": len(expected_rows),
        "manifest_count": len(manifest_names),
        "executed_count": len(result_rows),
        "passed_count": len(result_rows) - len(failed),
        "failed_count": len(failed),
        "skipped_count": len(skipped),
        "duplicate_manifest": duplicate_manifest,
        "duplicate_expected": duplicate_expected,
        "duplicate_results": duplicate_results,
        "missing": missing,
        "unexpected": unexpected,
        "missing_from_manifest": missing_from_manifest,
        "missing_from_input": missing_from_input,
        "failed": failed,
        "skipped": skipped,
    }
    report["passed"] = not any(
        (
            duplicate_manifest,
            duplicate_expected,
            duplicate_results,
            missing,
            unexpected,
            missing_from_manifest,
            missing_from_input,
            failed,
            skipped,
            len(expected_rows) != len(manifest_names),
            len(expected_rows) != len(result_rows),
        )
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
