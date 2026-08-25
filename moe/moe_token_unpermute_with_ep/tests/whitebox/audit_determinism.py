#!/usr/bin/env python3
"""Audit two full TTK executions of one deterministic Kernel case."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_single_result(path: Path, testcase_name: str) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0].get("testcase_name") != testcase_name:
        raise AssertionError(
            f"{path} must contain exactly {testcase_name}, got "
            f"{[row.get('testcase_name') for row in rows]}"
        )
    return rows[0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testcase", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--process-count", type=int, required=True)
    parser.add_argument("--run1-result", type=Path, required=True)
    parser.add_argument("--run2-result", type=Path, required=True)
    parser.add_argument("--run1-output", type=Path, required=True)
    parser.add_argument("--run2-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    run1 = read_single_result(args.run1_result, args.testcase)
    run2 = read_single_result(args.run2_result, args.testcase)
    outputs = (args.run1_output, args.run2_output)
    if any(not path.is_file() or path.stat().st_size == 0 for path in outputs):
        raise AssertionError("both raw output binaries must exist and be non-empty")

    hashes = tuple(digest(path) for path in outputs)
    statuses = (run1.get("precision_status"), run2.get("precision_status"))
    passed = statuses == ("PASS", "PASS") and hashes[0] == hashes[1]
    report = {
        "testcase_name": args.testcase,
        "seed": args.seed,
        "process_count": args.process_count,
        "run1_precision_status": statuses[0],
        "run2_precision_status": statuses[1],
        "run1_output_size_bytes": outputs[0].stat().st_size,
        "run2_output_size_bytes": outputs[1].stat().st_size,
        "run1_sha256": hashes[0],
        "run2_sha256": hashes[1],
        "bitwise_identical": hashes[0] == hashes[1],
        "raw_outputs": [str(path.resolve()) for path in outputs],
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
