# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
"""Acceptance gate: reject skipped, missing, duplicated or failed case results."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


def verify_junit(path, expected):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if expected <= 0 or len(cases) != expected:
        raise ValueError(f"Expected {expected} executed tests, got {len(cases)}")
    bad = [c.get("name") for c in cases if any(c.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    if bad:
        raise ValueError(f"Failed/error/skipped cases: {bad}")
    return len(cases)


def verify_ttk(path, source):
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        expected = [r["testcase_name"] for r in csv.DictReader(stream)]
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"testcase_name", "precision_status"}.issubset(reader.fieldnames or []):
            raise ValueError("TTK report requires testcase_name and precision_status columns")
        rows = list(reader)
    actual = [r["testcase_name"] for r in rows]
    if not expected or Counter(actual) != Counter(expected) or len(set(actual)) != len(actual):
        raise ValueError("TTK report has missing, extra or duplicated cases")
    bad = [(r["testcase_name"], r["precision_status"]) for r in rows if r["precision_status"].strip().upper() != "PASS"]
    if bad:
        raise ValueError(f"TTK cases without precision PASS: {bad}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--junit", type=Path)
    inputs.add_argument("--ttk-result", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--cases", type=Path, help="Exact TTK input CSV used for this run")
    args = parser.parse_args()
    if args.junit and args.expected is None:
        parser.error("--junit requires --expected")
    if args.ttk_result and not args.cases:
        parser.error("--ttk-result requires --cases")
    try:
        count = verify_junit(args.junit, args.expected) if args.junit else verify_ttk(args.ttk_result, args.cases)
    except (ValueError, OSError, ET.ParseError, KeyError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(f"PASS: {count} results verified, zero failures/errors/skips/missing cases")


if __name__ == "__main__":
    main()
