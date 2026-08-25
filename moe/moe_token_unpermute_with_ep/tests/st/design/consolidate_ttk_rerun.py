#!/usr/bin/env python3
"""Replace infrastructure-failed TTK rows with explicit PASS reruns."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    base_rows = read_rows(args.base)
    rerun_rows = read_rows(args.rerun)
    base_counts = Counter(row["testcase_name"] for row in base_rows)
    rerun_counts = Counter(row["testcase_name"] for row in rerun_rows)
    duplicate_base = sorted(name for name, count in base_counts.items() if count != 1)
    duplicate_rerun = sorted(name for name, count in rerun_counts.items() if count != 1)
    failed_base = sorted(
        row["testcase_name"] for row in base_rows if row.get("precision_status") != "PASS"
    )
    rerun_names = sorted(rerun_counts)
    rerun_not_pass = sorted(
        row["testcase_name"] for row in rerun_rows if row.get("precision_status") != "PASS"
    )
    errors = []
    if duplicate_base:
        errors.append("base contains duplicate testcase names")
    if duplicate_rerun:
        errors.append("rerun contains duplicate testcase names")
    if failed_base != rerun_names:
        errors.append("rerun testcase set does not exactly match base non-PASS testcase set")
    if rerun_not_pass:
        errors.append("one or more rerun rows are not PASS")
    if errors:
        raise SystemExit("; ".join(errors))

    replacements = {row["testcase_name"]: row for row in rerun_rows}
    consolidated = [replacements.get(row["testcase_name"], row) for row in base_rows]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(consolidated[0]))
        writer.writeheader()
        writer.writerows(consolidated)

    report = {
        "base": str(Path(args.base)),
        "base_sha256": sha256(args.base),
        "rerun": str(Path(args.rerun)),
        "rerun_sha256": sha256(args.rerun),
        "output": str(output),
        "base_count": len(base_rows),
        "rerun_count": len(rerun_rows),
        "output_count": len(consolidated),
        "replaced_testcases": rerun_names,
        "base_non_pass_count": len(failed_base),
        "rerun_pass_count": len(rerun_rows) - len(rerun_not_pass),
        "output_pass_count": sum(row.get("precision_status") == "PASS" for row in consolidated),
        "output_non_pass_count": sum(row.get("precision_status") != "PASS" for row in consolidated),
        "passed": not errors and all(row.get("precision_status") == "PASS" for row in consolidated),
    }
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
