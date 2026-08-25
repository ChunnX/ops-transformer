#!/usr/bin/env python3
"""Extract per-case ACLNN and AiCore timing from a TTK single-log run."""

import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


PROFILE = re.compile(
    r"^(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ .*"
    r"\[\d+ (?P<case>.+) OnProfiling\].*profiling\.py:450"
)
TIMING = re.compile(
    r"^\s*(?P<name>aclnnMoeTokenUnpermuteWithEp|MoeTokenUnpermuteWithEp_AiCore)\s+"
    r"(?P<total>[0-9.]+)\s+(?P<avg>[0-9.]+)\s+(?P<min>[0-9.]+)\s+"
    r"(?P<max>[0-9.]+)\s+(?P<calls>\d+)\s*$"
)


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summarize(values):
    return {
        "min_us": min(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p95_us": percentile(values, 0.95),
        "max_us": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log")
    parser.add_argument("--log-dir")
    parser.add_argument("--log-pattern", default="ttk-info-*.log")
    parser.add_argument("--since", required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--expected-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    if not args.log and not args.log_dir:
        parser.error("one of --log or --log-dir is required")

    since = datetime.strptime(args.since, "%Y-%m-%d %H:%M:%S")
    expected = list(csv.DictReader(Path(args.expected_csv).open(encoding="utf-8-sig", newline="")))
    expected_names = [row["testcase_name"] for row in expected]

    sources = []
    if args.log:
        sources.append(Path(args.log))
    if args.log_dir:
        sources.extend(sorted(Path(args.log_dir).glob(args.log_pattern)))

    records = {}
    for source in sources:
        current = None
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            match = PROFILE.match(line)
            if match:
                timestamp = datetime.strptime(match.group("time"), "%Y-%m-%d %H:%M:%S")
                current = match.group("case") if timestamp >= since else None
                if current is not None:
                    records[current] = {"testcase_name": current}
                continue
            if current is None:
                continue
            timing = TIMING.match(line)
            if timing is None:
                continue
            prefix = "api" if timing.group("name").startswith("aclnn") else "kernel"
            for field in ("total", "avg", "min", "max"):
                records[current][f"{prefix}_{field}_us"] = float(timing.group(field))
            records[current][f"{prefix}_calls"] = int(timing.group("calls"))

    complete = {
        name: row
        for name, row in records.items()
        if all(f"{prefix}_avg_us" in row for prefix in ("api", "kernel"))
    }
    missing = sorted(set(expected_names) - set(complete))
    unexpected = sorted(set(complete) - set(expected_names))
    ordered = [complete[name] for name in expected_names if name in complete]
    fields = [
        "testcase_name",
        "api_total_us",
        "api_avg_us",
        "api_min_us",
        "api_max_us",
        "api_calls",
        "kernel_total_us",
        "kernel_avg_us",
        "kernel_min_us",
        "kernel_max_us",
        "kernel_calls",
    ]
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)

    summary = {
        "source_log": str(Path(args.log)) if args.log else None,
        "source_log_dir": str(Path(args.log_dir)) if args.log_dir else None,
        "source_log_pattern": args.log_pattern if args.log_dir else None,
        "source_log_count": len(sources),
        "since": args.since,
        "expected_count": len(expected_names),
        "parsed_count": len(ordered),
        "missing": missing,
        "unexpected": unexpected,
        "api_avg": summarize([row["api_avg_us"] for row in ordered]) if ordered else None,
        "kernel_avg": summarize([row["kernel_avg_us"] for row in ordered]) if ordered else None,
        "passed": len(ordered) == len(expected_names) and not missing and not unexpected,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
