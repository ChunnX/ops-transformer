"""Pair msprof operator records with single-case execution records."""

import csv
import math
import statistics
from pathlib import Path

from common.fd_performance_runtime import RESULT_FIELDS, STATUS_OK

TIMING_FIELDS = ("metadata_median_us", "median_us", "p90_us", "tiling_key")
REPORT_FIELDS = RESULT_FIELDS + TIMING_FIELDS
OP_NAMES = {"smla": "sparseflashmla", "qsmla": "quantsparseflashmla",
            "mqsmla": "mixedquantsparseflashmla"}


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def collect_pairs(records, operator):
    base = OP_NAMES[operator]
    relevant = [r for r in records if r.get("OP Type", r.get("Op Type", "")).lower()
                in (base, base + "metadata")]
    if relevant and all(r.get("Task Start Time(us)") for r in relevant):
        relevant.sort(key=lambda r: float(r["Task Start Time(us)"]))
    pairs, pending = [], None
    for row in relevant:
        duration = float(row["Task Duration(us)"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid profiler duration: {duration}")
        if row.get("OP Type", row.get("Op Type", "")).lower().endswith("metadata"):
            pending = duration
        else:
            if pending is None:
                raise ValueError("Attention timing has no preceding metadata timing")
            pairs.append((pending, duration, row.get("Tiling Key", row.get("TilingKey", ""))))
            pending = None
    return pairs


def make_report(rows, records, operator):
    pairs = collect_pairs(records, operator)
    expected = sum(int(row["completed_calls"]) for row in rows)
    if len(pairs) != expected:
        raise ValueError(f"Profiler/execution mismatch: {len(pairs)} pairs, {expected} completed calls")
    cursor, report = 0, []
    for original in rows:
        row = dict(original)
        count = int(row["completed_calls"])
        samples = pairs[cursor:cursor + count]
        cursor += count
        row.update({key: "" for key in TIMING_FIELDS})
        if row["status"] == STATUS_OK:
            warmup, repeat = int(row["warmup"]), int(row["repeat"])
            if count != warmup + repeat:
                raise ValueError("Successful case has incomplete timing samples")
            samples = samples[warmup:]
            row.update(metadata_median_us=statistics.median(s[0] for s in samples),
                       median_us=statistics.median(s[1] for s in samples),
                       p90_us=percentile([s[1] for s in samples], .9),
                       tiling_key=";".join(sorted({s[2] for s in samples if s[2]})))
        report.append(row)
    return report


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def append_report(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REPORT_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
