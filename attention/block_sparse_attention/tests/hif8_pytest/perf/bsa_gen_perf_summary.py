#!/usr/bin/env python3
"""Extract msprof Task Duration and compare HIF8 with high-precision BSA."""

from __future__ import annotations

import csv
import glob
import os
import re
from datetime import datetime


HIF8_DIR = os.environ.get("HIF8_DIR", "./hif8_dir")
BASE_DIR = os.environ.get("BASE_DIR", "./highprec_half")
BASE_DTYPE = os.environ.get("BASE_DTYPE", "half")
OUTPUT = os.environ.get(
    "OUTPUT_CSV",
    os.path.join(os.path.dirname(HIF8_DIR), f"hif8_vs_{BASE_DTYPE}_{datetime.now():%Y%m%d_%H%M%S}.csv"),
)


def duration(case_dir):
    for path in glob.glob(os.path.join(case_dir, "**", "OpBasicInfo.csv"), recursive=True):
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as stream:
            for row in csv.DictReader(stream):
                column = next((key for key in row if re.sub(r"\s+", "", key).lower() == "taskduration(us)"), None)
                if column is None:
                    continue
                value = str(row[column]).strip().replace(",", "")
                try:
                    return float(value)
                except ValueError:
                    continue
    # Local .241 fallback when msprof is unavailable.  This is wall time for
    # the standalone process, not an operator Task Duration measurement.
    fallback = os.path.join(case_dir, "wall_time_us")
    try:
        with open(fallback, encoding="utf-8") as stream:
            return float(stream.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def has_task_csv(case_dir):
    return bool(case_dir and glob.glob(os.path.join(case_dir, "**", "OpBasicInfo.csv"), recursive=True))


def main():
    hif8_cases = {os.path.basename(path): path for path in glob.glob(os.path.join(HIF8_DIR, "case_*"))}
    base_cases = {os.path.basename(path): path for path in glob.glob(os.path.join(BASE_DIR, "case_*"))}
    rows = []
    for name in sorted(set(hif8_cases) | set(base_cases)):
        hif8_us = duration(hif8_cases[name]) if name in hif8_cases else None
        base_us = duration(base_cases[name]) if name in base_cases else None
        speedup = base_us / hif8_us if hif8_us and base_us is not None else None
        profile_kind = "task_duration" if any(
            has_task_csv(path) for path in (hif8_cases.get(name), base_cases.get(name))
        ) else "wall_time_fallback"
        rows.append({
            "case": name,
            "hif8_task_duration_us": hif8_us,
            f"highprec_{BASE_DTYPE}_task_duration_us": base_us,
            "highprec_over_hif8_speedup": speedup,
            "status": "complete" if hif8_us is not None and base_us is not None else "missing_profile",
            "measurement": profile_kind,
        })
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT)), exist_ok=True)
    with open(OUTPUT, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {os.path.abspath(OUTPUT)}")
    print(f"cases: {len(rows)} complete: {sum(row['status'] == 'complete' for row in rows)}")


if __name__ == "__main__":
    main()
